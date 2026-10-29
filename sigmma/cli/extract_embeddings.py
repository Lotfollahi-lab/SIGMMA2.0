"""Extract per-FOV SIGMMA image + spatial embeddings (all 3 scales) for a
retrieval config and save to npz, so we can measure representational COLLAPSE of
the SIGMMA output directly (vs the NEMO input). Mirrors benchmark_retrieval's
section_split dataloader + forward exactly; just dumps the embeddings instead of
only the cosine score matrix.

Usage:
  sigmma-extract-embeddings -config_file CFG -out_dir DIR
Outputs DIR/embeddings.npz with keys image_emb_s{0,1,2}, spatial_emb_s{0,1,2}
(raw, pre-normalization, float32, N x 512) + section_id.
"""
import os, argparse, numpy as np, torch, yaml
from ml_collections.config_dict import ConfigDict
from torch.utils.data import DataLoader, SequentialSampler
from sigmma.utils import load_dataset, load_model
from sigmma.datasets import multimodal_collate


@torch.no_grad()
def run(args):
    cfg = ConfigDict(yaml.load(open(args.config_file), yaml.FullLoader))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    mname = cfg.model.get("name")

    ds = load_dataset(cfg.dataset.query_dataset, is_train=False, model_name=mname)
    dl = DataLoader(ds, batch_size=cfg.test.batch_size, sampler=SequentialSampler(ds),
                    num_workers=cfg.test.num_workers, collate_fn=multimodal_collate)

    model = load_model(cfg, pretrained_image_encoder=cfg.model.get("pretrained_image_encoder", None),
                       pretrained_ckpt_path=cfg.model.get("checkpoint_path", None), train=False)
    model.eval().to(device)

    nscale = int(cfg.model.num_resolutions)
    img = {s: [] for s in range(nscale)}
    spa = {s: [] for s in range(nscale)}
    for batch in dl:
        if not isinstance(batch.get("image"), torch.Tensor) or batch["image"].numel() == 0:
            continue
        image = batch["image"].to(device)
        spatial = batch["spatial_graph"].to(device)
        with torch.amp.autocast(device.type, enabled=True):
            out = model(image, spatial)
        ie, se = out["image_embedding"], out["spatial_embedding"]   # lists over scales
        for s in range(nscale):
            img[s].append(ie[s].float().detach().cpu().numpy())
            spa[s].append(se[s].float().detach().cpu().numpy())

    save = {}
    for s in range(nscale):
        save[f"image_emb_s{s}"] = np.concatenate(img[s], 0)
        save[f"spatial_emb_s{s}"] = np.concatenate(spa[s], 0)
    np.savez_compressed(os.path.join(args.out_dir, "embeddings.npz"), **save)
    n = save["image_emb_s0"].shape[0]
    print(f"saved {n} FOV embeddings x {nscale} scales -> {args.out_dir}/embeddings.npz")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("-config_file")
    p.add_argument("-out_dir")
    run(p.parse_args(argv))


if __name__ == "__main__":
    main()
