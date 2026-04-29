import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch
import torch.nn.functional as F
import dgl
from graphviz import Digraph

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def plot_edges(adata,
               key='sigmma_connectivities',
               max_edges=100000):
    """
    
    """
    adata = adata.copy()
    A = adata.obsp[key]
    A = A.tocoo()

    # 좌표 선택
    if "spatial" in adata.obsm:
        XY = adata.obsm["spatial"]
    else:
        XY = adata.obs[["x_centroid", "y_centroid"]].to_numpy()

    x = XY[:, 0]
    y = XY[:, 1]

    # (선택) 중복 edge 줄이기: 무방향 그래프면 상삼각만 사용
    # A = sp.triu(A, k=1).tocoo()

    # (선택) weight threshold (값이 있을 때만)
    min_w = None  # 예: 0.1
    mask = np.ones(A.nnz, dtype=bool)
    if min_w is not None:
        mask &= (A.data >= min_w)

    rows = A.row[mask]
    cols = A.col[mask]
    w = A.data[mask]

    # 너무 많으면 랜덤 샘플링
    max_edges = max_edges 
    m = len(rows)
    if m > max_edges:
        rng = np.random.default_rng(0)
        idx = rng.choice(m, size=max_edges, replace=False)
        rows, cols, w = rows[idx], cols[idx], w[idx]

    # line segments 생성 (vectorized)
    segs = np.stack(
        [np.column_stack([x[rows], y[rows]]),
         np.column_stack([x[cols], y[cols]])],
        axis=1
    )

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.scatter(x, y, s=2, alpha=0.6)

    lc = LineCollection(segs, linewidths=0.2, alpha=1, color='red')  # alpha 낮게
    ax.add_collection(lc)

    ax.set_aspect("equal")
    ax.set_title("sigmma_connectivities (sampled edges)")
    ax.invert_yaxis()
    plt.show()

def line_plot(dict_metric, metric):
    """
    """
    # x-axis
    n_neighbors_list = sorted(dict_metric.keys())

    # y-axis values for each method
    sigmma_micro = [dict_metric[n]['sigmma_micro'] for n in n_neighbors_list]
    sigmma_meso  = [dict_metric[n]['sigmma_meso']  for n in n_neighbors_list]
    sigmma_macro = [dict_metric[n]['sigmma_macro'] for n in n_neighbors_list]
    nemo         = [dict_metric[n]['nemo']         for n in n_neighbors_list]

    # plot
    plt.figure()
    plt.plot(n_neighbors_list, sigmma_micro, label='sigmma_micro', marker='o')
    plt.plot(n_neighbors_list, sigmma_meso,  label='sigmma_meso', marker='o')
    plt.plot(n_neighbors_list, sigmma_macro, label='sigmma_macro', marker='o')
    plt.plot(n_neighbors_list, nemo,         label='nemo', marker='o')

    plt.xlabel('n_neighbors')
    plt.ylabel(f'{metric} score')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def build_index(items):
    by_id = {x["id"]: x for x in items}
    parent = {x["id"]: x["pId"] for x in items}
    names = {x["id"]: x.get("name", x["id"]) for x in items}
    return by_id, parent, names

def safe_id(raw_id: str) -> str:
    return raw_id.replace(":", "_")

def visualize_tree_graphviz(items, out_path="tree", fmt="png"):
    by_id, parent_map, names = build_index(items)

    dot = Digraph("tree")
    dot.attr(rankdir="TB")

    super_root_raw = "ROOTS"
    super_root = safe_id(super_root_raw)
    dot.node(super_root, super_root_raw, shape="box")

    for nid in by_id.keys():
        dot.node(
            safe_id(nid),
            f"{names.get(nid, nid)}\n({nid})"
        )

    for nid, pid in parent_map.items():
        if pid == "CL:0000000" or pid not in by_id:
            dot.edge(super_root, safe_id(nid))
        else:
            dot.edge(safe_id(pid), safe_id(nid))

    dot.format = fmt
    dot.render(out_path, cleanup=True)
    print(f"Saved: {out_path}.{fmt}")


def extract_edge_mask(each_graph, positive_only=True):
    mask = each_graph.edata['mask']
    if positive_only:
        eids = torch.nonzero(mask > 0, as_tuple=False).squeeze(1)
    else:
        eids = torch.nonzero(mask != 0, as_tuple=False).squeeze(1)
    return dgl.edge_subgraph(each_graph, eids, store_ids=True)

@torch.no_grad()
def preprocess_embeddings_for_cosine(x, k=1):
    x = torch.as_tensor(x).float()
    x = x - x.mean(dim=0, keepdim=True)

    if k > 0:
        U, S, Vh = torch.linalg.svd(x, full_matrices=False)
        Vk = Vh[:k]  # [k, D]
        # Remove top K
        x = x - (x @ Vk.T) @ Vk 
    x = F.normalize(x, dim=1)
    return x

@torch.no_grad()
def edge_vs_random_cosine_one_graph(
    g_masked: dgl.DGLGraph,
    feat_key: str = "learned_node_feat",
    seed: int = 0,
    device: str | torch.device = "cpu",
    undirected: bool = True,
    avoid_edges_in_random: bool = True,
    n_random: int | None = None,
    max_rounds: int = 200,
):
    g = g_masked.to(device)
    g = dgl.remove_self_loop(g)

    if feat_key not in g.ndata:
        raise KeyError(f"'{feat_key}' not found in g.ndata")
        
    # Normalize & Remove anisotropy 
    # x = preprocess_embeddings_for_cosine(g.ndata[feat_key], k=1).to(device)
    x = g.ndata[feat_key].to(device)

    src, dst = g.edges()
    E = int(src.numel())
    if E == 0:
        return None

    edge_cos = (x[src] * x[dst]).sum(dim=1)
    N = g.num_nodes()
    if N < 2:
        return None

    R = E if n_random is None else int(n_random)

    gen = torch.Generator(device=device).manual_seed(seed)

    # Sampling random pair
    rand_u = torch.empty((R,), dtype=torch.long, device=device)
    rand_v = torch.empty((R,), dtype=torch.long, device=device)

    edge_set = None
    if avoid_edges_in_random:
        if undirected:
            uu = torch.minimum(src, dst).tolist()
            vv = torch.maximum(src, dst).tolist()
        else:
            uu = src.tolist()
            vv = dst.tolist()
        edge_set = set(zip(uu, vv))

    filled = 0
    rounds = 0
    while filled < R:
        rounds += 1
        if rounds > max_rounds:
            need = R - filled
            u = torch.randint(0, N, (need,), generator=gen, device=device)
            v = torch.randint(0, N, (need,), generator=gen, device=device)
            bad = (u == v)
            while bad.any():
                k = int(bad.sum().item())
                u[bad] = torch.randint(0, N, (k,), generator=gen, device=device)
                v[bad] = torch.randint(0, N, (k,), generator=gen, device=device)
                bad = (u == v)
            rand_u[filled:] = u
            rand_v[filled:] = v
            filled = R
            break

        need = R - filled
        cand = max(need * 4, 1024)

        u = torch.randint(0, N, (cand,), generator=gen, device=device)
        v = torch.randint(0, N, (cand,), generator=gen, device=device)

        ok = (u != v)

        if avoid_edges_in_random:
            if undirected:
                a = torch.minimum(u, v).tolist()
                b = torch.maximum(u, v).tolist()
            else:
                a = u.tolist()
                b = v.tolist()

            ok2 = []
            for ai, bi, keep in zip(a, b, ok.tolist()):
                ok2.append(keep and ((ai, bi) not in edge_set))
            ok = torch.tensor(ok2, device=device, dtype=torch.bool)

        u_ok = u[ok]
        v_ok = v[ok]

        take = min(int(u_ok.numel()), need)
        if take > 0:
            rand_u[filled:filled + take] = u_ok[:take]
            rand_v[filled:filled + take] = v_ok[:take]
            filled += take

    rand_cos = (x[rand_u] * x[rand_v]).sum(dim=1)

    eps = 1e-6
    num_edge_cos1 = (edge_cos > 1 - eps).sum().item()
    frac_edge_cos1 = num_edge_cos1 / edge_cos.numel()

    summary = {
        "E": E,
        "R": int(R),
        "edge_mean": float(edge_cos.mean().item()),
        "edge_std": float(edge_cos.std(unbiased=False).item()),
        "rand_mean": float(rand_cos.mean().item()),
        "rand_std": float(rand_cos.std(unbiased=False).item()),
        "mean_diff(edge-rand)": float((edge_cos.mean() - rand_cos.mean()).item()),
        "edge_cos1_count": int(num_edge_cos1),
        "edge_cos1_frac": float(frac_edge_cos1),
    }

    return edge_cos.detach().cpu(), rand_cos.detach().cpu(), summary

@torch.no_grad()
def compare_many_graphs(
    graphs,
    feat_key: str = "learned_node_feat",
    positive_only: bool = True,
    n_random: int | None = None,
    avoid_edges_in_random: bool = True,
    base_seed: int = 0,
    device: str | torch.device = "cpu",
    collect_distributions: bool = False,
    undirected: bool = True,
):
    rows = []
    all_edge = []
    all_rand = []

    for i, g in enumerate(graphs):
        g_masked = extract_edge_mask(g, positive_only=positive_only)

        out = edge_vs_random_cosine_one_graph(
            g_masked,
            feat_key=feat_key,
            seed=base_seed + i,
            device=device,
            undirected=undirected,
            avoid_edges_in_random=avoid_edges_in_random,
            n_random=n_random,
        )

        if out is None:
            rows.append({"graph_idx": i, "skipped": True, "E": 0, "R": 0})
            continue

        edge_cos, rand_cos, summary = out
        rows.append({"graph_idx": i, "skipped": False, **summary})

        if collect_distributions:
            all_edge.append(edge_cos)
            all_rand.append(rand_cos)

    df = pd.DataFrame(rows)

    if collect_distributions:
        all_edge = torch.cat([torch.as_tensor(x) for x in all_edge], dim=0) if all_edge else torch.tensor([])
        all_rand = torch.cat([torch.as_tensor(x) for x in all_rand], dim=0) if all_rand else torch.tensor([])
        return df, all_edge, all_rand

    return df

def plot_edge_vs_random_hist(edge_cos, rand_cos, bins=60, title=None, density=True):
    """
    edge_cos, rand_cos: 1D tensor/np.array/list (cosine similarities)
    random: gray, edge-pair: purple
    """
    def to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    e = to_np(edge_cos).astype(np.float32)
    r = to_np(rand_cos).astype(np.float32)

    # NaN/inf 방지
    e = e[np.isfinite(e)]
    r = r[np.isfinite(r)]

    # 같은 bin range 쓰기 ([-1, 1]이 cosine의 자연 범위)
    hist_range = (-1.0, 1.0)

    plt.figure(figsize=(7, 5))
    plt.hist(r, bins=bins, range=hist_range, density=density, alpha=0.45,
             color="gray", label=f"Random pairs (n={len(r)})")
    plt.hist(e, bins=bins, range=hist_range, density=density, alpha=0.55,
             color="purple", label=f"Edge pairs (n={len(e)})")

    # median & IQR
    r_med = np.median(r)
    r_q1 = np.percentile(r, 25)
    r_q3 = np.percentile(r, 75)

    e_med = np.median(e)
    e_q1 = np.percentile(e, 25)
    e_q3 = np.percentile(e, 75)

    # median line
    plt.axvline(r_med, color="gray", linestyle="--", linewidth=1)
    plt.axvline(e_med, color="purple", linestyle="--", linewidth=1)

    # IQR band (optional but very informative)
    plt.axvspan(r_q1, r_q3, color="gray", alpha=0.15)
    plt.axvspan(e_q1, e_q3, color="purple", alpha=0.15)

    # print summary
    print(f"Random  median={r_med:.4f}, IQR=[{r_q1:.4f}, {r_q3:.4f}]")
    print(f"Edge    median={e_med:.4f}, IQR=[{e_q1:.4f}, {e_q3:.4f}]")
    print(f"Median diff (edge-random) = {e_med - r_med:.4f}")

    
    plt.xlabel("Cosine similarity")
    plt.ylabel("Density" if density else "Count")
    if title:
        plt.title(title)
    
    plt.ylim(0, 1.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_edge_vs_random_hist_stacked(
    edge_rand_list,   # [(edge_cos, rand_cos, title), ...]
    bins=70,
    pdf_path="edge_vs_random_stacked.pdf",
    density=True,
):

    n = len(edge_rand_list)
    fig, axes = plt.subplots(n, 1, figsize=(7, 4 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (edge_cos, rand_cos, title) in zip(axes, edge_rand_list):
        e = edge_cos.detach().cpu().numpy()
        r = rand_cos.detach().cpu().numpy()

        e = e[np.isfinite(e)]
        r = r[np.isfinite(r)]

        hist_range = (-1.0, 1.0)

        # histogram
        ax.hist(
            r, bins=bins, range=hist_range, density=density,
            color="gray", alpha=0.45, label=f"Random (n={len(r)})"
        )
        ax.hist(
            e, bins=bins, range=hist_range, density=density,
            color="purple", alpha=0.55, label=f"Edge (n={len(e)})"
        )

        # median & IQR
        r_med, r_q1, r_q3 = np.median(r), np.percentile(r, 25), np.percentile(r, 75)
        e_med, e_q1, e_q3 = np.median(e), np.percentile(e, 25), np.percentile(e, 75)

        ax.axvline(r_med, color="gray", linestyle="--", linewidth=1)
        ax.axvline(e_med, color="purple", linestyle="--", linewidth=1)

        ax.axvspan(r_q1, r_q3, color="gray", alpha=0.15)
        ax.axvspan(e_q1, e_q3, color="purple", alpha=0.15)

        ax.set_ylim(0, 1.5)
        # ---- 텍스트 박스 ----
        text = (
            f"Random:\n"
            f"  median = {r_med:.3f}\n"
            f"  IQR = [{r_q1:.3f}, {r_q3:.3f}]\n\n"
            f"Edge:\n"
            f"  median = {e_med:.3f}\n"
            f"  IQR = [{e_q1:.3f}, {e_q3:.3f}]"
        )

        ax.text(
            0.02, 0.05, text,
            transform=ax.transAxes,
            ha="left", va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85)
        )

        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel("Cosine similarity")
    axes[0].set_ylabel("Density" if density else "Count")

    plt.tight_layout()
    plt.savefig(pdf_path)
    plt.close(fig)

    print(f"Saved stacked histogram to: {pdf_path}")
