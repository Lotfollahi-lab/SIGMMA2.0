#BSUB -J gex2d-svg-2d22-test-eval[1-9]
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/gex2d-svg-2d22-test-eval.%J.%I.out
#BSUB -e logs/job_logs/gex2d-svg-2d22-test-eval.%J.%I.err
#BSUB -q training-parallel
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=40GB:num=1"
#BSUB -n 4
#BSUB -M 120GB
#BSUB -R "select[mem>120GB] rusage[mem=120GB] span[hosts=1]"

# Test-cohort eval (14 sections) on the new panel-aware SVG-100 cache
# (panel_2d22ba31). Reuses trained probe heads from
# outputs/gex_2d/svg_panel_2d22ba31_step1000/<condition>/{heads,heads_linear_sweep}/
# via symlink into a parallel _test out_root and runs --phase eval ONLY.
#
# Source train+eval (val cohort): jobscript_gex_2d_train_eval_svg_panel_2d22ba31.sh
# Source val_manifest: MANIFEST_panel_2d22ba31_val.csv (15 sec)
# Target val_manifest (this script): MANIFEST_panel_2d22ba31_test.csv (14 sec)
#
# Array idx → same as train+eval jobscript:
#   1 = uni2h baseline (no FT)
#   2-5 = sigmma a00/a01/a02/a04 (mlp head, 5 reprs each)
#   6-9 = bleep/clip/plip/omiclip FT (linear_sweep, 1 repr)

set -e
module load python-3.11.6
export PYTHONUNBUFFERED=1
export PATH="/nfs/users/nfs_d/dj16/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_dir="/nfs/team361/dj16/projects/SIGMMA"
ENV_PATH="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/activate"
[ -f $ENV_PATH ] && source $ENV_PATH
cd "${run_dir}"

cache_root="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache"
data_cache_dir="${cache_root}/data/gex_2d/svg_panel_2d22ba31_g14_fov256"
uni2h_cache_dir="${cache_root}/uni/gex_2d/uni2h_fov256_p14"
panel_dir="outputs/gex_2d/svg_panel_2d22ba31"

# Panel-restricted manifests
train_manifest="${run_dir}/outputs/gex_2d/svg_panel_2d22ba31/MANIFEST_panel_2d22ba31_train.csv"
test_manifest="${run_dir}/outputs/gex_2d/svg_panel_2d22ba31/MANIFEST_panel_2d22ba31_test.csv"

# Source (val-eval) and target (test-eval) roots
val_root="outputs/gex_2d/svg_panel_2d22ba31_step1000"
test_root="outputs/gex_2d/svg_panel_2d22ba31_step1000_test"

# Eval-mode args. --val_manifest is REUSED as the held-out side, i.e. test.
base_args=(
    --data_cache_dir "$data_cache_dir"
    --panel_dir "$panel_dir"
    --target_grid 14 --cache_target_grid 56 --fov_size 256
    --patch_size_hint 14
    --train_manifest "$train_manifest"
    --val_manifest "$test_manifest"
    --batch_size 64
)

declare -A SIGMMA_CKPT_TAG
SIGMMA_CKPT_TAG[a00]="26274_step1000"
SIGMMA_CKPT_TAG[a01]="25923_1_step1000"
SIGMMA_CKPT_TAG[a02]="25923_2_step1000"
SIGMMA_CKPT_TAG[a04]="25923_3_step1000"

# Mirror trained heads from val out_root into test out_root via symlinks so
# phase_eval finds best_seed*.pt without retraining.
link_heads () {
    local src_root="$1"  # e.g. val_root/sigmma_a01_step1000
    local dst_root="$2"  # e.g. test_root/sigmma_a01_step1000
    mkdir -p "$dst_root"
    for sub in heads heads_linear_sweep; do
        if [ -d "$src_root/$sub" ]; then
            ln -sfn "$(realpath "$src_root/$sub")" "$dst_root/$sub"
        fi
    done
}

run_eval_only () {
    local repr="$1"; shift
    echo "  [eval] repr=$repr (test cohort)"
    python -m scripts.probe_gex_2d --phase eval --input_repr "$repr" "$@"
}

case "${LSB_JOBINDEX}" in
    1)
        src_out="${val_root}/uni2h_no_ft"
        out_root="${test_root}/uni2h_no_ft"
        link_heads "$src_out" "$out_root"
        echo "=== idx 1: UNI2-h no FT (test) → $out_root ==="
        run_eval_only uni "${base_args[@]}" \
            --head_type linear_sweep \
            --uni_cache_dir "$uni2h_cache_dir" \
            --out_root "$out_root"
        ;;

    2|3|4|5)
        case "${LSB_JOBINDEX}" in
            2) alpha="a00" ;;
            3) alpha="a01" ;;
            4) alpha="a02" ;;
            5) alpha="a04" ;;
        esac
        ckpt_tag="${SIGMMA_CKPT_TAG[$alpha]}"
        sigmma_cache_dir="${cache_root}/sigmma/gex_2d/${ckpt_tag}"
        sigmma_config="${run_dir}/configs/cell_niche_transfer_dense_attnpool_${alpha}_step1000.yaml"
        src_out="${val_root}/sigmma_${alpha}_step1000"
        out_root="${test_root}/sigmma_${alpha}_step1000"
        link_heads "$src_out" "$out_root"
        cp "$sigmma_config" "${out_root}/" 2>/dev/null || true
        echo "=== idx ${LSB_JOBINDEX}: SIGMMA ${alpha} (${ckpt_tag}) test → $out_root ==="
        for repr in uni concat micro meso macro; do
            run_eval_only "$repr" "${base_args[@]}" \
                --head_type mlp \
                --uni_cache_dir "$uni2h_cache_dir" \
                --sigmma_cache_dir "$sigmma_cache_dir" \
                --sigmma_config "$sigmma_config" \
                --out_root "$out_root"
        done
        ;;

    6|7|8|9)
        case "${LSB_JOBINDEX}" in
            6) backbone="bleep_uni" ;;
            7) backbone="clip_uni" ;;
            8) backbone="plip_uni" ;;
            9) backbone="omiclip_uni" ;;
        esac
        ft_uni_cache_dir="${cache_root}/uni/gex_2d/${backbone}_step1000"
        src_out="${val_root}/${backbone}_step1000"
        out_root="${test_root}/${backbone}_step1000"
        link_heads "$src_out" "$out_root"
        echo "=== idx ${LSB_JOBINDEX}: baseline FT ${backbone} (test) → $out_root ==="
        run_eval_only uni "${base_args[@]}" \
            --head_type linear_sweep \
            --uni_cache_dir "$ft_uni_cache_dir" \
            --out_root "$out_root"
        ;;

    *)
        echo "Unknown LSB_JOBINDEX=${LSB_JOBINDEX}"; exit 1
        ;;
esac

echo "=== DONE — idx ${LSB_JOBINDEX} ==="
