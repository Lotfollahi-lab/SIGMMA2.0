#!/usr/bin/env bash
# Build TCGA-KIRC survival metadata (clinical + slide_survival.tsv) from the GDC
# API. CPU + network; compute nodes at Sanger have outbound HTTPS (the KIRC
# watcher already runs this same script on a farm node).
#BSUB -J kirc_survmeta
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/kirc_survmeta.%J.out
#BSUB -e logs/job_logs/kirc_survmeta.%J.err
#BSUB -q oversubscribed
#BSUB -n 1
#BSUB -M 4000
#BSUB -R "select[mem>4000] rusage[mem=4000] span[hosts=1]"
set -e
TP=/lustre/scratch126/cellgen/lotfollahi/dj16/data/public_TCGA
cd "$TP"
echo "[kirc] building survival metadata via GDC API"
/usr/bin/python3 make_survival_metadata.py TCGA-KIRC \
    --manifest TCGA-KIRC_dx_manifest.txt
echo "[kirc] done -> $TP/TCGA-KIRC_slide_survival.tsv"
ls -la TCGA-KIRC_slide_survival.tsv
