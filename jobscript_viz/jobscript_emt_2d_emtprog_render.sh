#!/bin/bash
#BSUB -J emt_2d_ep_render[1-54]
#BSUB -G team361
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/emt_2d_ep_render.%J.%I.out
#BSUB -e logs/job_logs/emt_2d_ep_render.%J.%I.err
#BSUB -q normal
#BSUB -n 4
#BSUB -M 100GB
#BSUB -R "select[mem>100GB] rusage[mem=100GB] span[hosts=1]"

# Pass 2 (render) for the clean emt_program gene set: uses anchors_emt_program.json
# and the --min_tumour_cells exclusion. Hardcode env (farm22 drops bsub-< env vars).
export MODE=render
export GENE_SET=emt_program
source jobscripts/jobscript_emt_2d_all.sh
