#!/bin/bash
#BSUB -J gex2d-cache-36852-full
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/gex2d-cache-36852-full.%J.out
#BSUB -e logs/job_logs/gex2d-cache-36852-full.%J.err
#BSUB -q inference
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=40GB:num=1"
#BSUB -n 4
#BSUB -M 160GB
#BSUB -R "select[mem>160GB] rusage[mem=160GB] span[hosts=1]"

# Thin wrapper: export MODE=full (env-passing via `VAR=val bsub <` fails
# silently on farm22) then run the main caching jobscript's FULL path.
export MODE=full
source /nfs/team361/dj16/projects/SIGMMA/jobscripts/jobscript_gex_2d_cache_36852_all.sh
