#!/bin/bash
#BSUB -J emt_2d_ep_dump[1-54]
#BSUB -G team361
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/emt_2d_ep_dump.%J.%I.out
#BSUB -e logs/job_logs/emt_2d_ep_dump.%J.%I.err
#BSUB -q normal
#BSUB -n 4
#BSUB -M 100GB
#BSUB -R "select[mem>100GB] rusage[mem=100GB] span[hosts=1]"

# Pass 1 (dump) for the clean emt_program gene set. farm22 LSF drops
# `VAR=val bsub < script` env vars, so we hardcode GENE_SET/MODE here and source
# the shared array body.
export MODE=dump
export GENE_SET=emt_program
source jobscripts/jobscript_emt_2d_all.sh
