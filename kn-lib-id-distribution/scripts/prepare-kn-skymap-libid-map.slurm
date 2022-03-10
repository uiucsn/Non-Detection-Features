#!/bin/bash -l
#SBATCH --partition=caps
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=deep1018@illinois.edu
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=41
#SBATCH --mem=32G
#SBATCH --output=./prog.out
#SBATCH --error=./prog.err

conda activate kn-rapid

python prepare-kn-skymap-libid-map.py
