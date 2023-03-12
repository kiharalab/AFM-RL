#!/usr/bin/env bash

module load alphafold/2.1.2
source /apps/miniconda38/etc/profile.d/conda.sh
conda activate alphafold_2.1.2

export PATH="/path/to/alphafold:$PATH"
export ALPHA_FOLD_DIR=/path/to/alphafold/
export ALPHA_FOLD_DATABASE_DIR=$ALPHA_FOLD_DIR/apps/alphafold/databases/
export TMPDIR=/tmp/

export CUDA_VISIBLE_DEVICES=-1

srun hostname
run_date=$(date +%F)
echo ${run_date}

run_alphafold.sh --fasta_paths=${pdb}.fasta --output_dir=${pdb} --max_template_date=2021-01-01 --model_preset=multimer --pdb_seqres_database_path=${ALPHA_FOLD_DATABASE_DIR}/pdb_seqres/pdb_seqres.txt --uniprot_database_path=${ALPHA_FOLD_DATABASE_DIR}/uniprot/uniprot.fasta --use_gpu_relax=False --only_run_msa=True
