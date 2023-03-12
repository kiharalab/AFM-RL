#!/bin/bash
module load anaconda/2020.02-py37
export PATH="$PATH:/path/to/miniconda3/bin"
echo $HOSTNAME
source /etc/profile.d/modules.sh
module load cuda/11.2.0.lua

INPUT_FASTA="${1}"

export TF_FORCE_UNIFIED_MEMORY='1'
export XLA_PYTHON_CLIENT_MEM_FRACTION='10.0'

data_dir="AF/alphafold_2.2.0/dl"
bfd_database_path="$data_dir/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt"
mgnify_database_path="$data_dir/mgnify/mgy_clusters.fa"
template_mmcif_dir="$data_dir/pdb_mmcif/mmcif_files"
obsolete_pdbs_path="$data_dir/pdb_mmcif/obsolete.dat"
pdb70_database_path="$data_dir/pdb70/pdb70"
uniclust30_database_path="$data_dir/uniclust30/uniclust30_2018_08/uniclust30_2018_08"
uniref90_database_path="$data_dir/uniref90/uniref90.fasta"
uniprot_database_path="$data_dir/uniprot/uniprot.fasta"
pdb_seqres_database_path="$data_dir/pdb_seqres/pdb_seqres.txt"
uniprot_database_path="$data_dir/uniprot/uniprot.fasta"


python3 alphafold_2.2.0/run_alphafold.py \
    --num_multimer_predictions_per_model=5 \
    --use_gpu_relax=True \
    --use_precomputed_msas \
    --bfd_database_path=$bfd_database_path \
    --uniclust30_database_path=$uniclust30_database_path \
    --mgnify_database_path=$mgnify_database_path \
    --template_mmcif_dir=$template_mmcif_dir \
    --obsolete_pdbs_path=$obsolete_pdbs_path \
    --uniref90_database_path=$uniref90_database_path \
    --data_dir=$data_dir \
    --fasta_paths=${INPUT_FASTA}.fasta \
    --output_dir=RESULT_220/${subname}/${INPUT_FASTA} \
    --max_template_date=2200-01-01 \
    --model_preset=multimer \
    --pdb_seqres_database_path=$pdb_seqres_database_path \
    --uniprot_database_path=$uniprot_database_path

####SBATCH --nodefile=node-a100
