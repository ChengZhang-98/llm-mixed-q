#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=30
#SBATCH --mem-per-cpu=3850
#SBATCH --gres=gpu:ampere_a100:3
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --account=su114-gpu
#SBATCH --signal=SIGUSR1@90
#SBATCH --output=log_bert-base-uncased_search_sst2.txt
#SBATCH --job-name=bert-base-uncased_search_sst2

module purge
module load CUDA/11.7.0 GCCcore/11.3.0 GCC/11.3.0 OpenMPI/4.1.4 Python/3.10.4
source /home/c/cz98/venvs/mase-sw/bin/activate

work_dir=$HOME/Projects/llm-mixed-q
env_name=mase-sw
run_dir=$work_dir/experiments/asplos/search

cd $run_dir
echo ========== Running BERT Base SST2 ==========
# conda run -n $env_name accelerate launch --multi_gpu fine_tune.py \
save_dir=$run_dir/bert_base_sst2
search_config=$work_dir/experiments/asplos/configs/search/bert_base_sst2.toml
python search_cls.py \
    --model_arch bert \
    --model_name bert-base-uncased \
    --task sst2 \
    --batch_size 64 \
    --padding max_length \
    --max_length 196 \
    --save_dir $save_dir \
    --search_config $search_config
echo ========== Done. ==========
