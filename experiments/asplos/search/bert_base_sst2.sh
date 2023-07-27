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

if [ -z $1 ]; then
    echo "❗Requires <search_tag> as \$0"
    exit
fi

if [ $USER = "cz98" ]; then
    # sulis
    module purge
    module load CUDA/11.7.0 GCCcore/11.3.0 GCC/11.3.0 OpenMPI/4.1.4 Python/3.10.4
    source /home/c/cz98/venvs/llm-mixed-q/bin/activate
fi

work_dir=$HOME/Projects/llm-mixed-q
env_name=llm-mixed-q
run_dir=$work_dir/experiments/asplos/search
cd $run_dir
echo ========== Running BERT Base SST2 ==========
search_tag=$1
save_dir=$work_dir/checkpoints/asplos/configs/search/bert_base_sst2/$search_tag
mkdir -p $save_dir

model_arch=bert
task=sst2
search_config=$work_dir/experiments/asplos/configs/search/block_fp/bert_base_sst2.toml
ckpt=$work_dir/checkpoints/asplos/fine_tune/bert_base_sst2
batch_size=256
max_length=196

if [ $USER = "cz98" ]; then
    conda run -n $env_name python search_cls.py \
        --model_arch $model_arch \
        --model_name $ckpt \
        --task $task \
        --batch_size $batch_size \
        --padding max_length \
        --max_length $max_length \
        --save_dir $save_dir \
        --search_config $search_config
else
    python search_cls.py \
        --model_arch $model_arch \
        --model_name $ckpt \
        --task $task \
        --batch_size $batch_size \
        --padding max_length \
        --max_length $max_length \
        --save_dir $save_dir \
        --search_config $search_config
fi
echo ========== Done. ==========
