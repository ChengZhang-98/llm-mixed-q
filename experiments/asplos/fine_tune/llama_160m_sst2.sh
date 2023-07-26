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
#SBATCH --output=log_llama-160m_sst2.txt
#SBATCH --job-name=llama-160m_sst2

if [ $USER = "cz98" ]; then
    module purge
    module load CUDA/11.7.0 GCCcore/11.3.0 GCC/11.3.0 OpenMPI/4.1.4 Python/3.10.4
    source /home/c/cz98/venvs/llm-mixed-q/bin/activate
fi

work_dir=$HOME/Projects/llm-mixed-q
env_name=llm-mixed-q

quant_config=$work_dir/experiments/asplos/configs/quantize/bypass.toml
run_dir=$work_dir/experiments/asplos/fine_tune
ckpt=$work_dir/checkpoints/asplos/fine_tune/llama_160m_sst2
cd $run_dir

echo ========== Running Llama 160M SST2 ==========
# conda run -n $env_name accelerate launch --multi_gpu fine_tune.py \
accelerate launch --multi_gpu fine_tune.py \
    --model_arch llama \
    --model_name Cheng98/llama-160m \
    --task sst2 \
    --quant_config $quant_config \
    --batch_size_train 16 \
    --batch_size_eval 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --output_dir $ckpt \
    --project_name llama_160m_sst2 \
    --with_tracking
echo ========== Done. ==========
