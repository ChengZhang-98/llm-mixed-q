#/bin/bash

work_dir=$HOME/Projects/llm-mixed-q
run_dir=$work_dir/experiments/asplos/block_size && cd $run_dir

# bert-base, bert-large, opt-125m, opt-350m, llama-160m
model_arch="llama"
model_name="huggyllama/llama-7b"
quant_config=$work_dir/experiments/asplos/configs/quantize/block_size/block_fp.toml
batch_size=1
save_dir=$work_dir/checkpoints/asplos/block_size/${model_name//\//_}

conda run -n llm-mixed-q --no-capture-output python block_size_search.py \
    --model_arch $model_arch \
    --model_name $model_name \
    --quant_config $quant_config \
    --batch_size $batch_size \
    --save_dir $save_dir
