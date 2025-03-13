#!/bin/bash
gpus_per_node=4
export CUDA_VISIBLE_DEVICES=4,5,6,7

save_root="../information_compress_results/"
output_name="llama2-7b-compress" 
model_name_or_path="llama2-7b-hf" #local model path
description=""

distributed_args="
    --nproc_per_node $gpus_per_node 
"
    
model_args="
    --model_name_or_path $model_name_or_path \
    --num_compress_token 5 \
    --bfloat16 True \
    --use_flash_attention_2 True 
"

data_args="
    --train_data_path ../data/pwc_train_unique.jsonl \
    --context_maxlen 4096 \
    --instruction_maxlen 512 \
    --output_maxlen 4096 \
    --instruction_left False 
"

compress_train_args="
    --lora_tune False \
    --lora_rank 32 \
    --lora_dropout 0.1 \
    --training True
"

train_args="
    --deepspeed ../config_zero1.json \
    --no_remove_unused_columns \
    --learning_rate 1e-5 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --num_train_epochs 1 \
    --bf16 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --dataloader_drop_last True \
    --logging_steps 1 \
    --save_steps 10000 \
    --gradient_checkpointing True \
    --output_dir $save_root/${output_name}
"

torchrun \
    $distributed_args \
    ./run_information_compress.py \
    $model_args \
    $data_args \
    $compress_train_args \
    $train_args

if test $node_rank = 0; then
echo "This is rank 0, copying $0"
cp $0 $save_root/$output_name/.
fi

set +x