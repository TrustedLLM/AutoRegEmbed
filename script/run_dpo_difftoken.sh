#!/bin/bash
export http_proxy=http://oversea-squid1.jp.txyun:11080 https_proxy=http://oversea-squid1.jp.txyun:11080 no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com

source /etc/ssd1/jiangzhongtao/anaconda3/bin/activate
echo $(which torchrun)

set -e

gpus_per_node=8
master_addr=10.82.138.12 # /etc/mpi/hostfile 第一个ip地址
master_port=12889
nnodes=4
node_rank=0
world_size=$(($gpus_per_node*$nnodes))
save_root="./results"
output_name="" # 
model_name_or_path="" # IC result
decoder_name_or_path="llama2-7b-hf" 
description=""

distributed_args="
    --nproc_per_node $gpus_per_node \

"
    

model_args="
    --model_name_or_path $model_name_or_path \
    --decoder_name_or_path $decoder_name_or_path \
    --num_compress_token 5 \
    --bfloat16 True \
    --use_flash_attention_2 True 
"

data_args="
    --train_data_path ../data/medi_bge.jsonl \
    --dpo_data_path ../data/llama2-7b-pwcunique_medi_bge_log_probs.pth \
    --context_maxlen 4096 \
    --instruction_maxlen 512 \
    --output_maxlen 4096 \
    --instruction_left False \
    --num_hn 1
"

compress_train_args="
    --lora_tune False \
    --lora_rank 32 \
    --lora_dropout 0.1 \
    --training True \
    --dpo_beta 0.1 \
    --dpo_label_smoothing 0.0 
"

train_args="
    --deepspeed ../config_zero1.json \
    --no_remove_unused_columns \
    --learning_rate 5e-6 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --num_train_epochs 4 \
    --bf16 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --dataloader_drop_last True \
    --logging_steps 1 \
    --save_steps 10000 \
    --gradient_checkpointing True \
    --output_dir $save_root/${output_name}
"
#    --eval_strategy epoch \
torchrun \
    $distributed_args \
   ./run_dpo_diff.py \
    $model_args \
    $data_args \
    $compress_train_args \
    $train_args

if test $node_rank = 0; then
echo "This is rank 0, copying $0"
cp $0 $save_root/$output_name/.
fi



set +x