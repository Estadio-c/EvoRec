export WANDB_MODE=disabled
export WANDB_DISABLED=true
output_dir=./checkpoint_update_after

if [ -d "$output_dir" ] && [ "$(ls -A $output_dir)" ]; then
    echo "文件夹 $output_dir 不为空，正在删除文件..."
    rm -rf "$output_dir"/*
    echo "文件夹中的所有文件已删除。"
fi

CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node 1 --master-port=60000 ./update/train.py \
    --stage sft \
    --do_train \
    --edit True \
    --adapter_path ./checkpoint_before \
    --model_name_or_path /data/opensource-model/Qwen2-7B-Instruct/ \
    --dataset train_after_1 \
    --eval_dataset train_val_1 \
    --dataset_dir ./unlearning_data \
    --template qwen \
    --finetuning_type lora \
    --lora_rank 16 \
    --lora_dropout 0.05 \
    --output_dir $output_dir \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1200 \
    --preprocessing_num_workers 150 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 20 \
    --save_steps 10 \
    --eval_steps 300 \
    --save_total_limit 100 \
    --seed 38 \
    --evaluation_strategy steps \
    --learning_rate 4e-4 \
    --warmup_ratio 0.05 \
    --num_train_epochs 2 \
    --bf16 \


