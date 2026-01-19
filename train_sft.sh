output_dir=./checkpoint_before/
export WANDB_MODE=disabled
export WANDB_DISABLED=true

if [ -d "$output_dir" ] && [ "$(ls -A $output_dir)" ]; then
    echo "文件夹 $output_dir 不为空，正在删除文件..."
    rm -rf "$output_dir"/*
    echo "文件夹中的所有文件已删除。"
fi


CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path  /data/opensource-model/Qwen2-7B-Instruct/ \
    --dataset all \
    --dataset_dir ./data/toys/1/llm_data \
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
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type constant \
    --logging_steps 20 \
    --save_steps 30 \
    --eval_steps 30 \
    --save_total_limit 100 \
    --seed 42 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --learning_rate 3e-4 \
    --warmup_ratio 0.05 \
    --num_train_epochs 1 \
    --val_size 0.01 \
    --plot_loss \
    --bf16 \

rm -rf ./checkpoint_before/checkpoint-*/  #只保留load_best_model_at_end的checkpoint

    



    