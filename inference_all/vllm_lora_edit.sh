test_file=../unlearning_data/edit_after_candidate_test.json
generate_file=../generate_data_all/edit_after.jsonl
MODEL_NAME=Qwen2-7B-Instruct-PATH
log_file=../generate_data_all/edit_after.log

lora_path=../checkpoint_edit_after/checkpoint-150

# Check and delete generated file
if [ -s "$generate_file" ]; then
    echo "File $generate_file is not empty, deleting..."
    rm "$generate_file"
    echo "File deleted."
fi

# Execute qwen_vllm.py script
CUDA_VISIBLE_DEVICES=4,5,6,7 python qwen_vllm.py \
    --batch 200 \
    --max_len 300 \
    --lora_path $lora_path \
    --model_name $MODEL_NAME \
    --save_name $generate_file \
    --test_file $test_file \
    >> $log_file 2>&1

echo "Loaded checkpoint: checkpoint-$i" >> $log_file

# Execute metric.py script
python ./inference_all/metric.py \
    --edit_user 602 \
    --test_file $test_file \
    --generate_file $generate_file \
    >> $log_file 2>&1



