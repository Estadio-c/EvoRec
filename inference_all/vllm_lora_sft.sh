
test_file=../data/toys/1/llm_data/edit_after_candidate_test_retain.json
generate_file=../generate_data_edit/before.jsonl
MODEL_NAME=Qwen2-7B-Instruct——PATH
log_file=../generate_data_edit/before.log


echo "" > $log_file

lora_path=../checkpoint_before

# Check and delete generated file
if [ -s "$generate_file" ]; then
    echo "File $generate_file is not empty, deleting..."
    rm "$generate_file"
    echo "File deleted."
fi

# Execute qwen_vllm.py script
CUDA_VISIBLE_DEVICES=0,1,2,3 python ./qwen_vllm.py \
    --batch 200 \
    --max_len 300 \
    --lora_path $lora_path \
    --model_name $MODEL_NAME \
    --save_name $generate_file \
    --test_file $test_file \
    >> $log_file 2>&1


# Log loaded checkpoint to log file
echo "Loaded checkpoint: checkpoint-$i" >> $log_file

# Execute metric.py script
python ./metric.py \
    --edit_user 602 \
    --test_file $test_file \
    --generate_file $generate_file \
    >> $log_file 2>&1


