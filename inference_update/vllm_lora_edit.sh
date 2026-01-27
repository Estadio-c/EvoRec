
test_file=../unlearning_data/edit_after_candidate_val.json
generate_file=../generate_data/update_after.jsonl
MODEL_NAME=YOUR_MODEL_PATH/Qwen2-7B-Instruct/
log_file=../generate_data/update_after.log
score_file=../generate_data/score.json

echo "" > $log_file

numbers=("150" "140" "130" "120" "110" "100" "90" "80" "70" "60" "50" "40")

for i in "${numbers[@]}"; do
    lora_path=../checkpoint_update_after/checkpoint-$i

    # Check and delete generated file
    if [ -s "$generate_file" ]; then
        echo "File $generate_file is not empty, deleting..."
        rm "$generate_file"
        echo "File deleted."
    fi

    # Execute qwen_vllm.py script
    CUDA_VISIBLE_DEVICES=2 python ./qwen_vllm.py \
        --batch 20 \
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
        --checkpoint "checkpoint-$i" \
        --score_file $score_file \
        --edit_user 602 \
        --test_file $test_file \
        --generate_file $generate_file \
        >> $log_file 2>&1

done

