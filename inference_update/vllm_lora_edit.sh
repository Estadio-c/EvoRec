
test_file=../unlearning_data/edit_after_candidate_val.json
generate_file=../generate_data/update_after.jsonl
MODEL_NAME=/data/opensource-model/Qwen2-7B-Instruct/
log_file=../generate_data/update_after.log
score_file=../generate_data/score.json

echo "" > $log_file

numbers=("150" "140" "130" "120" "110" "100" "90" "80" "70" "60" "50" "40")

for i in "${numbers[@]}"; do
    lora_path=../checkpoint_update_after/checkpoint-$i

    # 检查并删除生成文件
    if [ -s "$generate_file" ]; then
        echo "文件 $generate_file 不为空，正在删除..."
        rm "$generate_file"
        echo "文件已删除。"
    fi

    # 执行 qwen_vllm.py 脚本
    CUDA_VISIBLE_DEVICES=2 python ./qwen_vllm.py \
        --batch 20 \
        --max_len 300 \
        --lora_path $lora_path \
        --model_name $MODEL_NAME \
        --save_name $generate_file \
        --test_file $test_file \
        >> $log_file 2>&1


    # 打印加载的断点到日志文件
    echo "加载的断点: checkpoint-$i" >> $log_file

    # 执行 metric.py 脚本
    python ./metric.py \
        --checkpoint "checkpoint-$i" \
        --score_file $score_file \
        --edit_user 602 \
        --test_file $test_file \
        --generate_file $generate_file \
        >> $log_file 2>&1

done

