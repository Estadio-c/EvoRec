
test_file=../data/toys/1/llm_data/edit_after_candidate_test_retain.json
generate_file=../generate_data_edit/before.jsonl
MODEL_NAME=Qwen2-7B-Instruct——PATH
log_file=../generate_data_edit/before.log


echo "" > $log_file

lora_path=../checkpoint_before

# 检查并删除生成文件
if [ -s "$generate_file" ]; then
    echo "文件 $generate_file 不为空，正在删除..."
    rm "$generate_file"
    echo "文件已删除。"
fi

# 执行 qwen_vllm.py 脚本
CUDA_VISIBLE_DEVICES=0,1,2,3 python ./qwen_vllm.py \
    --batch 200 \
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
    --edit_user 602 \
    --test_file $test_file \
    --generate_file $generate_file \
    >> $log_file 2>&1


