test_file=../unlearning_data/edit_after_candidate_test.json
generate_file=../generate_data_all/edit_after.jsonl
MODEL_NAME=Qwen2-7B-Instruct-PATH
log_file=../generate_data_all/edit_after.log

lora_path=../checkpoint_edit_after/checkpoint-150

# 检查并删除生成文件
if [ -s "$generate_file" ]; then
    echo "文件 $generate_file 不为空，正在删除..."
    rm "$generate_file"
    echo "文件已删除。"
fi

# 执行 qwen_vllm.py 脚本
CUDA_VISIBLE_DEVICES=4,5,6,7 python qwen_vllm.py \
    --batch 200 \
    --max_len 300 \
    --lora_path $lora_path \
    --model_name $MODEL_NAME \
    --save_name $generate_file \
    --test_file $test_file \
    >> $log_file 2>&1

echo "加载的断点: checkpoint-$i" >> $log_file

# 执行 metric.py 脚本
python ./inference_all/metric.py \
    --edit_user 602 \
    --test_file $test_file \
    --generate_file $generate_file \
    >> $log_file 2>&1



