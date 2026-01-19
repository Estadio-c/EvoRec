#!/bin/bash

# 设置参数
DATA_BEFORE="./data/toys/1/llm_data/edit_before_candidate_train.json"
DATA_AFTER="./data/toys/1/llm_data/edit_after_candidate_train.json"
MODEL_PATH="/data/opensource-model/Qwen2-7B-Instruct/"
LORA_CHECKPOINT="./checkpoint_before"
OUTPUT_PATH="./locate/located_region.json"
BATCH_SIZE=2
MAX_LENGTH=1200

# 运行 Python 脚本
python ./update/localization.py \
    --data_before "$DATA_BEFORE" \
    --data_after "$DATA_AFTER" \
    --model_path "$MODEL_PATH" \
    --lora_checkpoint "$LORA_CHECKPOINT" \
    --output_path "$OUTPUT_PATH" \
    --batch_size "$BATCH_SIZE" \
    --max_length "$MAX_LENGTH"
