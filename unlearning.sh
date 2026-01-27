python ./unlearning/main.py

BEFORE_PATH="./data/toys/1/ID_data/edit_before_candidate_train.json"
AFTER_PATH="./data/toys/1/ID_data/edit_after_candidate_train.json"
VAL_PATH="./data/toys/1/ID_data/edit_after_candidate_val.json"
TEST_PATH="./data/toys/1/ID_data/edit_after_candidate_test.json"
ID2TITLE_PATH="./data/toys/id2title.json"
OUTPUT_TRAIN_PATH="./unlearning/new_data/edit_after_candidate_train.json"
OUTPUT_VAL_PATH="./unlearning/new_data/edit_after_candidate_val.json"
OUTPUT_TEST_PATH="./unlearning/new_data/edit_after_candidate_test.json"
UNLEARNING_DATA_PATH="./unlearning_data/"
TEST_LLM_DATA="./data/toys/1/llm_data/edit_after_candidate_test_retain.json"

GPU_ID="0"

# Run Python script
python ./unlearning/unlearning.py \
    --before_path "$BEFORE_PATH" \
    --after_path "$AFTER_PATH" \
    --val_path "$VAL_PATH" \
    --test_path "$TEST_PATH" \
    --id2title_path "$ID2TITLE_PATH" \
    --output_train_path "$OUTPUT_TRAIN_PATH" \
    --output_val_path "$OUTPUT_VAL_PATH" \
    --output_test_path "$OUTPUT_TEST_PATH" \
    --unlearning_data_path "$UNLEARNING_DATA_PATH" \
    --gpu_id "$GPU_ID" \
    --test_llm_path "$TEST_LLM_DATA"