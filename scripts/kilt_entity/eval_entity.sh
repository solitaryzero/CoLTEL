MODEL_NAME="Qwen3-8B"
MODEL_IDENTIFIER_OR_PATH="./models/Qwen3-8B"
SEED_LEN=2

CUDA_VISIBLE_DEVICES=7 python ./src/coltel/eval.py \
    --base_model ${MODEL_IDENTIFIER_OR_PATH} \
    --dictionary_path ./data/kilt \
    --eval_data_path ./data/kilt \
    --recipe entity \
    --seed_len ${SEED_LEN} \
    --projector_type 2xMLP \
    --bf16 \
    --use_lora \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0 \
    --tuned_model_path ./models/${MODEL_NAME}_stage1_entity \
    --save_result_path ./output/${MODEL_NAME}_stage1_entity \
    --num_examples 500000 \
    --max_length 2048 \
    --max_new_tokens 64 \
    --report_to none \
    --seed 42