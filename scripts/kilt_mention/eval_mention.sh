MODEL_NAME="Qwen3-8B"
MODEL_IDENTIFIER_OR_PATH="./models/Qwen3-8B"
SEED_LEN=8
LORA_R=128
LORA_ALPHA=128
LORA_DROPOUT=0.05

CUDA_VISIBLE_DEVICES=0 python ./src/coltel/eval.py \
    --base_model ${MODEL_IDENTIFIER_OR_PATH} \
    --dictionary_path ./data/kilt \
    --eval_data_path ./data/kilt \
    --recipe mention \
    --seed_len ${SEED_LEN} \
    --projector_type 2xMLP \
    --bf16 \
    --use_lora \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --tuned_model_path ./models/${MODEL_NAME}_stage2_mention_r${LORA_R}_a${LORA_ALPHA}_sl${SEED_LEN} \
    --save_result_path ./output/${MODEL_NAME}_stage2_mention_r${LORA_R}_a${LORA_ALPHA}_sl${SEED_LEN} \
    --num_examples 400000 \
    --max_length 2048 \
    --max_new_tokens 64 \
    --report_to none \
    --seed 42