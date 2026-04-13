MODEL_NAME="Qwen3-8B"
MODEL_IDENTIFIER_OR_PATH="./models/Qwen3-8B"
SEED_LEN=8
LORA_R=128
LORA_ALPHA=128
LORA_DROPOUT=0.05
TUNED_MODEL_PATH="./models/Qwen3-8B_stage1_entity_r${LORA_R}_a${LORA_ALPHA}_sl${SEED_LEN}"

accelerate launch --multi_gpu --num_processes=2 --gpu_ids 0,1 ./src/coltel/train.py \
    --base_model ${MODEL_IDENTIFIER_OR_PATH} \
    --dictionary_path ./data/kilt \
    --train_data_path ./data/kilt \
    --eval_data_path ./data/kilt \
    --tuned_model_path ${TUNED_MODEL_PATH} \
    --recipe entity \
    --seed_len ${SEED_LEN} \
    --projector_type 2xMLP \
    --accelerate \
    --bf16 \
    --use_lora \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --save_model_path ./models/${MODEL_NAME}_stage2_mention_r${LORA_R}_a${LORA_ALPHA}_sl${SEED_LEN} \
    --save_result_path ./output/${MODEL_NAME}_stage2_mention_r${LORA_R}_a${LORA_ALPHA}_sl${SEED_LEN} \
    --do_eval \
    --learning_rate 2e-4 \
    --grad_clip 1 \
    --train_batch_size 4 \
    --epoch 1 \
    --logging_steps 200 \
    --num_examples 400000 \
    --max_length 2048 \
    --max_new_tokens 64 \
    --report_to none \
    --seed 42