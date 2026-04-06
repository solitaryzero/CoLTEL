MODEL_NAME="Qwen3-8B"
MODEL_IDENTIFIER_OR_PATH="./models/Qwen3-8B"
SEED_LEN=2
LORA_R=64
LORA_ALPHA=128
LORA_DROPOUT=0.05

accelerate launch --multi_gpu --num_processes=2 --gpu_ids 2,3 ./src/coltel/train.py \
    --base_model ${MODEL_IDENTIFIER_OR_PATH} \
    --dictionary_path ./data/kilt \
    --train_data_path ./data/kilt \
    --eval_data_path ./data/kilt \
    --recipe entity \
    --seed_len ${SEED_LEN} \
    --projector_type 2xMLP \
    --accelerate \
    --bf16 \
    --use_lora \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --save_model_path ./models/${MODEL_NAME}_stage1_entity_r${LORA_R}_a${LORA_ALPHA} \
    --save_result_path ./output/${MODEL_NAME}_stage1_entity_r${LORA_R}_a${LORA_ALPHA} \
    --do_eval \
    --learning_rate 1e-5 \
    --grad_clip 1 \
    --train_batch_size 4 \
    --epoch 1 \
    --logging_steps 200 \
    --num_examples 400000 \
    --max_length 2048 \
    --max_new_tokens 64 \
    --report_to none \
    --seed 42