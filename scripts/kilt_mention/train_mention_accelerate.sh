MODEL_NAME="Qwen3-8B"
MODEL_IDENTIFIER_OR_PATH="./models/Qwen3-8B"
TUNED_MODEL_PATH="./models/Qwen3-8B_stage1_entity"
SEED_LEN=2

accelerate launch --multi_gpu --num_processes=2 ./src/coltel/train.py \
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
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0 \
    --save_model_path ./models/${MODEL_NAME}_stage1_entity \
    --save_result_path ./output/${MODEL_NAME}_stage1_entity \
    --do_eval \
    --learning_rate 1e-5 \
    --grad_clip 1 \
    --train_batch_size 4 \
    --epoch 1 \
    --logging_steps 200 \
    --num_examples 200000 \
    --max_length 2048 \
    --max_new_tokens 64 \
    --report_to none \
    --seed 42