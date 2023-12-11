source /opt/intel/oneapi/setvars.sh
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export WANDB_DISABLED=true

python dpo_clm_qlora.py \
  --model_name_or_path "./finetuned_model" \
  --output_dir "mistral_7b_dpo" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-4 \
  --max_steps 1000 \
  --save_steps 10 \
  --logging_steps 10 \
  --lora_alpha 16 \
  --lora_rank 16 \
  --lora_dropout 0.05 \
  --dataset_name ./data \
  --pad_max true \
  --bf16 \
  --use_auth_token True \
  --max_length 128 \
  --max_prompt_length 56 \
  --lr_scheduler_type "cosine" \
  --warmup_steps 100 \
  --device xpu \
  --gradient_checkpointing true