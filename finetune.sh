export HUGGINGFACE_HUB_CACHE=/workspace/CACHE

CUDA_VISIBLE_DEVICES=0 python train_by_text/train_delta_model.py \
    --ref_base_model_name_or_path meta-llama/Llama-2-7b-hf \
    --tgt_model_name_or_path nickypro/tinyllama-110M \
    --finetuning_type full \
    --do_train \
    --dataset alpaca_gpt4 \
    --max_source_length 1024 \
    --max_target_length 1024 \
    --template open-instruct \
    --output_dir results \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --save_strategy "no" \
    --learning_rate 2e-4 \
    --num_train_epochs 3 \
    --warmup_ratio 0.03 \
    --plot_loss \
    --bf16 \
    --tgt_model_train_from_config no \
    --padding_side right \
    --normalization "basenormalization"