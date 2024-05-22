Cross-model Control: Improving Multiple Large Language Models in One-time Training
==================================================================================

Download Models from Huggingface:
--------------
For instruction tuning: [Llama2-7b](https://huggingface.co/meta-llama/Llama-2-7b-hf), [Llama2-13b](https://huggingface.co/meta-llama/Llama-2-13b-hf), [Llama2-70b](https://huggingface.co/meta-llama/Llama-2-70b-hf), [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)

For unlearning: [tofu_ft_llama2-7b](https://huggingface.co/locuslab/tofu_ft_llama2-7b), [Llama2-13b-chat](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf), [
Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)

For delta model: [tinyllama-110M](https://huggingface.co/nickypro/tinyllama-110M), [tinyllama-42M](https://huggingface.co/nickypro/tinyllama-42M), [tinyllama-15M](https://huggingface.co/nickypro/tinyllama-15M)

Experiment on Instrution Tuning:
--------------
Train delta model with template model:
```bash
CUDA_VISIBLE_DEVICES=0 python train_by_text/train_delta_model.py \
    --ref_base_model_name_or_path ${llama7b_model_path} \
    --tgt_model_name_or_path ${tinyllama_model_path} \
    --finetuning_type full \
    --do_train \
    --dataset alpaca_gpt4 \
    --max_source_length 1024 \
    --max_target_length 1024 \
    --template open-instruct \
    --output_dir ${model_save_path} \
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
```

Evaluate on AlpacaEval for Llama vocabluary user models with delta model:
```bash
CUDA_VISIBLE_DEVICES=0 python eval/predict.py \
    --base_model_path ${user_model_path} \
    --dataset_file_path data/alpaca_eval.json \
    --delta_model_path ${delta_model_path}$ \
    --template open-instruct \
    --generator_name ${generator_name} \
    --predicted_out_filename ${predicted_out_filename} \
    --alpha 1.0

export OPENAI_API_KEY="openai key"
python eval/eval_alpacafarm.py \
    --model_results_file ${predicted_out_filename} \
    --output_file ${output_file} \
    --reference_outputs data/alpaca_farm_evaluation_merge.json 
```

Evaluate on AlpacaEval for other vocabluaries user models (e.g., Mistral-7B) with delta model:
```bash
CUDA_VISIBLE_DEVICES=0 python eval/predict_d.py \
    --base_model_path ${Mistral-7B_path} \
    --delta_model_path output/delta_model_from_text/alpaca_gpt4_all_110M_open-instruct_4epochs \
    --dataset_file_path data/alpaca_eval/alpaca_eval.json \
    --template open-instruct \
    --generator_name ${generator_name} \
    --predicted_out_filename ${output_file_path} \
    --base2delta_map_path token_maps/mistral2llama/one2one_mistral2llama \
    --alpha 1.0

export OPENAI_API_KEY="openai key"
python eval/eval_alpacafarm.py \
    --model_results_file ${predicted_out_filename} \
    --output_file ${output_file} \
    --reference_outputs data/alpaca_farm_evaluation_merge.json 
```

Experiment on Unlearning:
--------------
Finetune chat models on TOFU Forget Set and Retrain Set.
```bash
cd tofu/
# Fill in the model path in config/model_config.yaml first
split="full"
model="llama2-13b"
lr=1e-4
CUDA_VISIBLE_DEVICES=0 python finetune.py --config-name=finetune_lora.yaml split=${split} model_family=${model} lr=${lr}
```


Train delta model with template model:
```bash
cd tofu/
split="forget10"
model="llama2-7b"
lr=1e-4
model_path="../tofu_ft_llama2-7b"
num_epochs=20
batch_size=4
forget_loss="grad_diff"
normalization="basenormalization"
delta_model_name="${forget_loss}_delta_110M_${normalization}_lr${lr}_epoch${num_epochs}_bs${batch_size}"
save_dir="model_output/${delta_model_name}"
delta_model_path="../tinyllama-110M"


CUDA_VISIBLE_DEVICES=0 python forget.py --config-name=forget_delta.yaml split=${split} model_family=${model} lr=${lr} model_path=$model_path num_epochs=$num_epochs save_dir=$save_dir forget_loss=$forget_loss delta_model_path=$delta_model_path batch_size=$batch_size normalization=$normalization
```
Evaluate: 
```bash
cd tofu/
alpha=1.0
model_family="llama2-13b"
split="forget10_perturbed"
base_model_name="llama2-13b_epochs10_lr0.0001_bs1_lora"
model_path="paper_models/${base_model_name}"
save_dir="eval_results/${base_model_name}+${delta_model_name}_${alpha}"
delta_model_path="model_output/${delta_model_name}"
# Add this when evaluate with Misral-7B
# token_map="token_maps/mistral2llama/one2one_mistral2llama"


CUDA_VISIBLE_DEVICES=0 python evaluate_util.py \
 model_family=$model_family split=$split\
 model_path=$model_path save_dir=$save_dir delta_model_path=$delta_model_path token_map=$token_map batch_size=1 alpha=$alpha

path_to_aggregated_retain_result="eval_results/${base_model_name}+${delta_model_name}_${alpha}/eval_log_aggregated.json"
method_name="${base_model_name}+${delta_model_name}"
save_filename="eval_results/${base_model_name}+${delta_model_name}_${alpha}/aggregate_eval_stat.csv"

python aggregate_eval_stat.py retain_result=${path_to_aggregated_retain_result} ckpt_result=${path_to_aggregated_retain_result} \
 method_name=${method_name} save_file=${save_filename}
```

