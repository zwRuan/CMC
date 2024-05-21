from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments
)
import torch
import json
from tqdm import tqdm
from peft import PeftModel
import gc
import os


def load_model(model_path):
    config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": None,
        "revision": "main",
        "use_auth_token": None,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        **config_kwargs,
        padding_side="right"
    )
    config = AutoConfig.from_pretrained(model_path, **config_kwargs)
    print("loading {}...".format(model_path))
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        **config_kwargs
    )
    
    return model,tokenizer

def load_tokenizer(model_path):
    config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": None,
        "revision": "main",
        "use_auth_token": None,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        **config_kwargs,
        padding_side="right"
    )
    return tokenizer

# model,tokenizer = load_model("../Llama-2-13b-hf")
# model.to("cuda")
# print("Llama-2-13b-hf")
# for i in tqdm(range(50)):
#     directory="logits/13b_7b_alpaca_gpt4/{}".format(i)
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#     ids_query=torch.load("logits/13b_7b_alpaca_gpt4/input_output_ids/{}_input_vanilla.pt".format(i))
#     ids_response=torch.load("logits/13b_7b_alpaca_gpt4/input_output_ids/{}_output.pt".format(i))
#     ids_input = torch.cat([ids_query,ids_response[:-1]],dim=0).unsqueeze(0)
#     output_logits=model(ids_input).logits.to("cpu")
#     save_tensor=output_logits[0][len(ids_query):,:]
#     torch.save(save_tensor,"logits/13b_7b_alpaca_gpt4/{}/Llama-2-13b-hf.pt".format(i))

# model = PeftModel.from_pretrained(
#     model,
#     "output/lora/Llama-2-13b-hf_lora_alpaca_gpt4",
#     torch_dtype=torch.float16,
# )
# model = model.merge_and_unload()
# print("Llama-2-13b-hf+lora")
# for i in tqdm(range(50)):
#     ids_query=torch.load("logits/13b_7b_alpaca_gpt4/input_output_ids/{}_input_chat.pt".format(i))
#     ids_response=torch.load("logits/13b_7b_alpaca_gpt4/input_output_ids/{}_output.pt".format(i))
#     ids_input = torch.cat([ids_query,ids_response[:-1]],dim=0).unsqueeze(0)
#     output_logits=model(ids_input).logits.to("cpu")
#     save_tensor=output_logits[0][len(ids_query):,:]
#     torch.save(save_tensor,"logits/13b_7b_alpaca_gpt4/{}/Llama-2-13b-hf+lora.pt".format(i))
# del model
# gc.collect()
# torch.cuda.empty_cache()


# model,tokenizer = load_model("../Llama-2-7b-hf")
# model.to("cuda")
# print("Llama-2-7b-hf")
# for i in tqdm(range(50)):
#     ids_query=torch.load("logits/13b_7b_alpaca_gpt4/input_output_ids/{}_input_vanilla.pt".format(i))
#     ids_response=torch.load("logits/13b_7b_alpaca_gpt4/input_output_ids/{}_output.pt".format(i))
#     ids_input = torch.cat([ids_query,ids_response[:-1]],dim=0).unsqueeze(0)
#     output_logits=model(ids_input).logits.to("cpu")
#     save_tensor=output_logits[0][len(ids_query):,:]
#     torch.save(save_tensor,"logits/13b_7b_alpaca_gpt4/{}/Llama-2-7b-hf.pt".format(i))

# model = PeftModel.from_pretrained(
#     model,
#     "output/lora/Llama-2-7b-hf_lora_alpaca_gpt4",
#     torch_dtype=torch.float16,
# )
# print("Llama-2-7b-hf+lora")
# model = model.merge_and_unload()
# for i in tqdm(range(50)):
#     ids_query=torch.load("logits/13b_7b_alpaca_gpt4/input_output_ids/{}_input_chat.pt".format(i))
#     ids_response=torch.load("logits/13b_7b_alpaca_gpt4/input_output_ids/{}_output.pt".format(i))
#     ids_input = torch.cat([ids_query,ids_response[:-1]],dim=0).unsqueeze(0)
#     output_logits=model(ids_input).logits.to("cpu")
#     save_tensor=output_logits[0][len(ids_query):,:]
#     torch.save(save_tensor,"logits/13b_7b_alpaca_gpt4/{}/Llama-2-7b-hf+lora.pt".format(i))


model,tokenizer = load_model("../Llama-2-7b-hf")
model.to("cuda")

model = PeftModel.from_pretrained(
    model,
    "output/lora/Llama-2-7b-hf_gsm8k",
    torch_dtype=torch.float16,
)
print("Llama-2-7b-hf+lora")
model = model.merge_and_unload()
for i in tqdm(range(50)):
    ids_query=torch.load("logits/13b_7b_alpaca_gpt4/input_output_ids/{}_input_chat.pt".format(i))
    ids_response=torch.load("logits/13b_7b_alpaca_gpt4/input_output_ids/{}_output.pt".format(i))
    ids_input = torch.cat([ids_query,ids_response[:-1]],dim=0).unsqueeze(0)
    output_logits=model(ids_input).logits.to("cpu")
    save_tensor=output_logits[0][len(ids_query):,:]
    torch.save(save_tensor,"logits/13b_7b_alpaca_gpt4/{}/Llama-2-7b-hf+lora_gsm8k_long.pt".format(i))



