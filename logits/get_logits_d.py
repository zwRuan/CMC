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


# model,tokenizer = load_model("../Mistral-7B-v0.1")
# model.to("cuda")
# print("Mistral-7B-v0.1")
# for i in tqdm(range(50)):
#     directory="logits/mistral_llama_alpaca_gpt4/{}".format(i)
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#     ids_query=torch.load("logits/mistral_llama_alpaca_gpt4/input_output_ids/{}_input_vanilla.pt".format(i))
#     ids_response=torch.load("logits/mistral_llama_alpaca_gpt4/input_output_ids/{}_output.pt".format(i))
#     ids_input = torch.cat([ids_query,ids_response[:-1]],dim=0).unsqueeze(0)
#     output_logits=model(ids_input).logits.to("cpu")
#     save_tensor=output_logits[0][len(ids_query):,:]
#     torch.save(save_tensor,"logits/mistral_llama_alpaca_gpt4/{}/Mistral-7B-v0.1.pt".format(i))

# model = PeftModel.from_pretrained(
#     model,
#     "output/lora/Mistral-7B-v0.1_lora_alpaca_gpt4",
#     torch_dtype=torch.float16,
# )
# model = model.merge_and_unload()
# print("Mistral-7B-v0.1+lora")
# for i in tqdm(range(50)):
#     ids_query=torch.load("logits/mistral_llama_alpaca_gpt4/input_output_ids/{}_input_chat.pt".format(i))
#     ids_response=torch.load("logits/mistral_llama_alpaca_gpt4/input_output_ids/{}_output.pt".format(i))
#     ids_input = torch.cat([ids_query,ids_response[:-1]],dim=0).unsqueeze(0)
#     output_logits=model(ids_input).logits.to("cpu")
#     save_tensor=output_logits[0][len(ids_query):,:]
#     torch.save(save_tensor,"logits/mistral_llama_alpaca_gpt4/{}/Mistral-7B-v0.1+lora.pt".format(i))
# del model
# gc.collect()
# torch.cuda.empty_cache()


tokenizer_mistral = load_tokenizer("../Mistral-7B-v0.1")
model,tokenizer = load_model("../Llama-2-13b-hf")
model.to("cuda")
# print("Llama-2-7b-hf")
# for i in tqdm(range(50)):
#     ids_query=torch.load("logits/mistral_llama_alpaca_gpt4/input_output_ids/{}_input_vanilla.pt".format(i))
#     ids_response=torch.load("logits/mistral_llama_alpaca_gpt4/input_output_ids/{}_output.pt".format(i))
#     ids_input = torch.cat([ids_query,ids_response[:-1]],dim=0).unsqueeze(0)
#     @torch.no_grad()
#     def func():
#         len_query = len(ids_query)
#         len_input = len(ids_input[0])

#         llama_model_outputs_logits=[]
#         for j in range(len_query,len_input):
#             now_input=tokenizer_mistral.decode(ids_input[0][:j],skip_special_tokens=False)
#             input_llama=tokenizer.encode(now_input, return_tensors="pt")[:,1:].to("cuda")
#             output_logits=model(input_llama).logits[:,-1,:].unsqueeze(0)
#             llama_model_outputs_logits.append(output_logits)
#         llama_model_outputs_logits = torch.cat(llama_model_outputs_logits, dim=1).to("cpu")
#         torch.save(llama_model_outputs_logits[0],"logits/mistral_llama_alpaca_gpt4/{}/Llama-2-13b-hf.pt".format(i))

#     func()

model = PeftModel.from_pretrained(
    model,
    "output/lora/Llama-2-13b-hf_gsm8k",
    torch_dtype=torch.float16,
)
model = model.merge_and_unload()
print("Llama-2-13b-hf+lora")
for i in tqdm(range(50)):
    ids_query=torch.load("logits/mistral_llama_alpaca_gpt4/input_output_ids/{}_input_chat.pt".format(i))
    ids_response=torch.load("logits/mistral_llama_alpaca_gpt4/input_output_ids/{}_output.pt".format(i))
    ids_input = torch.cat([ids_query,ids_response[:-1]],dim=0).unsqueeze(0)
    @torch.no_grad()
    def func():
        len_query = len(ids_query)
        len_input = len(ids_input[0])

        llama_model_outputs_logits=[]
        for j in range(len_query,len_input):
            now_input=tokenizer_mistral.decode(ids_input[0][:j],skip_special_tokens=False)
            input_llama=tokenizer.encode(now_input, return_tensors="pt")[:,1:].to("cuda")
            output_logits=model(input_llama).logits[:,-1,:].unsqueeze(0)
            llama_model_outputs_logits.append(output_logits)
        llama_model_outputs_logits = torch.cat(llama_model_outputs_logits, dim=1).to("cpu")
        torch.save(llama_model_outputs_logits[0],"logits/mistral_llama_alpaca_gpt4/{}/Llama-2-13b-hf+lora_gsm8k_long.pt".format(i))

    func()

