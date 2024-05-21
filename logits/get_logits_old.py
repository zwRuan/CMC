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

# model,tokenizer = load_model("../Mistral-7B-Instruct-v0.1")
# model,tokenizer = load_model("../Mistral-7B-v0.1")
# tokenizer_mistral = load_tokenizer("../Mistral-7B-v0.1")
# tokenizer = load_tokenizer("../Llama-2-7b-hf")

model,tokenizer = load_model("../Llama-2-13b-hf")
# model = PeftModel.from_pretrained(
#     model,
#     "output/lora/Llama-2-7b-hf_lora_alpaca_gpt4",
#     torch_dtype=torch.float16,
# )
# model = model.merge_and_unload()
model.to("cuda")

# query = "<|user|>\nHow do I wrap a present neatly?\n<|assistant|>\n"
# ids_query = torch.tensor([    1,   529, 29989,  1792, 29989, 29958,    13,  5328,   437,   306,
#         12244,   263,  2198, 28539,   368, 29973,    13, 29966, 29989,   465,
#         22137, 29989, 29958,    13], device='cuda:0')
# query = "How do I wrap a present neatly?"
ids_query = torch.tensor([    1,  1128,   437,   306, 12244,   263,  2198, 28539,   368, 29973],
       device='cuda:0')
ids_response = torch.tensor([  399,   336,  3262,   263,  2198, 28539,   368,   508,   788,   304,
          278, 29163,   322, 23483,   362,   310,   278, 23957,   993, 29889,
         2266,   526,   777,  6576,   304,  1101,   304, 12244,   263,  2198,
        28539,   368, 29901,    13,    13, 29896, 29889, 14542,   852,   278,
         1492, 28489,  5650, 29901,  7605,   263,  5650,   393,  1614,  1860,
          278, 10039,   322,   278, 19797, 29889,   319,  5405, 15055,   411,
         3144,  5171,   470,  1701,   309,   408,   896,   508,   367,  5189,
          304,  3349, 29889,    13,    13, 29906, 29889,  2191,  3745,   278,
        19797, 29901,  2191,  3745,   278,  3309, 29892,  2920, 29892,   322,
         3171,   310,   278, 19797,   304,  8161,   278,  2159,   310,   278,
        28489,  5650,   366,   817, 29889,    13,    13, 29941, 29889,   315,
          329,   278,  5650, 29901,   315,   329,   278,  5650,   304,   278,
         8210,  2159, 29892, 10124,   263,  2846, 22831,  4805,   373,  1269,
         2625,   363,   900,  8497, 29889,    13,    13, 29946, 29889,   383,
         1025,   278,  5650, 29901,   383,  1025,   278,  5650,   297,  4203,
         3309,  3538, 29892,   769,   900, 29881,   278,  2246,   322,  5970,
        12770,   297,  1328,   304,  1653,   263, 17205,  8267, 29889,    13,
           13, 29945, 29889, 15484,   278, 19797, 29901,  7817,   278, 19797,
          373,   278,   900,  7176,  5650, 29892,  3907,  1854,   372,   338,
        11592,   368,   297,  2058, 29889,    13,    13, 29953, 29889,   399,
         2390,   278,  5650, 29901,  7370, 28489,   278,  5650,  2820,   278,
        19797, 29892,  3907,  1854,   304, 10597,   714,   738,  2358,   682,
          793,   408,   366,   748, 29889,    13,    13, 29955, 29889,   323,
         4085,   278,  5650, 29901,  4803,   260,  4085,   304, 11592,   278,
         5650,   304,   278, 19797, 29892,  3907,  1854,   304,   260,  4085,
          278, 12770,   322,   738, 23819, 10614, 29889,    13,    13, 29947,
        29889,  3462,   263, 12580, 29901,  3462,   263, 12580,   470, 18130,
         6718,   304,   278,  2246,   310,   278,  2198,   304,   788,   263,
        28321,  6023, 29889,    13,    13, 29929, 29889, 14350,   263,  4443,
        29901,   960,   366,   864,   304,  3160,   263,  4443, 29892,  2436,
          372,   373,   263,  8424,   310,  5650,   322,   260,  2707,   372,
         2768,   278,  2198, 29889,    13,    13,  2059,  1494,  1438,  6576,
        29892,   366,   508, 12244,   263,  2198, 28539,   368,   322,   788,
          304,   278, 29163,   310,   278, 23957,   993, 29889,     2],
       device='cuda:0')



ids_input = torch.cat([ids_query,ids_response[:-1]],dim=0).unsqueeze(0)

print(len(ids_response)-1)

output_logits=model(ids_input).logits.to("cpu")
save_tensor=output_logits[0][len(ids_query):,:]
print(save_tensor.shape)
torch.save(save_tensor,"logits/13b_7b_alpaca_gpt4/present/Llama-2-13b-hf.pt")

# @torch.no_grad()
# def func():
#     ids_query = tokenizer_mistral.encode(query, return_tensors="pt").to("cuda")
#     len_query = len(ids_query[0])
#     ids_input = tokenizer_mistral.encode(query+response, return_tensors="pt").to("cuda:1")
#     len_input = len(ids_input[0])
#     print(len_input-len_query)

#     llama_model_outputs_logits=[]
#     for i in tqdm(range(len_query,len_input)):
#         now_input=tokenizer_mistral.decode(ids_input[0][:i],skip_special_tokens=False)
#         input_llama=tokenizer.encode(now_input, return_tensors="pt")[:,1:].to("cuda:1")
#         output_logits=model(input_llama).logits[:,-1,:].unsqueeze(0)
#         llama_model_outputs_logits.append(output_logits)
#     llama_model_outputs_logits = torch.cat(llama_model_outputs_logits, dim=1).to("cpu")
#     print(llama_model_outputs_logits[0].shape)
#     torch.save(llama_model_outputs_logits[0],"logits/mistral_llama/Christmas/Llama-2-7b-hf.pt")

# func()


# ids_query = tokenizer.encode(query, return_tensors="pt").to("cuda")
# print(ids_query)
# output=model.generate(ids_query, temperature=1.0, top_p=1.0, do_sample=False, max_new_tokens=1000)
# print(output[0][:len_query])
# print(output[0][len_query:])

# print([tokenizer.decode(output[0][len_query:],skip_special_tokens=False)])
