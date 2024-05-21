import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import torch.nn as nn
from geomloss import SamplesLoss

def euclidean_distance(tensor1, tensor2):
    return torch.norm(tensor1 - tensor2).item()

def manhattan_distance(tensor1, tensor2):
    return torch.sum(torch.abs(tensor1 - tensor2))

def pearson_correlation(tensor1, tensor2):
    tensor1_mean = torch.mean(tensor1)
    tensor2_mean = torch.mean(tensor2)
    tensor1_centered = tensor1 - tensor1_mean
    tensor2_centered = tensor2 - tensor2_mean

    correlation = (tensor1_centered @ tensor2_centered) / (torch.sqrt((tensor1_centered**2).sum()) * torch.sqrt((tensor2_centered**2).sum()))
    return correlation

def jaccard_similarity(tensor1, tensor2):
    intersection = torch.logical_and(tensor1, tensor2).float().sum()
    union = torch.logical_or(tensor1, tensor2).float().sum()
    return intersection / union

with open("token_maps/mistral2llama/one2one_mistral2llama_id.json") as f:
    token_map=json.load(f)
token_map=torch.tensor(token_map)
map_fixed = torch.where(token_map == -1, torch.tensor(0), token_map)

temp=[]
for i in tqdm(range(26,27)):

    base_llama=torch.load("logits/mistral_llama_alpaca_gpt4/{}/Llama-2-13b-hf.pt".format(i)).detach()
    chat_llama=torch.load("logits/mistral_llama_alpaca_gpt4/{}/Llama-2-13b-hf+lora.pt".format(i)).detach()
    # chat_llama=torch.load("logits/mistral_llama_alpaca_gpt4/{}/Llama-2-7b-hf+lora_gsm8k_long.pt".format(i)).detach()
    base_mistral=torch.load("logits/mistral_llama_alpaca_gpt4/{}/Mistral-7B-v0.1.pt".format(i)).detach()
    chat_mistral=torch.load("logits/mistral_llama_alpaca_gpt4/{}/Mistral-7B-v0.1+lora.pt".format(i)).detach()

    base_llama=F.log_softmax(base_llama, dim=-1)
    chat_llama=F.log_softmax(chat_llama, dim=-1)
    base_mistral=F.log_softmax(base_mistral, dim=-1)
    chat_mistral=F.log_softmax(chat_mistral, dim=-1)

    base_llama = base_llama[:, map_fixed]
    chat_llama = chat_llama[:, map_fixed]
    base_llama[:, token_map == -1] = 0
    chat_llama[:, token_map == -1] = 0


    _, indices = torch.sort(chat_mistral, descending=True)

    top=100

    base_mistral = torch.gather(base_mistral, 1, indices[:,:top])
    chat_mistral = torch.gather(chat_mistral, 1, indices[:,:top])
    base_llama = torch.gather(base_llama, 1, indices[:,:top])
    chat_llama = torch.gather(chat_llama, 1, indices[:,:top])


    delta_mistra=chat_mistral-base_mistral
    delta_llama=chat_llama-base_llama

    # for j in range(delta_mistra.shape[0]):
    #     temp.append(manhattan_distance(delta_mistra[j],delta_llama[j])/top)

    # loss=F.cosine_similarity
    # sim=loss(chat_mistral, base_mistral)
    # temp.append(torch.mean(sim).item())

    loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=0.01)
    temp.append(loss_fn(delta_mistra,delta_llama)/top)


    _, indices_delta = torch.sort(delta_mistra, descending=True)

    top=50
    delta_mistra=torch.gather(delta_mistra, 1, indices_delta[:top,:])
    delta_llama=torch.gather(delta_llama, 1, indices_delta[:top,:])

    plt.figure(figsize=(5, 5))

    fs=12
    plt.subplot(2, 1, 1) 
    plt.title('logits shifts on Mistral-7B', fontsize=fs)
    plt.ylabel('time steps', fontsize=fs)
    plt.xlabel('token logits shifts (sorted)', fontsize=fs)
    plt.imshow(delta_mistra)
    plt.colorbar(pad=0.02) 
    plt.tick_params(axis='x', labelsize=fs)
    plt.tick_params(axis='y', labelsize=fs)


    plt.subplot(2, 1, 2)
    plt.title('logits shifts on Llama2-13b', fontsize=fs)
    plt.ylabel('time steps', fontsize=fs)
    plt.xlabel('token logits shifts (reordered by figure above)', fontsize=fs)
    plt.imshow(delta_llama)
    plt.colorbar(pad=0.02)
    plt.tick_params(axis='x', labelsize=fs)
    plt.tick_params(axis='y', labelsize=fs)

    plt.tight_layout()

    # plt.savefig("logits/mistral_llama_alpaca_gpt4/heatmap/{}.png".format(i))
    # plt.savefig("logits/mistral_llama_alpaca_gpt4/heatmap_13b/{}.png".format(i))
    plt.savefig("logits/mistral_llama_alpaca_gpt4/mistral_llama13b.pdf", bbox_inches='tight')
    plt.clf()

# print(temp)
# print(sum(temp)/len(temp))

# max_value = max(temp)
# max_index = temp.index(max_value)
# print(max_value)
# print(max_index)

