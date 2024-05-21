import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
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

temp=[]
for i in tqdm(range(8,9)):

    base_7b=torch.load("logits/13b_7b_alpaca_gpt4/{}/Llama-2-7b-hf.pt".format(i)).detach()
    chat_7b=torch.load("logits/13b_7b_alpaca_gpt4/{}/Llama-2-7b-hf+lora.pt".format(i)).detach()
    # chat_7b=torch.load("logits/13b_7b_alpaca_gpt4/{}/Llama-2-7b-hf+lora_gsm8k_long.pt".format(i)).detach()
    base_13b=torch.load("logits/13b_7b_alpaca_gpt4/{}/Llama-2-13b-hf.pt".format(i)).detach()
    chat_13b=torch.load("logits/13b_7b_alpaca_gpt4/{}/Llama-2-13b-hf+lora.pt".format(i)).detach()


    # base_7b=F.softmax(base_7b, dim=-1)
    # chat_7b=F.softmax(chat_7b, dim=-1)
    # base_13b=F.softmax(base_13b, dim=-1)
    # chat_13b=F.softmax(chat_13b, dim=-1)

    base_7b=F.log_softmax(base_7b, dim=-1)
    chat_7b=F.log_softmax(chat_7b, dim=-1)
    base_13b=F.log_softmax(base_13b, dim=-1)
    chat_13b=F.log_softmax(chat_13b, dim=-1)


    _, indices = torch.sort(base_13b, descending=True)

    top=100

    base_7b = torch.gather(base_7b, 1, indices[:,:top])
    chat_7b = torch.gather(chat_7b, 1, indices[:,:top])
    base_13b = torch.gather(base_13b, 1, indices[:,:top])
    chat_13b = torch.gather(chat_13b, 1, indices[:,:top])



    delta_7b=chat_7b-base_7b
    delta_13b=chat_13b-base_13b

    # for j in range(delta_13b.shape[0]):
    #     temp.append(manhattan_distance(delta_13b[j],delta_7b[j])/top)

    # loss=F.cosine_similarity
    # sim=loss(chat_13b, base_13b)
    # temp.append(torch.mean(sim).item())

    # loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=0.01)
    # temp.append(loss_fn(delta_13b,delta_7b)/top)


    _, indices_delta = torch.sort(delta_13b, descending=True)

    top=50
    delta_7b=torch.gather(delta_7b, 1, indices_delta[:top,:])
    delta_13b=torch.gather(delta_13b, 1, indices_delta[:top,:])

    plt.figure(figsize=(5, 5)) 

    fs=12
    plt.subplot(2, 1, 1) 
    plt.title('logits shifts on Llama2-13b', fontsize=fs)
    plt.ylabel('time steps', fontsize=fs)
    plt.xlabel('token logits shifts (sorted)', fontsize=fs)
    plt.imshow(delta_13b)
    plt.colorbar(pad=0.02) 
    plt.tick_params(axis='x', labelsize=fs)
    plt.tick_params(axis='y', labelsize=fs)



    plt.subplot(2, 1, 2)
    plt.imshow(delta_7b)
    plt.title('logits shifts on Llama2-7b', fontsize=fs)
    plt.ylabel('time steps', fontsize=fs)
    plt.xlabel('token logits shifts (reordered by figure above)', fontsize=fs)
    plt.colorbar(pad=0.02)
    plt.tick_params(axis='x', labelsize=fs)
    plt.tick_params(axis='y', labelsize=fs)

    plt.tight_layout()

    plt.savefig("logits/13b_7b_alpaca_gpt4/13b_7b.pdf", bbox_inches='tight')
    # # plt.savefig("logits/13b_7b_alpaca_gpt4/heatmap_1/{}.png".format(i))
    # # plt.savefig("logits/13b_7b_alpaca_gpt4/heatmap_2/{}.png".format(i))
    plt.clf()

# print(temp)
# print(sum(temp)/len(temp))
