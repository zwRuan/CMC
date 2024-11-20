from transformers import AutoTokenizer

# 加载两个模型的tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

# 提取词汇表
llama_vocab = set(llama_tokenizer.get_vocab().keys())
qwen_vocab = set(qwen_tokenizer.get_vocab().keys())

# 计算交集
common_vocab = llama_vocab.intersection(qwen_vocab)

# 计算重合度
overlap_ratio = len(common_vocab) / min(len(llama_vocab), len(qwen_vocab))

print(f"重合的词汇数量: {len(common_vocab)}")
print(f"重合比例: {overlap_ratio:.2%}")
