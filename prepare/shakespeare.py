import os
import numpy as np
import tiktoken

#直接复用char版下载好的原始文本
input_file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'shakespeare_char', 'input.txt')
if not os.path.exists(input_file_path):
    raise FileNotFoundError("找不到 input.txt，请先运行 python -m prepare.shakespeare_char")

#产物放到 data/shakespeare/，GPT-2的tokenizer由tiktoken提供，不需要meta.pkl
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'shakespeare')
os.makedirs(data_dir, exist_ok=True)

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

#构建训练和验证数据集
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

#用GPT-2的BPE编码，原始语料用encode_ordinary，不识别特殊token
enc = tiktoken.get_encoding("gpt2")
train_ids = np.array(enc.encode_ordinary(train_data), dtype=np.uint16)
val_ids = np.array(enc.encode_ordinary(val_data), dtype=np.uint16)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")
train_ids.tofile(os.path.join(data_dir, 'train.bin'))
val_ids.tofile(os.path.join(data_dir, 'val.bin'))
