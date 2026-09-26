import os
import requests
import numpy as np

from tokenizer.char_tokenizer import CharTokenizer

#产物统一放到 data/shakespeare_char/
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'shakespeare_char')
os.makedirs(data_dir, exist_ok=True)

#下载数据集
input_file_path = os.path.join(data_dir, 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    resp = requests.get(data_url, timeout=30)
    resp.raise_for_status()
    with open(input_file_path, 'w', encoding='utf-8', newline='') as f:
        f.write(resp.text)

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

#构建词表
tok = CharTokenizer.from_text(data)
print("all the unique characters:", ''.join(tok.chars))
print(f"vocab size: {tok.vocab_size}")

#构建训练和验证数据集
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

#数据集编码并输出bin文件
train_ids = np.array(tok.encode(train_data), dtype=np.uint16)
val_ids = np.array(tok.encode(val_data), dtype=np.uint16)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")
train_ids.tofile(os.path.join(data_dir, 'train.bin'))
val_ids.tofile(os.path.join(data_dir, 'val.bin'))

tok.save(os.path.join(data_dir, 'meta.pkl'))
