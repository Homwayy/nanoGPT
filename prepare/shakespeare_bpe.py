import os
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer.bpe_tokenizer import BPETokenizer

#直接复用char版下载好的原始文本
input_file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'shakespeare_char', 'input.txt')
if not os.path.exists(input_file_path):
    raise FileNotFoundError("找不到 input.txt,请先运行 python -m prepare.shakespeare_char")

#产物统一放到 data/shakespeare_bpe/
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'shakespeare_bpe')
os.makedirs(data_dir, exist_ok=True)

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

#构建训练和验证数据集
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

#只用训练集训练tokenizer，验证集不参与
vocab_size = 1024
tok = BPETokenizer.from_text(train_data, vocab_size, special_tokens=['<|endoftext|>'], verbose=True)
print(f"vocab size: {tok.vocab_size}")

#数据集编码并输出bin文件，原始语料用encode_ordinary，不把文本里的特殊token当真
train_ids = np.array(tok.encode_ordinary(train_data), dtype=np.uint16)
val_ids = np.array(tok.encode_ordinary(val_data), dtype=np.uint16)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")
print(f"compression: {len(data.encode('utf-8')) / (len(train_ids) + len(val_ids)):.2f} bytes/token")
train_ids.tofile(os.path.join(data_dir, 'train.bin'))
val_ids.tofile(os.path.join(data_dir, 'val.bin'))

tok.save(os.path.join(data_dir, 'meta.pkl'))
