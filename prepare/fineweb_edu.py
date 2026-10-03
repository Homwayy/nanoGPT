import os
import numpy as np
import pyarrow.parquet as pq
import tiktoken
from huggingface_hub import hf_hub_download

#FineWeb-Edu sample-10BT的第一个shard，约7.5亿token
REPO_ID = 'HuggingFaceFW/fineweb-edu'
SHARD = 'sample/10BT/000_00000.parquet'
TRAIN_TOKENS = 500_000_000
VAL_TOKENS = 5_000_000
VAL_PROB = 0.015  #文档划入val的概率
SEED = 1337

data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'fineweb_edu')
os.makedirs(data_dir, exist_ok=True)

#下载数据集，已下载则直接复用
parquet_path = hf_hub_download(REPO_ID, SHARD, repo_type='dataset', local_dir=os.path.join(data_dir, 'raw'))
pf = pq.ParquetFile(parquet_path)

#文档按抓取时间排序，打乱row group顺序后再随机划分train/val
rng = np.random.default_rng(SEED)
order = rng.permutation(pf.metadata.num_row_groups)

enc = tiktoken.get_encoding("gpt2")
eot = enc.eot_token  #50256

#train写入memmap临时文件，完成后再重命名
train_path = os.path.join(data_dir, 'train.bin')
train = np.memmap(train_path + '.tmp', dtype=np.uint16, mode='w+', shape=(TRAIN_TOKENS,))
val = []
n_train, n_val = 0, 0

for i, rg in enumerate(order):
    texts = pf.read_row_group(rg, columns=['text']).column('text').to_pylist()
    for ids in enc.encode_ordinary_batch(texts):
        #文档开头加<|endoftext|>分隔
        doc = np.array([eot] + ids, dtype=np.uint16)
        if n_val < VAL_TOKENS and rng.random() < VAL_PROB:
            val.append(doc)
            n_val += len(doc)
        elif n_train < TRAIN_TOKENS:
            #截断最后一篇文档，凑满TRAIN_TOKENS
            take = min(len(doc), TRAIN_TOKENS - n_train)
            train[n_train:n_train + take] = doc[:take]
            n_train += take
    if i % 50 == 0:
        print(f"row group {i}/{len(order)}: train {n_train:,}/{TRAIN_TOKENS:,}, val {n_val:,}/{VAL_TOKENS:,}")
    if n_train >= TRAIN_TOKENS and n_val >= VAL_TOKENS:
        break

if n_train < TRAIN_TOKENS or n_val < VAL_TOKENS:
    raise RuntimeError(f"shard不够用:train {n_train:,},val {n_val:,},需要调小TRAIN_TOKENS或者多下载一个shard")

train.flush()
del train
os.replace(train_path + '.tmp', train_path)
val_ids = np.concatenate(val)
val_ids.tofile(os.path.join(data_dir, 'val.bin'))
print(f"train has {n_train:,} tokens")
print(f"val has {len(val_ids):,} tokens ({len(val):,} docs)")
