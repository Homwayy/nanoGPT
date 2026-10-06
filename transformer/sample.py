import os
import time
import argparse
from contextlib import nullcontext

import torch
import tiktoken

from model import GPT, GPTConfig


ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

#top_k为0、top_p为1时不做对应的截断；temperature为0时每步取概率最大的token
#compile第一次运行要编译10秒以上，默认不开
defaults = dict(
    ckpt=os.path.join(ROOT, 'out', 'final', 'ckpt.pt'),
    prompt='<|endoftext|>',
    num_samples=3,
    max_new_tokens=200,
    temperature=0.8,
    top_k=0,
    top_p=0.9,
    seed=1337,
    kv_cache=True,
    cuda_graph=True,
    compile=False,
)

parser = argparse.ArgumentParser()
for k, v in defaults.items():
    if isinstance(v, bool):
        parser.add_argument(f'--{k}', action=argparse.BooleanOptionalAction, default=v)
    else:
        parser.add_argument(f'--{k}', type=type(v), default=v)
cfg = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(cfg.seed)
torch.set_float32_matmul_precision('high')
ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if device == 'cuda' else nullcontext()

ckpt = torch.load(cfg.ckpt, map_location='cpu')
#生成时用默认的注意力内核
model_args = dict(ckpt['model_args'], sdpa_backend='auto')
model = GPT(GPTConfig(**model_args))
model.load_state_dict(ckpt['model'])
model.to(device).eval()
print(f"loaded {os.path.normpath(cfg.ckpt)}: iter {ckpt['iter']}, val loss {ckpt['val_loss']:.4f}")
del ckpt

#和训练数据一样用GPT-2的tokenizer；默认的prompt <|endoftext|> 表示从头写一篇新文档
enc = tiktoken.get_encoding('gpt2')
start = enc.encode(cfg.prompt, allowed_special={'<|endoftext|>'})
if cfg.kv_cache and len(start) + cfg.max_new_tokens > model.config.block_size:
    parser.error(f"用 KV cache 时，prompt 加生成的长度不能超过 {model.config.block_size}，请减小 --max_new_tokens，或者加 --no-kv_cache")
x = torch.tensor(start, dtype=torch.long, device=device)[None].repeat(cfg.num_samples, 1)
#CUDA graph要用KV cache和GPU，不支持MoE
cuda_graph = cfg.cuda_graph and cfg.kv_cache and device == 'cuda' and model.config.n_expert == 0
if cfg.cuda_graph and not cuda_graph:
    print("cuda_graph is off: it needs --kv_cache, a GPU and a dense (non-MoE) model")
compile_step = cfg.compile and cfg.kv_cache
if cfg.compile and not compile_step:
    print("compile is off: it needs --kv_cache")

if device == 'cuda':
    torch.cuda.synchronize()
t0 = time.time()
with ctx:
    y = model.generate(x, cfg.max_new_tokens, temperature=cfg.temperature, top_k=cfg.top_k or None, top_p=cfg.top_p, use_cache=cfg.kv_cache, cuda_graph=cuda_graph, compile=compile_step)
if device == 'cuda':
    torch.cuda.synchronize()
dt = time.time() - t0

prompt_text = enc.decode([t for t in start if t != enc.eot_token])
for i in range(cfg.num_samples):
    tokens = y[i, len(start):].tolist()
    #遇到<|endoftext|>就截断
    if enc.eot_token in tokens:
        tokens = tokens[:tokens.index(enc.eot_token)]
    #词表补齐到50304，多出的id没有对应的文字
    print(prompt_text + enc.decode([t for t in tokens if t < enc.n_vocab]))
    print('---------------')
n = cfg.num_samples * cfg.max_new_tokens
print(f"generated {n} tokens in {dt:.2f} s, {n / dt:.0f} tokens/s (kv_cache={cfg.kv_cache}, cuda_graph={cuda_graph}, compile={compile_step})" + ("; the time includes compiling" if compile_step else ""))
