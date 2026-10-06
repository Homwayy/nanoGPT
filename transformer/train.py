import os
import csv
import json
import math
import time
import argparse

import numpy as np
import torch

from model import GPT, GPTConfig
from eval_gpt2 import evaluate


ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

defaults = dict(
    run_name='baseline',
    out_dir=os.path.join(ROOT, 'out'),
    overwrite=False,
    dataset='fineweb_edu',
    seed=1337,
    n_layer=12,
    n_head=12,
    n_embd=768,
    block_size=1024,
    vocab_size=50304,
    dropout=0.0,
    bias=True,
    mlp='gelu',
    pos_emb='learned',
    norm='layernorm',
    sdpa_backend='auto',
    n_expert=0,
    moe_topk=2,
    moe_aux_coef=0.01,
    batch_size=4,
    grad_accum_steps=32,
    max_tokens=200_000_000,
    max_iters=0,
    learning_rate=6e-4,
    min_lr=6e-5,
    warmup_iters=100,
    schedule='cosine',
    decay_frac=0.2,
    weight_decay=0.1,
    beta1=0.9,
    beta2=0.95,
    grad_clip=1.0,
    eval_interval=100,
    eval_tokens=500_000,
    eval_batch_size=4,
    log_interval=10,
    compile=False,
    wandb_log=False,
    wandb_project='nanogpt',
)

parser = argparse.ArgumentParser()
for k, v in defaults.items():
    if isinstance(v, bool):
        parser.add_argument(f'--{k}', action=argparse.BooleanOptionalAction, default=v)
    else:
        parser.add_argument(f'--{k}', type=type(v), default=v)
cfg = parser.parse_args()
if cfg.schedule not in ('cosine', 'wsd'):
    raise ValueError(f"未知的 schedule: {cfg.schedule}，可选 cosine 或 wsd")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_type = 'cuda' if device.startswith('cuda') else 'cpu'
torch.manual_seed(cfg.seed)
torch.set_float32_matmul_precision('high')
ctx = torch.autocast(device_type=device_type, dtype=torch.bfloat16)

tokens_per_iter = cfg.batch_size * cfg.block_size * cfg.grad_accum_steps
max_iters = cfg.max_iters if cfg.max_iters > 0 else cfg.max_tokens // tokens_per_iter

run_dir = os.path.join(cfg.out_dir, cfg.run_name)
if os.path.exists(os.path.join(run_dir, 'summary.json')) and not cfg.overwrite:
    raise FileExistsError(f"{os.path.normpath(run_dir)} 已有完成的实验结果，请换一个 --run_name，或者加 --overwrite 覆盖")
os.makedirs(run_dir, exist_ok=True)
with open(os.path.join(run_dir, 'config.json'), 'w', encoding='utf-8') as f:
    json.dump({**vars(cfg), 'tokens_per_iter': tokens_per_iter, 'max_iters': max_iters}, f, indent=2)
print(f"tokens per iter: {tokens_per_iter:,}, max iters: {max_iters:,}, total tokens: {tokens_per_iter * max_iters:,}")

data_dir = os.path.join(ROOT, 'data', cfg.dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
val_subset = val_data[:cfg.eval_tokens + 1]
shakespeare_path = os.path.join(ROOT, 'data', 'shakespeare', 'val.bin')
rng = np.random.default_rng(cfg.seed)


def get_batch():
    ix = rng.integers(0, len(train_data) - cfg.block_size, size=cfg.batch_size)
    x = torch.from_numpy(np.stack([train_data[i:i + cfg.block_size] for i in ix]).astype(np.int64))
    y = torch.from_numpy(np.stack([train_data[i + 1:i + 1 + cfg.block_size] for i in ix]).astype(np.int64))
    if device_type == 'cuda':
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


model_args = dict(
    block_size=cfg.block_size,
    vocab_size=cfg.vocab_size,
    n_layer=cfg.n_layer,
    n_head=cfg.n_head,
    n_embd=cfg.n_embd,
    dropout=cfg.dropout,
    bias=cfg.bias,
    mlp=cfg.mlp,
    pos_emb=cfg.pos_emb,
    norm=cfg.norm,
    sdpa_backend=cfg.sdpa_backend,
    n_expert=cfg.n_expert,
    moe_topk=cfg.moe_topk,
)
moe = cfg.n_expert > 0
model = GPT(GPTConfig(**model_args)).to(device)
print(f"number of parameters: {model.get_num_params() / 1e6:.2f}M")
if moe:
    print(f"active parameters per token: {model.get_num_active_params() / 1e6:.2f}M")
optimizer = model.configure_optimizers(cfg.weight_decay, cfg.learning_rate, (cfg.beta1, cfg.beta2), device_type)
raw_model = model
if cfg.compile:
    model = torch.compile(model)


def get_lr(it):
    if it < cfg.warmup_iters:
        return cfg.learning_rate * (it + 1) / cfg.warmup_iters
    if cfg.schedule == 'wsd':
        decay_start = max(cfg.warmup_iters, max_iters - int(cfg.decay_frac * max_iters))
        if it < decay_start:
            return cfg.learning_rate
        ratio = (it - decay_start) / max(1, max_iters - decay_start)
        return cfg.learning_rate - ratio * (cfg.learning_rate - cfg.min_lr)
    ratio = (it - cfg.warmup_iters) / max(1, max_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


def eval_loss(data):
    raw_model.eval()
    with ctx:
        loss, _ = evaluate(lambda x, y: raw_model(x, y)[1], data, cfg.block_size, cfg.eval_batch_size, device)
    raw_model.train()
    return loss


def save_checkpoint(it, val_loss):
    torch.save({
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'config': vars(cfg),
        'iter': it,
        'val_loss': val_loss,
    }, os.path.join(run_dir, 'ckpt.pt'))


log_file = open(os.path.join(run_dir, 'log.csv'), 'w', newline='', encoding='utf-8')
log_writer = csv.writer(log_file)
#MoE多记4列：各层负载均衡损失的平均和最大值，所有专家负载的最小和最大值
moe_cols = ['moe_aux', 'moe_aux_max', 'moe_load_min', 'moe_load_max'] if moe else []
log_writer.writerow(['iter', 'tokens', 'train_loss', 'val_loss', 'lr', 'grad_norm', 'dt_ms', 'tok_per_sec', 'mfu'] + moe_cols)
if moe:
    #每次评估时把各层的负载均衡损失和各专家的平均负载写进moe_load.jsonl
    moe_file = open(os.path.join(run_dir, 'moe_load.jsonl'), 'w', encoding='utf-8')
    aux_sum = torch.zeros(cfg.n_layer, device=device)
    load_sum = torch.zeros(cfg.n_layer, cfg.n_expert, device=device)
    moe_steps = 0

if cfg.wandb_log:
    import wandb
    wandb.init(project=cfg.wandb_project, name=cfg.run_name, config=vars(cfg))


def log_val(it, val_loss):
    tokens = it * tokens_per_iter
    log_writer.writerow([it, tokens, '', f'{val_loss:.4f}', '', '', '', '', ''] + [''] * len(moe_cols))
    log_file.flush()
    print(f"iter {it}: val loss {val_loss:.4f}")
    if cfg.wandb_log:
        wandb.log({'val/loss': val_loss, 'tokens': tokens}, step=it)


def log_moe_load(it):
    global moe_steps
    if not moe or moe_steps == 0:
        return
    record = dict(
        iter=it,
        aux=[round(v, 4) for v in (aux_sum / moe_steps).tolist()],
        load=[[round(v, 4) for v in row] for row in (load_sum / moe_steps).tolist()],
    )
    moe_file.write(json.dumps(record) + '\n')
    moe_file.flush()
    aux_sum.zero_()
    load_sum.zero_()
    moe_steps = 0


if device_type == 'cuda':
    torch.cuda.reset_peak_memory_stats()
start_time = time.time()
dts = []
x, y = get_batch()

for it in range(max_iters):
    if it % cfg.eval_interval == 0:
        val_loss = eval_loss(val_subset)
        log_val(it, val_loss)
        if it > 0:
            save_checkpoint(it, val_loss)
            log_moe_load(it)

    lr = get_lr(it)
    for group in optimizer.param_groups:
        group['lr'] = lr

    t0 = time.time()
    loss_accum = torch.zeros((), device=device)
    if moe:
        aux_accum = torch.zeros(cfg.n_layer, device=device)
        load_accum = torch.zeros(cfg.n_layer, cfg.n_expert, device=device)
    for micro_step in range(cfg.grad_accum_steps):
        with ctx:
            if moe:
                #train loss只记交叉熵；负载均衡损失各层相加后乘moe_aux_coef，只用于反向传播
                _, loss, aux, load = model(x, y, return_aux=True)
                aux_accum += aux.detach() / cfg.grad_accum_steps
                load_accum += load / cfg.grad_accum_steps
                total = (loss + cfg.moe_aux_coef * aux.sum()) / cfg.grad_accum_steps
            else:
                _, loss = model(x, y)
            loss = loss / cfg.grad_accum_steps
        loss_accum += loss.detach()
        x, y = get_batch()
        (total if moe else loss).backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), cfg.grad_clip).item()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if device_type == 'cuda':
        torch.cuda.synchronize()
    dt = time.time() - t0
    dts.append(dt)

    moe_stats = {}
    if moe:
        aux_sum += aux_accum
        load_sum += load_accum
        moe_steps += 1
        moe_stats = dict(zip(moe_cols, [aux_accum.mean().item(), aux_accum.max().item(), load_accum.min().item(), load_accum.max().item()]))
    train_loss = loss_accum.item()
    tokens = (it + 1) * tokens_per_iter
    tok_per_sec = tokens_per_iter / dt
    mfu = raw_model.estimate_mfu(cfg.batch_size * cfg.grad_accum_steps, dt)
    log_writer.writerow([it, tokens, f'{train_loss:.4f}', '', f'{lr:.3e}', f'{grad_norm:.4f}', f'{dt * 1000:.1f}', f'{tok_per_sec:.0f}', f'{mfu:.4f}'] + [f'{v:.4f}' for v in moe_stats.values()])
    if cfg.wandb_log:
        wandb.log({'train/loss': train_loss, 'lr': lr, 'grad_norm': grad_norm, 'tok_per_sec': tok_per_sec, 'mfu': mfu, 'tokens': tokens, **{f'moe/{k}': v for k, v in moe_stats.items()}}, step=it)
    if it % cfg.log_interval == 0:
        log_file.flush()
        print(f"iter {it}: loss {train_loss:.4f}, lr {lr:.2e}, norm {grad_norm:.2f}, dt {dt * 1000:.0f}ms, tok/s {tok_per_sec:,.0f}, mfu {mfu * 100:.1f}%" + (f", moe_aux {moe_stats['moe_aux']:.3f} (max {moe_stats['moe_aux_max']:.3f}), load {moe_stats['moe_load_min']:.2f}-{moe_stats['moe_load_max']:.2f}" if moe else ""))

val_loss = eval_loss(val_subset)
log_val(max_iters, val_loss)
save_checkpoint(max_iters, val_loss)
log_moe_load(max_iters)
log_file.close()
if moe:
    moe_file.close()

full_val_loss = eval_loss(val_data)
shakespeare_loss = None
if os.path.exists(shakespeare_path):
    shakespeare_loss = eval_loss(np.memmap(shakespeare_path, dtype=np.uint16, mode='r'))

steady = dts[5:] if len(dts) > 5 else dts
avg_dt = sum(steady) / len(steady)
summary = dict(
    run_name=cfg.run_name,
    iters=max_iters,
    tokens=max_iters * tokens_per_iter,
    val_loss=full_val_loss,
    shakespeare_loss=shakespeare_loss,
    tok_per_sec=tokens_per_iter / avg_dt,
    mfu=raw_model.estimate_mfu(cfg.batch_size * cfg.grad_accum_steps, avg_dt),
    wall_time_min=(time.time() - start_time) / 60,
    peak_memory_gb=torch.cuda.max_memory_allocated() / 1e9 if device_type == 'cuda' else None,
    peak_reserved_gb=torch.cuda.max_memory_reserved() / 1e9 if device_type == 'cuda' else None,
)
with open(os.path.join(run_dir, 'summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))

if cfg.wandb_log:
    wandb.summary.update(summary)
    wandb.finish()
