import os
import argparse
import numpy as np
import torch
from torch.nn import functional as F


#按不重叠的窗口遍历数据，计算平均每个token的loss
@torch.no_grad()
def evaluate(loss_fn, data, block_size, batch_size, device):
    n_windows = (len(data) - 1) // block_size
    total_loss, total_tokens = 0.0, 0
    for start in range(0, n_windows, batch_size):
        idx = range(start, min(start + batch_size, n_windows))
        x = torch.stack([torch.from_numpy(data[i*block_size : (i+1)*block_size].astype(np.int64)) for i in idx])
        y = torch.stack([torch.from_numpy(data[i*block_size+1 : (i+1)*block_size+1].astype(np.int64)) for i in idx])
        x, y = x.to(device), y.to(device)
        loss = loss_fn(x, y)
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
    return total_loss / total_tokens, total_tokens


if __name__ == '__main__':
    from transformers import GPT2LMHeadModel

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='shakespeare')
    parser.add_argument('--split', default='val')
    parser.add_argument('--model', default='gpt2')  #gpt2 / gpt2-medium / gpt2-large / gpt2-xl
    parser.add_argument('--block_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', args.dataset)

    if os.path.exists(os.path.join(data_dir, 'meta.pkl')):
        raise ValueError(f"{args.dataset} ")
    data = np.memmap(os.path.join(data_dir, f'{args.split}.bin'), dtype=np.uint16, mode='r')

    model = GPT2LMHeadModel.from_pretrained(args.model).to(device).eval()
    assert args.block_size <= model.config.n_positions

    def loss_fn(x, y):
        logits = model(x).logits
        return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

    loss, n_tokens = evaluate(loss_fn, data, args.block_size, args.batch_size, device)
    print(f"{args.model} on {args.dataset}/{args.split} (block_size={args.block_size}): loss {loss:.4f} over {n_tokens:,} tokens")
