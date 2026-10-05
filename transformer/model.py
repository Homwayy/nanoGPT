import math
import inspect
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel




@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    mlp: str = 'gelu'
    pos_emb: str = 'learned'
    norm: str = 'layernorm'
    sdpa_backend: str = 'auto'
    n_expert: int = 0
    moe_topk: int = 2

#RMSNorm：只按均方根缩放，不减均值，没有bias；eps和LayerNorm的默认值一样取1e-5
def make_norm(config):
    if config.norm == 'rmsnorm':
        return nn.RMSNorm(config.n_embd, eps=1e-5)
    return nn.LayerNorm(config.n_embd, bias=config.bias)

#RoPE：每个头的第i维和第i+hs/2维组成一对，按位置旋转，注意力分数只和相对位置有关
def apply_rope(x, cos, sin):
    d = x.size(-1) // 2
    x1, x2 = x[..., :d].float(), x[..., d:].float()
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).type_as(x)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resi_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        #SDPA默认的优先级是flash→memory-efficient→math→cuDNN，cuDNN排在math后面，不会被自动选中，要手动指定
        #cuDNN内核只支持fp16/bf16，要在autocast下用，fp32输入会直接报错
        self.cudnn = config.sdpa_backend == 'cudnn'
        self.rope = config.pos_emb == 'rope'
        if self.rope:
            hs = config.n_embd // config.n_head
            assert hs % 2 == 0, "RoPE needs an even head dim"
            inv_freq = 1.0 / (10000 ** (torch.arange(0, hs, 2).float() / hs))
            freqs = torch.outer(torch.arange(config.block_size).float(), inv_freq)
            self.register_buffer('rope_cos', freqs.cos(), persistent=False)
            self.register_buffer('rope_sin', freqs.sin(), persistent=False)


    def forward(self, x):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)# (B, T, C) -> (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)# (B, T, C) -> (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)# (B, T, C) -> (B, nh, T, hs)
        if self.rope:
            q = apply_rope(q, self.rope_cos[:T], self.rope_sin[:T])
            k = apply_rope(k, self.rope_cos[:T], self.rope_sin[:T])
        #Flash attention
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION) if self.cudnn else nullcontext():
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resi_dropout(self.c_proj(y))
        return y



class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

#SwiGLU：silu(xW1) * (xW3)，hidden取8/3*n_embd，权重参数量和GELU版MLP相同（每层多1024个bias）
class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = 8 * config.n_embd // 3
        #W1和W3合成一个Linear，一次矩阵乘法同时算出gate和up
        self.c_fc = nn.Linear(config.n_embd, 2*hidden, bias=config.bias)
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        gate, up = self.c_fc(x).chunk(2, dim=-1)
        x = F.silu(gate) * up
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

#MoE：每层n_expert个GELU专家，路由器给每个token选moe_topk个，输出按路由权重加权求和
#专家的hidden取4*n_embd/moe_topk，每个token激活的计算量和原来的MLP相同；不丢token
class MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_expert = config.n_expert
        self.topk = config.moe_topk
        hidden = 4 * config.n_embd // config.moe_topk
        self.router = nn.Linear(config.n_embd, config.n_expert, bias=False)
        #所有专家的权重放在一个张量里，形状(专家, 输入, 输出)，用分组矩阵乘法一次算完；只支持CUDA上的bf16
        self.w_fc = nn.Parameter(torch.randn(config.n_expert, config.n_embd, hidden) * 0.02)
        self.w_proj = nn.Parameter(torch.randn(config.n_expert, hidden, config.n_embd) * (0.02 / math.sqrt(2 * config.n_layer)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size()
        x = x.view(-1, C)
        #路由在fp32下算
        with torch.autocast(device_type=x.device.type, enabled=False):
            probs = F.softmax(self.router(x.float()), dim=-1)
        weight, expert = probs.topk(self.topk, dim=-1)
        weight = weight / weight.sum(dim=-1, keepdim=True)
        #负载均衡损失（Switch Transformer）：专家数 × sum(每个专家分到的比例 × 平均路由概率)，完全均衡时等于1
        counts = F.one_hot(expert, self.n_expert).sum(dim=(0, 1))
        aux = self.n_expert * (counts.float() / expert.numel() * probs.mean(dim=0)).sum()
        #按专家排序，同一个专家的token连在一起；offs是每个专家结束的位置
        order = expert.view(-1).argsort()
        token = order // self.topk
        offs = counts.cumsum(0).to(torch.int32)
        h = torch._grouped_mm(x.to(torch.bfloat16)[token], self.w_fc.to(torch.bfloat16), offs=offs)
        y = torch._grouped_mm(F.gelu(h), self.w_proj.to(torch.bfloat16), offs=offs)
        #排回token原来的顺序，每个token的topk个输出按路由权重加权求和
        y = y[order.argsort()].view(-1, self.topk, C)
        y = (y * weight.unsqueeze(-1).to(y.dtype)).sum(dim=1)
        return self.dropout(y.view(B, T, C)), aux

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.mlp in ('gelu', 'swiglu'), f"unknown mlp: {config.mlp}"
        self.ln_1 = make_norm(config)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = make_norm(config)
        self.moe = config.n_expert > 0
        if self.moe:
            assert config.mlp == 'gelu', "MoE的专家用GELU"
            self.mlp = MoE(config)
        else:
            self.mlp = SwiGLU(config) if config.mlp == 'swiglu' else MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        #MoE层还要返回负载均衡损失
        if self.moe:
            y, aux = self.mlp(self.ln_2(x))
            return x + y, aux
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.pos_emb in ('learned', 'rope'), f"unknown pos_emb: {config.pos_emb}"
        assert config.norm in ('layernorm', 'rmsnorm'), f"unknown norm: {config.norm}"
        assert config.sdpa_backend in ('auto', 'cudnn'), f"unknown sdpa_backend: {config.sdpa_backend}"
        assert config.n_expert == 0 or config.n_expert >= config.moe_topk >= 1, "需要 n_expert >= moe_topk >= 1"
        modules = dict(wte = nn.Embedding(config.vocab_size, config.n_embd))
        #RoPE不需要可学习的位置编码
        if config.pos_emb == 'learned':
            modules['wpe'] = nn.Embedding(config.block_size, config.n_embd)
        modules.update(
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = make_norm(config)
        )
        self.transformer = nn.ModuleDict(modules)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        #权值共享
        self.transformer.wte.weight = self.lm_head.weight
        #子模块初始化
        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0, std=0.02/math.sqrt(2 * config.n_layer))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and 'wpe' in self.transformer:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    #MoE每个token只激活moe_topk个专家，算FLOPs时只算激活的参数
    def get_num_active_params(self):
        n_params = self.get_num_params()
        cfg = self.config
        if cfg.n_expert > 0:
            hidden = 4 * cfg.n_embd // cfg.moe_topk
            n_params -= cfg.n_layer * (cfg.n_expert - cfg.moe_topk) * 2 * cfg.n_embd * hidden
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02) #std=1/sqrt(in_features)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, target=None, return_aux=False):
        device = idx.device

        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        if self.config.pos_emb == 'learned':
            x = tok_emb + self.transformer.wpe(pos)
        else:
            x = tok_emb

        x = self.transformer.drop(x)
        aux = 0.0
        for block in self.transformer.h:
            if block.moe:
                x, block_aux = block(x)
                aux = aux + block_aux
            else:
                x = block(x)
        x = self.transformer.ln_f(x)

        if target is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=-1)

        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        #return_aux时多返回各层负载均衡损失的平均值，只有MoE用
        if return_aux:
            return logits, loss, aux / self.config.n_layer
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
        

    def estimate_mfu(self, fwdbwd_per_iter, dt, flops_promised = 59e12):
        N = self.get_num_active_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt) 
        mfu = flops_achieved / flops_promised
        return mfu


    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            logits, _ = self(idx_cond)

            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits,dim=1)

            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx







