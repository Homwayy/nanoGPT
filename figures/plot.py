#画EXPERIMENTS.md和README用的图，数据都从results/读，图存在这个目录下
#用法：python figures/plot.py
import os
import csv
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RESULTS = os.path.join(ROOT, 'results')
OUT = os.path.dirname(os.path.abspath(__file__))

#中文字体：Windows上用微软雅黑，其他系统按顺序找能用的
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'PingFang SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

GRAY, BLUE, ORANGE, GREEN, RED, PURPLE = '#7f7f7f', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'


def read_log(run):
    train, val = {}, {}
    with open(os.path.join(RESULTS, run, 'log.csv'), encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['val_loss']:
                val[int(r['iter'])] = float(r['val_loss'])
            else:
                train[int(r['iter'])] = r
    return train, val


def final_val(run):
    with open(os.path.join(RESULTS, run, 'summary.json'), encoding='utf-8') as f:
        return json.load(f)['val_loss']


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"saved figures/{name}")


#图1：几次主要训练的val loss和学习率
def plot_val_loss():
    runs = [
        ('baseline', 'baseline：cosine，峰值 6e-4', GRAY),
        ('wsd', 'WSD，峰值 6e-4', BLUE),
        ('rope', 'RoPE（cosine，峰值 6e-4）', ORANGE),
        ('wsd_rope', 'WSD + RoPE，峰值 6e-4', GREEN),
        ('final', '最终：WSD + RoPE，峰值 9.6e-4', RED),
    ]
    fig, (ax, ax_lr) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.2]))
    for run, label, color in runs:
        train, val = read_log(run)
        steps = sorted(val)
        ax.plot(steps, [val[s] for s in steps], marker='o', markersize=3, color=color, label=f"{label}（{final_val(run):.4f}）")
    ax.set_ylim(3.6, 5.6)
    ax.set_ylabel('val loss')
    ax.set_title('val loss 和学习率（每次训练都是 2 亿 token、1525 步）')
    ax.legend(title='括号里是训练完以后完整验证集上的 val loss', title_fontsize=9, fontsize=9, loc='upper right')
    ax.text(0.01, 0.02, '曲线是每 100 步在验证集开头 50 万 token 上的评估', transform=ax.transAxes, fontsize=8, color=GRAY)

    #学习率：baseline和rope相同，wsd和wsd_rope相同
    for run, label, color in [('baseline', 'cosine，峰值 6e-4（baseline、RoPE）', GRAY), ('wsd', 'WSD，峰值 6e-4（WSD、WSD + RoPE）', BLUE), ('final', 'WSD，峰值 9.6e-4（最终）', RED)]:
        train, _ = read_log(run)
        steps = sorted(train)
        ax_lr.plot(steps, [float(train[s]['lr']) * 1e4 for s in steps], color=color, label=label)
    ax_lr.set_ylabel('学习率（×1e-4）')
    ax_lr.set_xlabel('步数')
    ax_lr.legend(fontsize=8, loc='lower left')
    for a in (ax, ax_lr):
        a.axvline(1220, color='black', linestyle='--', linewidth=0.8, alpha=0.6)
    ax_lr.text(1210, 7.6, 'WSD 从第 1220 步开始衰减 →', ha='right', fontsize=8)
    ax_lr.set_xlim(0, 1540)
    save(fig, 'val_loss.png')


#图2：SwiGLU和RoPE带来的val loss下降，随学习率提高的变化
def plot_arch_gain():
    settings = ['cosine，峰值 6e-4\n平均学习率 3.3e-4', 'WSD，峰值 6e-4\n平均学习率 5.3e-4', 'WSD，峰值 9.6e-4\n平均学习率 8.4e-4']
    #(不加这项改动, 加了这项改动)
    pairs = {
        'SwiGLU': [('baseline', 'swiglu'), ('wsd_rope', 'wsd_rope_swiglu'), ('wsd_rope_lr', 'wsd_rope_swiglu_lr')],
        'RoPE': [('baseline', 'rope'), ('wsd', 'wsd_rope'), ('wsd_lr', 'wsd_rope_lr')],
    }
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(settings))
    width = 0.36
    ax.axhspan(0, 0.01, color=GRAY, alpha=0.15, label='小于 0.01：在随机波动范围内')
    for i, (name, color) in enumerate([('SwiGLU', PURPLE), ('RoPE', ORANGE)]):
        gains = [final_val(a) - final_val(b) for a, b in pairs[name]]
        bars = ax.bar(x + (i - 0.5) * width, gains, width, color=color, label=name)
        for bar, g in zip(bars, gains):
            ax.text(bar.get_x() + bar.get_width() / 2, g + 0.003, f"{g:.4f}" if g < 0.01 else f"{g:.3f}", ha='center', fontsize=9)
    ax.set_xticks(x, settings)
    ax.set_ylabel('val loss 的下降')
    ax.set_title('学习率提高以后，SwiGLU 的收益消失，RoPE 的收益基本保留')
    ax.legend(fontsize=9)
    ax.text(0.0, -0.22, '第一组都是在 baseline 上单独测的；后两组里，SwiGLU 叠在 WSD + RoPE 上测，RoPE 叠在 WSD 上测。每组只有一个 seed。',
            transform=ax.transAxes, fontsize=8, color=GRAY)
    save(fig, 'arch_gain_vs_lr.png')


#图3：不同生成方式的速度随batch大小的变化
def plot_generation_speed():
    with open(os.path.join(RESULTS, 'benchmarks', 'generation_speed.json'), encoding='utf-8') as f:
        d = json.load(f)
    batches = d['batch_size']
    modes = list(d['modes'].items())
    colors = [GRAY, BLUE, ORANGE, PURPLE, RED]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(batches))
    width = 0.16
    for i, ((key, m), color) in enumerate(zip(modes, colors)):
        pos = x + (i - (len(modes) - 1) / 2) * width
        for p, v in zip(pos, m['tokens_per_s']):
            if v is None:
                continue
            ax.bar(p, v, width, color=color, label=m['label'] if p == pos[0] else None)
            ax.text(p, v * 1.08, f"{v:.0f}", ha='center', fontsize=7, rotation=90)
    ax.set_yscale('log')
    ax.set_ylim(50, 40000)
    ax.set_xticks(x, [f"batch {b}" for b in batches])
    ax.set_ylabel('token/s（对数刻度）')
    ax.set_title('生成速度（final 模型，bf16，每次生成 500 个 token，top-p 采样）')
    ax.legend(fontsize=8, loc='upper left', ncol=2)
    ax.text(1.0, -0.12, 'batch 64 时没有测不用 KV cache 的情况', transform=ax.transAxes, fontsize=8, color=GRAY, ha='right')
    save(fig, 'generation_speed.png')


#图4：MoE每一步分到最少和最多token的专家，以及每层每个专家每100步的平均负载
def plot_moe_load():
    train, _ = read_log('final_moe')
    steps = sorted(train)
    lo = [float(train[s]['moe_load_min']) for s in steps]
    hi = [float(train[s]['moe_load_max']) for s in steps]
    recs = []
    with open(os.path.join(RESULTS, 'final_moe', 'moe_load.jsonl'), encoding='utf-8') as f:
        for line in f:
            recs.append(json.loads(line))
    fig, (ax, ax_h) = plt.subplots(2, 1, figsize=(10, 9), gridspec_kw=dict(height_ratios=[1, 1.7]))
    ax.fill_between(steps, lo, hi, color=BLUE, alpha=0.25, linewidth=0)
    ax.plot(steps, hi, color=BLUE, linewidth=0.8, label='分到最多的专家')
    ax.plot(steps, lo, color=RED, linewidth=0.8, label='分到最少的专家')
    ax.axhline(1, color='black', linewidth=0.8, linestyle='--', label='完全均衡（1.0）')
    ax.axhline(2, color=GRAY, linewidth=0.8, linestyle=':')
    ax.text(1520, 2.03, '上限：每个 token 都选它', ha='right', fontsize=8, color=GRAY)
    ax.set_xlim(0, 1525)
    ax.set_ylim(0, 2.2)
    ax.set_xlabel('步数')
    ax.set_ylabel('分到的 token 数 / 平均值')
    ax.set_title('MoE 的负载（4 个专家，每个 token 选 2 个）：12 层 × 4 个专家里每一步的最小值和最大值')
    ax.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1, 0.9))

    #每层4个专家一行一行排下来，列是每100步的平均
    grid = np.array([[v for row in r['load'] for v in row] for r in recs]).T
    im = ax_h.imshow(grid, aspect='auto', cmap='RdBu_r', vmin=0.85, vmax=1.15, interpolation='nearest')
    n_layer, n_expert = len(recs[0]['load']), len(recs[0]['load'][0])
    ax_h.set_yticks([l * n_expert + (n_expert - 1) / 2 for l in range(n_layer)], [f"第 {l} 层" for l in range(n_layer)], fontsize=8)
    for l in range(1, n_layer):
        ax_h.axhline(l * n_expert - 0.5, color='white', linewidth=1.2)
    ax_h.set_xticks(range(len(recs)), [str(r['iter']) for r in recs], fontsize=7)
    ax_h.set_xlabel('步数（每一列是到这一步为止 100 步的平均，最后一列是最后 25 步）')
    ax_h.set_title('每层 4 个专家的负载（每层 4 行）：第 200 步以后都在 0.93–1.11 之间', fontsize=10)
    ax_h.grid(False)
    cb = fig.colorbar(im, ax=ax_h, fraction=0.03, pad=0.02)
    cb.set_label('分到的 token 数 / 平均值')
    fig.tight_layout()
    save(fig, 'moe_load.png')


#图5：MoE比dense慢在哪里
def plot_moe_speed():
    with open(os.path.join(RESULTS, 'benchmarks', 'moe_profile.json'), encoding='utf-8') as f:
        d = json.load(f)
    step = d['step_time_ms']
    per = d['per_microbatch_ms']
    fig, (ax_s, ax_b) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw=dict(width_ratios=[1, 1.8]))
    names = [('dense', 'dense', GRAY), ('moe', 'MoE', RED), ('moe_bmm', 'MoE\n（grouped_mm 换成 bmm）', ORANGE)]
    bars = ax_s.bar([n[1] for n in names], [step[n[0]] for n in names], color=[n[2] for n in names])
    for bar, (key, _, _) in zip(bars, names):
        v = step[key]
        extra = '' if key == 'dense' else f"\n+{v - step['dense']:.0f} ms（+{(v / step['dense'] - 1) * 100:.1f}%）"
        ax_s.text(bar.get_x() + bar.get_width() / 2, v + 40, f"{v:.0f} ms{extra}", ha='center', va='bottom', fontsize=8)
    ax_s.set_ylim(0, 4200)
    ax_s.set_ylabel('每步耗时（ms，32 个 micro-batch）')
    ax_s.set_title('每步耗时')

    #和dense比，每个micro-batch多出的时间按来源拆开；“其他”是总的多出时间减去前四项
    def extra(key):
        a, b = per[key], per['dense']
        parts = {
            'GPU 空闲': a['gpu_idle'] - b['gpu_idle'],
            '矩阵乘法': a['gemm'] - b['gemm'],
            'inductor 生成的 kernel\n（路由、token 重排等）': a['inductor'] - b['inductor'],
            '排序': a['sort'] - b['sort'],
        }
        parts['其他'] = (a['wall'] - b['wall']) - sum(parts.values())
        return parts
    cats = list(extra('moe'))
    x = np.arange(len(cats))
    width = 0.38
    for i, (key, label, color) in enumerate([('moe', f"MoE（每个 micro-batch 同步 {per['moe']['syncs']} 次）", RED), ('moe_bmm', 'MoE（grouped_mm 换成 bmm，不同步）', ORANGE)]):
        vals = list(extra(key).values())
        bs = ax_b.bar(x + (i - 0.5) * width, vals, width, color=color, label=f"{label}：共 +{sum(vals):.1f} ms")
        for bar, v in zip(bs, vals):
            ax_b.text(bar.get_x() + bar.get_width() / 2, v + (0.2 if v >= 0 else -0.8), f"{v:+.1f}", ha='center', fontsize=8)
    ax_b.axhline(0, color='black', linewidth=0.8)
    ax_b.set_ylim(-3, 14.5)
    ax_b.set_xticks(x, cats, fontsize=8)
    ax_b.set_ylabel('比 dense 多出的时间（ms / micro-batch）')
    ax_b.set_title('多出的时间从哪来（profiler）')
    ax_b.legend(fontsize=8)
    fig.suptitle('MoE 每个 token 的计算量和 dense 相同，却慢 22%', fontsize=12)
    fig.tight_layout()
    save(fig, 'moe_speed.png')


if __name__ == '__main__':
    plot_val_loss()
    plot_arch_gain()
    plot_generation_speed()
    plot_moe_load()
    plot_moe_speed()
