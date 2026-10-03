import pickle
from collections import Counter

import regex as re


#GPT-2的预切分正则
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


#统计相邻pair的出现次数，weight为chunk的重复次数
def get_stats(ids, counts=None, weight=1):
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + weight
    return counts


#把ids中所有的pair替换成idx
def merge(ids, pair, idx):
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


class BPETokenizer:
    def __init__(self, merges, pattern=GPT2_SPLIT_PATTERN, special_tokens=None):
        #merges: {(a, b): idx}，按训练顺序排列
        self.merges = dict(merges)
        #pattern为None时不做预切分
        self.pattern = pattern
        self.compiled_pattern = re.compile(pattern) if pattern else None

        #由merges构建vocab: id -> bytes
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), idx in self.merges.items():
            self.vocab[idx] = self.vocab[a] + self.vocab[b]

        #特殊token的id接在merges之后
        special_tokens = special_tokens or []
        start = 256 + len(self.merges)
        self.special_tokens = {tok: start + i for i, tok in enumerate(special_tokens)}
        self.inverse_special_tokens = {i: tok for tok, i in self.special_tokens.items()}
        if self.special_tokens:
            #特殊token按长度降序匹配
            specials = sorted(self.special_tokens, key=len, reverse=True)
            self.special_pattern = re.compile('(' + '|'.join(re.escape(s) for s in specials) + ')')
        else:
            self.special_pattern = None

        #chunk编码缓存
        self.cache = {}

    #从文本训练BPE
    @classmethod
    def from_text(cls, text, vocab_size, pattern=GPT2_SPLIT_PATTERN, special_tokens=None, verbose=False):
        special_tokens = special_tokens or []
        num_merges = vocab_size - 256 - len(special_tokens)
        assert num_merges >= 0, "vocab_size 至少要 256 + 特殊token数"

        #chunk去重，统计时按出现次数加权
        chunks = re.findall(pattern, text) if pattern else [text]
        chunk_counts = Counter(chunks)
        ids_list = [list(ch.encode('utf-8')) for ch in chunk_counts]
        weights = list(chunk_counts.values())

        merges = {}
        vocab = {i: bytes([i]) for i in range(256)}
        for i in range(num_merges):
            stats = {}
            for ids, w in zip(ids_list, weights):
                get_stats(ids, stats, w)
            if not stats:
                break  #没有可合并的pair

            #合并出现次数最多的pair
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids_list = [merge(ids, pair, idx) for ids in ids_list]
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            if verbose and ((i + 1) % 100 == 0 or i + 1 == num_merges):
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} {vocab[idx]!r} ({stats[pair]} occurrences)")

        return cls(merges, pattern, special_tokens)

    @property
    def vocab_size(self):
        return len(self.vocab) + len(self.special_tokens)

    #单个chunk的BPE编码
    def _encode_chunk(self, chunk):
        if chunk in self.cache:
            return self.cache[chunk]
        ids = list(chunk.encode('utf-8'))
        while len(ids) >= 2:
            stats = get_stats(ids)
            #优先合并最早学到的pair
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break  #没有可合并的pair
            ids = merge(ids, pair, self.merges[pair])
        self.cache[chunk] = ids
        return ids

    #字符串转换向量，不识别特殊token
    def encode_ordinary(self, text):
        chunks = self.compiled_pattern.findall(text) if self.compiled_pattern else [text]
        ids = []
        for chunk in chunks:
            ids.extend(self._encode_chunk(chunk))
        return ids

    #字符串转换向量，识别特殊token
    def encode(self, text):
        if not self.special_tokens:
            return self.encode_ordinary(text)
        ids = []
        for part in self.special_pattern.split(text):
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids

    #向量转换字符串
    def decode(self, ids):
        parts = []
        for i in ids:
            if i in self.vocab:
                parts.append(self.vocab[i])
            elif i in self.inverse_special_tokens:
                parts.append(self.inverse_special_tokens[i].encode('utf-8'))
            else:
                raise ValueError(f"invalid token id: {i}")
        #不完整的utf-8字节用替换字符表示
        return b''.join(parts).decode('utf-8', errors='replace')

    def save(self, path):
        meta = {
            'tokenizer': 'bpe',
            'vocab_size': self.vocab_size,
            'pattern': self.pattern,
            'merges': self.merges,
            'special_tokens': list(self.special_tokens),
        }
        with open(path, 'wb') as f:
            pickle.dump(meta, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            meta = pickle.load(f)
        return cls(meta['merges'], meta['pattern'], meta['special_tokens'])
