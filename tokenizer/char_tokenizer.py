import pickle


class CharTokenizer:
    def __init__(self, chars):
        self.chars = list(chars)
        #创建字符映射
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    #从文本构建词表
    @classmethod
    def from_text(cls, text):
        return cls(sorted(set(text)))

    @property
    def vocab_size(self):
        return len(self.chars)

    #字符转换向量
    def encode(self, s):
        return [self.stoi[c] for c in s]

    #向量转换字符
    def decode(self, ids):
        return ''.join([self.itos[i] for i in ids])

    def save(self, path):
        meta = {
            'tokenizer': 'char',
            'vocab_size': self.vocab_size,
            'itos': self.itos,
            'stoi': self.stoi,
        }
        with open(path, 'wb') as f:
            pickle.dump(meta, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            meta = pickle.load(f)
        itos = meta['itos']
        return cls([itos[i] for i in range(len(itos))])
