from collections import Counter
from typing import List, Dict

PAD = "<pad>"
SOS = "<sos>"
EOS = "<eos>"
UNK = "<unk>"

SPECIALS = [PAD, SOS, EOS, UNK]

class Vocab:
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.freqs = Counter()
        self.stoi: Dict[str, int] = {}
        self.itos: List[str] = []

    def build(self, tokenized_texts: List[List[str]]):
        for toks in tokenized_texts:
            self.freqs.update(toks)
        # Specials first
        itos = list(SPECIALS)
        for tok, freq in self.freqs.most_common():
            if freq >= self.min_freq and tok not in SPECIALS:
                itos.append(tok)
        self.itos = itos
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    @property
    def pad_idx(self):
        return self.stoi[PAD]

    @property
    def sos_idx(self):
        return self.stoi[SOS]

    @property
    def eos_idx(self):
        return self.stoi[EOS]

    @property
    def unk_idx(self):
        return self.stoi[UNK]

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.stoi.get(t, self.unk_idx) for t in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        return [self.itos[i] for i in ids if 0 <= i < len(self.itos)]