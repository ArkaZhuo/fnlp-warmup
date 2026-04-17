from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CharTokenizer:
    stoi: dict[str, int]
    itos: list[str]
    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"

    @classmethod
    def build(cls, corpus: list[str], extra_tokens: list[str] | None = None) -> "CharTokenizer":
        chars = set()
        for text in corpus:
            chars.update(list(text))
        specials = ["<pad>", "<bos>", "<eos>"]
        if extra_tokens:
            specials.extend(extra_tokens)
        ordered_chars = sorted(chars)
        itos = specials + ordered_chars
        stoi = {ch: i for i, ch in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    @property
    def pad_id(self) -> int:
        return self.stoi[self.pad_token]

    @property
    def bos_id(self) -> int:
        return self.stoi[self.bos_token]

    @property
    def eos_id(self) -> int:
        return self.stoi[self.eos_token]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend(self.stoi[c] for c in text)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        out: list[str] = []
        for i in ids:
            token = self.itos[i]
            if skip_special and token in {self.pad_token, self.bos_token, self.eos_token}:
                continue
            out.append(token)
        return "".join(out)

