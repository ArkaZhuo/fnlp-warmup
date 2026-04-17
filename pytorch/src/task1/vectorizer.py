from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable


class NgramVectorizer:
    """
    Simple count vectorizer for BoW / n-gram experiments.

    mode:
      - "bow": unigram only
      - "ngram": use n=1..ngram_n
    """

    def __init__(
        self,
        mode: str = "bow",
        ngram_n: int = 2,
        min_freq: int = 1,
        max_features: int = 20000,
        lowercase: bool = True,
    ) -> None:
        if mode not in {"bow", "ngram"}:
            raise ValueError("mode must be one of {'bow', 'ngram'}")
        if ngram_n < 1:
            raise ValueError("ngram_n must be >= 1")
        if min_freq < 1:
            raise ValueError("min_freq must be >= 1")
        if max_features < 1:
            raise ValueError("max_features must be >= 1")
        self.mode = mode
        self.ngram_n = ngram_n
        self.min_freq = min_freq
        self.max_features = max_features
        self.lowercase = lowercase
        self.vocab: dict[str, int] = {}
        self.idf: list[float] = []

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _tokenize(self, text: str) -> list[str]:
        text = text.strip()
        if self.lowercase:
            text = text.lower()
        return text.split()

    def _extract_features(self, tokens: list[str]) -> list[str]:
        if not tokens:
            return []
        if self.mode == "bow":
            return tokens

        feats: list[str] = []
        upper_n = max(1, self.ngram_n)
        for n in range(1, upper_n + 1):
            if len(tokens) < n:
                break
            for i in range(len(tokens) - n + 1):
                feats.append(" ".join(tokens[i : i + n]))
        return feats

    def fit(self, texts: Iterable[str]) -> None:
        counter: Counter[str] = Counter()
        doc_counter: Counter[str] = Counter()
        n_docs = 0
        for text in texts:
            tokens = self._tokenize(text)
            features = self._extract_features(tokens)
            counter.update(features)
            doc_counter.update(set(features))
            n_docs += 1

        filtered = [
            (feature, freq) for feature, freq in counter.items() if freq >= self.min_freq
        ]
        filtered.sort(key=lambda x: (-x[1], x[0]))
        filtered = filtered[: self.max_features]
        self.vocab = {feature: idx for idx, (feature, _) in enumerate(filtered)}
        self.idf = [0.0] * len(self.vocab)
        for feat, idx in self.vocab.items():
            df = doc_counter.get(feat, 0)
            self.idf[idx] = float((1.0 + n_docs) / (1.0 + df))

    def transform(
        self,
        texts: Iterable[str],
        weighting: str = "count",
    ) -> list[dict[int, float]]:
        if not self.vocab:
            raise RuntimeError("Vectorizer has no vocab. Call fit first.")
        if weighting not in {"count", "tfidf"}:
            raise ValueError("weighting must be one of {'count', 'tfidf'}")
        if weighting == "tfidf" and not self.idf:
            raise RuntimeError("No idf found. Call fit before tfidf transform.")
        outputs: list[dict[int, float]] = []
        for text in texts:
            tokens = self._tokenize(text)
            feats = self._extract_features(tokens)
            sample_counter: Counter[int] = Counter()
            for feat in feats:
                idx = self.vocab.get(feat)
                if idx is not None:
                    sample_counter[idx] += 1
            if weighting == "count":
                outputs.append({idx: float(cnt) for idx, cnt in sample_counter.items()})
            else:
                total = float(sum(sample_counter.values()))
                if total <= 0:
                    outputs.append({})
                else:
                    tfidf_map: dict[int, float] = {}
                    for idx, cnt in sample_counter.items():
                        tf = float(cnt) / total
                        tfidf_map[idx] = tf * float(self.idf[idx])
                    outputs.append(tfidf_map)
        return outputs

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        reverse_vocab = [None] * len(self.vocab)
        for token, idx in self.vocab.items():
            reverse_vocab[idx] = token
        payload = {
            "mode": self.mode,
            "ngram_n": self.ngram_n,
            "min_freq": self.min_freq,
            "max_features": self.max_features,
            "lowercase": self.lowercase,
            "vocab": reverse_vocab,
            "idf": self.idf,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "NgramVectorizer":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        vectorizer = cls(
            mode=payload["mode"],
            ngram_n=int(payload["ngram_n"]),
            min_freq=int(payload["min_freq"]),
            max_features=int(payload["max_features"]),
            lowercase=bool(payload["lowercase"]),
        )
        vocab_list = payload["vocab"]
        vectorizer.vocab = {token: i for i, token in enumerate(vocab_list)}
        vectorizer.idf = [float(x) for x in payload.get("idf", [1.0] * len(vocab_list))]
        return vectorizer
