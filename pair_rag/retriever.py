"""Dense retriever (E5 by default) over the clean corpus, with poisoned passages injected on the fly.

Clean embeddings are computed once and cached as .npy; for each question we cache its clean top-N,
so every PAIR iteration only has to embed the M candidate poisons.
"""
import os

import numpy as np

RETRIEVERS = {
    "e5": ("intfloat/e5-base-v2", "query: ", "passage: "),
    "e5-large": ("intfloat/e5-large-v2", "query: ", "passage: "),
    "contriever": ("facebook/contriever-msmarco", "", ""),  # mean pooling handled by sentence-transformers
}


def _device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DenseRetriever:
    def __init__(self, name, corpus, cache_dir, batch_size=128):
        from sentence_transformers import SentenceTransformer

        if name not in RETRIEVERS:
            raise ValueError(f"unknown retriever {name}; choose from {list(RETRIEVERS)}")
        model_id, self.q_prefix, self.p_prefix = RETRIEVERS[name]
        self.name = name
        self.model = SentenceTransformer(model_id, device=_device())
        self.texts = [d["text"] for d in corpus]
        self.batch_size = batch_size
        path = os.path.join(cache_dir, f"emb_{name}.npy")
        if os.path.exists(path) and np.load(path, mmap_mode="r").shape[0] == len(self.texts):
            self.emb = np.load(path)
        else:
            self.emb = self._encode([self.p_prefix + t for t in self.texts], show_progress=True)
            np.save(path, self.emb)
        self._clean_cache = {}
        self._replace = {}  # index -> text returned in its place (retrieval proxies)

    def add_documents(self, texts):
        """Append clean documents to the index."""
        texts = [t for t in texts if t]
        if not texts:
            return
        self.texts += texts
        self.emb = np.vstack([self.emb, self._encode([self.p_prefix + t for t in texts])])
        self._clean_cache = {}

    def add_proxies(self, proxies, replacements):
        """RAGForensics' benign text retrieval proxy (Appendix 8.3): index each proxy text (the user query) and
        return its replacement (the trigger-wrapped benign text) whenever the proxy is retrieved."""
        start = len(self.texts)
        self.add_documents(proxies)
        self._replace.update({start + i: r for i, r in enumerate(replacements)})

    def _encode(self, texts, show_progress=False):
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=show_progress).astype(np.float32)

    def _clean_topn(self, question, n):
        key = (question, n)
        if key not in self._clean_cache:
            q = self._encode([self.q_prefix + question])[0]
            scores = self.emb @ q
            idx = np.argpartition(-scores, min(n, len(scores) - 1))[:n]
            idx = idx[np.argsort(-scores[idx])]
            self._clean_cache[key] = (q, [(self._replace.get(i, self.texts[i]), float(scores[i]), False)
                                       for i in idx])
        return self._clean_cache[key]

    def rank(self, question, poison_texts, top_n=100):
        """Return [(text, score, is_poison)] for the top_n of clean corpus ∪ poison_texts, best first."""
        q, clean = self._clean_topn(question, top_n)
        merged = list(clean)
        if poison_texts:
            p = self._encode([self.p_prefix + t for t in poison_texts])
            merged += [(t, float(s), True) for t, s in zip(poison_texts, p @ q)]
        merged.sort(key=lambda x: -x[1])
        return merged[:top_n]
