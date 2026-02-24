"""
legal_indexer.py
────────────────
FAISS vector store for legal act documents.

Embeddings: uses transformers + torch directly.
NO sentence-transformers dependency — eliminates all version conflict errors
on Windows Python 3.10.

Model: BAAI/bge-small-en-v1.5  (384-dim, ~130MB, cached after first run)
"""

import os
import pickle
import logging
import numpy as np
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

VECTOR_STORE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "vector_store")
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "legal_index.faiss")
METADATA_PATH    = os.path.join(VECTOR_STORE_DIR, "legal_metadata.pkl")
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)


class EmbeddingModel:
    """
    Local embeddings via HuggingFace transformers directly.
    Mean-pooling + L2 normalisation over BAAI/bge-small-en-v1.5.
    """
    MODEL_NAME = "BAAI/bge-small-en-v1.5"
    DIMENSION  = 384

    def __init__(self):
        self.dimension = self.DIMENSION
        self._load()

    def _load(self):
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            self._tok   = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            self._model = AutoModel.from_pretrained(self.MODEL_NAME)
            self._model.eval()
            self._torch = torch
            logger.info("Embedding model loaded: %s", self.MODEL_NAME)
        except Exception as e:
            raise RuntimeError(
                f"Could not load embedding model. "
                f"Run: pip install transformers torch\nError: {e}"
            )

    def _pool(self, hidden, mask):
        m = mask.unsqueeze(-1).expand(hidden.size()).float()
        return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)

    def embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.array([], dtype=np.float32)
        torch = self._torch
        out   = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc   = self._tok(batch, padding=True, truncation=True,
                              max_length=512, return_tensors="pt")
            with torch.no_grad():
                h = self._model(**enc).last_hidden_state
            emb = self._pool(h, enc["attention_mask"])
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            out.append(emb.numpy().astype(np.float32))
        return np.vstack(out)

    def embed_single(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class LegalIndexer:
    """FAISS vector store. Fully local — no API key needed."""

    def __init__(self, embedding_provider: str = "huggingface"):
        self.embedder = EmbeddingModel()
        self.index    = None
        self.metadata: List[Dict] = []
        self._load_or_init()

    def _load_or_init(self):
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
            self._load_index()
        else:
            self._new_index()

    def _new_index(self):
        import faiss
        self.index    = faiss.IndexFlatIP(self.embedder.dimension)
        self.metadata = []

    def _load_index(self):
        import faiss
        self.index = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH, "rb") as f:
            self.metadata = pickle.load(f)
        logger.info("Loaded index: %d vectors", self.index.ntotal)

    def add_chunks(self, chunks: List[Dict]) -> int:
        if not chunks:
            return 0
        emb = self.embedder.embed([c["text"] for c in chunks])
        if emb.size == 0:
            return 0
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        self.index.add(emb)
        self.metadata.extend(chunks)
        return len(chunks)

    def clear_index(self):
        self._new_index()
        for p in [FAISS_INDEX_PATH, METADATA_PATH]:
            if os.path.exists(p):
                os.remove(p)

    def search(self, query: str, k: int = 5) -> List[Dict]:
        if not self.index or self.index.ntotal == 0:
            return []
        q = self.embedder.embed_single(query).reshape(1, -1)
        k = min(k, self.index.ntotal)
        scores, idxs = self.index.search(q, k)
        results = []
        for s, i in zip(scores[0], idxs[0]):
            if i == -1:
                continue
            c = dict(self.metadata[i])
            c["score"] = float(s)
            results.append(c)
        return results

    def save_index(self):
        import faiss
        faiss.write_index(self.index, FAISS_INDEX_PATH)
        with open(METADATA_PATH, "wb") as f:
            pickle.dump(self.metadata, f)

    @property
    def vector_count(self) -> int:
        return self.index.ntotal if self.index else 0

    @property
    def indexed_acts(self) -> List[str]:
        return sorted({m.get("act_name", "Unknown") for m in self.metadata})

    def get_stats(self) -> Dict:
        return {
            "total_vectors":   self.vector_count,
            "indexed_acts":    self.indexed_acts,
            "embedding_model": EmbeddingModel.MODEL_NAME,
            "dimension":       self.embedder.dimension,
        }


_indexer: Optional[LegalIndexer] = None

def get_indexer() -> LegalIndexer:
    global _indexer
    if _indexer is None:
        _indexer = LegalIndexer()
    return _indexer

def query_legal_database(query: str, k: int = 5) -> str:
    results = get_indexer().search(query, k=k)
    if not results:
        return "No results. Upload legal act PDFs first."
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(
            f"[{i}] {r.get('act_name','?')} — {r.get('section_id','?')}  "
            f"(score:{r.get('score',0):.3f})\n"
            f"{r.get('text','')[:600]}"
        )
    return "\n\n---\n\n".join(parts)
