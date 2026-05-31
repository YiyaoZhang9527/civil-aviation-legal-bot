"""BM25 + 向量混合检索。"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

from .types import LawDocument, LawNode
from .utils import normalize_text
from . import config

logger = logging.getLogger(__name__)


def tokenize_chinese(text: str) -> list[str]:
    tokens = jieba.lcut_for_search(text)
    return [t for t in tokens if len(t) >= 2 and not t.isspace()]


class BM25Index:
    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._corpus_keys: list[tuple[str, str]] = []

    def build(self, nodes: list[tuple[LawDocument, LawNode]]) -> None:
        corpus: list[list[str]] = []
        self._corpus_keys = []
        for doc, node in nodes:
            if node.type not in {"article", "law"}:
                continue
            text = " ".join(filter(None, [node.title, node.summary, " ".join(node.keywords), node.text]))
            tokens = tokenize_chinese(normalize_text(text))
            corpus.append(tokens)
            self._corpus_keys.append((doc.law_id, node.node_id))
        if corpus:
            self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = 50) -> list[tuple[tuple[str, str], float]]:
        if self._bm25 is None:
            return []
        tokens = tokenize_chinese(normalize_text(query))
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        scored = sorted(zip(self._corpus_keys, scores), key=lambda x: x[1], reverse=True)
        return [(key, score) for key, score in scored[:top_k] if score > 0]


class VectorIndex:
    def __init__(self) -> None:
        self._model = None
        self._embeddings = None
        self._corpus_keys: list[tuple[str, str]] = []
        self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def build(self, nodes: list[tuple[LawDocument, LawNode]]) -> bool:
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
        except ImportError:
            logger.warning("sentence_transformers 不可用，向量检索已禁用")
            return False

        texts: list[str] = []
        self._corpus_keys = []
        for doc, node in nodes:
            if node.type not in {"article", "law"}:
                continue
            text = " ".join(filter(None, [node.title, node.summary, " ".join(node.keywords)]))
            texts.append(text)
            self._corpus_keys.append((doc.law_id, node.node_id))

        sig = self._corpus_signature()
        if self._try_load_cache(sig, SentenceTransformer):
            self._available = True
            return True

        model_path = str(config.VECTOR_MODEL_PATH) if config.VECTOR_MODEL_PATH.exists() else config.VECTOR_MODEL_NAME
        try:
            self._model = SentenceTransformer(model_path)
            self._embeddings = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            self._available = True
            self._save_cache(sig)
            return True
        except Exception as exc:
            logger.warning("向量模型加载失败: %s", exc)
            return False

    def search(self, query: str, top_k: int = 50) -> list[tuple[tuple[str, str], float]]:
        if not self._available or self._model is None or self._embeddings is None:
            return []
        import numpy as np
        q_emb = self._model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        scores = (q_emb @ self._embeddings.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self._corpus_keys[i], float(scores[i])) for i in top_indices if scores[i] > config.VECTOR_SCORE_THRESHOLD]

    def _corpus_signature(self) -> str:
        keys_str = ",".join(f"{lid}:{nid}" for lid, nid in self._corpus_keys)
        return hashlib.md5(keys_str.encode()).hexdigest()

    def _try_load_cache(self, expected_sig: str, model_cls: type) -> bool:
        import numpy as np
        cache = config.VECTOR_CACHE_FILE
        if not cache.exists():
            return False
        try:
            data = np.load(str(cache), allow_pickle=False)
            cached_sig = str(data["signature"])
            if cached_sig != expected_sig:
                return False
            self._embeddings = data["embeddings"]
            keys_json = str(data["keys_json"])
            self._corpus_keys = [tuple(k) for k in json.loads(keys_json)]
            model_path = str(config.VECTOR_MODEL_PATH) if config.VECTOR_MODEL_PATH.exists() else config.VECTOR_MODEL_NAME
            self._model = model_cls(model_path)
            return True
        except Exception:
            return False

    def _save_cache(self, sig: str) -> None:
        import numpy as np
        config.VECTOR_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        keys_json = json.dumps(self._corpus_keys, ensure_ascii=False)
        np.savez(
            str(config.VECTOR_CACHE_FILE),
            embeddings=self._embeddings,
            keys_json=keys_json,
            signature=sig,
        )


def rrf_fuse(
    bm25_results: list[tuple[tuple[str, str], float]],
    vector_results: list[tuple[tuple[str, str], float]],
    hint_scores: dict[tuple[str, str], float],
    k: int = 60,
) -> dict[tuple[str, str], float]:
    """Reciprocal Rank Fusion：只看排名不看原始分数，鲁棒性远优于加权求和。"""
    w_bm25 = config.HYBRID_WEIGHT_BM25
    w_vector = config.HYBRID_WEIGHT_VECTOR
    w_hint = config.HYBRID_WEIGHT_HINT

    scores: dict[tuple[str, str], float] = {}
    for rank, (key, _) in enumerate(bm25_results, 1):
        scores[key] = scores.get(key, 0.0) + w_bm25 / (k + rank)
    for rank, (key, _) in enumerate(vector_results, 1):
        scores[key] = scores.get(key, 0.0) + w_vector / (k + rank)
    for key, hint_s in hint_scores.items():
        if hint_s > 0:
            scores[key] = scores.get(key, 0.0) + w_hint * min(hint_s / 12.0, 1.0)
    return scores
