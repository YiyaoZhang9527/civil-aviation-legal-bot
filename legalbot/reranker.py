"""Cross-Encoder 精排。"""

from __future__ import annotations

import logging

from . import config

logger = logging.getLogger(__name__)

_model = None


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _load_model():
    global _model
    if _model is not None:
        return _model
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        logger.warning("sentence_transformers 不可用，Cross-Encoder 精排已禁用")
        return None
    try:
        model_path = str(config.RERANKER_MODEL_PATH) if config.RERANKER_MODEL_PATH.exists() else config.RERANKER_MODEL_NAME
        device = "cuda" if _cuda_available() else "cpu"
        _model = CrossEncoder(model_path, device=device)
        return _model
    except Exception as exc:
        logger.warning("Cross-Encoder 模型加载失败: %s", exc)
        return None


def rerank(query: str, candidates: list[dict], top_n: int = 5, max_chars: int = 512) -> list[dict]:
    """Cross-Encoder 精排候选列表。

    candidates 每项需有 'title' 和 'text' 字段。
    返回按分数降序的新列表，每项附加 '_rerank_score'。
    """
    if not candidates or not config.RERANKER_ENABLED:
        return candidates[:top_n]

    model = _load_model()
    if model is None:
        return candidates[:top_n]

    pairs = []
    for c in candidates:
        doc = f"{c.get('title', '')} {c.get('text', '')}"
        if len(doc) > max_chars:
            half = max_chars // 2
            doc = doc[:half] + "..." + doc[-half:]
        pairs.append((query, doc))

    scores = model.predict(pairs)

    ranked = []
    for i, score in enumerate(scores):
        item = candidates[i].copy()
        item["_rerank_score"] = float(score)
        ranked.append(item)

    ranked.sort(key=lambda x: x.get("_rerank_score", 0), reverse=True)
    return ranked[:top_n]
