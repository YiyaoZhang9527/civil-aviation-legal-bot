"""三级树检索：法级 -> 章级 -> 条级，逐层剪枝。

向量缓存使用 numpy npz 格式（allow_pickle=False），不含 pickle 序列化。
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from . import config
from .hybrid_retrieval import rrf_fuse, tokenize_chinese
from .types import LawDocument, LawNode
from .utils import normalize_text

logger = logging.getLogger(__name__)


def _node_text(node: LawNode, parent_title: str = "", grandparent_title: str = "") -> str:
    """为节点生成编码文本，包含父级上下文。"""
    parts = [grandparent_title, parent_title, node.title, node.summary or ""]
    return " ".join(p for p in parts if p)


def _collect_tree(documents: list[LawDocument]):
    """遍历 LawDocument 树，收集三级节点及父子映射。"""
    law_keys: list[tuple[str, str]] = []
    law_texts: list[str] = []
    chapter_keys: list[tuple[str, str]] = []
    chapter_texts: list[str] = []
    chapter_to_law: list[int] = []
    article_keys: list[tuple[str, str]] = []
    article_texts: list[str] = []
    article_to_chapter: list[int] = []

    for doc in documents:
        law_idx = len(law_keys)
        law_keys.append((doc.law_id, doc.root.node_id))
        law_texts.append(_node_text(doc.root))

        for child in doc.root.children:
            if child.type == "chapter":
                _collect_chapter(child, doc, law_idx,
                                 chapter_keys, chapter_texts, chapter_to_law,
                                 article_keys, article_texts, article_to_chapter)
            elif child.type == "section":
                for grandchild in child.children:
                    if grandchild.type == "article":
                        _add_article(grandchild, doc, law_idx,
                                     article_keys, article_texts, article_to_chapter)
            elif child.type == "article":
                _add_article(child, doc, law_idx,
                             article_keys, article_texts, article_to_chapter)

    return (law_keys, law_texts, chapter_keys, chapter_texts, chapter_to_law,
            article_keys, article_texts, article_to_chapter)


def _collect_chapter(chapter, doc, law_idx,
                     ch_keys, ch_texts, ch_to_law,
                     art_keys, art_texts, art_to_ch):
    ch_idx = len(ch_keys)
    ch_keys.append((doc.law_id, chapter.node_id))
    ch_texts.append(_node_text(chapter, parent_title=doc.title))
    ch_to_law.append(law_idx)

    for child in chapter.children:
        if child.type == "section":
            for grandchild in child.children:
                if grandchild.type == "article":
                    _add_article(grandchild, doc, ch_idx,
                                 art_keys, art_texts, art_to_ch,
                                 parent_title=f"{doc.title} {chapter.title}")
        elif child.type == "article":
            _add_article(child, doc, ch_idx,
                         art_keys, art_texts, art_to_ch,
                         parent_title=doc.title)


def _add_article(node, doc, parent_idx,
                 art_keys, art_texts, art_to_ch, parent_title=""):
    art_keys.append((doc.law_id, node.node_id))
    art_texts.append(_node_text(node, parent_title=parent_title,
                                grandparent_title=doc.title))
    art_to_ch.append(parent_idx)


def _corpus_signature(keys):
    raw = ",".join(f"{lid}:{nid}" for lid, nid in sorted(keys))
    return hashlib.md5(raw.encode()).hexdigest()


class TreeRetriever:
    """三级树检索器：法级 -> 章级 -> 条级，逐层剪枝。"""

    def __init__(self) -> None:
        self._model = None
        self._available = False

        self._law_keys: list[tuple[str, str]] = []
        self._chapter_keys: list[tuple[str, str]] = []
        self._article_keys: list[tuple[str, str]] = []

        self._law_emb: np.ndarray | None = None
        self._chapter_emb: np.ndarray | None = None
        self._article_emb: np.ndarray | None = None

        self._chapter_to_law: np.ndarray | None = None
        self._article_to_chapter: np.ndarray | None = None

        self._law_bm25: BM25Okapi | None = None
        self._chapter_bm25: BM25Okapi | None = None
        self._article_bm25: BM25Okapi | None = None

        self._law_texts: list[str] = []

    @property
    def available(self) -> bool:
        return self._available

    # ── 构建索引 ─────────────────────────────────────────────

    def build(self, documents: list[LawDocument]) -> bool:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.warning("sentence_transformers 不可用，树检索禁用")
            return False

        (law_keys, law_texts, chapter_keys, chapter_texts, chapter_to_law,
         article_keys, article_texts, article_to_chapter) = _collect_tree(documents)

        self._law_keys = law_keys
        self._law_texts = law_texts
        self._chapter_keys = chapter_keys
        self._article_keys = article_keys
        self._chapter_to_law = np.array(chapter_to_law, dtype=np.int32)
        self._article_to_chapter = np.array(article_to_chapter, dtype=np.int32)

        self._law_bm25 = self._build_bm25(law_texts)
        self._chapter_bm25 = self._build_bm25(chapter_texts)
        self._article_bm25 = self._build_bm25(article_texts)

        sig = _corpus_signature(article_keys)
        if self._try_load_cache(sig, SentenceTransformer):
            self._available = True
            logger.info("tree vector cache loaded (%d laws, %d chapters, %d articles)",
                        len(law_keys), len(chapter_keys), len(article_keys))
            return True

        model_path = (str(config.VECTOR_MODEL_PATH)
                      if config.VECTOR_MODEL_PATH.exists() else config.VECTOR_MODEL_NAME)
        try:
            self._model = SentenceTransformer(model_path)
            self._law_emb = self._model.encode(law_texts, normalize_embeddings=True,
                                                show_progress_bar=False)
            self._chapter_emb = self._model.encode(chapter_texts, normalize_embeddings=True,
                                                   show_progress_bar=False)
            self._article_emb = self._model.encode(article_texts, normalize_embeddings=True,
                                                   show_progress_bar=False)
            self._available = True
            self._save_cache(sig)
            logger.info("tree vector built (%d laws, %d chapters, %d articles)",
                        len(law_keys), len(chapter_keys), len(article_keys))
            return True
        except Exception as exc:
            logger.warning("树检索向量构建失败: %s", exc)
            return False

    @staticmethod
    def _build_bm25(texts: list[str]) -> BM25Okapi:
        corpus = [tokenize_chinese(normalize_text(t)) for t in texts]
        return BM25Okapi(corpus) if corpus else BM25Okapi([[]])

    # ── 搜索 ─────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5,
               hint_law_ids: set[str] | None = None,
               hint_scores: dict[tuple[str, str], float] | None = None,
               tree_top_laws: int | None = None,
               ) -> list[tuple[tuple[str, str], float]]:
        """树检索：法->章->条逐层剪枝，返回 RRF 融合后的条级结果。"""
        if not self._available:
            return []

        top_laws = tree_top_laws or config.TREE_TOP_LAWS
        top_chapters = config.TREE_TOP_CHAPTERS
        article_candidate_count = min(top_k * 4, 30)

        # Level 0: 法级 (Dense + BM25, 可选 Cross-Encoder 三路融合)
        law_vec = self._level_vector_search(query, self._law_emb, top_laws)
        law_bm25 = self._level_bm25_search(query, self._law_bm25, top_laws)
        law_ce = self._level_ce_search(query, top_laws) if getattr(config, 'LEVEL0_CE_ENABLED', False) else None
        law_set = self._rrf_union(law_vec, law_bm25, top_laws, ce_results=law_ce)
        if not law_set:
            return []

        # A2: 法规级加权——hints 匹配到的法规加分，但不踢除非匹配法规
        if hint_law_ids:
            A2_BOOST = 0.3
            boosted = []
            for idx, s in law_set:
                if self._law_keys[idx][0] in hint_law_ids:
                    boosted.append((idx, s + A2_BOOST))
                else:
                    boosted.append((idx, s))
            law_set = boosted

        law_mask = np.zeros(len(self._law_keys), dtype=bool)
        for idx, _ in law_set:
            law_mask[idx] = True

        # Level 1: 章级（只在候选法内）
        ch_law_mask = np.isin(self._chapter_to_law, np.where(law_mask)[0])
        ch_vec = self._filtered_vector_search(query, self._chapter_emb, ch_law_mask, top_chapters)
        ch_bm25 = self._filtered_bm25_search(query, self._chapter_bm25, ch_law_mask, top_chapters)
        chapter_set = self._rrf_union(ch_vec, ch_bm25, top_chapters)
        if not chapter_set:
            return []

        ch_mask = np.zeros(len(self._chapter_keys), dtype=bool)
        for idx, _ in chapter_set:
            ch_mask[idx] = True

        # Level 2: 条级（只在候选章内）
        art_ch_mask = np.isin(self._article_to_chapter, np.where(ch_mask)[0])
        art_vec = self._filtered_vector_search(query, self._article_emb, art_ch_mask,
                                               article_candidate_count)
        art_bm25 = self._filtered_bm25_search(query, self._article_bm25, art_ch_mask,
                                              article_candidate_count)

        art_vec_results = [(self._article_keys[i], s) for i, s in art_vec]
        art_bm25_results = [(self._article_keys[i], s) for i, s in art_bm25]
        fused = rrf_fuse(art_bm25_results, art_vec_results, hint_scores or {})

        results = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return results

    # ── 搜索辅助 ─────────────────────────────────────────────

    def _level_vector_search(self, query, embeddings, top_k):
        if embeddings is None or self._model is None:
            return []
        q_emb = self._model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        scores = (q_emb @ embeddings.T).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_idx if scores[i] > config.VECTOR_SCORE_THRESHOLD]

    def _level_bm25_search(self, query, bm25_index, top_k):
        if bm25_index is None:
            return []
        tokens = tokenize_chinese(normalize_text(query))
        if not tokens:
            return []
        scores = bm25_index.get_scores(tokens)
        scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [(idx, s) for idx, s in scored if s > 0]

    def _level_ce_search(self, query, top_k):
        """Level 0 Cross-Encoder 粗筛：query vs 每个法规的代表性文本。"""
        if not self._law_keys:
            return []
        try:
            from .reranker import _load_model
            model = _load_model()
        except Exception:
            return []
        if not self._law_texts:
            return []
        pairs = [(query, text[:512]) for text in self._law_texts]
        scores = model.predict(pairs, show_progress_bar=False)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_idx if scores[i] > 0]

    def _filtered_vector_search(self, query, embeddings, mask, top_k):
        if embeddings is None or self._model is None:
            return []
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return []
        q_emb = self._model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        sub_emb = embeddings[indices]
        scores = (q_emb @ sub_emb.T).flatten()
        local_top = np.argsort(scores)[::-1][:top_k]
        return [(int(indices[i]), float(scores[i])) for i in local_top if scores[i] > config.VECTOR_SCORE_THRESHOLD]

    def _filtered_bm25_search(self, query, bm25_index, mask, top_k):
        if bm25_index is None:
            return []
        tokens = tokenize_chinese(normalize_text(query))
        if not tokens:
            return []
        scores = bm25_index.get_scores(tokens)
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return []
        sub_scores = scores[indices]
        local_top = np.argsort(sub_scores)[::-1][:top_k]
        return [(int(indices[i]), float(sub_scores[i])) for i in local_top if sub_scores[i] > 0]

    @staticmethod
    def _rrf_union(results_a, results_b, top_k, ce_results=None):
        """合并多组 (index, score) 为 RRF 排名，返回 top_k。支持可选的 CE 第三路信号。"""
        rrf_scores: dict[int, float] = {}
        k = config.RRF_K
        for rank, (idx, _) in enumerate(results_a, 1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank)
        for rank, (idx, _) in enumerate(results_b, 1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank)
        if ce_results:
            for rank, (idx, _) in enumerate(ce_results, 1):
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.2 / (k + rank)
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:top_k]

    # ── 向量缓存 ─────────────────────────────────────────────

    def _try_load_cache(self, expected_sig, model_cls) -> bool:
        cache = config.TREE_VECTOR_CACHE_FILE
        if not cache.exists():
            return False
        try:
            data = np.load(str(cache), allow_pickle=False)
            if str(data["signature"]) != expected_sig:
                return False
            self._law_emb = data["law_emb"]
            self._chapter_emb = data["chapter_emb"]
            self._article_emb = data["article_emb"]
            self._law_keys = [tuple(k) for k in json.loads(str(data["law_keys"]))]
            self._chapter_keys = [tuple(k) for k in json.loads(str(data["chapter_keys"]))]
            self._article_keys = [tuple(k) for k in json.loads(str(data["article_keys"]))]
            self._chapter_to_law = data["chapter_to_law"]
            self._article_to_chapter = data["article_to_chapter"]
            model_path = (str(config.VECTOR_MODEL_PATH)
                          if config.VECTOR_MODEL_PATH.exists() else config.VECTOR_MODEL_NAME)
            self._model = model_cls(model_path)
            return True
        except Exception:
            return False

    def _save_cache(self, sig: str) -> None:
        cache = config.TREE_VECTOR_CACHE_FILE
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(cache),
            law_emb=self._law_emb,
            law_keys=json.dumps(self._law_keys, ensure_ascii=False),
            chapter_emb=self._chapter_emb,
            chapter_keys=json.dumps(self._chapter_keys, ensure_ascii=False),
            chapter_to_law=self._chapter_to_law,
            article_emb=self._article_emb,
            article_keys=json.dumps(self._article_keys, ensure_ascii=False),
            article_to_chapter=self._article_to_chapter,
            signature=sig,
        )
