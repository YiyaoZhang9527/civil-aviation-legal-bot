"""检索与原文读取。"""

from __future__ import annotations

import json
import re
from pathlib import Path

from . import config
from .hybrid_retrieval import BM25Index, VectorIndex, rrf_fuse
from .index_builder import build_all_documents
from .logger import TerminalLogger
from .tree_retrieval import TreeRetriever
from .types import Evidence, LawDocument, LawNode
from .utils import normalize_text, extract_phrases


class IndexRepository:
    _documents: list[LawDocument] | None = None
    _flattened: list[tuple[LawDocument, LawNode]] | None = None
    _file_cache: dict[str, list[str]] = {}
    _bm25_index: BM25Index | None = None
    _vector_index: VectorIndex | None = None
    _tree_index: TreeRetriever | None = None

    @classmethod
    def load(cls) -> list[LawDocument]:
        if cls._documents is None:
            cls._documents = _dedupe_documents(build_all_documents())
            cls._flattened = None
            cls._file_cache = {}
            cls._bm25_index = None
            cls._vector_index = None
        return cls._documents

    @classmethod
    def read_file_lines(cls, source_file: str) -> list[str]:
        if source_file not in cls._file_cache:
            cls._file_cache[source_file] = Path(source_file).read_text(encoding="utf-8").splitlines()
        return cls._file_cache[source_file]

    @classmethod
    def documents(cls) -> list[LawDocument]:
        return cls.load()

    @classmethod
    def flattened_nodes(cls) -> list[tuple[LawDocument, LawNode]]:
        if cls._flattened is not None:
            return cls._flattened
        flattened: list[tuple[LawDocument, LawNode]] = []
        for doc in cls.load():
            for node in doc.flatten():
                flattened.append((doc, node))
        cls._flattened = flattened
        return flattened

    @classmethod
    def find_document(cls, law_hint: str) -> LawDocument | None:
        hint = normalize_text(law_hint)
        for doc in cls.load():
            title_n = normalize_text(doc.title)
            if title_n == hint:
                return doc
            # 子串匹配要求双方都 ≥3 字符，防止 title="1" 匹配 hint="ccar-91"
            if len(title_n) >= 3 and len(hint) >= 3 and (hint in title_n or title_n in hint):
                return doc
            stem_n = normalize_text(Path(doc.source_file).stem)
            if stem_n == hint:
                return doc
            if len(stem_n) >= 3 and len(hint) >= 3 and (hint in stem_n or stem_n in hint):
                return doc
        return None

    @classmethod
    def find_document_by_law_id(cls, law_id: str) -> LawDocument | None:
        for doc in cls.load():
            if normalize_text(doc.law_id) == normalize_text(law_id):
                return doc
        return None

    @classmethod
    def bm25_index(cls) -> BM25Index:
        if cls._bm25_index is None:
            idx = BM25Index()
            idx.build(cls.flattened_nodes())
            cls._bm25_index = idx
        return cls._bm25_index

    @classmethod
    def vector_index(cls) -> VectorIndex:
        if cls._vector_index is None:
            vi = VectorIndex()
            vi.build(cls.flattened_nodes())
            cls._vector_index = vi
        return cls._vector_index

    @classmethod
    def tree_index(cls) -> TreeRetriever | None:
        if not config.TREE_ENABLED:
            return None
        if cls._tree_index is not None:
            return cls._tree_index if cls._tree_index.available else None
        ti = TreeRetriever()
        if ti.build(cls.documents()):
            cls._tree_index = ti
            return ti
        cls._tree_index = ti
        return None


def _canonical_title(title: str) -> str:
    return normalize_text(title).replace("中华人民共和国", "")


def _dedupe_documents(docs: list[LawDocument]) -> list[LawDocument]:
    best: dict[str, LawDocument] = {}
    for doc in docs:
        key = _canonical_title(doc.title)
        current = best.get(key)
        if current is None:
            best[key] = doc
            continue
        # 评分：正式标题 > 子节点数 > 文本长度
        current_score = (
            1 if "中华人民共和国" in current.title else 0,
            len(current.root.children),
            len(current.root.text),
        )
        doc_score = (
            1 if "中华人民共和国" in doc.title else 0,
            len(doc.root.children),
            len(doc.root.text),
        )
        if doc_score > current_score:
            best[key] = doc
    return list(best.values())


def _law_matches(law: LawDocument, hint: str) -> bool:
    hint_clean = normalize_text(hint).replace("中华人民共和国", "")
    law_name = normalize_text(law.title).replace("中华人民共和国", "")
    law_stem = normalize_text(Path(law.source_file).stem).replace("中华人民共和国", "")
    if law_name == hint_clean or law_stem == hint_clean:
        return True
    if len(hint_clean) >= 3:
        if law_name.endswith(hint_clean) or law_stem.endswith(hint_clean):
            return True
    return False


def _direct_read_articles(law_hints: list[str], article_hints: list[str]) -> list[Evidence]:
    """根据 article_hints 精确直读条文，绕过向量/关键词检索。"""
    if not article_hints:
        return []

    # 解析 hint 中的 (法名, 条号) 对
    pairs: list[tuple[str, str]] = []  # [(law_hint, article_num_str), ...]
    for hint in article_hints:
        # 匹配 "法名第67.33条" / "法名 第39条" / "第67.33条" (支持小数条号)
        m = re.match(r"(.+?)\s*第([一二三四五六七八九十百零\d]+(?:\.\d+)*)\s*条", hint)
        if m:
            law_part = m.group(1).strip()
            article_str = m.group(2)
            if "." in article_str:
                pairs.append((law_part, article_str))
            else:
                from .utils import chinese_to_int
                num = chinese_to_int(article_str)
                if num is not None:
                    pairs.append((law_part, str(num)))
            continue
        # 纯数字条号：如 "第39条"
        m2 = re.match(r"第([一二三四五六七八九十百零\d]+(?:\.\d+)*)\s*条", hint)
        if m2:
            article_str = m2.group(1)
            if "." in article_str:
                pairs.append(("", article_str))
            else:
                from .utils import chinese_to_int
                num = chinese_to_int(article_str)
                if num is not None:
                    pairs.append(("", str(num)))

    if not pairs:
        return []

    results: list[Evidence] = []
    for law_hint, article_num in pairs:
        # 找法
        doc = IndexRepository.find_document(law_hint) if law_hint else None
        if doc is None and law_hints:
            for lh in law_hints:
                doc = IndexRepository.find_document(lh)
                if doc is not None:
                    break
        if doc is None:
            continue

        # 找条文
        node_id = f"article:{article_num}"
        node = next((n for n in doc.flatten() if n.node_id == node_id), None)
        if node is None:
            continue

        text = "\n".join(filter(None, [node.summary, node.text, node.title]))
        results.append(Evidence(
            law_id=doc.law_id,
            law_title=doc.title,
            node_id=node.node_id,
            article=node.title,
            text=text,
            score=1.0,  # 直读给最高分
            source_file=doc.source_file,
            source_anchor=node.source_anchor or node.title,
            verified=False,
        ))

    return results


# CCAR 编号→法规名映射（确定性，零LLM）
_CCR_TO_LAW = {
    "121": "大型飞机公共航空运输承运人运行合格审定规则",
    "135": "小型商业运输和空中游览运营人运行合格审定规则",
    "91": "一般运行和飞行规则",
    "92": "民用无人驾驶航空器运行安全管理规则",
    "141": "民用航空器驾驶员学校合格审定规则",
    "142": "飞行训练中心合格审定规则",
    "61": "民用航空器驾驶员",
    "67": "民用航空人员体检合格证管理规则",
    "145": "民用航空器维修单位合格审定规则",
    "43": "民用航空器维修人员执照管理规则",
}


def _extract_ccar_hints(query: str) -> list[str]:
    """从 query 中提取 CCAR-XXX 编号，映射到对应法规名。确定性，零LLM。"""
    hints = []
    for m in re.finditer(r"CCAR[\s\-]*(\d+)", query, re.IGNORECASE):
        num = m.group(1)
        if num in _CCR_TO_LAW:
            hints.append(_CCR_TO_LAW[num])
    return hints


# A2: 关键词→法规路由。优先从 law_routes.json（LLM离线生成）加载，
# 不存在时降级到硬编码规则。匹配只做增量，不阻断。
_KEYWORD_LAW_RULES_FALLBACK: list[dict] = [
    {"keywords": ["运行规范", "备降"], "hints": ["大型飞机公共航空运输承运人运行合格审定规则"]},
    {"keywords": ["121", "备降"], "hints": ["大型飞机公共航空运输承运人运行合格审定规则"]},
    {"keywords": ["签派", "放行"], "hints": ["大型飞机公共航空运输承运人运行合格审定规则"]},
    {"keywords": ["燃油"], "hints": ["大型飞机公共航空运输承运人运行合格审定规则"]},
    {"keywords": ["天气标准"], "hints": ["大型飞机公共航空运输承运人运行合格审定规则"]},
    {"keywords": ["结冰"], "hints": ["大型飞机公共航空运输承运人运行合格审定规则"]},
    {"keywords": ["135", "运行"], "hints": ["小型商业运输和空中游览运营人运行合格审定规则"]},
    {"keywords": ["91", "运行"], "hints": ["一般运行和飞行规则"]},
    {"keywords": ["无人机"], "hints": ["民用无人驾驶航空器运行安全管理规则"]},
    {"keywords": ["机场", "使用许可"], "hints": ["运输机场使用许可规定"]},
    {"keywords": ["机场", "运行安全"], "hints": ["运输机场运行安全管理规定"]},
    {"keywords": ["旅客", "延误"], "hints": ["公共航空运输旅客服务管理规定"]},
    {"keywords": ["行李", "赔偿"], "hints": ["公共航空运输旅客服务管理规定"]},
    {"keywords": ["拒载"], "hints": ["公共航空运输旅客服务管理规定"]},
    {"keywords": ["航班正常"], "hints": ["航班正常管理规定"]},
    {"keywords": ["民用航空法"], "hints": ["中华人民共和国民用航空法"]},
    {"keywords": ["适航指令"], "hints": ["民用航空器适航指令规定"]},
    {"keywords": ["适航"], "hints": ["民用航空产品和零部件合格审定规定"]},
    {"keywords": ["禁带物品"], "hints": ["民航旅客禁止随身携带和托运物品目录"]},
    {"keywords": ["禁带", "物品"], "hints": ["民航旅客禁止随身携带和托运物品目录"]},
    {"keywords": ["禁止", "带上飞机"], "hints": ["民航旅客禁止随身携带和托运物品目录", "民用航空安全检查规则"]},
    {"keywords": ["禁止", "携带"], "hints": ["民航旅客禁止随身携带和托运物品目录", "民用航空安全检查规则"]},
    {"keywords": ["安检"], "hints": ["民用航空安全检查规则"]},
    {"keywords": ["事故", "调查"], "hints": ["民用航空器事件技术调查规定"]},
]

# 路由表缓存（懒加载）
_law_routes_cache: dict | None = None


def _load_law_routes() -> dict:
    """加载 law_routes.json 路由表。"""
    global _law_routes_cache
    if _law_routes_cache is not None:
        return _law_routes_cache
    routes_path = Path(__file__).parent / "law_routes.json"
    if routes_path.exists():
        with open(routes_path, encoding="utf-8") as f:
            data = json.load(f)
        _law_routes_cache = data.get("routes", {})
        return _law_routes_cache
    _law_routes_cache = {}
    return _law_routes_cache


def _tokenize_for_routing(text: str) -> set[str]:
    """分词用于路由匹配。jieba分词后保留2字以上的词。"""
    import jieba
    tokens = set()
    for w in jieba.cut(text):
        w = w.strip()
        if len(w) >= 2:
            tokens.add(w)
    return tokens


def _resolve_law_by_keywords(query: str, law_hints: list[str]) -> list[str]:
    """基于关键词的确定性法规路由。匹配时追加 hint，未匹配时原样返回。"""
    q = normalize_text(query)
    q_tokens = _tokenize_for_routing(q)
    extra = []

    # 1. law_routes.json 路由（LLM 离线生成的完整路由表）
    routes = _load_law_routes()
    if routes:
        for law_id, info in routes.items():
            for kw_group in info.get("keywords", []):
                if isinstance(kw_group, str):
                    kw_tokens = _tokenize_for_routing(kw_group)
                    overlap = q_tokens & kw_tokens
                    if len(overlap) >= max(2, len(kw_tokens) // 2):
                        extra.append(law_id)
                        break
            if law_id not in extra:
                for pattern in info.get("question_patterns", []):
                    pat_tokens = _tokenize_for_routing(pattern)
                    overlap = q_tokens & pat_tokens
                    if len(overlap) >= 2:
                        extra.append(law_id)
                        break
            if law_id not in extra:
                for domain in info.get("domains", []):
                    dom_tokens = _tokenize_for_routing(domain)
                    overlap = q_tokens & dom_tokens
                    if len(overlap) >= max(2, len(dom_tokens) // 2):
                        extra.append(law_id)
                        break

    # 2. 硬编码 fallback 规则（精确关键词匹配，补充 law_routes.json 覆盖不到的场景）
    existing_extra = set(normalize_text(h) for h in extra)
    for rule in _KEYWORD_LAW_RULES_FALLBACK:
        if all(kw in q for kw in rule["keywords"]):
            for h in rule["hints"]:
                if normalize_text(h) not in existing_extra:
                    extra.append(h)
                    existing_extra.add(normalize_text(h))

    if extra:
        existing = set(normalize_text(h) for h in law_hints)
        for h in extra:
            if normalize_text(h) not in existing:
                law_hints = law_hints + [h]
    return law_hints


def _score_hints(node: LawNode, law: LawDocument, article_hints: list[str], law_hints: list[str]) -> float:
    """领域 hint 加分：法名精确匹配 + 条号精确匹配。"""
    score = 0.0
    normalized_article_hints = _normalize_article_hints(article_hints)
    if node.type == "law":
        for hint in law_hints:
            if _law_matches(law, hint):
                score += 6.0
    if node.type == "article":
        article_num = re.search(r"(\d+)", node.node_id)
        if article_num and article_num.group(1) in normalized_article_hints:
            score += 3.5
    if law_hints and any(_law_matches(law, hint) for hint in law_hints):
        score += 2.0
    if node.type == "law" and law_hints:
        score += 1.0
    return score


def _normalize_article_hints(article_hints: list[str]) -> set[str]:
    result: set[str] = set()
    for hint in article_hints:
        for num in re.findall(r"\d+", str(hint)):
            result.add(num)
        for cn in re.findall(r"第([一二三四五六七八九十百零]+)条", str(hint)):
            from .utils import chinese_to_int

            value = chinese_to_int(cn)
            if value is not None:
                result.add(str(value))
    return result


def _find_node(law_id: str, node_id: str) -> tuple[LawDocument | None, LawNode | None]:
    for doc, node in IndexRepository.flattened_nodes():
        if doc.law_id == law_id and node.node_id == node_id:
            return doc, node
    return None, None


def search_index_tree(query: str, top_k: int = config.DEFAULT_TOP_K, law_hints: list[str] | None = None, article_hints: list[str] | None = None, _logger: TerminalLogger | None = None) -> list[Evidence]:
    import time as _time

    law_hints = law_hints or []
    article_hints = article_hints or []
    t_start = _time.monotonic()

    # ── Phase 0: article_hints 精确直读 ──
    direct_evidence = _direct_read_articles(law_hints, article_hints)
    if direct_evidence and _logger:
        _logger.info("retrieval/检索", "Phase 0: 根据LLM推荐条号精确直读法条(跳过检索)",
                     f"直读{len(direct_evidence)}条: {', '.join(e.article + '(' + e.law_title + ')' for e in direct_evidence)}")

    # ── Phase 1: 检索（根据模式选择策略）补充证据 ──
    mode = config.RETRIEVAL_MODE
    if mode == "keyword_hints":
        if _logger:
            _logger.info("retrieval/检索", "Phase 1: 纯关键词+Hints检索(零GPU)", "")
        searched = _keyword_hints_search(query, top_k, law_hints, article_hints, _logger, t_start)
    elif mode == "flat":
        if _logger:
            _logger.info("retrieval/检索", "Phase 1: 扁平检索", "")
        searched = _flat_search(query, top_k, law_hints, article_hints, _logger, t_start)
    else:
        tree = IndexRepository.tree_index()
        if tree is not None:
            if _logger:
                _logger.info("retrieval/检索", "Phase 1: 三级树检索(法→章→条逐层剪枝)", "")
            searched = _tree_search(query, top_k, law_hints, article_hints, _logger, t_start, tree)
        else:
            if _logger:
                _logger.info("retrieval/检索", "Phase 1: 扁平检索(树检索不可用，降级)", "")
            searched = _flat_search(query, top_k, law_hints, article_hints, _logger, t_start)

    # ── 合并：直读优先，去重 ──
    if not direct_evidence:
        return searched

    seen = {(e.law_id, e.node_id) for e in direct_evidence}
    for e in searched:
        if (e.law_id, e.node_id) not in seen:
            direct_evidence.append(e)
            seen.add((e.law_id, e.node_id))

    return direct_evidence[:top_k]


def _keyword_hints_search(query, top_k, law_hints, article_hints, _logger, t_start):
    """纯 BM25 + LLM Hints 检索：零 GPU，最快路径。"""
    import time as _time

    t0 = _time.monotonic()
    bm25 = IndexRepository.bm25_index()
    bm25_results = bm25.search(query, top_k=config.BM25_RECALL_K)
    bm25_time = _time.monotonic() - t0

    if _logger:
        _logger.info("retrieval/关键词检索", f"BM25 召回: {len(bm25_results)} 条 ({bm25_time*1000:.0f}ms)")

    if not bm25_results:
        return []

    # Hint 加分
    t1 = _time.monotonic()
    hint_scores: dict[tuple[str, str], float] = {}
    hint_boosted = 0
    for (law_id, node_id), _ in bm25_results:
        doc, node = _find_node(law_id, node_id)
        if doc is None or node is None:
            continue
        hs = _score_hints(node, doc, article_hints, law_hints)
        if hs > 0:
            hint_scores[(law_id, node_id)] = hs
            hint_boosted += 1

    # RRF: 仅 BM25 + Hints（无向量通道）
    fused = rrf_fuse(bm25_results, [], hint_scores, k=config.RRF_K)

    # B3: 信号置信度截断（keyword_hints路径）
    if config.CONFIDENCE_CUTOFF_ENABLED:
        from .hybrid_retrieval import signal_cutoff
        fused = signal_cutoff(fused, bm25_results, [], hint_scores,
                              min_signals=config.CONFIDENCE_MIN_SIGNALS)

    candidates: list[Evidence] = []
    for (law_id, node_id), score in fused.items():
        doc, node = _find_node(law_id, node_id)
        if doc is None or node is None:
            continue
        if node.type not in {"law", "chapter", "section", "article"}:
            continue
        candidates.append(
            Evidence(
                law_id=doc.law_id,
                law_title=doc.title,
                node_id=node.node_id,
                article=node.title,
                text="\n".join(filter(None, [node.summary, node.text, node.title])),
                score=score,
                source_file=doc.source_file,
                source_anchor=node.source_anchor or node.title,
                verified=False,
            )
        )

    candidates.sort(key=lambda x: x.score, reverse=True)
    deduped: list[Evidence] = []
    seen: set[tuple[str, str]] = set()
    for item in candidates:
        key = (item.law_id, item.node_id)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    total_time = _time.monotonic() - t_start
    if _logger:
        _logger.success("retrieval/检索完成", f"[关键词+Hints] 返回{len(deduped)}条证据", f"总耗时{total_time*1000:.0f}ms")
    return deduped[:top_k]


def _tree_search(query, top_k, law_hints, article_hints, _logger, t_start, tree):
    """树检索主路径：法→章→条逐层剪枝。"""
    import time as _time

    # CCAR 编号识别：从 query 中提取 CCAR-XXX → law_hints
    ccr_hints = _extract_ccar_hints(query)
    if ccr_hints:
        existing = set(normalize_text(h) for h in law_hints)
        for h in ccr_hints:
            if normalize_text(h) not in existing:
                law_hints = law_hints + [h]

    # 关键词路由：确定性法规匹配（law_routes.json + fallback 硬编码规则）
    if getattr(config, 'KEYWORD_ROUTING_ENABLED', True):
        law_hints = _resolve_law_by_keywords(query, law_hints)

    # A2: 解析 hint_law_ids 用于 Level 0 前置过滤
    hint_law_ids: set[str] | None = None
    if law_hints:
        hint_law_ids = set()
        for hint in law_hints:
            hd = IndexRepository.find_document(hint)
            if hd:
                hint_law_ids.add(hd.law_id)
        if not hint_law_ids:
            hint_law_ids = None

    # A2: 预计算 hint_scores 用于 Level 2 条级融合
    pre_hint_scores: dict[tuple[str, str], float] = {}
    if article_hints or law_hints:
        normalized_articles = _normalize_article_hints(article_hints)
        for doc, node in IndexRepository.flattened_nodes():
            if node.type == "article":
                hs = _score_hints(node, doc, article_hints, law_hints)
                if hs > 0:
                    pre_hint_scores[(doc.law_id, node.node_id)] = hs

    t0 = _time.monotonic()
    tree_results = tree.search(query, top_k=top_k,
                               hint_law_ids=hint_law_ids,
                               hint_scores=pre_hint_scores if pre_hint_scores else None)
    tree_time = _time.monotonic() - t0

    if not tree_results:
        if _logger:
            _logger.warning("retrieval/树检索", "树检索返回0条结果，降级为扁平检索", "")
        return _flat_search(query, top_k, law_hints, article_hints, _logger, t_start)

    if _logger:
        _logger.info("retrieval/树检索", f"法→章→条逐层剪枝完成", f"候选{len(tree_results)}条 ({tree_time*1000:.0f}ms)")

    # ── hint scoring ──
    t2 = _time.monotonic()
    hint_scores: dict[tuple[str, str], float] = {}
    hint_boosted = 0
    for (law_id, node_id), score in tree_results:
        doc, node = _find_node(law_id, node_id)
        if doc is None or node is None:
            continue
        hs = _score_hints(node, doc, article_hints, law_hints)
        if hs > 0:
            hint_scores[(law_id, node_id)] = hs
            hint_boosted += 1

    # 重新排序：tree score + hint boost
    final_scores: dict[tuple[str, str], float] = {}
    for (law_id, node_id), score in tree_results:
        boost = hint_scores.get((law_id, node_id), 0.0)
        final_scores[(law_id, node_id)] = score + boost * config.HYBRID_WEIGHT_HINT / 12.0

    # B3: 信号置信度截断（树检索路径：tree=1路，tree+hint=2路）
    if config.CONFIDENCE_CUTOFF_ENABLED and hint_scores:
        tree_keys = {key for key, _ in tree_results}
        hint_keys = {key for key, s in hint_scores.items() if s > 0}
        before_count = len(final_scores)
        final_scores = {
            key: score for key, score in final_scores.items()
            if (key in tree_keys) + (key in hint_keys) >= config.CONFIDENCE_MIN_SIGNALS
        }
        if _logger:
            _logger.info("retrieval/信号截断",
                         f"树检索截断: {before_count}→{len(final_scores)}", "")

    # hint 匹配结果：如果 hint 能命中文档，则过滤非匹配项；否则不做过滤
    # 复用上面已计算的 hint_law_ids（Level 0 前置过滤已做，这里做后置双保险）
    candidates: list[Evidence] = []
    for (law_id, node_id), score in sorted(final_scores.items(), key=lambda x: x[1], reverse=True):
        doc, node = _find_node(law_id, node_id)
        if doc is None or node is None:
            continue
        if node.type not in {"law", "chapter", "section", "article"}:
            continue
        candidates.append(
            Evidence(
                law_id=doc.law_id,
                law_title=doc.title,
                node_id=node.node_id,
                article=node.title,
                text="\n".join(filter(None, [node.summary, node.text, node.title])),
                score=score,
                source_file=doc.source_file,
                source_anchor=node.source_anchor or node.title,
                verified=False,
            )
        )
    hint_time = _time.monotonic() - t2
    if _logger:
        _logger.info("retrieval/树检索", f"LLM推荐法条Hint加分: {hint_boosted}条证据获得额外权重", f"({hint_time*1000:.0f}ms)")

    # ── Cross-Encoder 精排 ──
    prerank = candidates[:min(top_k * 3, config.TREE_ARTICLE_CANDIDATES)]
    deduped: list[Evidence] = prerank
    rerank_time = 0.0
    if config.RERANKER_ENABLED and len(prerank) > 1:
        t3 = _time.monotonic()
        from .reranker import rerank
        to_rerank = [{"title": e.article, "text": e.text, "_evidence": e} for e in prerank]
        reranked = rerank(query, to_rerank, top_n=top_k)
        # 精排后按阈值过滤低分证据
        if config.RERANKER_MIN_SCORE > 0:
            filtered = [item for item in reranked if item.get("_rerank_score", 0) >= config.RERANKER_MIN_SCORE]
        else:
            filtered = reranked
        # 如果 reranker 全部过滤掉，回退到 raw top_k（避免零结果）
        if filtered:
            deduped = [item["_evidence"] for item in filtered if "_evidence" in item]
        else:
            deduped = prerank[:top_k]
        rerank_time = _time.monotonic() - t3
        if _logger:
            _logger.info("retrieval/精排", f"Cross-Engine精排: {len(prerank)}条 → {len(deduped)}条 (阈值>={config.RERANKER_MIN_SCORE})", f"({rerank_time*1000:.0f}ms)")
    else:
        deduped = prerank[:top_k]

    # Retrieve-then-Verify: 精排后结果不足时扩展重检索
    if len(deduped) < 3 and tree is not None:
        if _logger:
            _logger.warning("retrieval/扩展重检索", f"精排仅{len(deduped)}条，扩大Level 0候选重检索", "")
        expanded_results = tree.search(query, top_k=top_k * 2,
                                       hint_law_ids=None,
                                       hint_scores=None,
                                       tree_top_laws=config.TREE_TOP_LAWS)
        if expanded_results:
            extra_candidates: list[Evidence] = []
            for (law_id, node_id), score in expanded_results:
                doc, node = _find_node(law_id, node_id)
                if doc is None or node is None:
                    continue
                if node.type not in {"law", "chapter", "section", "article"}:
                    continue
                extra_candidates.append(
                    Evidence(
                        law_id=doc.law_id, law_title=doc.title,
                        node_id=node.node_id, article=node.title,
                        text="\n".join(filter(None, [node.summary, node.text, node.title])),
                        score=score, source_file=doc.source_file,
                        source_anchor=node.source_anchor or node.title,
                        verified=False,
                    )
                )
            if config.RERANKER_ENABLED and len(extra_candidates) > 1:
                to_rerank2 = [{"title": e.article, "text": e.text, "_evidence": e} for e in extra_candidates[:min(top_k * 3, config.TREE_ARTICLE_CANDIDATES)]]
                reranked2 = rerank(query, to_rerank2, top_n=top_k)
                if config.RERANKER_MIN_SCORE > 0:
                    reranked2 = [item for item in reranked2 if item.get("_rerank_score", 0) >= config.RERANKER_MIN_SCORE]
                expanded_deduped = [item["_evidence"] for item in reranked2 if "_evidence" in item]
            else:
                expanded_deduped = extra_candidates[:top_k]
            if len(expanded_deduped) > len(deduped):
                deduped = expanded_deduped
                if _logger:
                    _logger.info("retrieval/扩展重检索", f"扩展后获得{len(deduped)}条证据", "")

    total_time = _time.monotonic() - t_start
    if _logger:
        _logger.success("retrieval/检索完成", f"[树检索] 返回{len(deduped)}条证据", f"总耗时{total_time*1000:.0f}ms")
        for e in deduped[:5]:
            _logger.debug("retrieval", "result", f"{e.law_title} {e.article} score={e.score:.4f}")
    return deduped


def _flat_search(query, top_k, law_hints, article_hints, _logger, t_start):
    """扁平检索 fallback（原有逻辑）。"""
    import time as _time

    # ── Phase 1: 粗召回 ──
    t0 = _time.monotonic()
    bm25 = IndexRepository.bm25_index()
    vi = IndexRepository.vector_index()

    bm25_results = bm25.search(query, top_k=config.BM25_RECALL_K)
    bm25_time = _time.monotonic() - t0

    t1 = _time.monotonic()
    vector_results = vi.search(query, top_k=config.BM25_RECALL_K) if vi.available else []
    vector_time = _time.monotonic() - t1

    if _logger:
        _logger.info("retrieval", "bm25_recall", f"{len(bm25_results)} candidates ({bm25_time*1000:.0f}ms)")
        if vi.available:
            _logger.info("retrieval", "vector_recall", f"{len(vector_results)} candidates ({vector_time*1000:.0f}ms)")
        else:
            _logger.warning("retrieval", "vector_recall", "disabled (model unavailable)")

    if not bm25_results and not vector_results:
        if _logger:
            _logger.warning("retrieval", "fallback", "BM25+vector empty, falling back to legacy keyword search")
        return _legacy_search_index_tree(query, top_k, law_hints, article_hints)

    # ── Phase 2: RRF 融合 ──
    t2 = _time.monotonic()

    all_keys: set[tuple[str, str]] = set()
    for (law_id, node_id), _ in bm25_results:
        all_keys.add((law_id, node_id))
    for (law_id, node_id), _ in vector_results:
        all_keys.add((law_id, node_id))

    hint_scores: dict[tuple[str, str], float] = {}
    hint_boosted = 0
    for law_id, node_id in all_keys:
        doc, node = _find_node(law_id, node_id)
        if doc is None or node is None:
            continue
        hs = _score_hints(node, doc, article_hints, law_hints)
        if hs > 0:
            hint_scores[(law_id, node_id)] = hs
            hint_boosted += 1

    fused = rrf_fuse(bm25_results, vector_results, hint_scores, k=config.RRF_K)

    # B3: 信号置信度截断
    if config.CONFIDENCE_CUTOFF_ENABLED:
        from .hybrid_retrieval import signal_cutoff
        before_count = len(fused)
        fused = signal_cutoff(fused, bm25_results, vector_results, hint_scores,
                              min_signals=config.CONFIDENCE_MIN_SIGNALS)
        if _logger:
            _logger.info("retrieval/信号截断",
                         f"截断: {before_count}→{len(fused)} (至少{config.CONFIDENCE_MIN_SIGNALS}路信号一致)", "")

    candidates: list[Evidence] = []
    for (law_id, node_id), score in fused.items():
        doc, node = _find_node(law_id, node_id)
        if doc is None or node is None:
            continue
        if node.type not in {"law", "chapter", "section", "article"}:
            continue
        candidates.append(
            Evidence(
                law_id=doc.law_id,
                law_title=doc.title,
                node_id=node.node_id,
                article=node.title,
                text="\n".join(filter(None, [node.summary, node.text, node.title])),
                score=score,
                source_file=doc.source_file,
                source_anchor=node.source_anchor or node.title,
                verified=False,
            )
        )
    score_time = _time.monotonic() - t2

    if _logger:
        _logger.info("retrieval", "rrf_fuse", f"{len(fused)} fused from {len(all_keys)} candidates, {hint_boosted} hint-boosted ({score_time*1000:.0f}ms)")

    # ── Phase 3: 去重 + Cross-Encoder 精排 ──
    candidates.sort(key=lambda x: (x.score, len(x.text)), reverse=True)
    prerank: list[Evidence] = []
    seen: set[tuple[str, str]] = set()
    for item in candidates:
        key = (item.law_id, item.node_id)
        if key in seen:
            continue
        seen.add(key)
        prerank.append(item)
        if len(prerank) >= min(top_k * 3, config.TREE_ARTICLE_CANDIDATES):
            break

    deduped: list[Evidence] = prerank
    rerank_time = 0.0
    if config.RERANKER_ENABLED and len(prerank) > 1:
        t3 = _time.monotonic()
        from .reranker import rerank
        to_rerank = [{"title": e.article, "text": e.text, "_evidence": e} for e in prerank]
        reranked = rerank(query, to_rerank, top_n=top_k)
        # 精排后按阈值过滤低分证据
        if config.RERANKER_MIN_SCORE > 0:
            reranked = [item for item in reranked if item.get("_rerank_score", 0) >= config.RERANKER_MIN_SCORE]
        deduped = [item["_evidence"] for item in reranked if "_evidence" in item]
        rerank_time = _time.monotonic() - t3
        if _logger:
            _logger.info("retrieval", "rerank", f"{len(prerank)} -> {len(deduped)} after cross-encoder (threshold>={config.RERANKER_MIN_SCORE}, {rerank_time*1000:.0f}ms)")
    else:
        deduped = prerank[:top_k]

    total_time = _time.monotonic() - t_start
    if _logger:
        _logger.success("retrieval", "search_done", f"[flat] {len(deduped)} results ({total_time*1000:.0f}ms total)")
        for e in deduped[:5]:
            _logger.debug("retrieval", "result", f"{e.law_title} {e.article} rrf={e.score:.4f}")
    return deduped


def _legacy_search_index_tree(query: str, top_k: int = config.DEFAULT_TOP_K, law_hints: list[str] | None = None, article_hints: list[str] | None = None) -> list[Evidence]:
    """关键词检索兜底，BM25/向量均不可用时使用。"""
    law_hints = law_hints or []
    article_hints = article_hints or []
    candidates: list[Evidence] = []
    _hinted_law_ids_flat: set[str] | None = None
    if law_hints:
        _hinted_law_ids_flat = set()
        for hint in law_hints:
            hd = IndexRepository.find_document(hint)
            if hd:
                _hinted_law_ids_flat.add(hd.law_id)
        if not _hinted_law_ids_flat:
            _hinted_law_ids_flat = None

    for law, node in IndexRepository.flattened_nodes():
        if node.type not in {"law", "chapter", "section", "article"}:
            continue
        if _hinted_law_ids_flat is not None and law.law_id not in _hinted_law_ids_flat:
            continue
        score = _score_node(query, node, law, article_hints, law_hints)
        if score < 1.0:
            continue
        if node.type == "law" and score < 2.4:
            continue
        if node.type == "chapter" and score < 1.8:
            continue
        if node.type == "article" and score < 2.0:
            continue
        candidates.append(
            Evidence(
                law_id=law.law_id,
                law_title=law.title,
                node_id=node.node_id,
                article=node.title,
                text="\n".join(filter(None, [node.summary, node.text, node.title])),
                score=score,
                source_file=law.source_file,
                source_anchor=node.source_anchor or node.title,
                verified=False,
            )
        )
    candidates.sort(key=lambda x: (x.score, len(x.text)), reverse=True)
    deduped: list[Evidence] = []
    seen: set[tuple[str, str]] = set()
    for item in candidates:
        key = (item.law_id, item.node_id)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= top_k:
            break
    return deduped


def _score_node(question: str, node: LawNode, law: LawDocument, article_hints: list[str], law_hints: list[str]) -> float:
    q = normalize_text(question)
    haystack = normalize_text(" ".join([node.title, node.summary, " ".join(node.keywords), node.text]))
    score = 0.0
    if node.type == "law":
        for hint in law_hints:
            if _law_matches(law, hint):
                score += 6.0
    normalized_article_hints = _normalize_article_hints(article_hints)
    if node.type == "article":
        article_num = re.search(r"(\d+)", node.node_id)
        if article_num and article_num.group(1) in normalized_article_hints:
            score += 3.5
    for keyword in extract_phrases(q, 2):
        if keyword and keyword in haystack:
            score += 1.2
    for keyword in node.keywords:
        if keyword and normalize_text(keyword) in q:
            score += 1.4
    if law_hints and any(_law_matches(law, hint) for hint in law_hints):
        score += 2.0
    if node.type == "law" and law_hints:
        score += 1.0
    if node.type == "article" and node.summary and any(term in normalize_text(node.summary) for term in extract_phrases(q, 2)):
        score += 0.8
    if node.type == "law":
        score *= 0.7
    if node.type == "chapter":
        score *= 0.85
    return score


def resolve_node_text(source_file: str, node_id: str, include_context: bool = True) -> tuple[str, dict]:
    docs = IndexRepository.documents()
    doc = next((item for item in docs if item.source_file == source_file), None)
    if doc is None:
        return "", {}
    anchor = doc.anchor_map.get(node_id)
    if not anchor:
        # fallback by node_id suffix
        node = next((n for n in doc.flatten() if n.node_id == node_id), None)
        if node and node.line_start is not None and node.line_end is not None:
            anchor = {
                "node_id": node.node_id,
                "type": node.type,
                "title": node.title,
                "line_start": node.line_start,
                "line_end": node.line_end,
                "source_anchor": node.source_anchor,
            }
    if not anchor:
        return "", {}
    lines = IndexRepository.read_file_lines(source_file)
    start = max(0, int(anchor.get("line_start", 0)) - (2 if include_context else 0))
    end = min(len(lines), int(anchor.get("line_end", len(lines) - 1)) + 1 + (2 if include_context else 0))
    text = "\n".join(line.rstrip() for line in lines[start:end]).strip()
    return text, anchor


def read_law_node(law_id: str, node_id: str, include_context: bool = True) -> dict:
    doc = IndexRepository.find_document_by_law_id(law_id)
    if doc is None:
        return {"found": False, "reason": "law_not_found"}
    text, anchor = resolve_node_text(doc.source_file, node_id, include_context=include_context)
    if not text:
        node = next((n for n in doc.flatten() if n.node_id == node_id), None)
        if node:
            text = node.text or node.summary or node.title
            anchor = {
                "node_id": node.node_id,
                "type": node.type,
                "title": node.title,
                "source_anchor": node.source_anchor,
            }
    if not text:
        return {"found": False, "reason": "node_not_found"}
    return {
        "found": True,
        "law_id": doc.law_id,
        "law_title": doc.title,
        "node_id": node_id,
        "article": node_id.split(":")[-1],
        "text": text,
        "source_file": doc.source_file,
        "source_anchor": anchor.get("source_anchor", ""),
    }
