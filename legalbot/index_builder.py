"""索引树构建器。"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import config
from .parser import parse_law_text, render_index_markdown
from .summary_generator import enhance_summaries
from .types import LawDocument

logger = logging.getLogger(__name__)


def build_document(source_path: Path, llm=None) -> LawDocument:
    doc = parse_law_text(source_path)
    enhance_summaries(doc, llm)
    return doc


def build_all_documents() -> list[LawDocument]:
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)

    llm = _try_create_llm()
    max_workers = max(1, int(config.INDEX_MAX_WORKERS))

    files = sorted(config.LAW_DATA_DIR.glob("*.txt"))
    total = len(files)
    print(f"共 {total} 个法律文本，开始构建索引（{max_workers} 线程）...")

    def _process_one(source_path: Path) -> LawDocument:
        doc = build_document(source_path, llm=llm)
        md_path = config.INDEX_DIR / f"{source_path.stem}{config.INDEX_SUFFIX}"
        md_path.write_text(render_index_markdown(doc), encoding="utf-8")
        anchor_path = config.INDEX_DIR / f"{source_path.stem}{config.ANCHOR_SUFFIX}"
        anchor_path.write_text(json.dumps(doc.anchor_map, ensure_ascii=False, indent=2), encoding="utf-8")
        return doc

    documents: list[LawDocument] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process_one, f): f for f in files}
        for i, future in enumerate(as_completed(futures), 1):
            source_path = futures[future]
            try:
                doc = future.result()
                documents.append(doc)
            except Exception as exc:
                logger.error("构建索引失败: %s: %s", source_path.stem, exc)
            if i % 10 == 0 or i == total:
                print(f"[{i}/{total}] 已完成")

    print(f"索引构建完成，共处理 {len(documents)} 部法律")
    try:
        from .retrieval import IndexRepository

        IndexRepository._documents = None
        IndexRepository._flattened = None
    except Exception:
        pass
    return documents


def _try_create_llm():
    """尝试创建 LLMClient，失败则返回 None（使用规则摘要兜底）。"""
    try:
        from .llm import LLMClient
        return LLMClient()
    except Exception as exc:
        logger.info("LLM 不可用，索引摘要使用规则兜底: %s", exc)
        return None
