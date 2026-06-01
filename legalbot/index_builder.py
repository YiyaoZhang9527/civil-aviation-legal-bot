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

    # 同步生成法规路由表
    _generate_law_routes(documents, llm)

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


_ROUTES_PROMPT = """你是一个中国民航法规专家。根据以下法规信息，提取该法规的关键词路由信息。

法规标题: {title}
法规ID: {law_id}
前几条正文:
{preview}

请输出JSON格式（不要markdown代码块，直接输出JSON）：
{{
  "domains": ["该法规覆盖的领域，用2-4个短词描述"],
  "keywords": ["能标识该法规的关键词或短语，至少5个，包含简称、部号、核心主题词"],
  "question_patterns": ["用户可能提问的模式，包含口语化表达，至少3个"]
}}

要求：
1. keywords要包含常见的简称和口语表达（如"无人机"而非仅"民用无人驾驶航空器"）
2. keywords要包含CCAR-XX部号（如果适用）
3. question_patterns要模拟真实用户提问，用日常语言
4. 输出纯JSON，不要任何额外文字"""


def _generate_law_routes(documents: list[LawDocument], llm) -> None:
    """索引构建后增量生成 law_routes.json。只处理新增/变更的法规。"""
    if llm is None:
        logger.info("LLM 不可用，跳过法规路由表生成")
        return

    routes_path = config.BASE_DIR / "law_routes.json"

    # 加载已有路由表
    existing_routes = {}
    if routes_path.exists():
        try:
            with open(routes_path, encoding="utf-8") as f:
                existing = json.load(f)
            existing_routes = existing.get("routes", {})
        except Exception:
            existing_routes = {}

    # 计算差异
    current_ids = {doc.law_id for doc in documents}
    existing_ids = set(existing_routes.keys())

    new_ids = current_ids - existing_ids        # 新增法规
    removed_ids = existing_ids - current_ids    # 已删除法规
    unchanged_ids = current_ids & existing_ids  # 无变化

    if not new_ids and not removed_ids:
        print(f"法规路由表无变化 ({len(unchanged_ids)}部)，跳过生成")
        return

    print(f"法规路由表增量更新: +{len(new_ids)}新增, -{len(removed_ids)}删除, ={len(unchanged_ids)}不变")

    # 只对新增法规调用LLM
    results = dict(existing_routes)  # 复用已有路由
    for rid in removed_ids:
        results.pop(rid, None)
    failed = []

    for i, doc in enumerate(documents):
        if doc.law_id not in new_ids:
            continue

        title = doc.title[:80]
        law_id = doc.law_id
        preview = _get_doc_preview(doc)
        prompt = _ROUTES_PROMPT.format(title=title, law_id=law_id, preview=preview)

        try:
            response = llm.chat(messages=[{"role": "user", "content": prompt}], temperature=0)
            content = response.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            parsed = json.loads(content)
            results[law_id] = {
                "title": title,
                "domains": parsed.get("domains", []),
                "keywords": parsed.get("keywords", []),
                "question_patterns": parsed.get("question_patterns", []),
            }
        except Exception as exc:
            logger.debug("路由生成失败 %s: %s", law_id[:20], exc)
            failed.append(law_id)
            results[law_id] = {"title": title, "domains": [], "keywords": [title[:10]], "question_patterns": []}

    output = {
        "_meta": {"generated_by": "index_builder", "total_laws": len(documents), "failed": failed},
        "routes": results,
    }
    routes_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    if new_ids:
        gen_ok = len(new_ids) - len(failed)
        print(f"法规路由表已更新: +{gen_ok}新增, -{len(removed_ids)}删除, 共{len(results)}部")


def _get_doc_preview(doc: LawDocument, max_chars: int = 800) -> str:
    parts = []
    total = 0
    for child in doc.root.children[:5]:
        text = child.summary or child.text or child.title
        if text:
            parts.append(text[:200])
            total += len(text[:200])
            if total >= max_chars:
                break
    return "\n".join(parts)[:max_chars]
