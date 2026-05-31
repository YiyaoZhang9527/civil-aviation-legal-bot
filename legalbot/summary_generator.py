"""LLM 增强索引摘要：法级、章级语义 scope summary + 文件哈希缓存。"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from . import config
from .llm import LLMClient, LLMError
from .types import LawDocument

logger = logging.getLogger(__name__)

# ── Prompt 模板 ──────────────────────────────────────────────

LAW_PROMPT = """\
你是一个法律文档分析专家。请为以下法律文件生成一段 150 字以内的 scope summary。
要求涵盖：1）该法的核心调整对象 2）主要规制领域 3）适用主体范围。

法律标题：{title}
章节目录：
{outline}
正文节选（前 500 字）：
{text_head}

直接输出摘要，不要多余解释。"""

CHAPTER_PROMPT = """\
为以下法律章节生成 100 字以内的 scope summary。
要求说明该章的核心规制内容和涉及的典型法律问题。

所属法律：{law_title}
章节标题：{chapter_title}
该章节选（前 800 字）：
{chapter_text}

直接输出摘要。"""


# ── 缓存操作 ─────────────────────────────────────────────────

def _file_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_path(source_file: str) -> Path:
    config.SUMMARY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(source_file).stem
    return config.SUMMARY_CACHE_DIR / f"{stem}.summaries.json"


def _load_cache(source_file: str) -> dict:
    cp = _cache_path(source_file)
    if not cp.exists():
        return {}
    try:
        data = json.loads(cp.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(source_file: str, cache: dict) -> None:
    cp = _cache_path(source_file)
    cp.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


# ── LLM 摘要生成 ─────────────────────────────────────────────

def _call_llm(llm: LLMClient, prompt: str) -> str:
    return llm.chat([{"role": "user", "content": prompt}], temperature=0.1).strip()


def _build_law_outline(root) -> str:
    lines: list[str] = []
    for child in root.children:
        lines.append(f"  {child.title}")
        for grandchild in child.children:
            if grandchild.type == "section":
                lines.append(f"    {grandchild.title}")
    return "\n".join(lines) if lines else "（无章节结构）"


def _generate_law_summary(doc: LawDocument, llm: LLMClient) -> str:
    prompt = LAW_PROMPT.format(
        title=doc.title,
        outline=_build_law_outline(doc.root),
        text_head=(doc.root.text or "")[:500],
    )
    return _call_llm(llm, prompt)


def _generate_chapter_summary(doc: LawDocument, chapter, llm: LLMClient) -> str:
    prompt = CHAPTER_PROMPT.format(
        law_title=doc.title,
        chapter_title=chapter.title,
        chapter_text=(chapter.text or "")[:800],
    )
    return _call_llm(llm, prompt)


# ── 主入口 ────────────────────────────────────────────────────

def enhance_summaries(doc: LawDocument, llm: LLMClient | None = None) -> None:
    """用 LLM 增强 doc 中法级和章级的 summary，带缓存。"""
    if llm is None:
        return

    fhash = _file_hash(doc.source_file)
    cache = _load_cache(doc.source_file)
    dirty = False

    # ── 法级 ──
    law_key = "law"
    cached = cache.get(law_key)
    if cached and cached.get("hash") == fhash:
        doc.root.summary = cached["summary"]
        logger.info("summary cache hit: %s (law)", doc.title)
    else:
        try:
            summary = _generate_law_summary(doc, llm)
            if summary:
                doc.root.summary = summary
                cache[law_key] = {"hash": fhash, "summary": summary}
                dirty = True
                logger.info("summary generated: %s (law)", doc.title)
        except LLMError as exc:
            logger.warning("LLM summary failed for %s (law): %s", doc.title, exc)

    # ── 章级 ──
    for child in doc.root.children:
        if child.type != "chapter":
            continue
        ch_key = child.node_id
        cached = cache.get(ch_key)
        if cached and cached.get("hash") == fhash:
            child.summary = cached["summary"]
        else:
            try:
                summary = _generate_chapter_summary(doc, child, llm)
                if summary:
                    child.summary = summary
                    cache[ch_key] = {"hash": fhash, "summary": summary}
                    dirty = True
                    logger.info("summary generated: %s (%s)", doc.title, ch_key)
            except LLMError as exc:
                logger.warning("LLM summary failed for %s (%s): %s", doc.title, ch_key, exc)

    if dirty:
        _save_cache(doc.source_file, cache)
