"""法律文本解析器。"""

from __future__ import annotations

import re
from pathlib import Path

from .types import LawDocument, LawNode
from .utils import article_label, article_number_to_label, normalize_text, split_sentences, top_keywords, strip_heading, chinese_to_int


# ── 正则：章节 / 条文 / 条款 ──────────────────────────────────────
CHAPTER_RE = re.compile(
    r"^(?:"
    r"第([一二三四五六七八九十百零\d]+)章"      # 第X章
    r"|([A-Za-z])\s*章"                        # A章 / A 章
    r"|([A-Za-z])\s*(?:部分|分部)"              # A部分 / A 分部
    r")\s*(.*)$"
)
SECTION_RE = re.compile(r"^第([一二三四五六七八九十百零\d]+)节\s*(.*)$")
ARTICLE_RE = re.compile(
    r"^(?:"
    r"第([一二三四五六七八九十百零\d]+(?:\.\d+)*)\s*条"   # 第X条 / 第67.33条（允许条前空格）
    r"|(\d{2,3}\.\d+)\s+"                                 # 142.1（dot编号，后必须跟非空内容）
    r")\s*(.*)$"
)
ITEM_RE = re.compile(r"^(?:（[一二三四五六七八九十百零\d]+）|[一二三四五六七八九十百零\d]+[、\.])\s*(.*)$")

# 页码分隔符：纯数字行、dash/em-dash 包裹
PAGE_SEP_RE = re.compile(r"^(?:\d+|[-—]+\s*\d+\s*[-—]+)$")

# 标题提取时应该跳过的非法规名第一行
_SKIP_TITLE_RE = re.compile(
    r"^(?:"
    r"\d+$"                                                  # 纯数字页码
    r"|[-—]+\s*\d+\s*[-—]+$"                               # dash 页码
    r"|(?:中华人民共和国)?(?:交通运输部|中国民用航空总局|民航总局)令"  # 令抬头
    r"|\d{4}\s*年?第\d+\s*号"                               # 年号令编号
    r"|交通运输部(?:关于|规章)"                               # 修改决定
    r"|民航总局令第"                                          # 民航总局令
    r"|[（(]\s*(?:CCAR|ccar)[-\s]?\d+\s*[）)]"              # (CCAR-73)
    r"|(?:CCAR|ccar)[-\s]?\d+"                              # CCAR-73
    r"|附件\s*[A-Z\d]*$"                                    # 附件标记
    r"|[【\[][^】\]]+[】\]]"                                 # 【颁布日期】
    r"|[A-Za-z]\s*(章|部分?|分部)"                           # A章 / A部分 / A分部
    r"|第[一二三四五六七八九十百零\d]+[章节条款]"              # 第X章/节/条/款
    r"|交通运输部(?:令|关于)"                                 # 交通运输部令
    r")"
)


def _node_id(kind: str, num: int | str, law_id: str) -> str:
    return f"{kind}:{num}" if kind != "law" else f"law:{law_id}"


def _clean_line(line: str) -> str:
    line = line.rstrip().replace("\u3000", " ")
    # 全角 ASCII → 半角（U+FF01–U+FF5E → U+0021–U+007E）
    line = re.sub(r"[\uff01-\uff5e]", lambda m: chr(ord(m.group()) - 0xfee0), line)
    return line.strip()


def _summary_from_text(text: str, fallback: str = "") -> str:
    sentences = split_sentences(text)
    if sentences:
        return sentences[0][:120]
    return fallback


def _keywords_from_text(text: str, title: str = "") -> list[str]:
    combined = f"{title} {text}"
    words = top_keywords(combined, max_n=6)
    if title:
        for token in re.findall(r"[\u4e00-\u9fff]{2,}", title):
            if token not in words:
                words.insert(0, token)
    return words[:8]



def parse_law_text(source_path: Path) -> LawDocument:
    raw = source_path.read_text(encoding="utf-8")
    lines = [_clean_line(line) for line in raw.splitlines()]
    non_empty = [line for line in lines if line]
    title = source_path.stem
    for candidate in non_empty[:10]:
        if candidate and not re.fullmatch(r"[=—\-_*•·\s]+", candidate) and not _SKIP_TITLE_RE.match(candidate):
            title = candidate
            break
    law_id = normalize_text(source_path.stem)

    root = LawNode(
        node_id=_node_id("law", law_id, law_id),
        type="law",
        title=title,
        summary="",
        keywords=top_keywords(raw, 8),
        source_file=str(source_path),
        source_anchor=title,
        text="",
        line_start=0,
        line_end=max(len(lines) - 1, 0),
    )
    anchor_map: dict[str, dict] = {
        root.node_id: {
            "node_id": root.node_id,
            "type": root.type,
            "title": root.title,
            "line_start": root.line_start,
            "line_end": root.line_end,
            "source_anchor": root.source_anchor,
        }
    }

    current_chapter: LawNode | None = None
    current_section: LawNode | None = None
    current_article: LawNode | None = None
    current_article_lines: list[str] = []
    current_article_start: int | None = None
    pending_intro: list[str] = []

    def close_article(end_idx: int) -> None:
        nonlocal current_article, current_article_lines, current_article_start
        if current_article is None:
            return
        text = "\n".join([line for line in current_article_lines if line]).strip()
        current_article.text = text
        current_article.line_end = end_idx
        current_article.summary = _summary_from_text(text, current_article.title)
        current_article.keywords = _keywords_from_text(text, current_article.title)
        anchor_map[current_article.node_id] = {
            "node_id": current_article.node_id,
            "type": current_article.type,
            "title": current_article.title,
            "line_start": current_article.line_start,
            "line_end": current_article.line_end,
            "source_anchor": current_article.source_anchor,
        }
        if current_section is not None:
            current_section.children.append(current_article)
        elif current_chapter is not None:
            current_chapter.children.append(current_article)
        else:
            root.children.append(current_article)
        current_article = None
        current_article_lines = []
        current_article_start = None

    def close_section(end_idx: int) -> None:
        nonlocal current_section
        if current_section is not None:
            current_section.line_end = end_idx
            current_section.summary = _summary_from_text(current_section.text, current_section.title)
            current_section.keywords = _keywords_from_text(current_section.text, current_section.title)
            anchor_map[current_section.node_id] = {
                "node_id": current_section.node_id,
                "type": current_section.type,
                "title": current_section.title,
                "line_start": current_section.line_start,
                "line_end": current_section.line_end,
                "source_anchor": current_section.source_anchor,
            }
            if current_chapter is not None and current_section not in current_chapter.children:
                current_chapter.children.append(current_section)
            current_section = None

    def close_chapter(end_idx: int) -> None:
        nonlocal current_chapter
        if current_chapter is not None:
            current_chapter.line_end = end_idx
            current_chapter.summary = _summary_from_text(current_chapter.text, current_chapter.title)
            current_chapter.keywords = _keywords_from_text(current_chapter.text, current_chapter.title)
            anchor_map[current_chapter.node_id] = {
                "node_id": current_chapter.node_id,
                "type": current_chapter.type,
                "title": current_chapter.title,
                "line_start": current_chapter.line_start,
                "line_end": current_chapter.line_end,
                "source_anchor": current_chapter.source_anchor,
            }
            if current_chapter not in root.children:
                root.children.append(current_chapter)
            current_chapter = None

    for idx, raw_line in enumerate(lines[1:], start=1):
        line = _clean_line(raw_line)
        if not line:
            if current_article is not None:
                current_article_lines.append("")
            continue
        if PAGE_SEP_RE.match(line):
            continue

        chapter_match = CHAPTER_RE.match(line)
        section_match = SECTION_RE.match(line)
        article_match = ARTICLE_RE.match(line)

        if chapter_match:
            close_article(idx - 1)
            close_section(idx - 1)
            close_chapter(idx - 1)
            # group(1)=中文章号, group(2)=字母章, group(3)=字母部分/分部, group(4)=名称
            cn_num = chapter_match.group(1)
            letter = chapter_match.group(2) or chapter_match.group(3)
            name = chapter_match.group(4).strip()
            if cn_num:
                chapter_num = chinese_to_int(cn_num) or 1
                raw_label = f"第{article_number_to_label(chapter_num)}章"
            else:
                chapter_num = letter.upper()
                raw_label = f"{letter}章"
            if not name:
                name = raw_label
            current_chapter = LawNode(
                node_id=_node_id("chapter", chapter_num, law_id),
                type="chapter",
                title=f"{raw_label} {name}".strip(),
                source_file=str(source_path),
                source_anchor=raw_label,
                line_start=idx,
                text=line,
            )
            continue

        if section_match:
            close_article(idx - 1)
            close_section(idx - 1)
            num = section_match.group(1)
            section_num = chinese_to_int(num) or 1
            name = section_match.group(2).strip() or f"第{article_number_to_label(int(num) if num.isdigit() else 1)}节"
            current_section = LawNode(
                node_id=_node_id("section", section_num, law_id),
                type="section",
                title=f"第{article_number_to_label(section_num)}节 {name}".strip(),
                source_file=str(source_path),
                source_anchor=f"第{article_number_to_label(section_num)}节",
                line_start=idx,
                text=line,
            )
            continue

        if article_match:
            close_article(idx - 1)
            num = article_match.group(1) or article_match.group(2)
            title_tail = article_match.group(3).strip()
            # 小数点编号（如 67.33）保留原始字符串作为标识
            if "." in num:
                article_num = num
                raw_label = f"第{num}条"
            else:
                article_num = chinese_to_int(num) or 1
                raw_label = f"第{article_number_to_label(article_num)}条"
            current_article = LawNode(
                node_id=_node_id("article", article_num, law_id),
                type="article",
                title=f"{raw_label} {title_tail}".strip() if title_tail else raw_label,
                summary=title_tail[:120],
                source_file=str(source_path),
                source_anchor=raw_label,
                line_start=idx,
            )
            current_article_lines = [line]
            current_article_start = idx
            continue

        if current_article is not None:
            current_article_lines.append(line)
        elif current_section is not None:
            current_section.text += ("\n" if current_section.text else "") + line
        elif current_chapter is not None:
            current_chapter.text += ("\n" if current_chapter.text else "") + line
        else:
            pending_intro.append(line)

    close_article(len(lines) - 1)
    close_section(len(lines) - 1)
    close_chapter(len(lines) - 1)

    if pending_intro:
        root.text = "\n".join(pending_intro)
    root.summary = _summary_from_text(raw, title)
    root.keywords = _keywords_from_text(raw, title)

    # 为章/节补充 summary/keywords 与 tree 结构
    for chapter in root.children:
        if chapter.type == "chapter" and not chapter.summary:
            chapter.summary = _summary_from_text(chapter.text, chapter.title)
        if chapter.type == "chapter" and not chapter.keywords:
            chapter.keywords = _keywords_from_text(chapter.text, chapter.title)
        for child in chapter.children:
            if child.type == "section" and not child.summary:
                child.summary = _summary_from_text(child.text, child.title)
                child.keywords = _keywords_from_text(child.text, child.title)

    return LawDocument(
        law_id=law_id,
        title=title,
        source_file=str(source_path),
        root=root,
        anchor_map=anchor_map,
    )


def render_index_markdown(doc: LawDocument) -> str:
    lines: list[str] = []

    def render(node: LawNode, depth: int = 0) -> None:
        heading = "#" * (depth + 1)
        lines.append(f"{heading} {node.title}")
        lines.append(f"- node_id: {node.node_id}")
        lines.append(f"- type: {node.type}")
        if node.source_file:
            lines.append(f"- source_file: {node.source_file}")
        if node.source_anchor:
            lines.append(f"- source_anchor: {node.source_anchor}")
        if node.summary:
            lines.append(f"- summary: {node.summary}")
        if node.keywords:
            lines.append(f"- keywords: {', '.join(node.keywords)}")
        if node.children:
            lines.append(f"- children: {', '.join(child.node_id for child in node.children)}")
        lines.append("")
        for child in node.children:
            render(child, depth + 1)

    render(doc.root)
    return "\n".join(lines).rstrip() + "\n"
