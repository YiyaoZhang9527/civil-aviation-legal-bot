"""静态分析（无 LLM）：引用有效性 + 拒答率。

替代：test_citation_validity.py + test_summary.py
输入：test_*.csv（test_30questions.py / test_100questions.py 输出）
输出：{out_dir}/<stem>_static.json

用法：
    .venv/bin/python tests/analyze_static.py tests/test30_20260604_180203.csv
"""

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from legalbot.retrieval import IndexRepository

# ── 拒答关键词 ──
REFUSAL_KEYWORDS = [
    "无法确定", "无法回答", "证据不足", "未包含相关",
    "未涉及", "未找到", "没有找到", "未能找到",
    "无法提供", "无法确认", "无法判断", "现有证据不足以",
]


def is_refusal(answer: str) -> bool:
    return any(kw in answer for kw in REFUSAL_KEYWORDS)


# ── 引用提取 ──
CITATION_RE = re.compile(
    r"《([^》]+)》[^。；\n]*?第([一二三四五六七八九十百千零\d]+(?:\.\d+)*)\s*条"
)
# 模式 2: 无书名号引用（法名必须以规则/规定/办法/法律等关键词结尾）
CITATION_RE_NO_BRACKETS = re.compile(
    r"(?:^|[。\n])\s*([^。\n《》]{2,30}?(?:规则|规定|办法|条例|法律|法|标准|细则|规范))\s*第([一二三四五六七八九十百千零\d]+(?:\.\d+)*)\s*条"
)


def extract_citations(answer: str) -> list[dict]:
    """从答案中提取法条引用（两种模式：书名号/无书名号）。"""
    citations = []
    for m in CITATION_RE.finditer(answer):
        citations.append({"law_name": m.group(1).strip(), "article": m.group(2)})
    # 模式 2：无书名号（必须以规则类关键词结尾）
    for m in CITATION_RE_NO_BRACKETS.finditer(answer):
        citations.append({"law_name": m.group(1).strip(), "article": m.group(2)})
    return citations


def chinese_to_int(s: str) -> int:
    """简单的中文数字转 int（用于条号比较）。"""
    cn_map = {"零": 0, "〇": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9, "十": 10, "百": 100, "千": 1000}
    if s.isdigit():
        return int(s)
    if s in cn_map:
        return cn_map[s]
    # 简单累加（如"二十三"=20+3=23）
    total = 0
    current = 0
    for c in s:
        if c in cn_map:
            v = cn_map[c]
            if v >= 10:
                total += (current if current else 1) * v
                current = 0
            else:
                current = v
        else:
            break
    return total + current


def article_num_to_str(article: str) -> str:
    """条号标准化：'一二三' / '123' / '1.2' → 用于比较。"""
    if "." in article:
        parts = article.split(".")
        return ".".join(str(chinese_to_int(p) if not p.isdigit() else int(p)) for p in parts)
    try:
        return str(int(article)) if article.isdigit() else str(chinese_to_int(article))
    except (ValueError, KeyError):
        return article


def validate_citation(citation: dict, law_titles: set[str], law_articles: dict[str, set[int]] = None) -> dict:
    """检查引用是否真实存在（法名 + 条号双重验证）。

    Args:
        law_titles: 所有法名集合（模糊匹配）
        law_articles: {normalized_law_title: {article_int, ...}} 用于条号验证
    """
    law_name = citation["law_name"]
    # 第一步：法名模糊匹配
    matched_title = None
    for title in law_titles:
        if law_name in title or title in law_name:
            matched_title = title
            break

    result = {
        "citation": citation,
        "valid": matched_title is not None,
        "law_matched": matched_title,
    }

    # 第二步：条号验证（如果提供了 law_articles）
    if law_articles is not None and matched_title:
        # 找法名对应的 article 集合
        articles = None
        for title, arts in law_articles.items():
            if title == matched_title or law_name in title or title in law_name:
                articles = arts
                break
        if articles is not None:
            # 标准化引用条号（中文数字 → 阿拉伯数字）
            norm_article = article_num_to_str(citation["article"])
            result["article_exists"] = norm_article in articles
            if not result["article_exists"]:
                result["valid"] = False  # 法名对但条号错

    return result


# ── 主流程 ──
def analyze_csv(csv_path: Path) -> dict:
    csv_path = Path(csv_path)
    print(f"分析: {csv_path}")

    # 加载法规索引（法名集合 + 条号集合）
    print("  加载法规索引...")
    docs = IndexRepository.documents()
    law_titles = set()
    # {normalized_law_title: {article_int, ...}} 用于条号验证
    law_articles: dict[str, set[int]] = {}
    for doc in docs:
        if doc.title:
            law_titles.add(doc.title)
            normalized = doc.title.replace("《", "").replace("》", "")
            law_titles.add(normalized)
            # 收集所有 article 节点的条号（支持整数和小数，如 article:91 / article:91.103）
            article_nums = set()
            for node in doc.flatten():
                if node.type == "article":
                    m = re.search(r"article:([\d.]+)", node.node_id or "")
                    if m:
                        article_nums.add(m.group(1))  # 保留字符串（"91" 或 "91.103"）
            if article_nums:
                law_articles[doc.title] = article_nums
                law_articles[normalized] = article_nums
    print(f"  法规数: {len(docs)}, 法名变体: {len(law_titles)}, 含条号: {len(law_articles) // 2}")

    # 读取 CSV
    print("  读取答案...")
    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    refusals = []
    all_citations = []
    citation_results = []

    for r in rows:
        qid = r.get("question_id", "")
        answer = r.get("answer_full", "") or r.get("conclusion_preview", "")
        if is_refusal(answer):
            refusals.append({
                "qid": qid,
                "category": r.get("category", ""),
                "question": r.get("question", ""),
                "answer_preview": answer[:300].replace("\n", " "),
                "answer_len": len(answer),
            })
        # 按 (law_name, article) 去重——同一引用重复出现只算 1 个
        seen_in_answer = set()
        for c in extract_citations(answer):
            key = (c["law_name"], c["article"])
            if key in seen_in_answer:
                continue
            seen_in_answer.add(key)
            v = validate_citation(c, law_titles, law_articles)
            v["qid"] = qid
            all_citations.append(v)

    total_citations = len(all_citations)
    valid_citations = sum(1 for c in all_citations if c["valid"])
    law_not_found = sum(1 for c in all_citations if c["valid"] and not c.get("article_exists", True))
    article_not_found = sum(1 for c in all_citations if c.get("article_exists") is False)
    invalid = [c for c in all_citations if not c["valid"]]

    # 拒答题统计
    refusal_count = len(refusals)
    refused_questions = refusals

    # 法规名频次（被引用最多/最少的）
    cited_laws = Counter(c["citation"]["law_name"] for c in all_citations)

    result = {
        "source": str(csv_path),
        "n_questions": total,
        "refusals": {
            "count": refusal_count,
            "rate": refusal_count / total if total else 0,
            "details": refused_questions,  # 完整列表（每个含 qid/category/question/preview/len）
        },
        "citation_validity": {
            "total_citations": total_citations,
            "valid_citations": valid_citations,
            "law_not_found": law_not_found,
            "article_not_found": article_not_found,
            "validity_rate": valid_citations / total_citations if total_citations else 0,
            "invalid_examples": [
                {
                    "qid": c["qid"],
                    "law": c["citation"]["law_name"],
                    "article": c["citation"]["article"],
                    "reason": "article_not_found" if c.get("article_exists") is False else "law_not_found",
                }
                for c in invalid[:10]
            ],
        },
        "cited_law_frequency": dict(cited_laws.most_common(20)),
    }

    # 写报告
    stem = csv_path.stem
    out_path = csv_path.parent / f"{stem}_static.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告: {out_path}")
    print(f"  拒答率: {result['refusals']['rate']:.1%} ({refusal_count}/{total})")
    print(f"  引用有效率: {result['citation_validity']['validity_rate']:.1%} ({valid_citations}/{total_citations})")
    if law_not_found:
        print(f"    法名不存在: {law_not_found}")
    if article_not_found:
        print(f"    条号不存在: {article_not_found}")
    return result


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("csv_path", nargs="?", default=None, help="test_*.csv 路径（默认找最新）")
    args = p.parse_args()

    if args.csv_path:
        csv_path = Path(args.csv_path)
    else:
        # 默认找最新的 test*.csv
        candidates = sorted(
            list((PROJECT_ROOT / "tests").glob("test30_*.csv"))
            + list((PROJECT_ROOT / "tests").glob("test100_*.csv")),
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            print("ERROR: 找不到 test*.csv")
            return
        csv_path = candidates[-1]
        print(f"使用最新: {csv_path}")

    if not csv_path.exists():
        print(f"ERROR: 文件不存在: {csv_path}")
        return

    analyze_csv(csv_path)


if __name__ == "__main__":
    main()
