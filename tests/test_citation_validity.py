"""第 2 层评测：引用真实性检查（纯代码，零 LLM）

从已有测试结果 JSON 中读取答案，解析法条引用，
对照 129 部法规的 index 验证引用是否存在。

用法：
    .venv/bin/python tests/test_citation_validity.py [结果JSON路径]
    默认读取 tests/test30_20260601_235055.json
"""

import json
import re
import sys
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
    """判断答案是否为拒答。"""
    return any(kw in answer for kw in REFUSAL_KEYWORDS)


def extract_citations(answer: str) -> list[dict]:
    """从答案中提取法条引用。

    匹配模式：
    1. 《法名》第X条
    2. 《法名》第X.Y条（小数条号，CCAR 风格）
    3. 法名 第X条（无书名号）
    """
    citations = []
    seen = set()

    # 模式 1 & 2: 《法名》第X条 / 《法名》第X.Y条
    for m in re.finditer(r"《([^》]+)》[^。；\n]*?第([一二三四五六七八九十百千零\d]+(?:\.\d+)*)\s*条", answer):
        law_name = m.group(1).strip()
        article = m.group(2)
        key = (law_name, article)
        if key not in seen:
            seen.add(key)
            citations.append({"law_name": law_name, "article": article, "raw": m.group(0)})

    # 模式 3: 法名 第X条（补充捕获无书名号的引用）
    for m in re.finditer(
        r"(?:^|[。\n])\s*([^。\n《》]{2,30}?(?:规则|规定|办法|条例|法律|法|标准))\s*第([一二三四五六七八九十百千零\d]+(?:\.\d+)*)\s*条",
        answer,
    ):
        law_name = m.group(1).strip()
        article = m.group(2)
        key = (law_name, article)
        if key not in seen:
            seen.add(key)
            citations.append({"law_name": law_name, "article": article, "raw": m.group(0)})

    return citations


def article_num_to_str(article: str) -> str:
    """将中文数字条号转为阿拉伯数字（用于匹配 node_id）。"""
    from legalbot.utils import chinese_to_int
    if "." in article:
        return article  # 小数条号直接返回
    try:
        num = chinese_to_int(article)
        return str(num) if num is not None else article
    except Exception:
        return article


def validate_citation(citation: dict, documents) -> dict:
    """验证单条引用的真实性。

    返回：
        valid: 引用对应的法规+条文是否存在
        law_found: 法规是否找到
        article_found: 具体条文是否找到
        matched_law: 匹配到的法规标题
        matched_node: 匹配到的 node_id
    """
    law_name = citation["law_name"]
    article_raw = citation["article"]
    article_num = article_num_to_str(article_raw)

    # 查找法规
    law_found = False
    matched_law = ""
    matched_doc = None

    for doc in documents:
        from legalbot.utils import normalize_text
        title_n = normalize_text(doc.title)
        query_n = normalize_text(law_name)
        # 精确匹配或子串匹配
        if title_n == query_n or (len(query_n) >= 4 and query_n in title_n) or (len(title_n) >= 4 and title_n in query_n):
            law_found = True
            matched_law = doc.title
            matched_doc = doc
            break

    if not law_found:
        return {
            "valid": False,
            "law_found": False,
            "article_found": False,
            "matched_law": "",
            "matched_node": "",
        }

    # 查找条文
    article_found = False
    matched_node = ""
    if matched_doc:
        for node in matched_doc.flatten():
            # node_id 格式: "article:123" 或 "article:121.557"
            nid = node.node_id
            if nid.startswith("article:"):
                nid_num = nid.split(":")[1]
                if nid_num == article_num:
                    article_found = True
                    matched_node = nid
                    break

    return {
        "valid": law_found and article_found,
        "law_found": law_found,
        "article_found": article_found,
        "matched_law": matched_law,
        "matched_node": matched_node,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", nargs="?",
                        default=str(PROJECT_ROOT / "tests" / "test30_20260601_235055.json"))
    args = parser.parse_args()

    json_path = Path(args.json_path)
    if not json_path.exists():
        print(f"文件不存在: {json_path}")
        return

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])

    # 加载法规索引
    print("加载法规索引...")
    documents = IndexRepository.documents()
    print(f"已加载 {len(documents)} 部法规\n")

    # 统计
    total_questions = len(results)
    refusal_count = 0
    total_citations = 0
    valid_citations = 0
    law_not_found = 0
    article_not_found = 0
    per_question = []

    for r in results:
        qid = r["question_id"]
        question = r["question"]
        answer = r.get("answer_full", "")

        refusal = is_refusal(answer)
        if refusal:
            refusal_count += 1

        citations = extract_citations(answer)
        q_valid = 0
        q_law_nf = 0
        q_art_nf = 0
        cit_details = []

        for cit in citations:
            result = validate_citation(cit, documents)
            total_citations += 1
            if result["valid"]:
                valid_citations += 1
                q_valid += 1
            elif not result["law_found"]:
                law_not_found += 1
                q_law_nf += 1
            else:
                article_not_found += 1
                q_art_nf += 1
            cit_details.append({**cit, **result})

        per_question.append({
            "qid": qid,
            "question": question[:40],
            "refusal": refusal,
            "total_cit": len(citations),
            "valid": q_valid,
            "law_not_found": q_law_nf,
            "article_not_found": q_art_nf,
            "details": cit_details,
        })

    # 输出
    print("=" * 70)
    print("引用真实性检查报告")
    print("=" * 70)
    print(f"总题数: {total_questions}")
    print(f"拒答: {refusal_count} ({refusal_count/total_questions*100:.0f}%)")
    print(f"有效回答: {total_questions - refusal_count} ({(total_questions-refusal_count)/total_questions*100:.0f}%)")
    print()

    answered = [q for q in per_question if not q["refusal"]]
    answered_cit = sum(q["total_cit"] for q in answered)
    answered_valid = sum(q["valid"] for q in answered)
    answered_law_nf = sum(q["law_not_found"] for q in answered)
    answered_art_nf = sum(q["article_not_found"] for q in answered)

    print(f"有效回答中的引用:")
    print(f"  总引用数: {answered_cit}")
    print(f"  有效引用: {answered_valid} ({answered_valid/answered_cit*100:.1f}%)" if answered_cit else "  无引用")
    print(f"  法规不存在: {answered_law_nf}")
    print(f"  条文不存在: {answered_art_nf}")
    print(f"  引用有效率: {answered_valid/answered_cit*100:.1f}%" if answered_cit else "  N/A")
    print()

    # 每题详情
    print("-" * 70)
    print(f"{'题号':>4} {'拒答':>4} {'引用':>4} {'有效':>4} {'法不存在':>6} {'条不存在':>6}  问题")
    print("-" * 70)
    for q in per_question:
        refusal_mark = "是" if q["refusal"] else ""
        print(f"{q['qid']:>4} {refusal_mark:>4} {q['total_cit']:>4} {q['valid']:>4} {q['law_not_found']:>6} {q['article_not_found']:>6}  {q['question']}")

    # 伪造引用详情
    fabricated = []
    for q in per_question:
        for d in q["details"]:
            if not d["valid"]:
                fabricated.append((q["qid"], d["law_name"], d["article"], d["raw"],
                                  "法规不存在" if not d["law_found"] else "条文不存在"))
    if fabricated:
        print()
        print("=" * 70)
        print("可疑引用详情（法规或条文不存在）")
        print("=" * 70)
        for qid, law, art, raw, reason in fabricated:
            print(f"  {qid} [{reason}] {raw[:80]}")

    # 保存 JSON
    output_path = json_path.parent / (json_path.stem + "_citation_validity.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "source": json_path.name,
            "summary": {
                "total_questions": total_questions,
                "refusals": refusal_count,
                "answered": total_questions - refusal_count,
                "total_citations": answered_cit,
                "valid_citations": answered_valid,
                "law_not_found": answered_law_nf,
                "article_not_found": answered_art_nf,
                "citation_validity_rate": answered_valid / answered_cit if answered_cit else None,
            },
            "per_question": per_question,
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n详细结果已保存: {output_path}")


if __name__ == "__main__":
    main()
