"""第 3 层评测：拒答统计 + 三层汇总报告

汇总第 1 层（Faithfulness）和第 2 层（引用真实性）的评测结果，
输出对标商用系统的综合报告。

用法：
    .venv/bin/python tests/test_summary.py [结果JSON路径]
    默认读取 tests/test30_20260601_235055
"""

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

REFUSAL_KEYWORDS = [
    "无法确定", "无法回答", "证据不足", "未包含相关",
    "未涉及", "未找到", "没有找到", "未能找到",
    "无法提供", "无法确认", "无法判断", "现有证据不足以",
]


def is_refusal(answer: str) -> bool:
    return any(kw in answer for kw in REFUSAL_KEYWORDS)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("base_path", nargs="?",
                        default=str(PROJECT_ROOT / "tests" / "test30_20260601_235055"))
    args = parser.parse_args()

    base = Path(args.base_path)
    suffix = base.suffix  # .json
    stem = base.stem if suffix else str(base)

    # 加载原始测试结果
    raw_path = Path(str(base) + ".json") if not suffix else base
    if not raw_path.exists():
        raw_path = Path(str(base) + ".json")
    if not raw_path.exists():
        print(f"找不到测试结果: {base}")
        return

    with open(raw_path, encoding="utf-8") as f:
        raw = json.load(f)
    results = raw.get("results", [])

    # 第 2 层：引用真实性
    cit_path = raw_path.parent / (raw_path.stem + "_citation_validity.json")
    cit_data = None
    if cit_path.exists():
        with open(cit_path, encoding="utf-8") as f:
            cit_data = json.load(f)

    # 第 1 层：Faithfulness
    faith_path = raw_path.parent / (raw_path.stem + "_faithfulness.json")
    faith_data = None
    if faith_path.exists():
        with open(faith_path, encoding="utf-8") as f:
            faith_data = json.load(f)

    # ── 汇总 ──
    total = len(results)

    # 拒答统计
    refusal_ids = []
    for r in results:
        if is_refusal(r.get("answer_full", "")):
            refusal_ids.append(r["question_id"])
    refusal_count = len(refusal_ids)
    answered = total - refusal_count

    print("=" * 70)
    print("三层评测汇总报告")
    print("=" * 70)
    print(f"测试文件: {raw_path.name}")
    print()

    # ── 旧指标（对比用） ──
    old_sup = sum(r["supported"] for r in results)
    old_par = sum(r["partial"] for r in results)
    old_uns = sum(r["unsupported"] for r in results)
    old_total = old_sup + old_par + old_uns
    print("── 旧指标（系统自评 CE 校验） ──")
    print(f"  CE supported: {old_sup}/{old_total} ({old_sup/old_total*100:.1f}%)")
    print(f"  CE unsupported: {old_uns}/{old_total} ({old_uns/old_total*100:.1f}%)")
    print()

    # ── 拒答 ──
    print("── 拒答统计 ──")
    print(f"  拒答: {refusal_count}/{total} ({refusal_count/total*100:.1f}%)")
    if refusal_ids:
        print(f"  拒答题: {', '.join(refusal_ids)}")
    print(f"  有效回答: {answered}/{total} ({answered/total*100:.1f}%)")
    print()

    # ── 第 2 层：引用真实性 ──
    if cit_data:
        s = cit_data["summary"]
        print("── 第 2 层：引用真实性（纯代码检查） ──")
        print(f"  总引用: {s['total_citations']}")
        print(f"  有效引用: {s['valid_citations']} ({s['citation_validity_rate']*100:.1f}%)")
        print(f"  法规不存在: {s['law_not_found']}")
        print(f"  条文不存在: {s['article_not_found']}")
        print()
    else:
        print("── 第 2 层：引用真实性 — 未运行 ──")
        print(f"  运行命令: .venv/bin/python tests/test_citation_validity.py")
        print()

    # ── 第 1 层：Faithfulness + Completeness ──
    if faith_data:
        s = faith_data["summary"]
        print("── 第 1 层：Faithfulness + Completeness（独立 LLM 评测） ──")
        print(f"  评测模型: {faith_data.get('evaluator_model', 'N/A')}")
        print(f"  拒答: {s['refusals']}/{s['total_questions']}")
        print(f"  有效回答: {s['answered']}")
        print(f"  总声明数: {s['total_claims']}")
        if s['total_claims']:
            tc = s['total_claims']
            print(f"  faithful:      {s['faithful']:3d} ({s['faithful']/tc*100:5.1f}%)")
            print(f"  partial:       {s['partial']:3d} ({s['partial']/tc*100:5.1f}%)")
            unver = s.get('unverifiable', 0)
            print(f"  unverifiable:  {unver:3d} ({unver/tc*100:5.1f}%)")
            print(f"  hallucinated:  {s['hallucinated']:3d} ({s['hallucinated']/tc*100:5.1f}%)")
            print(f"  missing:       {s.get('total_missing', 0):3d} 条遗漏信息")
            print()
            print(f"  ★ 真实幻觉率: {s['hallucination_rate']*100:.1f}%")
            print(f"  ★ 检索缺口率: {s.get('retrieval_gap_rate', 0)*100:.1f}%")
            print(f"  ★ 忠实率:     {s['faithfulness_rate']*100:.1f}%")
            if s.get('completeness_recall') is not None:
                print(f"  ★ 完整率:     {s['completeness_recall']*100:.1f}%")
            if s.get('f1') is not None:
                print(f"  ★ F1:         {s['f1']*100:.1f}%")
        print()
    else:
        print("── 第 1 层：Faithfulness — 未运行 ──")
        print(f"  运行命令: .venv/bin/python tests/test_faithfulness.py")
        print()

    # ── 对标商用系统 ──
    print("── 对标商用系统 ──")
    print(f"  {'系统':<25} {'幻觉率':>8} {'拒答率':>8} {'完整率':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'通用 GPT-4':<25} {'58-82%':>8} {'~0%':>8} {'N/A':>8}")
    if faith_data and faith_data["summary"]["total_claims"] > 0:
        hr = f"{faith_data['summary']['hallucination_rate']*100:.1f}%"
        cr = f"{faith_data['summary'].get('completeness_recall', 0)*100:.1f}%" if faith_data['summary'].get('completeness_recall') else "N/A"
    else:
        hr = "N/A"
        cr = "N/A"
    rr = f"{refusal_count/total*100:.1f}%"
    print(f"  {'我们':<25} {hr:>8} {rr:>8} {cr:>8}")
    print(f"  {'Westlaw AI-AR':<25} {'33%':>8} {'25%':>8} {'N/A':>8}")
    print(f"  {'Lexis+ AI':<25} {'17%':>8} {'18%':>8} {'N/A':>8}")
    print()

    # 保存汇总
    output_path = raw_path.parent / (raw_path.stem + "_summary.json")
    summary = {
        "source": raw_path.name,
        "old_metrics": {
            "ce_supported": old_sup,
            "ce_unsupported": old_uns,
            "ce_total": old_total,
            "ce_supported_rate": old_sup / old_total if old_total else None,
        },
        "refusals": {"count": refusal_count, "ids": refusal_ids},
        "citation_validity": cit_data["summary"] if cit_data else None,
        "faithfulness": faith_data["summary"] if faith_data else None,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"汇总已保存: {output_path}")


if __name__ == "__main__":
    main()
