"""一键全面测试：30 题 + 三层评测 + 报告归档。

用法：
    .venv/bin/python tests/test_comprehensive.py

流程：
    1. 跑 30 题（约 50 分钟）
    2. 跑 Layer 2 引用真实性（<1 分钟）
    3. 跑 Layer 1 Faithfulness（5-10 分钟）
    4. 跑 Summary
    5. 把所有结果归档到 tests/对话过程/

输出（按时间戳）：
    tests/test30_YYYYMMDD_HHMMSS.csv       (30 题完整答案)
    tests/test30_YYYYMMDD_HHMMSS.json
    tests/test30_YYYYMMDD_HHMMSS_citation_validity.json
    tests/test30_YYYYMMDD_HHMMSS_faithfulness.json
    tests/test30_YYYYMMDD_HHMMSS_summary.json
    tests/对话过程/YYYYMMDD_HHMMSS_全面测试报告.md
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = PROJECT_ROOT / "tests"
DIALOG_DIR = TESTS_DIR / "对话过程"
DIALOG_DIR.mkdir(exist_ok=True)


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _run(cmd: list[str], label: str, timeout: int | None = None) -> int:
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    print(f"$ {' '.join(cmd)}")
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=PROJECT_ROOT, timeout=timeout)
        elapsed = time.time() - t0
        print(f"[{label}] exit={r.returncode}, 耗时 {elapsed:.0f}s")
        return r.returncode
    except subprocess.TimeoutExpired:
        print(f"[{label}] TIMEOUT after {timeout}s")
        return 1


def main() -> int:
    ts = _ts()
    base = f"test30_{ts}"
    print(f"全面测试启动: {ts}")
    print(f"输出文件: tests/{base}.{{csv,json,json.*}}")

    # 1. 跑 30 题
    r1 = _run(
        [".venv/bin/python", str(TESTS_DIR / "test_30questions.py")],
        "Step 1/4 — 30 题端到端",
        timeout=4500,
    )
    if r1 != 0:
        print("Step 1 失败，中止")
        return r1

    # 找新生成的文件
    json_path = TESTS_DIR / f"{base}.json"
    if not json_path.exists():
        print(f"找不到 {json_path}")
        return 1

    # 2. 跑 Layer 2
    r2 = _run(
        [".venv/bin/python", str(TESTS_DIR / "test_citation_validity.py"), str(json_path)],
        "Step 2/4 — Layer 2 引用真实性",
        timeout=300,
    )
    if r2 != 0:
        print("Step 2 失败，继续")

    # 3. 跑 Layer 1
    r3 = _run(
        [".venv/bin/python", str(TESTS_DIR / "test_faithfulness.py"), str(json_path)],
        "Step 3/4 — Layer 1 Faithfulness",
        timeout=1500,
    )
    if r3 != 0:
        print("Step 3 失败，继续")

    # 4. 跑 Summary
    r4 = _run(
        [".venv/bin/python", str(TESTS_DIR / "test_summary.py"), base],
        "Step 4/4 — Summary",
        timeout=300,
    )

    # 5. 归档综合报告
    _write_report(ts, base)

    print(f"\n{'='*70}\n全面测试完成: {ts}\n{'='*70}")
    print(f"归档报告: {DIALOG_DIR}/{ts}_全面测试报告.md")
    return 0


def _write_report(ts: str, base: str) -> None:
    """汇总所有指标写到对话过程目录。"""
    summary_path = TESTS_DIR / f"{base}_summary.json"
    if not summary_path.exists():
        print("Summary 不存在，跳过归档")
        return

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    csv_path = TESTS_DIR / f"{base}.csv"
    json_path = TESTS_DIR / f"{base}.json"

    # 统计答案长度
    results = json.loads(json_path.read_text(encoding="utf-8")).get("results", [])
    avg_len = sum(r.get("answer_len", 0) for r in results) // max(1, len(results))
    sup_total = sum(int(r.get("supported", 0)) for r in results)
    claim_total = sum(
        int(r.get("supported", 0)) + int(r.get("partial", 0)) + int(r.get("unsupported", 0))
        for r in results
    )

    lines = [
        f"# 全面测试综合报告 ({ts})",
        "",
        "## 文件清单",
        f"- `tests/{csv_path.name}` (30 题完整答案)",
        f"- `tests/{json_path.name}` (结构化结果)",
        f"- `tests/{base}_citation_validity.json` (Layer 2)",
        f"- `tests/{base}_faithfulness.json` (Layer 1)",
        f"- `tests/{summary_path.name}` (汇总)",
        "",
        "## 关键指标",
        "",
        f"- 完成: **{len(results)}/{len(results)}** 题",
        f"- 平均答案长度: **{avg_len}** 字",
        f"- 整体 CE supported: **{sup_total}/{claim_total}** ({sup_total/max(1,claim_total)*100:.1f}%)",
        "",
    ]

    # 三层指标
    if "faithfulness" in summary:
        f = summary["faithfulness"]
        lines.extend([
            "### Layer 1 — Faithfulness",
            f"- 总声明: {f.get('total_claims', 0)}",
            f"- faithful: {f.get('faithful', 0)} ({f.get('faithfulness_rate', 0)*100:.1f}%)",
            f"- partial: {f.get('partial', 0)}",
            f"- unverifiable: {f.get('unverifiable', 0)} ({f.get('retrieval_gap_rate', 0)*100:.1f}%)",
            f"- hallucinated: {f.get('hallucinated', 0)} ({f.get('hallucination_rate', 0)*100:.1f}%)",
            f"- missing: {f.get('total_missing', 0)}",
            f"- 完整率 Recall: **{f.get('completeness_recall', 0)*100:.1f}%**",
            f"- 精确率 Precision: **{f.get('precision', 0)*100:.1f}%**",
            f"- **F1: {f.get('f1', 0)*100:.1f}%**",
            "",
        ])
    if "citation_validity" in summary:
        c = summary["citation_validity"]
        lines.extend([
            "### Layer 2 — 引用真实性",
            f"- 总引用: {c.get('total_citations', 0)}",
            f"- 有效: {c.get('valid_citations', 0)} ({c.get('citation_validity_rate', 0)*100:.1f}%)",
            f"- 法规不存在: {c.get('law_not_found', 0)}",
            f"- 条文不存在: {c.get('article_not_found', 0)}",
            "",
        ])
    if "refusals" in summary:
        r = summary["refusals"]
        total = len(results)
        ref = r.get("count", 0)
        lines.extend([
            "### Layer 3 — 拒答",
            f"- 拒答: {ref}/{total} ({ref/max(1,total)*100:.1f}%)",
            "",
        ])

    # Top 5 题目
    def _rate(r):
        s = int(r.get("supported", 0))
        t = s + int(r.get("partial", 0)) + int(r.get("unsupported", 0))
        return s / max(1, t)
    ranked = sorted(results, key=_rate, reverse=True)
    lines.extend([
        "## Top 5 题目（CE supported 率）",
        "",
    ])
    for r in ranked[:5]:
        lines.append(f"- **{r['question_id']}** ({_rate(r):.0%}) {r['question']}")
    lines.append("")
    lines.append("## Bottom 5 题目（CE supported 率）")
    for r in ranked[-5:]:
        lines.append(f"- **{r['question_id']}** ({_rate(r):.0%}) {r['question']}")

    out = DIALOG_DIR / f"{ts}_全面测试报告.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"归档: {out}")


if __name__ == "__main__":
    sys.exit(main())
