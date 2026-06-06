"""LLM 分析：Faithfulness + LeMAJ + MQS（含 F1/recall 聚合）。

替代：test_faithfulness.py + eval_quality.py（合二为一）
输入：test_*.csv（test_30questions.py / test_100questions.py 输出）
输出：{out_dir}/<stem>_llm.json

指标：
1. Faithfulness 4 档（faithful / partial / unverifiable / hallucinated）
2. LeMAJ correct（准确率）+ supported + relevant
3. MQS weighted（0-100）+ 5 维
4. F1 / Recall（由 faithfulness 聚合）
5. 可选：inter-rater 一致性（双 LLM 校验）

用法：
    .venv/bin/python tests/analyze_llm.py tests/test30_20260604_180203.csv
    .venv/bin/python tests/analyze_llm.py tests/test30_20260604_180203.csv --ir=5
    .venv/bin/python tests/analyze_llm.py tests/test30_20260604_180203.csv --offset 30 --n 70
"""

import argparse
import csv
import json
import os
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from legalbot.llm import LLMClient, LLMConfig, load_env

load_env()

# ── 拒答关键词 ──
REFUSAL_KEYWORDS = [
    "无法确定", "无法回答", "证据不足", "未包含相关",
    "未涉及", "未找到", "没有找到", "未能找到",
    "无法提供", "无法确认", "无法判断", "现有证据不足以",
]


def is_refusal(answer: str) -> bool:
    return any(kw in answer for kw in REFUSAL_KEYWORDS)


def make_judge_llm(model: str = None, provider: str = "deepseek") -> LLMClient:
    """评测 LLM：默认 deepseek-v4-pro，可切换 Qwen 做 inter-rater。"""
    if provider == "custom1":
        return LLMClient(LLMConfig(
            provider="custom1",
            api_key=os.getenv("CUSTOM_API_KEY_1", ""),
            base_url=os.getenv("CUSTOM_API_BASE_URL_1", "").rstrip("/"),
            model=model or "Qwen/Qwen2.5-72B-Instruct",
        ))
    if model:
        os.environ["LEGALBOT_LLM_MODEL"] = model
    return LLMClient()


# ════════════════════════════════════════════════
# 评测 1: Faithfulness（4 档忠实性）
# ════════════════════════════════════════════════
FAITHFULNESS_SYSTEM = """你是法律答案忠实性评估器。把答案分解为 atomic claims（最小独立事实声明），对每条判断：
- faithful: 声明能从证据直接找到支撑
- partial: 核心信息有证据，细节无法验证
- unverifiable: 证据完全没有覆盖该声明涉及的话题
- hallucinated: 证据覆盖了该话题但与声明矛盾，或声明编造了证据中不存在的数字/细节

关键区别：
- 证据沉默 → unverifiable（声明可能正确也可能错误）
- 证据矛盾 → hallucinated

输出 JSON:
{"claims": [{"text": "声明", "status": "faithful|partial|unverifiable|hallucinated", "reason": "≤30字"}],
 "summary": {"total_claims": N, "faithful_pct": 0.X, "partial_pct": 0.X, "unverifiable_pct": 0.X, "hallucinated_pct": 0.X}}
"""


def faithfulness_score(question: str, answer: str, evidence: str, llm: LLMClient) -> dict:
    messages = [
        {"role": "system", "content": FAITHFULNESS_SYSTEM},
        {"role": "user", "content": f"问题：{question}\n\n答案：{answer}\n\n证据全文：{evidence}"},
    ]
    try:
        data = llm.json(messages)
        if not isinstance(data, dict): return {"claims": [], "summary": _empty_summary()}
        claims = data.get("claims", [])
        if not isinstance(claims, list): claims = []
        total = len(claims)
        if total == 0:
            return {"claims": [], "summary": _empty_summary()}
        buckets = {"faithful": 0, "partial": 0, "unverifiable": 0, "hallucinated": 0}
        for c in claims:
            s = str(c.get("status", "unverifiable")).strip().lower()
            if s in buckets:
                buckets[s] += 1
            else:
                buckets["unverifiable"] += 1
        return {
            "claims": claims,
            "summary": {
                "total_claims": total,
                # 绝对计数（用于 F1 lenient 公式）
                "faithful": buckets["faithful"],
                "partial": buckets["partial"],
                "unverifiable": buckets["unverifiable"],
                "hallucinated": buckets["hallucinated"],
                # 百分比（用于展示）
                "faithful_pct": round(buckets["faithful"] / total, 3),
                "partial_pct": round(buckets["partial"] / total, 3),
                "unverifiable_pct": round(buckets["unverifiable"] / total, 3),
                "hallucinated_pct": round(buckets["hallucinated"] / total, 3),
            },
        }
    except Exception:
        return {"claims": [], "summary": _empty_summary()}


def _empty_summary():
    return {"total_claims": 0, "faithful": 0, "partial": 0, "unverifiable": 0, "hallucinated": 0,
            "faithful_pct": 0, "partial_pct": 0, "unverifiable_pct": 0, "hallucinated_pct": 0}


# ════════════════════════════════════════════════
# 评测 2: LeMAJ（Legal Data Points 拆解 + 准确率）
# ════════════════════════════════════════════════
LEMAJ_SYSTEM = """你是民航法律答案分解与评估器。模仿律师审查答案的方式：
1. 把答案拆成"Legal Data Points"（LDPs），每条 LDP 是独立的法律事实声明。
2. 对每条 LDP 三维评分（每个维度 true/false）：
   - supported: 证据中是否有该 LDP 的支撑
   - correct: 该 LDP 是否符合民航法律常识（基于你的训练知识）
   - relevant: 该 LDP 是否与用户问题相关

输出 JSON:
{"ldps": [{"text": "声明", "correct": true/false, "supported": true/false, "relevant": true/false}],
 "summary": {"total_ldps": N, "supported_pct": 0.X, "correct_pct": 0.X, "relevant_pct": 0.X}}
"""


def lemaj_score(question: str, answer: str, evidence: str, llm: LLMClient) -> dict:
    messages = [
        {"role": "system", "content": LEMAJ_SYSTEM},
        {"role": "user", "content": f"用户问题：{question}\n\n答案：{answer}\n\n参考证据：{evidence}"},
    ]
    try:
        data = llm.json(messages)
        if not isinstance(data, dict): return {"ldps": [], "summary": _empty_lemaj()}
        ldps = data.get("ldps", [])
        if not isinstance(ldps, list): ldps = []
        total = len(ldps)
        if total == 0:
            return {"ldps": [], "summary": _empty_lemaj()}
        supported = sum(1 for l in ldps if l.get("supported"))
        correct = sum(1 for l in ldps if l.get("correct"))
        relevant = sum(1 for l in ldps if l.get("relevant"))
        return {
            "ldps": ldps,
            "summary": {
                "total_ldps": total,
                "supported_pct": round(supported / total, 3),
                "correct_pct": round(correct / total, 3),
                "relevant_pct": round(relevant / total, 3),
            },
        }
    except Exception:
        return {"ldps": [], "summary": _empty_lemaj()}


def _empty_lemaj():
    return {"total_ldps": 0, "supported_pct": 0, "correct_pct": 0, "relevant_pct": 0}


# ════════════════════════════════════════════════
# 评测 3: MQS（5 维质量分）
# ════════════════════════════════════════════════
MQS_SYSTEM = """你是民航法律答案质量评估器。严格按照 5 维评估，每个维度 0/1/2 分。

1. Q-Match（答对题了吗？）0=答非所问 1=部分答 2=直接准确
2. Law-Correctness（引用法规对吗？）0=完全无关 1=大致相关但不准 2=完全匹配
3. Coverage（覆盖全吗？）0=漏关键 1=缺边缘 2=完整
4. Calibration（诚实承认不确定吗？）0=没证据硬编 1=部分承认 2=清楚说未找到
5. Format（写得清楚吗？）0=混乱 1=可读 2=有清晰结构

权重：Q-Match 30% + Law-Correct 30% + Coverage 20% + Calib 10% + Format 10%

输出 JSON:
{"q_match": 0-2, "law_correct": 0-2, "coverage": 0-2, "calibration": 0-2, "format": 0-2,
 "reasons": {"q_match": "≤30字", "law_correct": "...", "coverage": "...", "calibration": "...", "format": "..."}}
"""


def mqs_score(question: str, answer: str, evidence: str, llm: LLMClient) -> dict:
    messages = [
        {"role": "system", "content": MQS_SYSTEM},
        {"role": "user", "content": f"用户问题：{question}\n\n答案：{answer}\n\n参考证据：{evidence[:1500]}"},
    ]
    try:
        data = llm.json(messages)
        if not isinstance(data, dict):
            return _empty_mqs()
        dims = {k: int(data.get(k, 0)) for k in ["q_match", "law_correct", "coverage", "calibration", "format"]}
        weights = {"q_match": 0.3, "law_correct": 0.3, "coverage": 0.2, "calibration": 0.1, "format": 0.1}
        weighted = sum(dims[k] * weights[k] for k in dims)
        return {
            **dims,
            "weighted_score": round(weighted / 2 * 100, 1),
            "reasons": data.get("reasons", {}),
        }
    except Exception:
        return _empty_mqs()


def _empty_mqs():
    return {"q_match": 0, "law_correct": 0, "coverage": 0, "calibration": 0, "format": 0, "weighted_score": 0}


# ════════════════════════════════════════════════
# 评测 4: Missing 检测（"应提未提"信息数）
# ════════════════════════════════════════════════
MISSING_SYSTEM = """你是法律答案完整性评估器。基于问题+参考证据+答案，检查"答案漏掉了哪些关键信息"。

任务：
1. 根据问题+证据，列出"理想答案应该包含的关键信息点"（5-10 条）
2. 对比 bot 答案，标记哪些"应该提但没提"（missing）

输出 JSON:
{
  "expected_points": ["关键点1", "关键点2", ...],
  "missing_points": ["bot 漏掉的关键点1", ...],
  "missing_count": N
}
"""


def detect_missing(question: str, answer: str, evidence: str, llm: LLMClient) -> dict:
    """检测答案漏掉的关键信息数。"""
    messages = [
        {"role": "system", "content": MISSING_SYSTEM},
        {"role": "user", "content": f"问题：{question}\n\n参考证据：{evidence[:1500]}\n\nBot 答案：{answer[:1500]}"},
    ]
    try:
        data = llm.json(messages)
        if not isinstance(data, dict): return {"missing_count": 0, "missing_points": [], "expected_points": []}
        missing = data.get("missing_points", [])
        if not isinstance(missing, list): missing = []
        return {
            "missing_count": len(missing),
            "missing_points": missing,
            "expected_points": data.get("expected_points", []),
        }
    except Exception:
        return {"missing_count": 0, "missing_points": [], "expected_points": []}


# ════════════════════════════════════════════════
# 评测 5: Inter-rater 一致性（Pro vs Qwen）
# ════════════════════════════════════════════════
def inter_rater_check(question: str, answer: str, evidence: str, n_questions: list) -> dict:
    """对前 N 题用 Pro + Qwen 双裁判，比较 MQS 一致性。"""
    pro = make_judge_llm(provider="deepseek")
    qwen = make_judge_llm(provider="custom1", model="Qwen/Qwen2.5-72B-Instruct")

    pro_mqs = mqs_score(question, answer, evidence, pro)
    qwen_mqs = mqs_score(question, answer, evidence, qwen)

    dims = ["q_match", "law_correct", "coverage", "calibration", "format", "weighted_score"]
    agreements = {}
    for d in dims:
        a_val = pro_mqs.get(d, 0)
        b_val = qwen_mqs.get(d, 0)
        agreements[d] = {
            "pro": a_val, "qwen": b_val, "delta": abs(a_val - b_val),
            "agree": abs(a_val - b_val) <= 1,
        }
    n_agree_dims = sum(1 for a in agreements.values() if a["agree"])
    return {
        "pro": pro_mqs, "qwen": qwen_mqs,
        "n_agree_dims": n_agree_dims, "n_total_dims": len(dims),
        "agreement_rate": round(n_agree_dims / len(dims), 3),
    }


# ════════════════════════════════════════════════
# 加载 + 主流程
# ════════════════════════════════════════════════
def load_bot_answers(csv_path: Path, offset: int = 0, n: int = None) -> list[dict]:
    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if offset:
        rows = rows[offset:]
    if n is not None:
        rows = rows[:n]
    return [
        {
            "question_id": r.get("question_id", ""),
            "category": r.get("category", ""),
            "question": r.get("question", ""),
            "bot_answer": r.get("answer_full", ""),
            "evidence_summary": r.get("evidence_articles", ""),
        }
        for r in rows if r.get("answer_full") or r.get("conclusion_preview")
    ]


def build_evidence_summaries(evidence_articles: str) -> str:
    if not evidence_articles:
        return "（无可用证据摘要）"
    return " | ".join(evidence_articles.split(" | ")[:8])


def find_latest_csv() -> Path:
    candidates = sorted(
        list((PROJECT_ROOT / "tests").glob("test30_*.csv"))
        + list((PROJECT_ROOT / "tests").glob("test100_*.csv")),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError("找不到 test30_*.csv 或 test100_*.csv")
    return candidates[-1]


def aggregate_results(results: list[dict]) -> dict:
    """汇总所有题目的指标。"""
    n = len(results)
    if n == 0:
        return {}

    # MQS 各维度均值
    mqs_means = {}
    for dim in ["q_match", "law_correct", "coverage", "calibration", "format", "weighted_score"]:
        vals = [r["mqs"].get(dim, 0) for r in results if r.get("mqs")]
        mqs_means[dim] = round(statistics.mean(vals), 2) if vals else 0

    # LeMAJ 平均
    lemaj_means = {}
    for dim in ["supported_pct", "correct_pct", "relevant_pct"]:
        vals = [r["lemaj"]["summary"].get(dim, 0) for r in results if r.get("lemaj")]
        lemaj_means[dim] = round(statistics.mean(vals), 3) if vals else 0

    # Faithfulness 平均
    faith_means = {}
    for dim in ["faithful_pct", "partial_pct", "unverifiable_pct", "hallucinated_pct"]:
        vals = [r["faith"]["summary"].get(dim, 0) for r in results if r.get("faith")]
        faith_means[dim] = round(statistics.mean(vals), 3) if vals else 0

    # Missing 平均
    missing_means = {}
    for dim in ["missing_count"]:
        vals = [r.get("missing", {}).get(dim, 0) for r in results if r.get("missing")]
        missing_means[dim] = round(statistics.mean(vals), 1) if vals else 0
    missing_means["total_missing"] = sum(r.get("missing", {}).get("missing_count", 0) for r in results)

    # ★ 三种 F1 / Recall 变体（供对比）
    if faith_means:
        f = faith_means["faithful_pct"]
        s = faith_means["partial_pct"]
        h = faith_means["hallucinated_pct"]
        u = faith_means["unverifiable_pct"]
        total_missing = missing_means.get("total_missing", 0)  # 绝对总数

        # ── 变体 1（严）：correct = faithful, recall den = total_claims ──
        p1 = f / (f + s + h) if (f + s + h) > 0 else 0
        r1 = f
        f1_strict = 2 * p1 * r1 / (p1 + r1) if (p1 + r1) > 0 else 0

        # ── 变体 2（宽，复刻旧 test_faithfulness.py 公式）：绝对计数 ──
        sum_total_claims = 0
        sum_correct = 0
        sum_faithful_abs = 0
        sum_hallucinated_abs = 0
        for r in results:
            if r.get("faith"):
                fsum = r["faith"]["summary"]
                tc = fsum.get("total_claims", 0)
                f_count = fsum.get("faithful", 0)
                p_count = fsum.get("partial", 0)
                h_count = fsum.get("hallucinated", 0)
                sum_total_claims += tc
                sum_correct += f_count + p_count
                sum_faithful_abs += f_count
                sum_hallucinated_abs += h_count
        if sum_total_claims > 0:
            p2 = sum_correct / sum_total_claims
            r2 = sum_correct / (sum_correct + total_missing) if (sum_correct + total_missing) > 0 else 0
        else:
            p2 = r2 = 0
        f1_lenient = 2 * p2 * r2 / (p2 + r2) if (p2 + r2) > 0 else 0

        # ── 变体 3（中）：correct = faithful + partial, recall den = total_claims ──
        p3 = f + s
        r3 = f + s
        f1_mid = 2 * p3 * r3 / (p3 + r3) if (p3 + r3) > 0 else 0

        # ── 变体 4（标准 IR，Wikipedia/Stanford 教材）：严格 TP/FP/FN 分类 ──
        # 用绝对计数（更符合 IR 教材定义）
        if (sum_faithful_abs + sum_hallucinated_abs) > 0:
            p4 = sum_faithful_abs / (sum_faithful_abs + sum_hallucinated_abs)
            r4 = sum_faithful_abs / (sum_faithful_abs + total_missing) if (sum_faithful_abs + total_missing) > 0 else 0
        else:
            p4 = r4 = 0
        f1_standard = 2 * p4 * r4 / (p4 + r4) if (p4 + r4) > 0 else 0

        faith_means["f1_variants"] = {
            "strict": {
                "definition": "correct=faithful, recall den=total_claims",
                "precision": round(p1, 3),
                "recall": round(r1, 3),
                "f1": round(f1_strict, 3),
            },
            "lenient": {
                "definition": "正确复刻旧 test_faithfulness.py: correct=faithful+partial (绝对), recall den=correct+missing (绝对)",
                "precision": round(p2, 3),
                "recall": round(r2, 3),
                "f1": round(f1_lenient, 3),
            },
            "middle": {
                "definition": "correct=faithful+partial, recall den=total_claims",
                "precision": round(p3, 3),
                "recall": round(r3, 3),
                "f1": round(f1_mid, 3),
            },
            "standard": {
                "definition": "标准 IR (Wikipedia/Stanford): TP=faithful, FP=hallucinated, FN=missing",
                "precision": round(p4, 3),
                "recall": round(r4, 3),
                "f1": round(f1_standard, 3),
            },
        }
        # 兼容旧字段
        faith_means["f1"] = round(f1_lenient, 3)

    return {
        "n_questions": n,
        "mqs": mqs_means,
        "lemaj": lemaj_means,
        "faithfulness": faith_means,
        "missing": missing_means,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv_path", nargs="?", default=None)
    p.add_argument("--ir", type=int, default=0, help="inter-rater 跑前 N 题（0=不跑）")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--max-workers", type=int, default=int(os.environ.get("ANALYZE_PARALLEL", "10")))
    args = p.parse_args()

    csv_path = Path(args.csv_path) if args.csv_path else find_latest_csv()
    print(f"输入: {csv_path}")
    print(f"评测模型: deepseek-v4-pro（主裁判）")
    print(f"并发: {args.max_workers}")

    items = load_bot_answers(csv_path, offset=args.offset, n=args.n)
    if not items:
        print("ERROR: 没有数据")
        return
    print(f"题数: {len(items)} (offset={args.offset})")

    judge = make_judge_llm()

    def eval_one(item):
        evidence = build_evidence_summaries(item["evidence_summary"])
        return {
            "question_id": item["question_id"],
            "category": item["category"],
            "question": item["question"],
            "mqs": mqs_score(item["question"], item["bot_answer"], evidence, judge),
            "lemaj": lemaj_score(item["question"], item["bot_answer"], evidence, judge),
            "faith": faithfulness_score(item["question"], item["bot_answer"], evidence, judge),
            "missing": detect_missing(item["question"], item["bot_answer"], evidence, judge),
        }

    per_question = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(eval_one, item): item["question_id"] for item in items}
        done = 0
        for future in as_completed(futures):
            result = future.result()
            per_question.append(result)
            done += 1
            if done % 5 == 0 or done == len(items):
                print(f"  进度: {done}/{len(items)}", flush=True)

    # Inter-rater（可选）
    ir_results = []
    if args.ir > 0:
        print(f"\n跑 inter-rater（前 {args.ir} 题）...")
        for i, item in enumerate(items[:args.ir], 1):
            evidence = build_evidence_summaries(item["evidence_summary"])
            try:
                ir = inter_rater_check(item["question"], item["bot_answer"], evidence, list(range(5)))
                ir_results.append({"question_id": item["question_id"], "inter_rater": ir})
                print(f"  IR {i}/{args.ir} {item['question_id']}: {ir['agreement_rate']:.1%}")
            except Exception as e:
                print(f"  IR {i} 失败: {e}")

    # 汇总
    summary = aggregate_results(per_question)
    if ir_results:
        valid_ir = [r["inter_rater"] for r in ir_results if "inter_rater" in r]
        if valid_ir:
            avg_agree = sum(r["agreement_rate"] for r in valid_ir) / len(valid_ir)
            summary["inter_rater"] = {
                "n_questions": len(valid_ir),
                "avg_agreement_rate": round(avg_agree, 3),
            }

    # 输出报告
    report = {
        "evaluated_at": __import__("datetime").datetime.now().isoformat(),
        "bot_source": str(csv_path),
        "judge_model": judge.config.model,
        "summary": summary,
        "per_question": per_question,
        "inter_rater": ir_results,
    }
    out_path = csv_path.parent / f"{csv_path.stem}_llm.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 控制台汇总
    print(f"\n{'='*60}")
    print(f"### LLM 评测汇总 ({len(items)} 题) ###")
    print(f"{'='*60}")
    print(f"\n【MQS】(0-100 分)")
    for k, v in summary.get("mqs", {}).items():
        print(f"  {k:20s} {v:>6.1f}")
    print(f"\n【LeMAJ】(% LDP)")
    for k, v in summary.get("lemaj", {}).items():
        print(f"  {k:20s} {v*100:>6.1f}%")
    print(f"\n【Faithfulness】(% claims)")
    for k, v in summary.get("faithfulness", {}).items():
        if k == "f1_variants":
            print(f"  {k}:")
            for variant, metrics in v.items():
                print(f"    [{variant}] {metrics['definition']}")
                print(f"      precision={metrics['precision']:.3f}, recall={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
        elif isinstance(v, (int, float)):
            print(f"  {k:20s} {v*100:>6.1f}%")
    print(f"\n【Missing】(LLM 评测)")
    m = summary.get("missing", {})
    print(f"  avg_missing_count: {m.get('missing_count', 0)}")
    print(f"  total_missing:     {m.get('total_missing', 0)}")
    if "inter_rater" in summary:
        print(f"\n【Inter-rater】")
        print(f"  Pro vs Qwen 一致率: {summary['inter_rater']['avg_agreement_rate']*100:.1f}%")
    print(f"\n报告: {out_path}")


if __name__ == "__main__":
    main()
