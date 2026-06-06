"""完整质量评测套件：MQS + LeMAJ + Faithfulness + Inter-rater。

参考学术基础：
- LeMAJ (arXiv 2510.07243): 把答案拆成 Legal Data Points，逐条评
- RAGAS Faithfulness: 拆 claim + 逐条 verify
- MT-Bench / AlpacaEval: LLM-as-judge 范式（80%+ 与人类一致）
- LLM-RUBRIC (ACL 2024): 多维 rubric + 校准

⚠️ 注意：same-model judge 同一答案会有 ~28% 标签翻转（即使 temperature=0）。
   本评测用 Pro 当主裁判，Qwen 跑 5 题 inter-rater 验证。
"""

import json
import os
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from legalbot.llm import LLMClient, LLMConfig


# ── 评测模型配置 ────────────────────────────────────────


def make_judge_llm(model: str = "deepseek-v4-pro", provider: str = "deepseek") -> LLMClient:
    """创建评测用 LLM。

    默认：DeepSeek Pro（推理更强）
    可选：Qwen（cross-family，bias 校验用）
    """
    if provider == "custom1":
        # 硅基流动 / Qwen
        from legalbot.llm import load_env
        load_env()
        return LLMClient(LLMConfig(
            provider="custom1",
            api_key=os.getenv("CUSTOM_API_KEY_1", ""),
            base_url=os.getenv("CUSTOM_API_BASE_URL_1", "").rstrip("/"),
            model=model if "Qwen" in model or "DeepSeek" in model else "Qwen/Qwen2.5-72B-Instruct",
        ))
    os.environ["LEGALBOT_LLM_MODEL"] = model
    return LLMClient()


# ── 数据加载 ─────────────────────────────────────────────


def find_latest_bot_results() -> Path:
    matches = (
        list(PROJECT_ROOT.glob("tests/test100_*.csv")) +
        list(PROJECT_ROOT.glob("tests/test30_*.csv"))
    )
    if not matches:
        raise FileNotFoundError("找不到 test30_*.csv 或 test100_*.csv，请先跑测试")
    return max(matches, key=lambda p: p.stat().st_mtime)


def load_bot_answers(csv_path: Path) -> list[dict]:
    """从 CSV 加载 bot 答案和证据摘要。"""
    import csv
    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return [
        {
            "question_id": r["question_id"],
            "category": r["category"],
            "question": r["question"],
            "bot_answer": r["answer_full"],
            "evidence_summary": r.get("evidence_articles", ""),
        }
        for r in rows
        if r.get("answer_full")
    ]


def build_evidence_summaries(evidence_articles: str) -> str:
    """从 CSV 已有字段构造 evidence summary。"""
    if not evidence_articles:
        return "（无可用证据摘要）"
    return " | ".join(evidence_articles.split(" | ")[:8])


# ── 指标 1: MQS（5 维质量分）────────────────────────────


MQS_SYSTEM = """你是民航法律答案质量评估器。严格按照以下 5 个维度评估，每个维度 0/1/2 分。

## 评估维度

### 1. Q-Match（答对题了吗？）
- 0: 答非所问 / 完全跑题（用户在问 A，答案讲 B）
- 1: 部分回答了问题（提到一些相关内容但没直接答）
- 2: 直接、准确回答了用户问题

### 2. Law-Correctness（引用的法规对吗？）
- 0: 引用了完全不相关的法规（如问"行李丢失"却引用"航空器优先权"）
- 1: 引用了大致相关法规，但具体条号/内容不准确
- 2: 引用准确，法规名和条号都匹配问题

### 3. Coverage（覆盖全吗？）
- 0: 漏了关键情形 / 主要条件
- 1: 覆盖了主要但缺边缘情况
- 2: 完整覆盖关键情形和例外

### 4. Calibration（诚实承认不确定吗？）
- 0: 没证据时硬编或编造内容
- 1: 部分承认证据有限，但仍给出可能答案
- 2: 证据不足时清楚说"未找到"/"建议咨询专业人士"

### 5. Format（写得清楚吗？）
- 0: 混乱、无结构、难读
- 1: 可读但无清晰分段
- 2: 有清晰结构（结论/依据/适用/风险等）

## 输出 JSON
{
  "q_match": 0-2,
  "law_correct": 0-2,
  "coverage": 0-2,
  "calibration": 0-2,
  "format": 0-2,
  "reasons": {"q_match": "简短理由（≤30字）", "law_correct": "...", ...}
}
"""


def mqs_score(question: str, answer: str, evidence_summary: str, llm: LLMClient) -> dict:
    """5 维 MQS 评分。返回原始维度分 + 0-100 加权总分。"""
    messages = [
        {"role": "system", "content": MQS_SYSTEM},
        {"role": "user", "content": (
            f"用户问题：{question}\n\n"
            f"Bot 答案：{answer}\n\n"
            f"参考证据（来自检索，仅供你参考判断准确性）：\n{evidence_summary}"
        )},
    ]
    data = llm.json(messages)
    dims = {
        "q_match": int(data.get("q_match", 0)),
        "law_correct": int(data.get("law_correct", 0)),
        "coverage": int(data.get("coverage", 0)),
        "calibration": int(data.get("calibration", 0)),
        "format": int(data.get("format", 0)),
    }
    # 加权：Q-match 30% + Law-correct 30% + Coverage 20% + Calib 10% + Format 10%
    weights = {"q_match": 0.3, "law_correct": 0.3, "coverage": 0.2, "calibration": 0.1, "format": 0.1}
    weighted = sum(dims[k] * weights[k] for k in dims)  # 0-2 scale
    score_100 = round(weighted / 2 * 100, 1)  # normalize to 0-100
    return {
        **dims,
        "weighted_score": score_100,
        "reasons": data.get("reasons", {}),
    }


# ── 指标 2: LeMAJ（法律数据点级评测）──────────────────


LEMAJ_SYSTEM = """你是民航法律答案分解与评估器。模仿律师审查答案的方式：

1. 把答案拆成"Legal Data Points"（LDPs），每条 LDP 是独立的法律事实声明。
   - 例如："航空公司可在用户使用伪造证件时拒载"→一条 LDP
   - 概括性结论（如"以上为可能相关法规"）不算 LDP

2. 对每条 LDP 三维评分（每个维度 true/false）：
   - supported: 证据中是否有该 LDP 的支撑
   - correct: 该 LDP 是否符合民航法律常识（基于你的训练知识）
   - relevant: 该 LDP 是否与用户问题相关

## 输出 JSON
{
  "ldps": [
    {"text": "声明内容", "supported": true/false, "correct": true/false, "relevant": true/false}
  ],
  "summary": {
    "total_ldps": N,
    "supported_pct": 0.X (3 位小数),
    "correct_pct": 0.X,
    "relevant_pct": 0.X
  }
}
"""


def lemaj_score(question: str, answer: str, evidence_summary: str, llm: LLMClient) -> dict:
    """LeMAJ 风格的 LDP 级评测。"""
    messages = [
        {"role": "system", "content": LEMAJ_SYSTEM},
        {"role": "user", "content": (
            f"用户问题：{question}\n\n"
            f"Bot 答案：{answer}\n\n"
            f"检索到的证据（用于判断 supported）：\n{evidence_summary}"
        )},
    ]
    data = llm.json(messages)
    ldps = data.get("ldps", [])
    if not isinstance(ldps, list):
        ldps = []
    total = len(ldps)
    if total == 0:
        return {"ldps": [], "summary": {"total_ldps": 0, "supported_pct": 0, "correct_pct": 0, "relevant_pct": 0}}
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
        }
    }


# ── 指标 3: Faithfulness（4 档忠实性）────────────────────


FAITHFULNESS_SYSTEM = """你是法律答案忠实性评估器。把答案分解为 atomic claims（最小独立事实声明），对每条判断：

- faithful: 声明能从证据直接找到支撑
- partial: 声明核心信息有证据支撑，但部分细节无法验证
- unverifiable: 证据完全没有覆盖该声明涉及的话题（不是矛盾，只是没说）
- hallucinated: 证据覆盖了该话题但与声明矛盾，或声明编造了证据中不存在的数字/细节

## 关键区别
- 证据沉默 → unverifiable（声明可能正确也可能错误，无法判断）
- 证据矛盾 → hallucinated（声明歪曲或编造）

## 输出 JSON
{
  "claims": [
    {"text": "声明", "status": "faithful|partial|unverifiable|hallucinated", "reason": "判断依据（≤30字）"}
  ],
  "summary": {
    "total_claims": N,
    "faithful_pct": 0.X,
    "partial_pct": 0.X,
    "unverifiable_pct": 0.X,
    "hallucinated_pct": 0.X
  }
}
"""


def faithfulness_score(question: str, answer: str, evidence_summary: str, llm: LLMClient) -> dict:
    """4 档忠实性评测。"""
    messages = [
        {"role": "system", "content": FAITHFULNESS_SYSTEM},
        {"role": "user", "content": (
            f"用户问题：{question}\n\n"
            f"Bot 答案：{answer}\n\n"
            f"证据全文：\n{evidence_summary}"
        )},
    ]
    data = llm.json(messages)
    claims = data.get("claims", [])
    if not isinstance(claims, list):
        claims = []
    total = len(claims)
    if total == 0:
        return {"claims": [], "summary": {"total_claims": 0, "faithful_pct": 0, "partial_pct": 0, "unverifiable_pct": 0, "hallucinated_pct": 0}}
    buckets = {"faithful": 0, "partial": 0, "unverifiable": 0, "hallucinated": 0}
    for c in claims:
        s = str(c.get("status", "unverifiable")).strip().lower()
        if s in buckets:
            buckets[s] += 1
        else:
            # 未知状态：保守归为 unverifiable
            buckets["unverifiable"] += 1
    return {
        "claims": claims,
        "summary": {
            "total_claims": total,
            "faithful_pct": round(buckets["faithful"] / total, 3),
            "partial_pct": round(buckets["partial"] / total, 3),
            "unverifiable_pct": round(buckets["unverifiable"] / total, 3),
            "hallucinated_pct": round(buckets["hallucinated"] / total, 3),
        }
    }


# ── 指标 4: Inter-rater 验证（Pro vs Qwen）───────────────


def inter_rater_mqs(question: str, answer: str, evidence_summary: str,
                    model_a: str = "deepseek-v4-pro",
                    model_b: str = "custom1:Qwen/Qwen2.5-72B-Instruct") -> dict:
    """两个不同模型跑 MQS，比较一致性。"""
    # Model A
    provider_a = "custom1" if "custom1" in model_a else "deepseek"
    model_a_name = model_a.split(":", 1)[-1]
    llm_a = make_judge_llm(model=model_a_name, provider=provider_a)
    result_a = mqs_score(question, answer, evidence_summary, llm_a)

    # Model B
    provider_b = "custom1" if "custom1" in model_b else "deepseek"
    model_b_name = model_b.split(":", 1)[-1]
    llm_b = make_judge_llm(model=model_b_name, provider=provider_b)
    result_b = mqs_score(question, answer, evidence_summary, llm_b)

    dims = ["q_match", "law_correct", "coverage", "calibration", "format", "weighted_score"]
    agreements = {}
    for d in dims:
        a_val = result_a.get(d, 0)
        b_val = result_b.get(d, 0)
        agreements[d] = {
            "model_a": a_val,
            "model_b": b_val,
            "delta": abs(a_val - b_val),
            "agree": abs(a_val - b_val) <= 1,  # within 1 point tolerance
        }

    return {
        "model_a": model_a,
        "model_b": model_b,
        "per_dim": agreements,
        "n_agree_dims": sum(1 for a in agreements.values() if a["agree"]),
        "n_total_dims": len(dims),
    }


# ── 主流程 ─────────────────────────────────────────────


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

    return {
        "n_questions": n,
        "mqs": mqs_means,
        "lemaj": lemaj_means,
        "faithfulness": faith_means,
    }


def main(n_questions: int | None = None, run_inter_rater: int = 5, offset: int = 0):
    """主流程：跑所有 3 个指标 + inter-rater 验证。

    offset: 跳过前 N 道题（与 n_questions 配合使用，跑中间段）
    """
    bot_path = find_latest_bot_results()
    items = load_bot_answers(bot_path)
    if offset:
        items = items[offset:]
    if n_questions is not None:
        items = items[:n_questions]
    print(f"加载 {len(items)} 题 ({bot_path.name}, offset={offset})")

    judge = make_judge_llm()  # 默认 Pro
    print(f"主裁判模型: {judge.config.provider}/{judge.config.model}")

    per_question = []
    for i, item in enumerate(items, 1):
        print(f"[{i}/{len(items)}] {item['question_id']} 评估中...", flush=True)
        evidence_summary = build_evidence_summaries(item["evidence_summary"])
        try:
            mqs = mqs_score(item["question"], item["bot_answer"], evidence_summary, judge)
            lemaj = lemaj_score(item["question"], item["bot_answer"], evidence_summary, judge)
            faith = faithfulness_score(item["question"], item["bot_answer"], evidence_summary, judge)
        except Exception as e:
            print(f"  错误: {e}")
            mqs = lemaj = faith = {"error": str(e)}

        per_question.append({
            "question_id": item["question_id"],
            "category": item["category"],
            "question": item["question"],
            "mqs": mqs,
            "lemaj": lemaj,
            "faith": faith,
        })

    # Inter-rater on first N samples
    ir_results = []
    if run_inter_rater > 0:
        print(f"\n跑 {run_inter_rater} 题 inter-rater (Pro vs Qwen)...")
        for i, item in enumerate(items[:run_inter_rater], 1):
            print(f"  IR [{i}/{run_inter_rater}] {item['question_id']}...", flush=True)
            try:
                ir = inter_rater_mqs(
                    item["question"],
                    item["bot_answer"],
                    build_evidence_summaries(item["evidence_summary"]),
                )
                ir_results.append({"question_id": item["question_id"], "inter_rater": ir})
            except Exception as e:
                ir_results.append({"question_id": item["question_id"], "error": str(e)})

    # 汇总
    summary = aggregate_results(per_question)
    if ir_results:
        valid_ir = [r["inter_rater"] for r in ir_results if "inter_rater" in r]
        if valid_ir:
            agree_dims = sum(r["n_agree_dims"] for r in valid_ir)
            total_dims = sum(r["n_total_dims"] for r in valid_ir)
            summary["inter_rater"] = {
                "n_questions": len(valid_ir),
                "agreement_rate": round(agree_dims / total_dims, 3) if total_dims else 0,
            }

    # 输出报告
    report = {
        "evaluated_at": __import__("datetime").datetime.now().isoformat(),
        "bot_source": bot_path.name,
        "judge_model": judge.config.model,
        "summary": summary,
        "per_question": per_question,
        "inter_rater": ir_results,
    }

    from datetime import datetime
    out_path = PROJECT_ROOT / "tests" / f"quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 控制台汇总
    print(f"\n{'='*60}")
    print(f"### 评测汇总 ({len(items)} 题)")
    print(f"{'='*60}")
    print(f"\n【MQS】(0-100 分)")
    for k, v in summary.get("mqs", {}).items():
        print(f"  {k:20s} {v:>6.1f}")
    print(f"\n【LeMAJ】(% LDP 满足条件)")
    for k, v in summary.get("lemaj", {}).items():
        print(f"  {k:20s} {v*100:>6.1f}%")
    print(f"\n【Faithfulness】(% claims 落档)")
    for k, v in summary.get("faithfulness", {}).items():
        print(f"  {k:20s} {v*100:>6.1f}%")
    if "inter_rater" in summary:
        print(f"\n【Inter-rater】")
        print(f"  Pro vs Qwen 一致率: {summary['inter_rater']['agreement_rate']*100:.1f}%")
    print(f"\n报告：{out_path}")
    return report


if __name__ == "__main__":
    n = None
    ir = 5
    offset = 0
    for arg in sys.argv[1:]:
        if arg.startswith("--n="):
            n = int(arg.split("=", 1)[1])
        elif arg.startswith("--ir="):
            ir = int(arg.split("=", 1)[1])
        elif arg.startswith("--offset="):
            offset = int(arg.split("=", 1)[1])
    main(n_questions=n, run_inter_rater=ir, offset=offset)
