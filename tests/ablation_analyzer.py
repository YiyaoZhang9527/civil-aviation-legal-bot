"""消融测试分析器：读原始数据，计算全部 10 项指标 + 多角度对比。

不重新跑任何 LLM 答题，只读取 ablation_runs/ 下的 JSON。

计算的指标：
1. supported_rate           - CE 通过率（启发式，无需 LLM）
2. evidence_count            - 平均证据数
3. elapsed_sec              - 平均耗时
4. answer_len                - 平均答案长度
5. reflexion_iterations      - 平均自检轮数
6. refusal_rate              - 拒答率
7. citation_validity_rate    - 引用真实存在率（正则 + index 对照）
8. LeMAJ correct（准确率）   - LLM 裁判（独立 LLM）
9. MQS weighted              - LLM 裁判（5 维质量分）
10. Faithful/Hallucinated    - LLM 裁判（4 档忠实性）

用法:
    .venv/bin/python tests/ablation_analyzer.py <out_dir>
    .venv/bin/python tests/ablation_analyzer.py <out_dir> --skip-llm  # 不调 LLM 裁判（快速）
    .venv/bin/python tests/ablation_analyzer.py <out_dir> --top 5      # 只评测 top 5 配置
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from collections import defaultdict
from statistics import mean, stdev

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from legalbot.retrieval import IndexRepository

# ── 拒答关键词 ──
REFUSAL_KEYWORDS = [
    "无法确定", "无法回答", "证据不足", "未包含相关",
    "未涉及", "未找到", "没有找到", "未能找到",
    "无法提供", "无法确认", "无法判断", "现有证据不足以",
]

# ── 引用提取正则 ──
CITATION_RE = re.compile(r"《([^》]+)》[^。；\n]*?第([一二三四五六七八九十百千零\d]+(?:\.\d+)*)\s*条")


def load_runs(out_dir):
    """加载所有 ablation 原始数据。"""
    out_dir = Path(out_dir)
    files = sorted(out_dir.glob("*_q*.json"))
    runs = []
    for f in files:
        if "ERROR" in f.name:
            continue
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
            r["_file"] = str(f)
            runs.append(r)
        except Exception as e:
            print(f"读 {f} 失败: {e}")
    return runs


def group_by_config(runs):
    """按 config_id 分组。"""
    by_config = defaultdict(list)
    for r in runs:
        by_config[r["config_id"]].append(r)
    return dict(by_config)


def compute_basic_metrics(runs):
    """无需 LLM 裁判的基础指标。"""
    if not runs:
        return {}
    valid = [r for r in runs if not r.get("error")]
    if not valid:
        return {}

    n = len(valid)
    sup_rates = []
    for r in valid:
        cc = r.get("citation_count", 0)
        sc = r.get("supported_count", 0)
        if cc > 0:
            sup_rates.append(sc / cc)
    elapsed_list = [r.get("elapsed_sec", 0) for r in valid]
    answer_lens = [r.get("answer_len", 0) for r in valid]
    ev_counts = [r.get("evidence_count", 0) for r in valid]
    reflexion_iters = [r.get("reflexion_iterations", 0) for r in valid]
    refusals = sum(1 for r in valid if r.get("is_refusal"))
    errors = sum(1 for r in runs if r.get("error"))

    return {
        "n_questions": n,
        "n_errors": errors,
        "supported_rate": mean(sup_rates) if sup_rates else 0,
        "avg_elapsed_sec": mean(elapsed_list) if elapsed_list else 0,
        "stdev_elapsed_sec": stdev(elapsed_list) if len(elapsed_list) > 1 else 0,
        "avg_answer_len": mean(answer_lens) if answer_lens else 0,
        "avg_evidence_count": mean(ev_counts) if ev_counts else 0,
        "avg_reflexion_iters": mean(reflexion_iters) if reflexion_iters else 0,
        "refusal_rate": refusals / n if n else 0,
    }


def extract_citations_from_answer(answer):
    """从答案中提取法条引用。"""
    citations = []
    for m in CITATION_RE.finditer(answer):
        citations.append({"law_name": m.group(1).strip(), "article": m.group(2)})
    return citations


def compute_citation_validity(runs):
    """第二层：引用真实性（正则 + index 对照）。"""
    docs = IndexRepository.documents()  # list[LawDocument]
    # 获取所有 (law_title) 集合
    law_titles = set()
    for doc in docs:
        if doc.title:
            law_titles.add(doc.title)
            # 也加去书名号版本
            law_titles.add(doc.title.replace("《", "").replace("》", ""))

    total_citations = 0
    valid_citations = 0
    invalid = []

    for r in runs:
        if r.get("error"): continue
        citations = extract_citations_from_answer(r.get("answer_full", ""))
        for cit in citations:
            total_citations += 1
            law_name = cit["law_name"]
            found = any(law_name in name or name in law_name for name in law_titles)
            if found:
                valid_citations += 1
            else:
                invalid.append((r["config_id"], r["qid"], law_name, cit["article"]))

    return {
        "total_citations": total_citations,
        "valid_citations": valid_citations,
        "validity_rate": valid_citations / total_citations if total_citations else 0,
        "invalid_examples": invalid[:10],
    }


def run_llm_judges(runs, max_configs=None):
    """调 LLM 裁判算 LeMAJ correct / MQS / Faithful。

    与 eval_quality.py 共用评测逻辑，但只针对 ablation 的小数据集。

    Args:
        runs: 全部 run 记录
        max_configs: 限制评测的配置数（None=全部）
    """
    from legalbot.llm import LLMClient
    from legalbot.llm import LLMConfig, load_env
    load_env()

    # 评测 LLM：默认 Pro
    judge = LLMClient()
    print(f"评测 LLM: {judge.config.provider}/{judge.config.model}")

    # 按 config 分组
    by_config = group_by_config(runs)

    # 限制数量
    if max_configs:
        # 按 supported_rate 选 top max_configs
        ranked = sorted(
            by_config.items(),
            key=lambda x: compute_basic_metrics(x[1]).get("supported_rate", 0),
            reverse=True,
        )
        by_config = dict(ranked[:max_configs])
        print(f"评测 top {len(by_config)} 配置（按 supported_rate 排序）")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    # LLM 评测并行配置
    PARALLEL_WORKERS = int(os.environ.get("ABLATION_PARALLEL", "10"))  # 默认 10 并发
    print(f"LLM 评测并发: {PARALLEL_WORKERS}")

    def judge_one_question(item, judge):
        """对单题跑 3 个 LLM 评测（在单个线程内串行）。"""
        try:
            lemaj = lemaj_score_judge(item["question"], item["answer"], item["evidence"], judge)
            mqs = mqs_score_judge(item["question"], item["answer"], item["evidence"], judge)
            faith = faithfulness_score_judge(item["question"], item["answer"], item["evidence"], judge)
            return {
                "qid": item["qid"],
                "lemaj": lemaj, "mqs": mqs,
                "faithful_pct": faith.get("faithful_pct", 0),
                "hallucinated_pct": faith.get("hallucinated_pct", 0),
            }
        except Exception as e:
            print(f"  {item['qid']} 评测失败: {e}", flush=True)
            return None

    results = {}
    for cfg_id, cfg_runs in by_config.items():
        print(f"\n评测配置 {cfg_id} ({len(cfg_runs)} 题)...", flush=True)
        # 取 evidence 摘要
        evidence_summary = []
        for r in cfg_runs:
            ev_titles = [f"{e['law_title']} {e['article']}" for e in r.get("evidence", [])][:5]
            evidence_summary.append({
                "qid": r["qid"],
                "question": r["question"],
                "answer": r["answer_full"],
                "evidence": " | ".join(ev_titles) if ev_titles else "（无）",
            })

        # ★ 并行评测：每个 question 一个线程 ★
        question_results = []
        with ThreadPoolExecutor(max_workers=min(PARALLEL_WORKERS, len(evidence_summary))) as pool:
            futures = {pool.submit(judge_one_question, item, judge): item["qid"] for item in evidence_summary}
            for future in as_completed(futures):
                r = future.result()
                if r:
                    question_results.append(r)
                    print(f"  {r['qid']}: LeMAJ={r['lemaj']:.2f} MQS={r['mqs']:.1f}", flush=True)

        # 聚合
        if question_results:
            lemaj_correct = [r["lemaj"] for r in question_results]
            mqs_scores = [r["mqs"] for r in question_results]
            faithful_pct = [r["faithful_pct"] for r in question_results]
            hallucinated_pct = [r["hallucinated_pct"] for r in question_results]
        else:
            lemaj_correct = mqs_scores = faithful_pct = hallucinated_pct = []

        results[cfg_id] = {
            "n_evaluated": len(lemaj_correct),
            "lemaj_correct": mean(lemaj_correct) if lemaj_correct else 0,
            "mqs_weighted": mean(mqs_scores) if mqs_scores else 0,
            "faithful_pct": mean(faithful_pct) if faithful_pct else 0,
            "hallucinated_pct": mean(hallucinated_pct) if hallucinated_pct else 0,
        }
        print(f"  → LeMAJ correct={results[cfg_id]['lemaj_correct']:.3f}, "
              f"MQS={results[cfg_id]['mqs_weighted']:.1f}, "
              f"faithful={results[cfg_id]['faithful_pct']:.3f}, "
              f"hallucinated={results[cfg_id]['hallucinated_pct']:.3f}")

    return results


def lemaj_score_judge(question, answer, evidence, judge):
    """LeMAJ 准确率评测（对齐 eval_quality.py 的 LDP 分解方法）。"""
    messages = [
        {"role": "system", "content": (
            "你是民航法律答案分解与评估器。模仿律师审查答案的方式:\n"
            "1. 把答案拆成\"Legal Data Points\"（LDPs），每条 LDP 是独立的法律事实声明。\n"
            "   - 例如：\"航空公司可在用户使用伪造证件时拒载\"→一条 LDP\n"
            "2. 对每条 LDP 三维评分（每个维度 true/false）：\n"
            "   - correct: 该 LDP 是否符合民航法律常识（基于你的训练知识）\n"
            "   - supported: 证据中是否有该 LDP 的支撑\n"
            "   - relevant: 该 LDP 是否与用户问题相关\n\n"
            "输出 JSON:\n"
            '{"ldps": [{"text": "声明", "correct": true/false, "supported": true/false, "relevant": true/false}]}'
        )},
        {"role": "user", "content": f"用户问题：{question}\n\n答案：{answer[:2000]}\n\n参考证据：{evidence[:500]}"},
    ]
    try:
        data = judge.json(messages)
        if not isinstance(data, dict):
            return 0.0
        ldps = data.get("ldps", [])
        if not ldps:
            return 0.0
        correct = sum(1 for l in ldps if l.get("correct"))
        return round(correct / len(ldps), 3)
    except Exception:
        return 0.0


def mqs_score_judge(question, answer, evidence, judge):
    """MQS 加权分（0-100）。"""
    messages = [
        {"role": "system", "content": (
            "你是民航法律答案质量评估器。\n"
            "从 5 维评估：Q-Match(0-2) + Law-Correct(0-2) + Coverage(0-2) + Calibration(0-2) + Format(0-2)。\n"
            "权重：Q-Match 30% + Law-Correct 30% + Coverage 20% + Calibration 10% + Format 10%。\n"
            "输出 JSON: {\"q_match\": 0-2, \"law_correct\": 0-2, \"coverage\": 0-2, \"calibration\": 0-2, \"format\": 0-2}\n"
            "只输出 JSON。"
        )},
        {"role": "user", "content": f"问题：{question}\n\n答案：{answer[:1500]}"},
    ]
    try:
        data = judge.json(messages)
        if not isinstance(data, dict): return 0
        weights = {"q_match": 0.3, "law_correct": 0.3, "coverage": 0.2, "calibration": 0.1, "format": 0.1}
        weighted = sum(data.get(k, 0) * w for k, w in weights.items())
        return round(weighted / 2 * 100, 1)
    except Exception:
        return 0.0


def faithfulness_score_judge(question, answer, evidence, judge):
    """Faithfulness 4 档评测。"""
    messages = [
        {"role": "system", "content": (
            "你是民航法律答案忠实性评估器。\n"
            "判断答案内容与参考证据的对齐度：\n"
            "- faithful_pct: 与证据直接一致的声明占比\n"
            "- partial_pct: 部分一致的占比\n"
            "- unverifiable_pct: 证据未覆盖的占比\n"
            "- hallucinated_pct: 与证据矛盾的占比\n"
            "输出 JSON: {\"faithful_pct\": 0.0-1.0, \"partial_pct\": 0.0-1.0, \"unverifiable_pct\": 0.0-1.0, \"hallucinated_pct\": 0.0-1.0}\n"
            "四个数之和=1.0。只输出 JSON。"
        )},
        {"role": "user", "content": f"问题：{question}\n\n答案：{answer[:1500]}\n\n参考证据：{evidence[:500]}"},
    ]
    try:
        data = judge.json(messages)
        if not isinstance(data, dict): return {"faithful_pct": 0, "hallucinated_pct": 0}
        return data
    except Exception:
        return {"faithful_pct": 0, "hallucinated_pct": 0}


def print_table(by_config, all_basic, citation_validity, llm_results=None):
    """打印综合对比表。"""
    print(f"\n{'='*120}")
    print(f"消融测试综合报告")
    print(f"{'='*120}")
    cv_rate = citation_validity.get("validity_rate", 0)

    headers = ["config_id", "n", "sup_rate", "elapsed", "ev_cnt", "ans_len", "reflex", "refusal", "cv_rate"]
    if llm_results:
        headers += ["lemaj_corr", "mqs", "faithful", "hallucinated"]
    print(" | ".join(f"{h:>12}" for h in headers))
    print("-" * 140)

    rows = []
    for cfg_id, runs in by_config.items():
        b = all_basic.get(cfg_id, {})
        row = {
            "config_id": cfg_id[:40],
            "n": b.get("n_questions", 0),
            "sup_rate": b.get("supported_rate", 0),
            "elapsed": b.get("avg_elapsed_sec", 0),
            "ev_cnt": b.get("avg_evidence_count", 0),
            "ans_len": b.get("avg_answer_len", 0),
            "reflex": b.get("avg_reflexion_iters", 0),
            "refusal": b.get("refusal_rate", 0),
            "cv_rate": cv_rate,  # 同一个数据集所以相同
        }
        if llm_results and cfg_id in llm_results:
            lr = llm_results[cfg_id]
            row.update({
                "lemaj_corr": lr.get("lemaj_correct", 0),
                "mqs": lr.get("mqs_weighted", 0),
                "faithful": lr.get("faithful_pct", 0),
                "hallucinated": lr.get("hallucinated_pct", 0),
            })
        rows.append(row)

    # 按 supported_rate 排序
    rows.sort(key=lambda r: -r["sup_rate"])

    for r in rows:
        line = (
            f"{r['config_id']:>40} | {r['n']:>12} | {r['sup_rate']:>12.3f} | "
            f"{r['elapsed']:>12.1f} | {r['ev_cnt']:>12.1f} | {r['ans_len']:>12.0f} | "
            f"{r['reflex']:>12.2f} | {r['refusal']:>12.3f} | {r['cv_rate']:>12.3f}"
        )
        if llm_results and "lemaj_corr" in r:
            line += f" | {r['lemaj_corr']:>12.3f} | {r['mqs']:>12.1f} | {r['faithful']:>12.3f} | {r['hallucinated']:>12.3f}"
        print(line)

    print("-" * 140)
    print(f"Citation validity: {citation_validity.get('valid_citations', 0)}/{citation_validity.get('total_citations', 0)} = {cv_rate:.1%}")
    if citation_validity.get("invalid_examples"):
        print(f"无效引用示例（前 10）:")
        for ex in citation_validity["invalid_examples"]:
            print(f"  {ex[0]} {ex[1]}: 《{ex[2]}》 第{ex[3]}条")

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", help="ablation_runs 目录路径")
    parser.add_argument("--skip-llm", action="store_true", help="跳过 LLM 评测（只用基础指标 + 引用）")
    parser.add_argument("--top", type=int, default=None, help="只评测 top N 配置（按 supported_rate 排）")
    parser.add_argument("--report", type=str, default=None, help="报告输出 JSON 路径")
    args = parser.parse_args()

    print(f"=" * 70)
    print(f"消融测试分析器")
    print(f"=" * 70)
    print(f"输入: {args.out_dir}")

    # 1. 加载所有 run
    runs = load_runs(args.out_dir)
    print(f"加载 {len(runs)} 条记录")

    if not runs:
        print("ERROR: 没有数据")
        return

    by_config = group_by_config(runs)
    print(f"配置数: {len(by_config)}")
    print(f"题数: {len(set(r['qid'] for r in runs))}")

    # 2. 基础指标（无需 LLM）
    print("\n计算基础指标...")
    all_basic = {cfg_id: compute_basic_metrics(rs) for cfg_id, rs in by_config.items()}

    # 3. 引用有效性（无需 LLM）
    print("计算引用有效性...")
    cv = compute_citation_validity(runs)

    # 4. LLM 评测（可选）
    llm_results = None
    if not args.skip_llm:
        print(f"\n启动 LLM 评测（top={args.top}）...")
        llm_results = run_llm_judges(runs, max_configs=args.top)

    # 5. 打印报告
    rows = print_table(by_config, all_basic, cv, llm_results)

    # 6. 写 JSON 报告
    if args.report:
        report = {
            "out_dir": args.out_dir,
            "n_configs": len(by_config),
            "n_runs": len(runs),
            "citation_validity": cv,
            "per_config": {
                cfg_id: {
                    "basic": all_basic[cfg_id],
                    "llm": (llm_results or {}).get(cfg_id),
                }
                for cfg_id in by_config
            },
        }
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n报告: {args.report}")


if __name__ == "__main__":
    main()
