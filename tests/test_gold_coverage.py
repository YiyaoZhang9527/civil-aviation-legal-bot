"""Gold Answer Coverage 评测：LLM 读相关法条自动生成标准答案，评估 bot 覆盖率。

解决盲区：
- 现有测试只能抓"瞎编"(citation validity)和"不忠实"(faithfulness)
- 抓不到"答偏"(答了证据里有但与问题无关的内容)
- 抓不到"答漏"(漏了关键要点)

原理：
1. 对每道题，用更宽的检索（top_k=30）拉证据
2. 用一个独立 LLM 读证据后生成"权威答案"和 5-15 个"关键要点"
3. 另一个 LLM judge 评估 bot 答案覆盖了哪些关键要点
4. coverage_rate = 覆盖要点 / 总要点
"""

import datetime
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import os

from legalbot.agents import (
    IssueAnalysis,
    RewriteAgent,
    SubjectAnalysis,
    SynthesisAgent,
)
from legalbot.llm import LLMClient, LLMConfig
from legalbot.retrieval import IndexRepository, search_index_tree
from legalbot.utils import normalize_text


def make_evaluator_llm() -> LLMClient:
    """评测用 LLM：默认与 bot 相同（LEGALBOT_LLM_MODEL），可被环境变量覆盖。

    环境变量：
    - EVAL_LLM_PROVIDER：deepseek / custom1（硅基流动 Qwen）
    - EVAL_LLM_MODEL：模型名（覆盖 DEEPSEEK_MODELS 第一项）

    用法：
    - JUDGE=flash: LEGALBOT_LLM_MODEL=deepseek-v4-flash
    - JUDGE=pro: EVAL_LLM_MODEL=deepseek-v4-pro
    - JUDGE=qwen: EVAL_LLM_PROVIDER=custom1 EVAL_LLM_MODEL=Qwen/Qwen2.5-72B-Instruct
    """
    provider_override = os.getenv("EVAL_LLM_PROVIDER")
    model_override = os.getenv("EVAL_LLM_MODEL")

    if provider_override == "custom1":
        # 硅基流动：用 Qwen 当 cross-family judge
        load_env = __import__("legalbot.llm", fromlist=["load_env"]).load_env
        load_env()
        return LLMClient(LLMConfig(
            provider="custom1",
            api_key=os.getenv("CUSTOM_API_KEY_1", ""),
            base_url=os.getenv("CUSTOM_API_BASE_URL_1", "").rstrip("/"),
            model=model_override or _first_model(os.getenv("CUSTOM_API_MODELS_1", "")),
        ))

    if model_override:
        # 同公司但不同模型（flash vs pro）
        os.environ["LEGALBOT_LLM_MODEL"] = model_override
    return LLMClient()


def _first_model(s: str) -> str:
    """从逗号分隔的模型列表取第一个。"""
    for token in s.split(","):
        token = token.strip()
        if token:
            return token
    return ""


def _ensure_tree_ready(retries: int = 5, delay: float = 2.0) -> None:
    """确保 TreeRetriever 构建完成（否则 search 会 fallback 到 flat 返回垃圾）。"""
    import time
    for _ in range(retries):
        tree = IndexRepository.tree_index()
        if tree is not None and tree.available:
            return
        time.sleep(delay)


def _hinted_search(query: str, law_hints: list[str], article_hints: list[str], top_k: int) -> list:
    """带 hint 的检索——bot 的核心能力，gold 必须复刻。"""
    return search_index_tree(
        query=query,
        top_k=top_k,
        law_hints=law_hints or None,
        article_hints=article_hints or None,
    )


def _multi_query_search(
    question: str,
    queries: list[str],
    law_hints: list[str],
    article_hints: list[str],
    top_k: int,
) -> list:
    """Multi-query + hint retrieval：与 bot 检索路径对齐。"""
    all_evidence = []
    seen: set[tuple[str, str]] = set()
    for q in queries:
        if not q or not q.strip():
            continue
        evs = _hinted_search(q, law_hints, article_hints, top_k)
        for ev in evs:
            key = (ev.law_id, ev.node_id)
            if key not in seen:
                seen.add(key)
                all_evidence.append(ev)
    all_evidence.sort(key=lambda e: e.score, reverse=True)
    return all_evidence


# ── Gold 生成 ────────────────────────────────────────────


def generate_gold_answer(question: str, llm: LLMClient, top_k: int = 30) -> dict:
    """对单道题生成 gold 答案 + 关键要点。

    检索策略（与 bot 完全对齐 + 扩展）：
    1. RewriteAgent 改写 query 并推荐 law_hints / article_hints
    2. 每个 query 用 hints 检索 top_k 条
    3. 合并去重，按分数排序取 top_k
    """
    _ensure_tree_ready()

    rewrite_agent = RewriteAgent(llm)
    try:
        rewrite = rewrite_agent.rewrite_queries(
            question, SubjectAnalysis(), IssueAnalysis()
        )
        queries = [normalize_text(question)] + [
            normalize_text(q) for q in rewrite.queries if q.strip()
        ]
        # 去重
        seen = set()
        unique_queries = []
        for q in queries:
            if q and q not in seen:
                seen.add(q)
                unique_queries.append(q)
        queries = unique_queries[:4]
        law_hints = list(rewrite.law_hints or [])
        article_hints = list(rewrite.article_hints or [])
    except Exception:
        queries = [question]
        law_hints = []
        article_hints = []

    evidence = _multi_query_search(question, queries, law_hints, article_hints, top_k=top_k)
    evidence = evidence[:top_k]

    evidence_text = "\n\n".join(
        f"[{i+1}] 《{ev.law_title}》{ev.article}\n{ev.text[:1500]}"
        for i, ev in enumerate(evidence)
    )

    if not evidence_text.strip():
        return {"answer": "（无证据）", "key_points": []}

    messages = [
        {"role": "system", "content": (
            "你是中国民航法律专家。基于给定证据，对用户问题给出最全面、最权威的答案，"
            "并提取关键要点。\n\n"
            "要求：\n"
            "1. key_points 必须是 5-15 条具体、可独立验证的事实声明\n"
            "2. 不要给法律结论（如'可能违法''应当赔偿'），只列客观事实\n"
            "3. 答案应穷举证据中所有相关内容（不只是 top 1）\n"
            "4. 关键要点覆盖：法律依据（法规名+条号）、具体情形、数字条件、程序要求、例外情形"
        )},
        {"role": "user", "content": (
            f"问题：{question}\n\n证据（共 {len(evidence)} 条）：\n{evidence_text}\n\n"
            "请输出 JSON：\n"
            '{"answer": "完整答案", "key_points": ["要点1", "要点2", ...]}'
        )},
    ]
    data = llm.json(messages)
    return {
        "answer": str(data.get("answer", "")),
        "key_points": [str(p).strip() for p in data.get("key_points", []) if str(p).strip()],
    }


# ── Coverage 评测 ────────────────────────────────────────


def evaluate_coverage(bot_answer: str, key_points: list[str], llm: LLMClient) -> dict:
    """判断 bot 答案覆盖了 gold 关键要点的哪些。

    确定性预检查：bot 拒答 → 0% 覆盖，不调 LLM。
    """
    if not key_points:
        return {"covered_indices": [], "missing_indices": [], "coverage_rate": 0.0, "reason": "no key points"}

    # 预检查：bot 拒答 → 全未覆盖
    if SynthesisAgent._looks_like_refusal(bot_answer):
        return {
            "covered_indices": [],
            "missing_indices": list(range(1, len(key_points) + 1)),
            "coverage_rate": 0.0,
            "reason": "bot answer is a refusal",
        }

    points_block = "\n".join(f"{i+1}. {p}" for i, p in enumerate(key_points))

    messages = [
        {"role": "system", "content": (
            "你是法律答案覆盖度评估器。判断 bot 答案覆盖了 gold 关键要点的哪些。\n"
            "覆盖标准：该要点的核心信息（事实/数字/法规条号/情形）在 bot 答案中能找到（语义等价即可，不必逐字相同）。\n"
            "注意：\n"
            "- 答案字面没提但隐含包含 → 算覆盖\n"
            "- 答案提了但歪曲/编造 → 仍算覆盖（忠实性另测）\n"
            "- bot 答案拒答或跑题 → 全算未覆盖"
        )},
        {"role": "user", "content": (
            f"Gold 关键要点：\n{points_block}\n\n"
            f"Bot 答案：\n{bot_answer}\n\n"
            "请输出 JSON：\n"
            '{"covered": [1, 3, 5], "missing": [2, 4], "reason": "简要说明"}'
        )},
    ]
    data = llm.json(messages)
    covered = [int(i) for i in data.get("covered", []) if str(i).isdigit()]
    missing = [int(i) for i in data.get("missing", []) if str(i).isdigit()]
    # 容错：补全、补扣
    all_indices = set(range(1, len(key_points) + 1))
    covered_set = set(covered) & all_indices
    # missing = (模型报的 missing) ∪ (没在 covered 也没在 missing 的) - (covered 的)
    missing_set = (set(missing) | (all_indices - covered_set - set(missing))) - covered_set
    return {
        "covered_indices": sorted(covered_set),
        "missing_indices": sorted(missing_set),
        "coverage_rate": len(covered_set) / len(key_points) if key_points else 0.0,
        "reason": str(data.get("reason", "")),
    }


# ── 主流程 ────────────────────────────────────────────────


def find_latest_bot_results() -> Path:
    """找最新的 test30_*.json（bot 答案）。"""
    matches = sorted(PROJECT_ROOT.glob("tests/test30_*.json"))
    # 排除 faithfulness 和 gold 中间产物
    candidates = [m for m in matches if "faithfulness" not in m.name]
    if not candidates:
        raise FileNotFoundError("找不到 test30_*.json，请先跑 test_30questions.py")
    return candidates[-1]


def load_bot_answers(json_path: Path) -> list[dict]:
    """加载 bot 答案列表。"""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    results = data.get("results", [])
    # 过滤掉 error
    return [
        {
            "question_id": r.get("question_id"),
            "category": r.get("category"),
            "question": r.get("question"),
            "bot_answer": r.get("answer_full", ""),
        }
        for r in results
        if r.get("answer_full")
    ]


def main(regen_gold: bool = False, n_questions: int | None = None):
    """主流程：加载 bot → 加载/生成 gold → 评测 coverage → 输出报告。

    n_questions：限制处理的题数（用于快速验证）。
    """
    bot_path = find_latest_bot_results()
    bot_answers = load_bot_answers(bot_path)
    if n_questions is not None:
        bot_answers = bot_answers[:n_questions]
    print(f"加载 bot 答案：{len(bot_answers)} 题 ({bot_path.name})")

    eval_llm = make_evaluator_llm()
    print(f"评测 LLM：provider={eval_llm.config.provider}, model={eval_llm.config.model}")

    # gold 缓存路径（基于 bot 路径 + 评测模型，避免不同模型结果混用）
    model_tag = eval_llm.config.model.replace("/", "_").replace(":", "_")
    gold_path = PROJECT_ROOT / "tests" / f"gold_{bot_path.stem}_{model_tag}.json"
    if gold_path.exists() and not regen_gold:
        gold_data = json.loads(gold_path.read_text(encoding="utf-8"))
        print(f"加载缓存 gold：{len(gold_data['golds'])} 题 ({gold_path.name})")
    else:
        golds = []
        for i, item in enumerate(bot_answers):
            print(f"生成 gold [{i+1}/{len(bot_answers)}] {item['question_id']}...", flush=True)
            gold = generate_gold_answer(item["question"], eval_llm)
            golds.append({
                "question_id": item["question_id"],
                "question": item["question"],
                "gold_answer": gold["answer"],
                "gold_key_points": gold["key_points"],
            })
        gold_data = {
            "generated_at": datetime.datetime.now().isoformat(),
            "bot_source": bot_path.name,
            "judge_model": eval_llm.config.model,
            "golds": golds,
        }
        gold_path.write_text(
            json.dumps(gold_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"已保存 gold 缓存：{gold_path}")

    # coverage 评测
    llm = eval_llm
    coverage_results = []
    for item, gold in zip(bot_answers, gold_data["golds"]):
        cov = evaluate_coverage(item["bot_answer"], gold["gold_key_points"], llm)
        coverage_results.append({
            "question_id": item["question_id"],
            "category": item["category"],
            "question": item["question"],
            "bot_answer": item["bot_answer"],
            "gold_answer": gold["gold_answer"],
            "gold_key_points": gold["gold_key_points"],
            "covered_indices": cov["covered_indices"],
            "missing_indices": cov["missing_indices"],
            "coverage_rate": cov["coverage_rate"],
            "reason": cov["reason"],
        })

    # 汇总
    rates = [r["coverage_rate"] for r in coverage_results]
    avg = sum(rates) / len(rates) if rates else 0.0
    fully = sum(1 for r in rates if r >= 0.9)
    partial = sum(1 for r in rates if 0.5 <= r < 0.9)
    poor = sum(1 for r in rates if r < 0.5)
    refused = sum(1 for r in coverage_results if r["bot_answer"].startswith("【结论】\n以下为系统检索到的可能相关法规"))

    report = {
        "evaluated_at": datetime.datetime.now().isoformat(),
        "bot_source": bot_path.name,
        "gold_source": gold_path.name,
        "summary": {
            "total": len(coverage_results),
            "avg_coverage": round(avg, 3),
            "fully_covered_(>=90%)": fully,
            "partial_(50-90%)": partial,
            "poor_(<50%)": poor,
            "refused_fallback": refused,
        },
        "results": coverage_results,
    }

    out_path = PROJECT_ROOT / "tests" / f"coverage_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== Coverage Summary ===")
    print(f"  Total: {len(coverage_results)}")
    print(f"  Avg coverage: {avg:.1%}")
    print(f"  Fully covered (≥90%): {fully}")
    print(f"  Partial (50-90%): {partial}")
    print(f"  Poor (<50%): {poor}")
    print(f"  Refused (fallback): {refused}")
    print(f"\n  报告：{out_path}")

    return report


if __name__ == "__main__":
    regen = "--regen" in sys.argv
    n = None
    for arg in sys.argv[1:]:
        if arg.startswith("--n="):
            n = int(arg.split("=", 1)[1])
    main(regen_gold=regen, n_questions=n)
