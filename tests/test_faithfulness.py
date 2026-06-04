"""第 1 层评测：Faithfulness + Completeness 自动评测

用独立 LLM 做答案级 faithfulness + completeness 检测。
流程：answer → 分解 claims → 四档评判 → 检测遗漏

四档：faithful / partial / unverifiable / hallucinated
遗漏：基于问题+证据，检测答案应该提但没提的关键信息

用法：
    .venv/bin/python tests/test_faithfulness.py [结果JSON路径]
    默认读取 tests/test30_20260602_111235.json

环境变量：
    FAITHFULNESS_API_KEY — 评测用 LLM API key
    FAITHFULNESS_BASE_URL — 评测用 LLM base URL
    FAITHFULNESS_MODEL — 评测用模型名
"""

import csv
import json
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from legalbot.llm import LLMClient, LLMConfig, load_env
import legalbot.config as config

load_env()

# ── 拒答关键词 ──
REFUSAL_KEYWORDS = [
    "无法确定", "无法回答", "证据不足", "未包含相关",
    "未涉及", "未找到", "没有找到", "未能找到",
    "无法提供", "无法确认", "无法判断", "现有证据不足以",
]


def is_refusal(answer: str) -> bool:
    return any(kw in answer for kw in REFUSAL_KEYWORDS)


def get_evaluator_llm() -> LLMClient:
    """获取评测用 LLM（优先使用与系统不同的模型）。"""
    provider = os.getenv("FAITHFULNESS_LLM_PROVIDER", "")
    api_key = os.getenv("FAITHFULNESS_API_KEY", "")
    base_url = os.getenv("FAITHFULNESS_BASE_URL", "")
    model = os.getenv("FAITHFULNESS_MODEL", "")

    if api_key and base_url and model:
        config = LLMConfig(provider="custom_eval", api_key=api_key,
                           base_url=base_url, model=model)
        return LLMClient(config)

    # 降级：复用系统 LLM（不是最佳实践，但能跑）
    print("警告: 未配置 FAITHFULNESS_* 环境变量，使用系统 LLM 做评测")
    print("建议配置不同的模型以避免自我肯定偏差")
    return LLMClient()


def extract_claims(llm: LLMClient, question: str, answer: str) -> list[str]:
    """Step 1: 把答案分解为 atomic claims。"""
    messages = [
        {"role": "system", "content": (
            "你是法律答案声明分解器。只输出 JSON。\n"
            "把给定答案分解为最小独立的事实声明（atomic claims）。\n"
            "规则：\n"
            "1. 每条声明必须是独立可验证的单句\n"
            "2. 不要包含结论性/概括性声明，只保留具体事实\n"
            "3. 如果答案是拒答（如'无法确定'），返回空列表\n"
            "4. 忽略格式性文本（如【结论】【法律依据】等标签）"
        )},
        {"role": "user", "content": (
            f"用户问题：{question}\n\n"
            f"答案：\n{answer}\n\n"
            "输出 JSON：{\"claims\": [\"声明1\", \"声明2\", ...]}"
        )},
    ]
    data = llm.json(messages)
    return [str(c).strip() for c in data.get("claims", []) if str(c).strip()]


def judge_claims(llm: LLMClient, question: str, claims: list[str],
                 evidence_texts: list[str]) -> list[dict]:
    """Step 2: 对每条 claim 判断 evidence 是否支持。"""
    if not claims:
        return []

    evidence_block = "\n\n".join(
        f"[证据{i+1}] {t[:config.EVAL_JUDGE_MAX_CHARS]}" for i, t in enumerate(evidence_texts)
    )
    claims_block = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))

    messages = [
        {"role": "system", "content": (
            "你是法律答案忠实性评估器。只输出 JSON。\n"
            "任务：判断答案中的每条声明是否能从提供的证据中推导出来。\n\n"
            "判定标准（四档）：\n"
            "- faithful: 声明能从证据中直接找到支撑（允许合理的语义等价推断）\n"
            "- partial: 声明的核心信息有证据支撑，但部分细节无法验证\n"
            "- unverifiable: 证据完全没有覆盖该声明涉及的主题。"
            "证据对该话题沉默，声明可能正确也可能错误，无法判断。\n"
            "  典型场景：声明引用了某法条的具体内容，但证据中没有该法条的原文。\n"
            "- hallucinated: 证据覆盖了该主题但与声明矛盾，"
            "或声明编造了证据中不存在的具体数字/时间/细节。\n"
            "  关键区别：证据讨论了同一话题但给出了不同信息 → hallucinated；"
            "证据根本没讨论这个话题 → unverifiable。\n\n"
            "【来源标签处理】声明前如有 [合理推断] 或 [建议] 标签：\n"
            "- 这是 LLM 自身承认的'非直接证据'声明（基于原则的合理延伸或通用法律知识）\n"
            "- 默认判 partial（不判 unverifiable，因为 LLM 已声明这是推断而非证据缺失）\n"
            "- 例外：若声明内容与证据主题完全无关、完全是自由发挥 → 仍判 unverifiable\n\n"
            "注意：\n"
            "- 严格区分 unverifiable 和 hallucinated，这是最重要的区分\n"
            "- 如果声明引用了具体的法条（法规名+条号），但证据中没有该法条原文 → unverifiable\n"
            "- 如果声明是泛泛而谈的法律常识而非来自证据 → unverifiable\n"
            "- 只有当证据明确包含相关内容但声明歪曲或编造了细节时 → hallucinated"
        )},
        {"role": "user", "content": (
            f"用户问题：{question}\n\n"
            f"证据：\n{evidence_block}\n\n"
            f"待评估声明（注意每条声明前的 [合理推断]/[建议] 标签）：\n{claims_block}\n\n"
            "输出 JSON：\n"
            '{"judgments": [{"claim_id": 1, "status": "faithful|partial|unverifiable|hallucinated", '
            '"reason": "判断依据（简短）"}]}'
        )},
    ]
    data = llm.json(messages)
    return data.get("judgments", [])


def detect_missing(llm: LLMClient, question: str, answer: str,
                   evidence_texts: list[str]) -> list[str]:
    """Step 3: 检测答案遗漏的关键信息（LeMAJ LDP missing）。"""
    if not evidence_texts:
        return []

    evidence_block = "\n\n".join(
        f"[证据{i+1}] {t[:400]}" for i, t in enumerate(evidence_texts)
    )

    messages = [
        {"role": "system", "content": (
            "你是法律答案完整性评估器。只输出 JSON。\n"
            "任务：基于用户问题和提供的证据，找出答案中**遗漏的关键信息**。\n\n"
            "判定遗漏的标准：\n"
            "- 证据中包含与问题直接相关的具体规定/数字/条件，但答案没有提到\n"
            "- 问题是列举类（'有哪些''什么要求'），但答案遗漏了证据中的项\n"
            "- 不要把以下情况标为遗漏：\n"
            "  - 答案已提到但表述不同（语义等价不算遗漏）\n"
            "  - 证据中有但与问题无关的信息\n"
            "  - 过于细碎的程序性/格式性条款\n"
            "  - 证据中也没有覆盖的信息（那是检索缺口，不是答案遗漏）\n\n"
            "每条遗漏用一句简洁的话描述。"
        )},
        {"role": "user", "content": (
            f"用户问题：{question}\n\n"
            f"答案：\n{answer[:2000]}\n\n"
            f"证据：\n{evidence_block}\n\n"
            "输出 JSON：{\"missing\": [\"遗漏1\", \"遗漏2\", ...]}\n"
            "如果没有遗漏，返回空列表。"
        )},
    ]
    data = llm.json(messages)
    return [str(m).strip() for m in data.get("missing", []) if str(m).strip()]


def load_evidence_texts(result: dict) -> list[str]:
    """从测试结果中加载证据文本。

    优先使用 answer_full 中引用的证据，
    降级使用 evidence_articles 作为参考。
    """
    # 从系统测试结果中，我们只有 evidence_articles（标题列表）
    # 需要从 IndexRepository 读取原文
    articles = result.get("evidence_articles", "")
    if not articles:
        return []

    from legalbot.retrieval import IndexRepository, read_law_node

    docs = IndexRepository.documents()
    texts = []
    for art in articles.split(" | ")[:12]:
        art = art.strip()
        if not art:
            continue
        for doc in docs:
            for node in doc.flatten():
                if node.title and node.title.strip() == art.strip():
                    # 优先读原文（text），再降级 summary
                    text = node.text or node.summary or ""
                    if not text or text.strip() == node.title.strip():
                        # 尝试从源文件读取完整条文
                        node_result = read_law_node(doc.law_id, node.node_id,
                                                     include_context=False)
                        if node_result.get("found"):
                            text = node_result["text"]
                    if text:
                        texts.append(f"{doc.title} {node.title}: {text[:config.EVAL_EVIDENCE_MAX_CHARS]}")
                        break
            if len(texts) >= 12:
                break
        if len(texts) >= 12:
            break
    return texts


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", nargs="?",
                        default=str(PROJECT_ROOT / "tests" / "test30_20260602_111235.json"))
    parser.add_argument("--skip", type=int, default=0,
                        help="跳过前 N 题（用于断点续跑）")
    parser.add_argument("--only", type=str, default="",
                        help="只跑指定题号，逗号分隔，如 Q03,Q28")
    args = parser.parse_args()

    json_path = Path(args.json_path)
    if not json_path.exists():
        print(f"文件不存在: {json_path}")
        return

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    llm = get_evaluator_llm()
    print(f"评测模型: {llm.config.model}\n")

    # 筛选题号
    if args.only:
        only_set = set(args.only.split(","))
        results = [r for r in results if r["question_id"] in only_set]
    if args.skip:
        results = results[args.skip:]

    all_judgments = []
    total_claims = 0
    total_missing = 0
    faithful = 0
    partial = 0
    unverifiable = 0
    hallucinated = 0
    refusals = 0

    for i, r in enumerate(results):
        qid = r["question_id"]
        question = r["question"]
        answer = r.get("answer_full", "")

        refusal = is_refusal(answer)
        if refusal:
            refusals += 1
            print(f"[{qid}] 拒答，跳过")
            all_judgments.append({
                "qid": qid, "question": question,
                "refusal": True, "claims": [], "judgments": [],
            })
            continue

        # 加载证据
        print(f"[{qid}] {question[:40]}... ", end="", flush=True)
        evidence_texts = load_evidence_texts(r)
        print(f"证据={len(evidence_texts)}条 ", end="", flush=True)

        # Step 1: 分解 claims
        try:
            claims = extract_claims(llm, question, answer)
        except Exception as e:
            print(f"claim分解失败: {e}")
            claims = []

        if not claims:
            print("无声明")
            all_judgments.append({
                "qid": qid, "question": question,
                "refusal": False, "claims": [], "judgments": [],
            })
            continue

        print(f"claims={len(claims)} ", end="", flush=True)

        # Step 2: 评判
        try:
            judgments = judge_claims(llm, question, claims, evidence_texts)
        except Exception as e:
            print(f"评判失败: {e}")
            judgments = []

        # 统计
        q_faithful = sum(1 for j in judgments if j.get("status") == "faithful")
        q_partial = sum(1 for j in judgments if j.get("status") == "partial")
        q_unverifiable = sum(1 for j in judgments if j.get("status") == "unverifiable")
        q_hallucinated = sum(1 for j in judgments if j.get("status") == "hallucinated")

        # Step 3: 检测遗漏
        try:
            missing = detect_missing(llm, question, answer, evidence_texts)
        except Exception as e:
            print(f"(missing检测失败: {e}) ", end="", flush=True)
            missing = []

        q_missing = len(missing)
        total_claims += len(claims)
        total_missing += q_missing
        faithful += q_faithful
        partial += q_partial
        unverifiable += q_unverifiable
        hallucinated += q_hallucinated

        print(f"→ f={q_faithful} p={q_partial} u={q_unverifiable} h={q_hallucinated} missing={q_missing}")

        all_judgments.append({
            "qid": qid,
            "question": question,
            "refusal": False,
            "claims": claims,
            "judgments": judgments,
            "missing": missing,
        })

        time.sleep(1)  # 避免速率限制

    # 汇总
    answered = len(results) - refusals
    print("\n" + "=" * 70)
    print("Faithfulness 评测报告")
    print("=" * 70)
    print(f"总题数: {len(results)}")
    print(f"拒答: {refusals} ({refusals/len(results)*100:.0f}%)")
    print(f"有效回答: {answered}")
    print()
    if total_claims > 0:
        print(f"总声明数: {total_claims}")
        print(f"  faithful:      {faithful:3d} ({faithful/total_claims*100:5.1f}%)")
        print(f"  partial:       {partial:3d} ({partial/total_claims*100:5.1f}%)")
        print(f"  unverifiable:  {unverifiable:3d} ({unverifiable/total_claims*100:5.1f}%)")
        print(f"  hallucinated:  {hallucinated:3d} ({hallucinated/total_claims*100:5.1f}%)")
        print(f"  missing:       {total_missing:3d} 条遗漏信息")
        print()
        print(f"★ 真实幻觉率 (hallucinated/total): {hallucinated/total_claims*100:.1f}%")
        print(f"★ 检索缺口率 (unverifiable/total): {unverifiable/total_claims*100:.1f}%")
        print(f"★ 忠实率 (faithful/total): {faithful/total_claims*100:.1f}%")
        print(f"★ 可接受率 (faithful+partial/total): {(faithful+partial)/total_claims*100:.1f}%")
        # LeMAJ Recall: correct / (correct + missing)
        correct = faithful + partial
        if correct + total_missing > 0:
            recall = correct / (correct + total_missing)
            precision = correct / total_claims
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            print(f"★ 完整率 Recall (correct/(correct+missing)): {recall*100:.1f}%")
            print(f"★ 精确率 Precision (correct/total_claims): {precision*100:.1f}%")
            print(f"★ F1: {f1*100:.1f}%")

    # 保存
    correct = faithful + partial
    prec = correct / total_claims if total_claims else 0
    rec = correct / (correct + total_missing) if (correct + total_missing) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    output_path = json_path.parent / (json_path.stem + "_faithfulness.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "source": json_path.name,
            "evaluator_model": llm.config.model,
            "summary": {
                "total_questions": len(results),
                "refusals": refusals,
                "answered": answered,
                "total_claims": total_claims,
                "total_missing": total_missing,
                "faithful": faithful,
                "partial": partial,
                "unverifiable": unverifiable,
                "hallucinated": hallucinated,
                "faithfulness_rate": faithful / total_claims if total_claims else None,
                "hallucination_rate": hallucinated / total_claims if total_claims else None,
                "retrieval_gap_rate": unverifiable / total_claims if total_claims else None,
                "completeness_recall": rec if rec else None,
                "precision": prec if prec else None,
                "f1": f1 if f1 else None,
            },
            "per_question": all_judgments,
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n详细结果已保存: {output_path}")


if __name__ == "__main__":
    main()
