"""重复10次同一问题，分析法律引用一致性。"""

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

# 确保项目根目录在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from legalbot.agents import LegalOrchestrator
from legalbot.llm import LLMClient


QUESTION = "不在《运行规范》的机场是否可以去备降"
ROUNDS = 10


def extract_citations_from_text(text: str) -> list[str]:
    """从答案文本中提取所有看起来像法条引用的片段。"""
    patterns = [
        r'《([^》]+)》',                           # 书名号法条名
        r'第[一二三四五六七八九十百千\d]+条',       # 第X条
        r'第[一二三四五六七八九十百千\d]+章',       # 第X章
        r'第[一二三四五六七八九十百千\d]+款',       # 第X款
        r'第[一二三四五六七八九十百千\d]+项',       # 第X项
    ]
    found = []
    for pat in patterns:
        found.extend(re.findall(pat, text))
    return found


def extract_law_article_pairs(text: str) -> list[str]:
    """提取 '《法名》第X条' 这样的完整引用对。"""
    pairs = re.findall(r'《([^》]+)》[^。；\n]*?(第[一二三四五六七八九十百千\d]+条)', text)
    return [f"《{name}》{article}" for name, article in pairs]


def run_once(orchestrator: LegalOrchestrator, round_num: int,
             total_start: float, per_round_times: list[float]) -> dict:
    """执行一轮问答，返回结构化结果。"""
    print(f"\n{'='*60}")
    print(f"[{round_num}/{ROUNDS}] 第 {round_num} 轮开始...", flush=True)

    start = time.time()
    result = orchestrator.answer(QUESTION)
    elapsed = time.time() - start
    per_round_times.append(elapsed)

    # 累计耗时 & 预估剩余
    total_elapsed = time.time() - total_start
    avg_per_round = total_elapsed / round_num
    eta = avg_per_round * (ROUNDS - round_num)
    print(f"[{round_num}/{ROUNDS}] 本轮耗时: {elapsed:.1f}s | 累计: {total_elapsed:.1f}s | 预估剩余: {eta:.0f}s", flush=True)

    answer = result.answer

    # 从结构化数据提取
    evidence_laws = []
    evidence_articles = []
    for ev in (result.evidence or []):
        evidence_laws.append(ev.law_title)
        evidence_articles.append(f"{ev.law_title} {ev.article}")

    citation_nodes = []
    for cit in (result.citations or []):
        citation_nodes.append({
            "claim": cit.claim,
            "law_id": cit.law_id,
            "node_id": cit.node_id,
            "status": cit.status,
            "confidence": cit.confidence,
        })

    # 从文本提取引用
    text_laws = extract_citations_from_text(answer)
    text_pairs = extract_law_article_pairs(answer)

    summary = {
        "round": round_num,
        "elapsed": round(elapsed, 1),
        "answer_len": len(answer),
        "answer_preview": answer[:300],
        "evidence_count": len(result.evidence or []),
        "evidence_laws": evidence_laws,
        "evidence_articles": evidence_articles,
        "citation_count": len(result.citations or []),
        "citation_nodes": citation_nodes,
        "text_law_refs": text_laws,
        "text_law_article_pairs": text_pairs,
        "reflexion_iterations": result.reflexion_iterations,
        "conflicts": [
            {"laws": c.law_titles, "reason": c.reason}
            for c in (result.conflicts or [])
        ],
    }

    print(f"[{round_num}/{ROUNDS}] 答案{len(answer)}字 | 证据{len(result.evidence or [])}条 | 校验{len(result.citations or [])}条 | 自检{result.reflexion_iterations}轮", flush=True)
    print(f"[{round_num}/{ROUNDS}] 文本法条: {text_pairs}", flush=True)

    return summary


def analyze(results: list[dict]) -> str:
    """生成一致性分析报告。"""
    lines = []
    lines.append("=" * 70)
    lines.append(f"法律引用一致性分析报告")
    lines.append(f"测试问题: 「{QUESTION}」")
    lines.append(f"测试轮数: {ROUNDS}")
    lines.append("=" * 70)

    # ── 1. 基础统计 ──
    lines.append("\n## 一、基础统计")
    lengths = [r["answer_len"] for r in results]
    lines.append(f"  答案长度: 最短={min(lengths)}字, 最长={max(lengths)}字, 均值={sum(lengths)/len(lengths):.0f}字")
    times = [r["elapsed"] for r in results]
    lines.append(f"  响应时间: 最快={min(times):.1f}s, 最慢={max(times):.1f}s, 均值={sum(times)/len(times):.1f}s")
    ev_counts = [r["evidence_count"] for r in results]
    lines.append(f"  证据数量: 最少={min(ev_counts)}, 最多={max(ev_counts)}, 均值={sum(ev_counts)/len(ev_counts):.1f}")
    cit_counts = [r["citation_count"] for r in results]
    lines.append(f"  引用校验: 最少={min(cit_counts)}, 最多={max(cit_counts)}, 均值={sum(cit_counts)/len(cit_counts):.1f}")
    ref_iters = [r["reflexion_iterations"] for r in results]
    lines.append(f"  自检轮数: {Counter(ref_iters)}")

    # ── 2. 证据层面：检索到的法条一致性 ──
    lines.append("\n## 二、证据层一致性（检索阶段）")
    all_evidence_articles = []
    for r in results:
        all_evidence_articles.extend(r["evidence_articles"])
    ev_counter = Counter(all_evidence_articles)
    lines.append(f"  检索到的法条共 {len(ev_counter)} 种，出现频次:")
    for art, count in ev_counter.most_common(20):
        rounds_with = sum(1 for r in results if art in r["evidence_articles"])
        lines.append(f"    {art}: 出现{count}次 (覆盖{rounds_with}/{ROUNDS}轮)")

    # 每轮证据完全一致率
    ev_sets = [frozenset(r["evidence_articles"]) for r in results]
    unique_ev_sets = set(ev_sets)
    lines.append(f"  证据组合唯一数: {len(unique_ev_sets)} (10轮中有{len(unique_ev_sets)}种不同的证据组合)")
    if len(unique_ev_sets) == 1:
        lines.append(f"  ✅ 所有轮次检索到完全相同的证据")
    else:
        lines.append(f"  ⚠️ 证据组合不一致")

    # ── 3. 引用校验层面 ──
    lines.append("\n## 三、引用校验层一致性")
    # 按 node_id 统计校验状态
    node_status_map: dict[str, list[str]] = {}
    node_conf_map: dict[str, list[float]] = {}
    for r in results:
        for cn in r["citation_nodes"]:
            nid = cn["node_id"]
            if nid:
                node_status_map.setdefault(nid, []).append(cn["status"])
                node_conf_map.setdefault(nid, []).append(cn["confidence"])

    if node_status_map:
        lines.append(f"  被校验的 node_id 共 {len(node_status_map)} 个:")
        for nid, statuses in sorted(node_status_map.items()):
            status_counter = Counter(statuses)
            confs = node_conf_map.get(nid, [])
            avg_conf = sum(confs) / len(confs) if confs else 0
            status_str = ", ".join(f"{s}:{c}次" for s, c in status_counter.most_common())
            lines.append(f"    {nid}: [{status_str}] 平均置信度={avg_conf:.2f}")
            if len(status_counter) > 1:
                lines.append(f"      ⚠️ 同一法条在不同轮次中校验状态不一致!")
    else:
        lines.append("  无结构化引用校验数据")

    # ── 4. 文本层面：最终答案中引用的法律 ──
    lines.append("\n## 四、答案文本引用一致性")
    # 法律名引用
    all_text_laws = []
    for r in results:
        all_text_laws.extend(r["text_law_refs"])
    text_law_counter = Counter(all_text_laws)
    lines.append(f"  答案中引用的法律名共 {len(text_law_counter)} 种:")
    for law, count in text_law_counter.most_common():
        rounds_with = sum(1 for r in results if law in r["text_law_refs"])
        lines.append(f"    《{law}》: {count}次 (覆盖{rounds_with}/{ROUNDS}轮)")

    # 法条引用对
    all_pairs = []
    for r in results:
        all_pairs.extend(r["text_law_article_pairs"])
    pair_counter = Counter(all_pairs)
    lines.append(f"\n  答案中引用的具体法条共 {len(pair_counter)} 种:")
    for pair, count in pair_counter.most_common():
        rounds_with = sum(1 for r in results if pair in r["text_law_article_pairs"])
        lines.append(f"    {pair}: {count}次 (覆盖{rounds_with}/{ROUNDS}轮)")

    # ── 5. 每轮答案引用完整性对比 ──
    lines.append("\n## 五、逐轮引用详情")
    for r in results:
        pairs_str = ", ".join(r["text_law_article_pairs"]) if r["text_law_article_pairs"] else "(无明确法条引用)"
        lines.append(f"  轮{r['round']}: {pairs_str}")

    # ── 6. 不一致性根源分析 ──
    lines.append("\n## 六、不一致性根源分析")

    # 检索层差异
    ev_per_round = [set(r["evidence_articles"]) for r in results]
    all_evs = set()
    for s in ev_per_round:
        all_evs |= s
    always_present = all_evs
    for s in ev_per_round:
        always_present &= s
    sometimes_present = all_evs - always_present
    lines.append(f"  1. 检索层:")
    lines.append(f"     每轮都出现的法条({len(always_present)}): {sorted(always_present) if always_present else '无'}")
    lines.append(f"     部分轮次出现的法条({len(sometimes_present)}): {sorted(sometimes_present) if sometimes_present else '无'}")
    if sometimes_present:
        lines.append(f"     → 检索阶段的不确定性来源: BM25/向量检索的排序存在微小差异，top-k 截断导致边界法条被随机纳入/排除")

    # LLM层差异
    lines.append(f"  2. LLM生成层:")
    pair_sets = [set(r["text_law_article_pairs"]) for r in results]
    all_pair_sets = set()
    for s in pair_sets:
        all_pair_sets |= s
    always_pairs = all_pair_sets
    for s in pair_sets:
        always_pairs &= s
    variable_pairs = all_pair_sets - always_pairs
    if variable_pairs:
        lines.append(f"     LLM稳定引用的法条: {sorted(always_pairs) if always_pairs else '无'}")
        lines.append(f"     LLM不稳定引用的法条: {sorted(variable_pairs)}")
        lines.append(f"     → 即使temperature=0.0，DeepSeek API仍可能因服务端负载均衡、KV cache命中率等因素产生非确定性输出")
    else:
        lines.append(f"     ✅ 所有轮次引用的法条完全一致")

    # ── 7. 结论 ──
    lines.append("\n## 七、结论")

    # 计算一致性分数
    if always_pairs and all_pair_sets:
        stability = len(always_pairs) / len(all_pair_sets) * 100
    elif all_pair_sets:
        stability = 0
    else:
        stability = 100

    lines.append(f"  法条引用稳定率: {stability:.0f}% ({len(always_pairs)}/{len(all_pair_sets)} 个引用在所有轮次中都出现)")

    # 检查核心法律结论是否一致
    lines.append("\n  各轮答案核心观点:")
    for r in results:
        preview = r["answer_preview"].replace("\n", " ")[:200]
        lines.append(f"    轮{r['round']}: {preview}...")

    lines.append(f"\n{'='*70}")
    lines.append("报告结束")
    lines.append(f"{'='*70}")

    return "\n".join(lines)


def main():
    total_start = time.time()
    print(f"开始一致性测试: 问题=「{QUESTION}」, 轮数={ROUNDS}", flush=True)

    orchestrator = LegalOrchestrator(logger=None)

    results = []
    per_round_times: list[float] = []
    for i in range(1, ROUNDS + 1):
        r = run_once(orchestrator, i, total_start, per_round_times)
        results.append(r)
        if i < ROUNDS:
            print(f"[{i}/{ROUNDS}] 暂停2秒...", flush=True)
            time.sleep(2)

    total_elapsed = time.time() - total_start
    print(f"\n全部{ROUNDS}轮完成, 总耗时: {total_elapsed:.1f}s, 每轮: {per_round_times}", flush=True)

    # 生成分析报告
    report = analyze(results)

    # 保存原始数据
    output_dir = PROJECT_ROOT / "data" / "trace"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "consistency_raw.json"
    report_path = output_dir / "consistency_report.txt"

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n原始数据已保存: {raw_path}")
    print(f"分析报告已保存: {report_path}")
    print(f"\n{report}")


if __name__ == "__main__":
    main()
