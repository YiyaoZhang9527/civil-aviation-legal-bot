"""渐进式一致性测试：5种flag组合，每种3轮。"""

import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import legalbot.config as cfg
from legalbot.agents import LegalOrchestrator
from legalbot.types import Evidence

QUESTION = "不在《运行规范》的机场是否可以去备降"
ROUNDS = 3

COMBINATIONS = [
    {"name": "Baseline (all off)", "flags": {}},
    {"name": "A1 only", "flags": {"QUERY_GATE_ENABLED": True}},
    {"name": "A1+A2+C1", "flags": {"QUERY_GATE_ENABLED": True, "CROSS_ENCODER_CITATION": True}},
    {"name": "A1+C1+B3", "flags": {"QUERY_GATE_ENABLED": True, "CROSS_ENCODER_CITATION": True, "CONFIDENCE_CUTOFF_ENABLED": True}},
    {"name": "A1+C1+B3+E1 (all on)", "flags": {"QUERY_GATE_ENABLED": True, "CROSS_ENCODER_CITATION": True, "CONFIDENCE_CUTOFF_ENABLED": True, "LEXICAL_REFLEXION_ENABLED": True}},
]


def extract_citation_pairs(text: str) -> list[str]:
    pairs = re.findall(r'《([^》]+)》[^。；\n]*?(第[一二三四五六七八九十百千\d]+条)', text)
    return [f"《{n}》{a}" for n, a in pairs]


def extract_article_numbers(evidence: list[Evidence]) -> list[str]:
    return [ev.article for ev in evidence if ev.article]


def run_one(orch: LegalOrchestrator, round_num: int) -> dict:
    start = time.time()
    result = orch.answer(QUESTION)
    elapsed = time.time() - start

    ev_articles = extract_article_numbers(result.evidence or [])
    cit_statuses = [(c.node_id, c.status, round(c.confidence, 2)) for c in (result.citations or [])]
    text_refs = extract_citation_pairs(result.answer)

    return {
        "round": round_num,
        "elapsed": round(elapsed, 1),
        "answer_len": len(result.answer),
        "evidence_count": len(result.evidence or []),
        "evidence_articles": ev_articles,
        "citation_count": len(result.citations or []),
        "citation_statuses": cit_statuses,
        "text_refs": text_refs,
        "reflexion_iterations": result.reflexion_iterations,
        "answer_preview": result.answer[:200],
    }


def analyze_combo(results: list[dict], combo_name: str) -> str:
    lines = [f"\n{'='*60}", f"组合: {combo_name}", f"{'='*60}"]

    # 每轮摘要
    for r in results:
        lines.append(f"  轮{r['round']}: {r['elapsed']}s, {r['answer_len']}字, "
                      f"证据{r['evidence_count']}条, 校验{r['citation_count']}条, "
                      f"自检{r['reflexion_iterations']}轮")
        lines.append(f"    证据法条: {r['evidence_articles'][:6]}")
        lines.append(f"    文本引用: {r['text_refs']}")

    # 证据组合一致性
    ev_sets = [frozenset(r["evidence_articles"]) for r in results]
    unique_sets = set(ev_sets)
    if len(unique_sets) == 1:
        lines.append(f"  ✅ 证据组合: 3轮完全一致")
    else:
        common = set(ev_sets[0])
        for s in ev_sets[1:]:
            common &= s
        lines.append(f"  ⚠️ 证据组合: {len(unique_sets)}种不同, 交集={sorted(common) if common else '空'}")

    # 引用校验一致性
    all_cit_by_node: dict[str, list[str]] = {}
    for r in results:
        for nid, status, conf in r["citation_statuses"]:
            all_cit_by_node.setdefault(nid, []).append(status)
    inconsistent = {nid: statuses for nid, statuses in all_cit_by_node.items() if len(set(statuses)) > 1}
    if inconsistent:
        lines.append(f"  ⚠️ 校验不一致的node({len(inconsistent)}个):")
        for nid, statuses in list(inconsistent.items())[:5]:
            lines.append(f"    {nid}: {statuses}")
    else:
        lines.append(f"  ✅ 校验一致性: 所有node跨轮次状态相同")

    # 文本引用一致性
    all_refs = []
    for r in results:
        all_refs.extend(r["text_refs"])
    ref_counter = {}
    for ref in all_refs:
        ref_counter[ref] = ref_counter.get(ref, 0) + 1
    stable_refs = [ref for ref, count in ref_counter.items() if count == ROUNDS]
    lines.append(f"  文本法条引用: 共{len(ref_counter)}种, 3轮都出现={len(stable_refs)}")

    # 结论一致性
    for i, r in enumerate(results):
        lines.append(f"  轮{r['round']}结论: {r['answer_preview'][:100]}...")

    return "\n".join(lines)


def main():
    print(f"渐进式一致性测试: {ROUNDS}轮 × {len(COMBINATIONS)}组合")
    print(f"问题: {QUESTION}\n")

    all_results = {}
    report_lines = [f"渐进式一致性测试报告", f"日期: 2026-05-31",
                     f"问题: {QUESTION}", f"每组合轮数: {ROUNDS}", ""]

    for combo in COMBINATIONS:
        # 重置所有flags
        cfg.QUERY_GATE_ENABLED = False
        cfg.CROSS_ENCODER_CITATION = False
        cfg.CONFIDENCE_CUTOFF_ENABLED = False
        cfg.LEXICAL_REFLEXION_ENABLED = False
        # 开启当前组合的flags
        for flag_name, flag_val in combo["flags"].items():
            setattr(cfg, flag_name, flag_val)

        flag_desc = ", ".join(f"{k}=True" for k in combo["flags"]) if combo["flags"] else "all flags=False"
        print(f"\n>>> 组合: {combo['name']} ({flag_desc})")

        orch = LegalOrchestrator(logger=None)
        combo_results = []
        for i in range(1, ROUNDS + 1):
            print(f"  轮{i}...", end=" ", flush=True)
            r = run_one(orch, i)
            combo_results.append(r)
            print(f"完成 {r['elapsed']}s, {r['answer_len']}字")
            time.sleep(1)

        all_results[combo["name"]] = combo_results
        analysis = analyze_combo(combo_results, combo["name"])
        report_lines.append(analysis)
        print(analysis)

    # 总结对比表
    report_lines.append(f"\n{'='*60}")
    report_lines.append("总结对比")
    report_lines.append(f"{'='*60}")
    report_lines.append(f"{'组合':<25} {'证据一致':<12} {'校验一致':<12} {'文本引用':<15} {'平均耗时':<10}")
    report_lines.append("-" * 74)
    for combo in COMBINATIONS:
        results = all_results[combo["name"]]
        ev_sets = [frozenset(r["evidence_articles"]) for r in results]
        unique_ev = len(set(ev_sets))
        ev_str = "✅全部一致" if unique_ev == 1 else f"⚠️{unique_ev}种"

        all_cit = {}
        for r in results:
            for nid, status, _ in r["citation_statuses"]:
                all_cit.setdefault(nid, []).append(status)
        incit = sum(1 for s in all_cit.values() if len(set(s)) > 1)
        cit_str = f"✅全一致" if incit == 0 else f"⚠️{incit}个不一致"

        refs = []
        for r in results:
            refs.extend(r["text_refs"])
        stable = sum(1 for ref in set(refs) if refs.count(ref) == ROUNDS)
        ref_str = f"{stable}/{len(set(refs))}稳定"

        avg_time = sum(r["elapsed"] for r in results) / len(results)
        report_lines.append(f"{combo['name']:<25} {ev_str:<12} {cit_str:<12} {ref_str:<15} {avg_time:.0f}s")

    report_text = "\n".join(report_lines)

    # 保存
    output_dir = PROJECT_ROOT / "tests"
    raw_path = output_dir / "组合测试原始数据.json"
    report_path = output_dir / "组合测试报告.md"

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n原始数据: {raw_path}")
    print(f"测试报告: {report_path}")
    print(f"\n{report_text}")


if __name__ == "__main__":
    main()
