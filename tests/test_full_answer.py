"""单组合完整答案测试。用法: python test_full_answer.py <组合名> <A1=0|1> <C1=0|1> <B3=0|1> <E1=0|1>"""

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


def extract_citation_pairs(text: str) -> list[str]:
    pairs = re.findall(r'《([^》]+)》[^。；\n]*?(第[一二三四五六七八九十百千\d]+条)', text)
    return [f"《{n}》{a}" for n, a in pairs]


def extract_article_numbers(evidence: list[Evidence]) -> list[str]:
    return [ev.article for ev in evidence if ev.article]


def run_one(orch: LegalOrchestrator, round_num: int) -> dict:
    start = time.time()
    result = orch.answer(QUESTION)
    elapsed = time.time() - start

    return {
        "round": round_num,
        "elapsed": round(elapsed, 1),
        "answer_len": len(result.answer),
        "evidence_count": len(result.evidence or []),
        "evidence_articles": extract_article_numbers(result.evidence or []),
        "citation_count": len(result.citations or []),
        "citation_statuses": [(c.node_id, c.status, round(c.confidence, 2)) for c in (result.citations or [])],
        "text_refs": extract_citation_pairs(result.answer),
        "reflexion_iterations": result.reflexion_iterations,
        "answer_full": result.answer,
    }


def main():
    combo_name = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    a1 = sys.argv[2] == "1" if len(sys.argv) > 2 else False
    c1 = sys.argv[3] == "1" if len(sys.argv) > 3 else False
    b3 = sys.argv[4] == "1" if len(sys.argv) > 4 else False
    e1 = sys.argv[5] == "1" if len(sys.argv) > 5 else False

    cfg.QUERY_GATE_ENABLED = a1
    cfg.CROSS_ENCODER_CITATION = c1
    cfg.CONFIDENCE_CUTOFF_ENABLED = b3
    cfg.LEXICAL_REFLEXION_ENABLED = e1

    print(f"[{combo_name}] A1={a1} C1={c1} B3={b3} E1={e1}")

    orch = LegalOrchestrator(logger=None)
    results = []
    for i in range(1, ROUNDS + 1):
        print(f"[{combo_name}] 轮{i}...", end=" ", flush=True)
        r = run_one(orch, i)
        results.append(r)
        print(f"完成 {r['elapsed']}s, {r['answer_len']}字")
        time.sleep(1)

    output_path = PROJECT_ROOT / "tests" / f"full_{combo_name.replace(' ', '_').replace('+', '_')}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"combo": combo_name, "question": QUESTION, "rounds": results},
                  f, ensure_ascii=False, indent=2, default=str)
    print(f"[{combo_name}] 结果已保存: {output_path}")


if __name__ == "__main__":
    main()
