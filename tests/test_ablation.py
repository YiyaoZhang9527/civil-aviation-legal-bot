"""消融实验：8题 x 6组合，找出最优配置。

组合设计（逐层叠加）：
  A: Baseline (A1+C1 only)
  B: +检索层 (WRRF + AdaptiveK)
  C: +校验层 (Claim-NLI反转)
  D: +生成层 (JSON + Set-Membership)
  E: Full (全部开启)
  F: Full-JSON (全开但不用JSON模式，测试JSON对准确性的影响)

指标：supported数、supported率、耗时、每题详情
"""

import csv
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

# 8题代表性测试集：覆盖好题/差题/各类法规
QUESTIONS = [
    {"id": "Q01", "category": "CCAR-121", "question": "不在运行规范的机场是否可以去备降"},
    {"id": "Q04", "category": "CCAR-121", "question": "运行规范中需要包含哪些内容"},
    {"id": "Q07", "category": "CCAR-121", "question": "签派员放行飞机时需要检查哪些事项"},
    {"id": "Q14", "category": "CCAR-92", "question": "无人机飞行的安全距离要求是什么"},
    {"id": "Q18", "category": "旅客服务", "question": "航班延误超过多长时间航空公司需要为旅客提供餐饮"},
    {"id": "Q19", "category": "旅客服务", "question": "航空公司拒载旅客的合法理由有哪些"},
    {"id": "Q23", "category": "航空安全", "question": "航空器发生事故后谁来负责调查"},
    {"id": "Q25", "category": "适航管理", "question": "航空器维修单位需要什么资质"},
]

# 组合配置
COMBOS = {
    "A_Baseline": {
        "QUERY_GATE_ENABLED": True,
        "CROSS_ENCODER_CITATION": True,
        "WRRF_ENABLED": False,
        "TREE_ADAPTIVE_K_ENABLED": False,
        "SYNTHESIS_JSON_MODE": False,
        "SET_MEMBERSHIP_CHECK": False,
        "CLAIM_LEVEL_CITATION": False,
    },
    "B_Retrieval": {
        "QUERY_GATE_ENABLED": True,
        "CROSS_ENCODER_CITATION": True,
        "WRRF_ENABLED": True,
        "TREE_ADAPTIVE_K_ENABLED": True,
        "SYNTHESIS_JSON_MODE": False,
        "SET_MEMBERSHIP_CHECK": False,
        "CLAIM_LEVEL_CITATION": False,
    },
    "C_ClaimNLI": {
        "QUERY_GATE_ENABLED": True,
        "CROSS_ENCODER_CITATION": True,
        "WRRF_ENABLED": False,
        "TREE_ADAPTIVE_K_ENABLED": False,
        "SYNTHESIS_JSON_MODE": False,
        "SET_MEMBERSHIP_CHECK": False,
        "CLAIM_LEVEL_CITATION": True,
    },
    "D_JSON_SM": {
        "QUERY_GATE_ENABLED": True,
        "CROSS_ENCODER_CITATION": True,
        "WRRF_ENABLED": False,
        "TREE_ADAPTIVE_K_ENABLED": False,
        "SYNTHESIS_JSON_MODE": True,
        "SET_MEMBERSHIP_CHECK": True,
        "CLAIM_LEVEL_CITATION": False,
    },
    "E_Full": {
        "QUERY_GATE_ENABLED": True,
        "CROSS_ENCODER_CITATION": True,
        "WRRF_ENABLED": True,
        "TREE_ADAPTIVE_K_ENABLED": True,
        "SYNTHESIS_JSON_MODE": True,
        "SET_MEMBERSHIP_CHECK": True,
        "CLAIM_LEVEL_CITATION": True,
    },
    "F_FullNoJSON": {
        "QUERY_GATE_ENABLED": True,
        "CROSS_ENCODER_CITATION": True,
        "WRRF_ENABLED": True,
        "TREE_ADAPTIVE_K_ENABLED": True,
        "SYNTHESIS_JSON_MODE": False,
        "SET_MEMBERSHIP_CHECK": False,
        "CLAIM_LEVEL_CITATION": True,
    },
}


def apply_config(combo: dict):
    for key, value in combo.items():
        setattr(cfg, key, value)


def run_question(orch, q):
    start = time.time()
    result = orch.answer(q["question"])
    elapsed = time.time() - start
    s = sum(1 for c in result.citations if c.status == "supported")
    p = sum(1 for c in result.citations if c.status == "partial")
    u = sum(1 for c in result.citations if c.status == "unsupported")
    total = s + p + u
    return {
        "id": q["id"],
        "category": q["category"],
        "question": q["question"],
        "elapsed": round(elapsed, 1),
        "evidence": len(result.evidence),
        "citations": total,
        "supported": s,
        "partial": p,
        "unsupported": u,
        "rate": f"{s}/{total}" if total else "0/0",
        "claims": len(result.structured_claims),
        "removed": result.unsupported_claims_removed,
        "answer_preview": result.answer[:120].replace("\n", " "),
    }


def main():
    import datetime
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "tests"
    csv_path = output_dir / f"ablation_{date_str}.csv"
    json_path = output_dir / f"ablation_{date_str}.json"

    print(f"消融实验: {len(QUESTIONS)}题 x {len(COMBOS)}组合")
    print(f"输出: {csv_path}")
    print()

    all_results = {}
    csv_rows = []

    for combo_name, combo_config in COMBOS.items():
        print(f"\n{'='*60}")
        print(f"组合: {combo_name}")
        print(f"{'='*60}")
        apply_config(combo_config)

        orch = LegalOrchestrator(logger=None)
        combo_results = []

        for q in QUESTIONS:
            print(f"  [{q['id']}] {q['question'][:30]}...", end=" ", flush=True)
            try:
                r = run_question(orch, q)
                combo_results.append(r)
                print(f"{r['elapsed']}s s={r['rate']} claims={r['claims']} rm={r['removed']}")
            except Exception as e:
                print(f"FAIL: {e}")
                combo_results.append({
                    "id": q["id"], "category": q["category"],
                    "question": q["question"], "elapsed": 0,
                    "evidence": 0, "citations": 0, "supported": 0,
                    "partial": 0, "unsupported": 0, "rate": "0/0",
                    "claims": 0, "removed": 0,
                    "answer_preview": f"ERROR: {e}",
                })

            time.sleep(1)

        # 汇总
        total_s = sum(r["supported"] for r in combo_results)
        total_c = sum(r["citations"] for r in combo_results)
        total_t = sum(r["elapsed"] for r in combo_results)
        avg_t = total_t / len(combo_results) if combo_results else 0
        print(f"  汇总: supported={total_s}/{total_c} ({total_s*100//max(total_c,1)}%) "
              f"总耗时={total_t:.0f}s 平均={avg_t:.0f}s")

        all_results[combo_name] = {
            "config": combo_config,
            "results": combo_results,
            "summary": {
                "total_supported": total_s,
                "total_citations": total_c,
                "supported_rate": f"{total_s}/{total_c}",
                "total_time": round(total_t, 1),
                "avg_time": round(avg_t, 1),
            },
        }

        for r in combo_results:
            row = {"combo": combo_name, **r}
            for k, v in combo_config.items():
                row[f"cfg_{k}"] = v
            csv_rows.append(row)

    # 写CSV
    if csv_rows:
        fields = list(csv_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(csv_rows)

    # 写JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

    # 对比表
    print(f"\n{'='*60}")
    print("消融实验对比表")
    print(f"{'='*60}")
    print(f"{'组合':<18} {'supported':<12} {'率':<8} {'总耗时':<10} {'平均耗时':<10}")
    print("-" * 58)
    for name, data in all_results.items():
        s = data["summary"]["total_supported"]
        c = data["summary"]["total_citations"]
        pct = f"{s*100//max(c,1)}%"
        tt = data["summary"]["total_time"]
        at = data["summary"]["avg_time"]
        print(f"{name:<18} {s}/{c:<10} {pct:<8} {tt}s{'':<5} {at}s")

    print(f"\nCSV: {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
