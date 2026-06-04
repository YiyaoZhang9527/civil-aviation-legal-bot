"""补测 G 组合 + 阈值扫描。

G 组合 = D_JSON+SM + WRRF + AdaptiveK（消融实验缺失的组合）
阈值扫描 = 在 G 组合上遍历 threshold 0.20~0.50，找最优
"""

import csv
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import legalbot.config as cfg
from legalbot.agents import LegalOrchestrator

QUESTIONS = [
    {"id": "Q01", "question": "不在运行规范的机场是否可以去备降"},
    {"id": "Q04", "question": "运行规范中需要包含哪些内容"},
    {"id": "Q07", "question": "签派员放行飞机时需要检查哪些事项"},
    {"id": "Q14", "question": "无人机飞行的安全距离要求是什么"},
    {"id": "Q18", "question": "航班延误超过多长时间航空公司需要为旅客提供餐饮"},
    {"id": "Q19", "question": "航空公司拒载旅客的合法理由有哪些"},
    {"id": "Q23", "question": "航空器发生事故后谁来负责调查"},
    {"id": "Q25", "question": "航空器维修单位需要什么资质"},
]

def apply_config(overrides: dict):
    for k, v in overrides.items():
        setattr(cfg, k, v)

def run_question(orch, q):
    start = time.time()
    result = orch.answer(q["question"])
    elapsed = time.time() - start
    s = sum(1 for c in result.citations if c.status == "supported")
    p = sum(1 for c in result.citations if c.status == "partial")
    u = sum(1 for c in result.citations if c.status == "unsupported")
    total = s + p + u
    return {
        "id": q["id"], "elapsed": round(elapsed, 1),
        "evidence": len(result.evidence), "citations": total,
        "supported": s, "partial": p, "unsupported": u,
        "rate": f"{s}/{total}" if total else "0/0",
        "claims": len(result.structured_claims),
        "removed": result.unsupported_claims_removed,
        "answer_preview": result.answer[:200].replace("\n", " "),
    }

def run_combo(name, config_overrides, orch=None):
    apply_config(config_overrides)
    if orch is None:
        orch = LegalOrchestrator(logger=None)
    results = []
    for q in QUESTIONS:
        print(f"  [{q['id']}] {q['question'][:25]}...", end=" ", flush=True)
        try:
            r = run_question(orch, q)
            print(f"{r['elapsed']}s s={r['rate']} claims={r['claims']}")
        except Exception as e:
            print(f"FAIL: {e}")
            r = {"id": q["id"], "elapsed": 0, "evidence": 0, "citations": 0,
                 "supported": 0, "partial": 0, "unsupported": 0, "rate": "0/0",
                 "claims": 0, "removed": 0, "answer_preview": f"ERROR: {e}"}
        results.append(r)
        time.sleep(1)

    total_s = sum(r["supported"] for r in results)
    total_c = sum(r["citations"] for r in results)
    total_t = sum(r["elapsed"] for r in results)
    avg_t = total_t / max(len(results), 1)
    print(f"  >> {name}: supported={total_s}/{total_c} ({total_s*100//max(total_c,1)}%) "
          f"time={total_t:.0f}s avg={avg_t:.0f}s")
    return {"name": name, "config": config_overrides, "results": results,
            "summary": {"supported": total_s, "citations": total_c,
                        "rate_pct": total_s*100//max(total_c,1),
                        "total_time": round(total_t, 1), "avg_time": round(avg_t, 1)}}

def main():
    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    G_CONFIG = {
        "QUERY_GATE_ENABLED": True,
        "CROSS_ENCODER_CITATION": True,
        "WRRF_ENABLED": True,
        "TREE_ADAPTIVE_K_ENABLED": True,
        "SYNTHESIS_JSON_MODE": True,
        "SET_MEMBERSHIP_CHECK": True,
        "CLAIM_LEVEL_CITATION": False,  # 旧校验方向
    }

    all_data = {}

    # ---- Part 1: G 组合 ----
    print("=" * 60)
    print("G: JSON+SM + WRRF + AdaptiveK (旧校验方向)")
    print("=" * 60)
    all_data["G_JSON_SM_Retrieval"] = run_combo("G_JSON_SM_Retrieval", G_CONFIG)

    # ---- Part 2: E_Full 对比（已有数据从消融实验，这里重跑确认） ----
    print("\n" + "=" * 60)
    print("E_Full (全部开启，新校验方向)")
    print("=" * 60)
    E_CONFIG = {
        "QUERY_GATE_ENABLED": True,
        "CROSS_ENCODER_CITATION": True,
        "WRRF_ENABLED": True,
        "TREE_ADAPTIVE_K_ENABLED": True,
        "SYNTHESIS_JSON_MODE": True,
        "SET_MEMBERSHIP_CHECK": True,
        "CLAIM_LEVEL_CITATION": True,  # 新校验方向
    }
    all_data["E_Full"] = run_combo("E_Full", E_CONFIG)

    # ---- Part 3: 阈值扫描（在 G 组合上） ----
    print("\n" + "=" * 60)
    print("阈值扫描 (G 组合, threshold 0.20~0.50)")
    print("=" * 60)
    for threshold in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        config_t = {**G_CONFIG, "CROSS_ENCODER_CITATION_THRESHOLD": threshold}
        name = f"G_threshold_{threshold:.2f}"
        all_data[name] = run_combo(name, config_t)

    # ---- 汇总 ----
    print("\n" + "=" * 60)
    print("对比表")
    print("=" * 60)
    print(f"{'组合':<25s} {'supported':<12s} {'rate':<6s} {'总耗时':<8s} {'平均':<6s}")
    print("-" * 57)
    for name, data in all_data.items():
        s = data["summary"]["supported"]
        c = data["summary"]["citations"]
        pct = data["summary"]["rate_pct"]
        tt = data["summary"]["total_time"]
        at = data["summary"]["avg_time"]
        print(f"{name:<25s} {s:>3d}/{c:<6d} {pct:>3d}%   {tt:>5.0f}s   {at:>4.0f}s")

    # 写文件
    csv_path = PROJECT_ROOT / "tests" / f"ablation2_{ts}.csv"
    json_path = PROJECT_ROOT / "tests" / f"ablation2_{ts}.json"
    rows = []
    for name, data in all_data.items():
        for r in data["results"]:
            row = {"combo": name, **r}
            for k, v in data["config"].items():
                row[f"cfg_{k}"] = v
            rows.append(row)
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2, default=str)

    print(f"\nCSV: {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
