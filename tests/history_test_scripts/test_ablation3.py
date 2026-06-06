"""消融测试 v3：Level 0 法规选择 + 通用条文降权，8 组 × 30 题。

用法: .venv/bin/python tests/test_ablation3.py [combo_names]

combo_names 可选，逗号分隔，如: A,B,F 。默认跑全部 8 组。
"""

import csv
import datetime
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import legalbot.config as cfg
from legalbot.agents import LegalOrchestrator
from legalbot.types import Evidence

# ── 30 道测试题（与 test_30questions.py 相同） ──
QUESTIONS = [
    {"id": "Q01", "category": "CCAR-121", "question": "不在《运行规范》的机场是否可以去备降"},
    {"id": "Q02", "category": "CCAR-121", "question": "飞机起飞前需要满足哪些燃油量要求"},
    {"id": "Q03", "category": "CCAR-121", "question": "航空公司在什么情况下可以低于最低天气标准运行"},
    {"id": "Q04", "category": "CCAR-121", "question": "运行规范中需要包含哪些内容"},
    {"id": "Q05", "category": "CCAR-121", "question": "飞行机组的值勤期限制是多少"},
    {"id": "Q06", "category": "CCAR-121", "question": "飞机延程运行EDTO需要满足什么条件"},
    {"id": "Q07", "category": "CCAR-121", "question": "签派员放行飞机时需要检查哪些事项"},
    {"id": "Q08", "category": "CCAR-121", "question": "飞机在结冰条件下运行有什么要求"},
    {"id": "Q09", "category": "CCAR-91", "question": "机长在紧急情况下有哪些权力"},
    {"id": "Q10", "category": "CCAR-91", "question": "民用航空器在什么情况下可以进行特技飞行"},
    {"id": "Q11", "category": "CCAR-135", "question": "小型航空器商业运输的备降机场要求是什么"},
    {"id": "Q12", "category": "CCAR-135", "question": "CCAR-135部运营人的飞行员资质要求是什么"},
    {"id": "Q13", "category": "CCAR-92", "question": "无人机在什么情况下需要申请空域许可"},
    {"id": "Q14", "category": "CCAR-92", "question": "无人机飞行的安全距离要求是什么"},
    {"id": "Q15", "category": "机场管理", "question": "机场使用许可证的申请条件是什么"},
    {"id": "Q16", "category": "机场管理", "question": "机场运行安全管理中谁来负责飞行区安全"},
    {"id": "Q17", "category": "机场管理", "question": "航班备降时机场运营人有什么义务"},
    {"id": "Q18", "category": "旅客服务", "question": "航班延误超过多长时间航空公司需要为旅客提供餐饮"},
    {"id": "Q19", "category": "旅客服务", "question": "航空公司拒载旅客的合法理由有哪些"},
    {"id": "Q20", "category": "旅客服务", "question": "旅客行李丢失后航空公司如何赔偿"},
    {"id": "Q21", "category": "航空安全", "question": "航空安全检查中哪些物品禁止带上飞机"},
    {"id": "Q22", "category": "航空安全", "question": "民用航空安全信息报告的时限要求是什么"},
    {"id": "Q23", "category": "航空安全", "question": "航空器发生事故后谁来负责调查"},
    {"id": "Q24", "category": "适航管理", "question": "民用航空器适航指令是做什么的"},
    {"id": "Q25", "category": "适航管理", "question": "航空器维修单位需要什么资质"},
    {"id": "Q26", "category": "空管", "question": "空中交通管制服务由哪些单位提供"},
    {"id": "Q27", "category": "空管", "question": "飞行程序设计和运行最低标准由谁审批"},
    {"id": "Q28", "category": "民用航空法", "question": "中华人民共和国对领空享有什么权利"},
    {"id": "Q29", "category": "民用航空法", "question": "民用航空器所有权的取得和转让有什么要求"},
    {"id": "Q30", "category": "人员资质", "question": "飞行员执照的种类有哪些"},
]

# ── 8 组消融配置 ──
# 基础配置（所有组共享）
BASE_CONFIG = {
    "QUERY_GATE_ENABLED": True,
    "CROSS_ENCODER_CITATION": True,
    "CROSS_ENCODER_CITATION_THRESHOLD": 0.25,
    "RERANKER_ENABLED": True,
    "RERANKER_MIN_SCORE": 0.05,
    "SYNTHESIS_JSON_MODE": True,
    "SET_MEMBERSHIP_CHECK": True,
    "CLAIM_LEVEL_CITATION": False,
    "WRRF_ENABLED": False,
    "TREE_ADAPTIVE_K_ENABLED": False,
    "CONFIDENCE_CUTOFF_ENABLED": False,
    "LEXICAL_REFLEXION_ENABLED": False,
    "LEVEL0_CE_ENABLED": False,
    "TREE_TOP_LAWS": 30,
    "TREE_TOP_CHAPTERS": 30,
    "TREE_CHAPTER_PER_LAW": 10,
    "TREE_LAW_PRIOR_WEIGHT": 0.2,
    "TREE_HINT_BOOST": 0.3,
}

COMBOS = {
    "A_Baseline": {
        **BASE_CONFIG,
        "KEYWORD_ROUTING_ENABLED": False,
        "TREE_GENERIC_ARTICLE_PENALTY": 1.0,  # 无降权
    },
    "B_KW_Routing": {
        **BASE_CONFIG,
        "KEYWORD_ROUTING_ENABLED": True,
        "TREE_GENERIC_ARTICLE_PENALTY": 1.0,
    },
    "C_Generic_Penalty": {
        **BASE_CONFIG,
        "KEYWORD_ROUTING_ENABLED": False,
        "TREE_GENERIC_ARTICLE_PENALTY": 0.5,
    },
    "D_KW_Generic": {
        **BASE_CONFIG,
        "KEYWORD_ROUTING_ENABLED": True,
        "TREE_GENERIC_ARTICLE_PENALTY": 0.5,
    },
    "E_KW_Rules": {
        **BASE_CONFIG,
        "KEYWORD_ROUTING_ENABLED": True,
        "TREE_GENERIC_ARTICLE_PENALTY": 1.0,
    },
    "F_Full": {
        **BASE_CONFIG,
        "KEYWORD_ROUTING_ENABLED": True,
        "TREE_GENERIC_ARTICLE_PENALTY": 0.5,
    },
    "G_Full_L0_50": {
        **BASE_CONFIG,
        "KEYWORD_ROUTING_ENABLED": True,
        "TREE_GENERIC_ARTICLE_PENALTY": 0.5,
        "TREE_TOP_LAWS": 50,
    },
    "H_Full_Penalty_03": {
        **BASE_CONFIG,
        "KEYWORD_ROUTING_ENABLED": True,
        "TREE_GENERIC_ARTICLE_PENALTY": 0.3,
    },
}


def apply_config(combo: dict):
    """应用配置到全局 config 模块。"""
    for key, value in combo.items():
        setattr(cfg, key, value)


def run_question(orch: LegalOrchestrator, q: dict) -> dict:
    start = time.time()
    result = orch.answer(q["question"])
    elapsed = time.time() - start

    ev_articles = [ev.article for ev in (result.evidence or []) if ev.article]
    text_refs = []
    import re
    for name, art in re.findall(r'《([^》]+)》[^。；\n]*?(第[一二三四五六七八九十百千\d]+条)', result.answer):
        text_refs.append(f"《{name}》{art}")

    cit_statuses = [(c.node_id, c.status) for c in (result.citations or [])]
    supported = sum(1 for _, s in cit_statuses if s == "supported")
    partial = sum(1 for _, s in cit_statuses if s == "partial")
    unsupported = sum(1 for _, s in cit_statuses if s == "unsupported")

    return {
        "question_id": q["id"],
        "category": q["category"],
        "question": q["question"],
        "elapsed_sec": round(elapsed, 1),
        "answer_len": len(result.answer),
        "evidence_count": len(result.evidence or []),
        "evidence_articles": " | ".join(ev_articles),
        "citation_count": len(cit_statuses),
        "supported": supported,
        "partial": partial,
        "unsupported": unsupported,
        "supported_rate": f"{supported}/{len(cit_statuses)}",
        "text_refs": " | ".join(text_refs),
        "reflexion_iterations": result.reflexion_iterations,
        "conclusion_preview": result.answer.strip()[:150].replace("\n", " "),
        "answer_full": result.answer,
    }


def main():
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 确定要跑的组合
    if len(sys.argv) > 1:
        selected = [n.strip() for n in sys.argv[1].split(",")]
    else:
        selected = list(COMBOS.keys())

    print(f"消融测试 v3: Level 0 法规选择 + 通用条文降权")
    print(f"时间: {date_str}")
    print(f"组合: {', '.join(selected)} ({len(selected)}组 × 30题 = {len(selected)*30}次)\n")

    csv_path = PROJECT_ROOT / "tests" / f"ablation3_{date_str}.csv"
    json_path = PROJECT_ROOT / "tests" / f"ablation3_{date_str}.json"
    all_combo_results = {}

    for combo_name in selected:
        if combo_name not in COMBOS:
            print(f"跳过未知组合: {combo_name}")
            continue

        combo = COMBOS[combo_name]
        apply_config(combo)

        print(f"\n{'='*60}")
        print(f"组合: {combo_name}")
        print(f"  KEYWORD_ROUTING = {combo.get('KEYWORD_ROUTING_ENABLED')}")
        print(f"  GENERIC_PENALTY = {combo.get('TREE_GENERIC_ARTICLE_PENALTY')}")
        print(f"  TREE_TOP_LAWS   = {combo.get('TREE_TOP_LAWS')}")
        print(f"{'='*60}")

        orch = LegalOrchestrator(logger=None)
        combo_results = []

        for q in QUESTIONS:
            print(f"  [{q['id']}] {q['question'][:35]:35s}", end=" ", flush=True)
            try:
                r = run_question(orch, q)
                r["combo"] = combo_name
                for k, v in combo.items():
                    r[f"cfg_{k}"] = v
                combo_results.append(r)
                print(f"{r['elapsed_sec']:5.1f}s  sup={r['supported_rate']:6s}  ev={r['evidence_count']:2d}")
            except Exception as e:
                print(f"ERROR: {e}")
                combo_results.append({
                    "question_id": q["id"], "category": q["category"],
                    "question": q["question"], "combo": combo_name,
                    "conclusion_preview": f"ERROR: {e}",
                    **{f"cfg_{k}": v for k, v in combo.items()},
                })
            time.sleep(0.5)

        all_combo_results[combo_name] = combo_results

        # 每组完成后输出汇总
        ok = [r for r in combo_results if not r.get("conclusion_preview", "").startswith("ERROR")]
        if ok:
            s = sum(r.get("supported", 0) for r in ok)
            c = sum(r.get("citation_count", 0) for r in ok)
            t = sum(r.get("elapsed_sec", 0) for r in ok) / len(ok)
            print(f"  → 汇总: {len(ok)}题, supported={s}/{c} ({s/max(c,1)*100:.0f}%), 平均{t:.0f}s")

    # 写 CSV
    csv_cols = ["combo", "question_id", "category", "question", "elapsed_sec",
                "answer_len", "evidence_count", "citation_count", "supported",
                "partial", "unsupported", "supported_rate", "reflexion_iterations",
                "conclusion_preview",
                "cfg_KEYWORD_ROUTING_ENABLED", "cfg_TREE_GENERIC_ARTICLE_PENALTY",
                "cfg_TREE_TOP_LAWS"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=csv_cols, extrasaction="ignore")
        writer.writeheader()
        for combo_name, results in all_combo_results.items():
            for r in results:
                writer.writerow(r)

    # 写 JSON（含 answer_full）
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "test_date": date_str,
            "combos": {name: results for name, results in all_combo_results.items()},
        }, f, ensure_ascii=False, indent=2, default=str)

    # 最终对比汇总
    print(f"\n{'='*70}")
    print(f"{'组合':25s} | {'supported':>10s} | {'avg_time':>8s}")
    print(f"{'-'*70}")
    for combo_name, results in all_combo_results.items():
        ok = [r for r in results if not r.get("conclusion_preview", "").startswith("ERROR")]
        if ok:
            s = sum(r.get("supported", 0) for r in ok)
            c = sum(r.get("citation_count", 0) for r in ok)
            t = sum(r.get("elapsed_sec", 0) for r in ok) / len(ok)
            print(f"{combo_name:25s} | {s:3d}/{c:3d} ({s/max(c,1)*100:4.0f}%) | {t:6.1f}s")

    print(f"\nCSV: {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
