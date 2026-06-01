"""30题完整测试：输出CSV格式结果。"""

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

# ── 30道测试题 ──
# 覆盖：CCAR-121(8题)、CCAR-91(2题)、CCAR-135(2题)、CCAR-92/无人机(2题)、
#       机场管理(3题)、旅客服务(3题)、航空安全(3题)、适航(2题)、
#       空管(2题)、民用航空法(2题)、人员资质(1题)
QUESTIONS = [
    # CCAR-121 大型航空器运输
    {"id": "Q01", "category": "CCAR-121", "question": "不在《运行规范》的机场是否可以去备降"},
    {"id": "Q02", "category": "CCAR-121", "question": "飞机起飞前需要满足哪些燃油量要求"},
    {"id": "Q03", "category": "CCAR-121", "question": "航空公司在什么情况下可以低于最低天气标准运行"},
    {"id": "Q04", "category": "CCAR-121", "question": "运行规范中需要包含哪些内容"},
    {"id": "Q05", "category": "CCAR-121", "question": "飞行机组的值勤期限制是多少"},
    {"id": "Q06", "category": "CCAR-121", "question": "飞机延程运行EDTO需要满足什么条件"},
    {"id": "Q07", "category": "CCAR-121", "question": "签派员放行飞机时需要检查哪些事项"},
    {"id": "Q08", "category": "CCAR-121", "question": "飞机在结冰条件下运行有什么要求"},
    # CCAR-91 一般运行
    {"id": "Q09", "category": "CCAR-91", "question": "机长在紧急情况下有哪些权力"},
    {"id": "Q10", "category": "CCAR-91", "question": "民用航空器在什么情况下可以进行特技飞行"},
    # CCAR-135 小型运输
    {"id": "Q11", "category": "CCAR-135", "question": "小型航空器商业运输的备降机场要求是什么"},
    {"id": "Q12", "category": "CCAR-135", "question": "CCAR-135部运营人的飞行员资质要求是什么"},
    # CCAR-92 无人机
    {"id": "Q13", "category": "CCAR-92", "question": "无人机在什么情况下需要申请空域许可"},
    {"id": "Q14", "category": "CCAR-92", "question": "无人机飞行的安全距离要求是什么"},
    # 机场管理
    {"id": "Q15", "category": "机场管理", "question": "机场使用许可证的申请条件是什么"},
    {"id": "Q16", "category": "机场管理", "question": "机场运行安全管理中谁来负责飞行区安全"},
    {"id": "Q17", "category": "机场管理", "question": "航班备降时机场运营人有什么义务"},
    # 旅客服务
    {"id": "Q18", "category": "旅客服务", "question": "航班延误超过多长时间航空公司需要为旅客提供餐饮"},
    {"id": "Q19", "category": "旅客服务", "question": "航空公司拒载旅客的合法理由有哪些"},
    {"id": "Q20", "category": "旅客服务", "question": "旅客行李丢失后航空公司如何赔偿"},
    # 航空安全
    {"id": "Q21", "category": "航空安全", "question": "航空安全检查中哪些物品禁止带上飞机"},
    {"id": "Q22", "category": "航空安全", "question": "民用航空安全信息报告的时限要求是什么"},
    {"id": "Q23", "category": "航空安全", "question": "航空器发生事故后谁来负责调查"},
    # 适航管理
    {"id": "Q24", "category": "适航管理", "question": "民用航空器适航指令是做什么的"},
    {"id": "Q25", "category": "适航管理", "question": "航空器维修单位需要什么资质"},
    # 空中交通管理
    {"id": "Q26", "category": "空管", "question": "空中交通管制服务由哪些单位提供"},
    {"id": "Q27", "category": "空管", "question": "飞行程序设计和运行最低标准由谁审批"},
    # 民用航空法
    {"id": "Q28", "category": "民用航空法", "question": "中华人民共和国对领空享有什么权利"},
    {"id": "Q29", "category": "民用航空法", "question": "民用航空器所有权的取得和转让有什么要求"},
    # 人员资质
    {"id": "Q30", "category": "人员资质", "question": "飞行员执照的种类有哪些"},
]


def extract_citation_pairs(text: str) -> list[str]:
    pairs = re.findall(r'《([^》]+)》[^。；\n]*?(第[一二三四五六七八九十百千\d]+条)', text)
    return [f"《{n}》{a}" for n, a in pairs]


def extract_article_numbers(evidence: list[Evidence]) -> list[str]:
    return [ev.article for ev in evidence if ev.article]


def run_question(orch: LegalOrchestrator, q: dict) -> dict:
    start = time.time()
    result = orch.answer(q["question"])
    elapsed = time.time() - start

    ev_articles = extract_article_numbers(result.evidence or [])
    cit_statuses = [(c.node_id, c.status, round(c.confidence, 2)) for c in (result.citations or [])]
    text_refs = extract_citation_pairs(result.answer)

    # 统计
    supported = sum(1 for _, s, _ in cit_statuses if s == "supported")
    partial = sum(1 for _, s, _ in cit_statuses if s == "partial")
    unsupported = sum(1 for _, s, _ in cit_statuses if s == "unsupported")

    # 结论提取（取答案第一句或前100字）
    answer_clean = result.answer.strip()
    conclusion = answer_clean[:150].replace("\n", " ")

    return {
        "question_id": q["id"],
        "category": q["category"],
        "question": q["question"],
        "elapsed_sec": round(elapsed, 1),
        "answer_len": len(result.answer),
        "evidence_count": len(result.evidence or []),
        "evidence_articles": " | ".join(ev_articles),
        "citation_count": len(result.citations or []),
        "supported": supported,
        "partial": partial,
        "unsupported": unsupported,
        "supported_rate": f"{supported}/{len(cit_statuses)}" if cit_statuses else "0/0",
        "text_refs": " | ".join(text_refs) if text_refs else "",
        "reflexion_iterations": result.reflexion_iterations,
        "conclusion_preview": conclusion,
        "answer_full": result.answer,
        "config": f"A1={cfg.QUERY_GATE_ENABLED}/C1={cfg.CROSS_ENCODER_CITATION}/B3={cfg.CONFIDENCE_CUTOFF_ENABLED}/E1={cfg.LEXICAL_REFLEXION_ENABLED}/reranker_min={cfg.RERANKER_MIN_SCORE}",
    }


CSV_COLUMNS = [
    # 基础信息
    "question_id",       # 题号 Q01-Q30
    "category",          # 法规分类
    "question",          # 问题
    # 性能指标
    "elapsed_sec",       # 耗时(秒)
    "answer_len",        # 答案字数
    # Agent执行信息
    "evidence_count",    # 证据条数
    "reflexion_iterations",  # 自检迭代次数
    # 检索结果
    "evidence_articles", # 检索到的法条列表
    "text_refs",         # 答案中的文本引用
    # 引用校验
    "citation_count",    # 校验条数
    "supported",         # supported数
    "partial",           # partial数
    "unsupported",       # unsupported数
    "supported_rate",    # supported率
    # 答案
    "conclusion_preview", # 结论预览(前150字)
    "config",            # 运行配置快照
]


def main():
    import datetime
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"30题完整测试")
    print(f"配置: {cfg.QUERY_GATE_ENABLED=}, {cfg.CROSS_ENCODER_CITATION=}, {cfg.RERANKER_MIN_SCORE=}")
    print(f"开始时间: {date_str}\n")

    csv_path = PROJECT_ROOT / "tests" / f"test30_{date_str}.csv"
    json_path = PROJECT_ROOT / "tests" / f"test30_{date_str}.json"

    orch = LegalOrchestrator(logger=None)
    all_results = []

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()

        for i, q in enumerate(QUESTIONS):
            print(f"[{q['id']}] {q['question'][:40]}...", end=" ", flush=True)
            try:
                r = run_question(orch, q)
                writer.writerow(r)
                all_results.append(r)
                print(f"完成 {r['elapsed_sec']}s, {r['answer_len']}字, supported={r['supported_rate']}")
            except Exception as e:
                print(f"失败: {e}")
                error_row = {col: "" for col in CSV_COLUMNS}
                error_row["question_id"] = q["id"]
                error_row["category"] = q["category"]
                error_row["question"] = q["question"]
                error_row["conclusion_preview"] = f"ERROR: {e}"
                writer.writerow(error_row)
                all_results.append(error_row)
            time.sleep(1)

    # 保存完整JSON（含answer_full）
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "test_date": date_str,
            "config": all_results[0]["config"] if all_results else "",
            "results": all_results,
        }, f, ensure_ascii=False, indent=2, default=str)

    # 汇总统计
    print(f"\n{'='*60}")
    print("汇总统计")
    print(f"{'='*60}")
    ok = [r for r in all_results if not r.get("conclusion_preview", "").startswith("ERROR")]
    if ok:
        avg_time = sum(r["elapsed_sec"] for r in ok) / len(ok)
        avg_len = sum(r["answer_len"] for r in ok) / len(ok)
        total_supported = sum(r["supported"] for r in ok)
        total_citations = sum(r["citation_count"] for r in ok)
        print(f"完成: {len(ok)}/30")
        print(f"平均耗时: {avg_time:.1f}s")
        print(f"平均答案长度: {avg_len:.0f}字")
        print(f"整体supported率: {total_supported}/{total_citations} ({total_supported/max(total_citations,1)*100:.0f}%)")

        # 按分类统计
        by_cat = {}
        for r in ok:
            cat = r["category"]
            by_cat.setdefault(cat, []).append(r)
        print(f"\n按分类:")
        for cat, rows in sorted(by_cat.items()):
            s = sum(r["supported"] for r in rows)
            c = sum(r["citation_count"] for r in rows)
            t = sum(r["elapsed_sec"] for r in rows) / len(rows)
            print(f"  {cat:15s}: {len(rows)}题, supported={s}/{c}, 平均{t:.0f}s")

    print(f"\nCSV: {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
