"""消融测试 worker：只跑配置、存原始数据，不算任何指标。

接收参数（通过环境变量）：
- ABLATION_GRID: JSON 列表 [[param, value], ...]
- ABLATION_QUESTIONS: JSON 列表（探针题）
- ABLATION_OUT_DIR: 原始数据输出目录
- ABLATION_WORKER_ID: worker 编号（用于日志）

每跑完一道题，写一个 JSON 到 OUT_DIR，文件名 = {config_id}_q{qid}.json
内容 = 完整原始数据（answer / evidence / citations / elapsed 等）

不在 worker 里算指标——交给 analyzer.py 统一分析。
"""

import os
import sys
import json
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from legalbot import config as cfg
from legalbot.agents import LegalOrchestrator

# 拒绝词（与 test_citation_validity / test_summary 保持一致）
REFUSAL_KEYWORDS = [
    "无法确定", "无法回答", "证据不足", "未包含相关",
    "未涉及", "未找到", "没有找到", "未能找到",
    "无法提供", "无法确认", "无法判断", "现有证据不足以",
]


def is_refusal(answer: str) -> bool:
    return any(kw in answer for kw in REFUSAL_KEYWORDS)


def serialize_evidence(ev) -> dict:
    return {
        "law_id": getattr(ev, "law_id", ""),
        "law_title": getattr(ev, "law_title", ""),
        "article": getattr(ev, "article", ""),
        "node_id": getattr(ev, "node_id", ""),
        "text": getattr(ev, "text", ""),
        "score": getattr(ev, "score", 0.0),
    }


def serialize_citation(c) -> dict:
    return {
        "node_id": getattr(c, "node_id", ""),
        "law_id": getattr(c, "law_id", ""),
        "claim": getattr(c, "claim", ""),
        "status": getattr(c, "status", ""),
        "confidence": getattr(c, "confidence", 0.0),
        "reason": getattr(c, "reason", ""),
    }


def main():
    grid = json.loads(os.environ["ABLATION_GRID"])
    questions = json.loads(os.environ["ABLATION_QUESTIONS"])
    out_dir = os.environ["ABLATION_OUT_DIR"]
    worker_id = os.environ.get("ABLATION_WORKER_ID", "0")
    project_root = os.environ.get("PROJECT_ROOT", str(PROJECT_ROOT))

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print(f"[W{worker_id}] 启动：处理 {len(grid)} 个配置 × {len(questions)} 题", flush=True)
    print(f"[W{worker_id}] 输出目录: {out_dir}", flush=True)

    orch = LegalOrchestrator(logger=None)
    start_all = time.time()

    for cfg_idx, (param, value) in enumerate(grid):
        # 应用配置（仅在 cfg 模块上 setattr，不修改磁盘文件）
        setattr(cfg, param, value)

        # config_id 用于文件命名
        val_str = str(value)
        if isinstance(value, bool):
            val_str = "True" if value else "False"
        elif isinstance(value, float) and value == int(value):
            val_str = f"{value:.1f}"
        config_id = f"{param}={val_str}"
        safe_id = config_id.replace(".", "_").replace("=", "_").replace(" ", "_")

        for q_idx, q in enumerate(questions):
            qid = q["id"]
            out_path = Path(out_dir) / f"{safe_id}_q{qid}.json"

            t0 = time.time()
            error = None
            try:
                result = orch.answer(q["question"])
                elapsed = time.time() - t0
                answer = result.answer or ""
                evidence = result.evidence or []
                citations = result.citations or []
            except Exception as e:
                elapsed = time.time() - t0
                answer = ""
                evidence = []
                citations = []
                error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

            record = {
                "config_id": config_id,
                "config_param": param,
                "config_value": value,
                "qid": qid,
                "category": q.get("category", ""),
                "question": q["question"],
                "answer_full": answer,
                "evidence": [serialize_evidence(e) for e in evidence],
                "citations": [serialize_citation(c) for c in citations],
                "elapsed_sec": round(elapsed, 2),
                "answer_len": len(answer),
                "evidence_count": len(evidence),
                "citation_count": len(citations),
                "supported_count": sum(1 for c in citations if c.status == "supported"),
                "partial_count": sum(1 for c in citations if c.status == "partial"),
                "unsupported_count": sum(1 for c in citations if c.status == "unsupported"),
                "reflexion_iterations": getattr(result, "reflexion_iterations", 0) if not error else 0,
                "is_refusal": is_refusal(answer),
                "error": error,
                "timestamp": time.time(),
                "worker_id": worker_id,
            }

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)

            status = "OK" if not error else "ERR"
            print(
                f"[W{worker_id}] [{cfg_idx+1}/{len(grid)}] {config_id} | {qid} | "
                f"{elapsed:.1f}s | ans={len(answer)}c | ev={len(evidence)} | cit={len(citations)} | {status}",
                flush=True,
            )

    # 还原配置（仅清理这一批的影响——不写回磁盘）
    # 注意：cfg 改动只影响当前进程，下个 worker 各自加载自己的 Python 进程，无干扰
    total_elapsed = time.time() - start_all
    print(f"[W{worker_id}] 完成 {len(grid)} 配置，用时 {total_elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()
