"""消融测试主控：2 worker 并行跑配置，写原始数据到 OUT_DIR。

用法:
    .venv/bin/python tests/ablation_runner.py [--workers 2] [--quick] [--grid PARAM=VAL,VAL,VAL ...]

默认: 2 worker, 5 题探针, 17 个配置
  --quick: 用 5 题探针（默认）
  --full:  24 题探针
  --workers N: 并行 worker 数（8GB GPU 安全 2 个）
  --grid: 覆盖默认 grid（如 --grid RERANKER_MIN_SCORE=0.0,0.1）

只跑配置存数据，不算指标。指标分析在 ablation_analyzer.py。
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ablation_grids import (
    QUICK_PROBE, PROBE_QUESTIONS,
    PARAMETER_GRID, expand_grid, chunk_for_workers,
)


def parse_args():
    p = argparse.ArgumentParser(description="消融测试主控（2 worker 并行）")
    p.add_argument("--workers", type=int, default=2, help="并行 worker 数（8GB GPU 推荐 2）")
    p.add_argument("--quick", action="store_true", default=True, help="用 5 题快筛（默认）")
    p.add_argument("--full", action="store_true", help="用 24 题完整探针")
    p.add_argument("--out-dir", type=str, default=None, help="输出目录（默认时间戳）")
    p.add_argument("--grid", type=str, default=None, help="覆盖 grid，如 PARAM=0.0,0.1,0.2")
    return p.parse_args()


def parse_grid_override(override_str):
    """解析 --grid RERANKER_MIN_SCORE=0.0,0.1"""
    out = {}
    if not override_str:
        return out
    for item in override_str.split(";"):
        if "=" not in item:
            continue
        key, vals = item.split("=", 1)
        out[key.strip()] = [parse_value(v.strip()) for v in vals.split(",") if v.strip()]
    return out


def parse_value(s):
    if s.lower() == "true": return True
    if s.lower() == "false": return False
    try:
        if "." in s: return float(s)
        return int(s)
    except ValueError:
        return s


def main():
    args = parse_args()

    # 选探针题集
    questions = PROBE_QUESTIONS if args.full else QUICK_PROBE

    # 选 grid
    if args.grid:
        grid_dict = parse_grid_override(args.grid)
        if not grid_dict:
            print("ERROR: --grid 解析失败")
            sys.exit(1)
    else:
        grid_dict = PARAMETER_GRID

    grid = expand_grid(grid_dict)
    print(f"=" * 70)
    print(f"消融测试 v2 - 配置与探针分离")
    print(f"=" * 70)
    print(f"配置数: {len(grid)}")
    print(f"探针题数: {len(questions)}")
    print(f"Worker 数: {args.workers}")
    print(f"8GB GPU: 建议 workers=2（每 worker ~2.5GB）")

    # 输出目录
    if args.out_dir:
        out_dir = args.out_dir
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = f"tests/ablation_runs/{ts}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {out_dir}")

    # 把 grid 写入 meta.json（analyzer 读这个）
    meta = {
        "timestamp": datetime.now().isoformat(),
        "workers": args.workers,
        "total_configs": len(grid),
        "total_questions": len(questions),
        "out_dir": out_dir,
        "grid": [{"param": p, "value": v} for p, v in grid],
        "questions": questions,
        "questions_total": PROBE_QUESTIONS.__len__(),
    }
    with open(Path(out_dir) / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 切分 grid 给 worker
    chunks = chunk_for_workers(grid, args.workers)
    print(f"切分: {[len(c) for c in chunks]} (per worker)")

    # 启动 worker
    procs = []
    for i, chunk in enumerate(chunks):
        env = os.environ.copy()
        env["ABLATION_GRID"] = json.dumps(chunk)
        env["ABLATION_QUESTIONS"] = json.dumps(questions)
        env["ABLATION_OUT_DIR"] = out_dir
        env["ABLATION_WORKER_ID"] = str(i)
        env["PROJECT_ROOT"] = str(PROJECT_ROOT)
        # 共享 HF 缓存，避免重新下载
        env["HF_HOME"] = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

        # 错开启动避免同时初始化
        if i > 0:
            time.sleep(5)

        cmd = [str(PROJECT_ROOT / ".venv" / "bin" / "python"),
               str(PROJECT_ROOT / "tests" / "run_ablation_worker.py")]
        print(f"启动 worker {i} (配置数={len(chunk)})...", flush=True)
        p = subprocess.Popen(cmd, env=env, cwd=str(PROJECT_ROOT))
        procs.append((i, p))

    # 等待所有 worker
    start = time.time()
    for i, p in procs:
        ret = p.wait()
        print(f"Worker {i} 退出码: {ret}", flush=True)
    elapsed = time.time() - start

    # 统计
    files_written = list(Path(out_dir).glob("*.json"))
    files_written = [f for f in files_written if f.name != "meta.json"]

    print(f"\n{'='*70}")
    print(f"消融测试完成")
    print(f"{'='*70}")
    print(f"总耗时: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"原始数据文件: {len(files_written)}")
    print(f"输出目录: {out_dir}")
    print(f"\n下一步: 跑全面测评")
    print(f"  .venv/bin/python tests/ablation_analyzer.py {out_dir}")


if __name__ == "__main__":
    main()
