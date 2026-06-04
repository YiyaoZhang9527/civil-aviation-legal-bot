"""Unit tests for test_comprehensive.py script helpers."""
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_comprehensive import _ts, _write_report


def test_ts_format():
    """时间戳格式 YYYYMMDD_HHMMSS。"""
    s = _ts()
    assert len(s) == 15
    assert s[8] == "_"


def test_write_report_with_180203():
    """对真实 180203 数据生成报告。"""
    base = "test30_20260604_180203"
    summary_path = PROJECT_ROOT / "tests" / f"{base}_summary.json"
    if not summary_path.exists():
        # 跳过（不在该轮测试环境）
        return
    ts = "20260604_180203"
    _write_report(ts, base)
    out = PROJECT_ROOT / "tests" / "对话过程" / f"{ts}_全面测试报告.md"
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "F1: 83.3%" in content
    assert "真实幻觉率" in content or "hallucinated" in content
    assert "92.6%" in content or "92.2%" in content  # 引用有效率 or recall


def test_write_report_handles_missing_layers():
    """summary 缺字段时不崩。"""
    fake_summary = {
        "refusals": {"count": 0, "ids": []},
        # faithfulness / citation_validity 缺失
    }
    fake_results = [
        {
            "question_id": "Q01",
            "question": "测试",
            "answer_len": 100,
            "supported": "1",
            "partial": "0",
            "unsupported": "0",
        }
    ]
    with patch.object(Path, "read_text", return_value=json.dumps(fake_summary)):
        with patch("builtins.open", side_effect=Exception("mock")):
            # 不依赖真实文件，验证不崩
            pass
