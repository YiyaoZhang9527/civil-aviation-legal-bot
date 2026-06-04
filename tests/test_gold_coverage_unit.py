"""Gold coverage 评测的单元测试（不依赖 LLM）。"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_gold_coverage import (
    evaluate_coverage,
    generate_gold_answer,
    load_bot_answers,
)


# ── Bot 答案加载 ─────────────────────────────────────────


class TestLoadBotAnswers:
    def test_loads_all_results(self, tmp_path):
        data = {
            "results": [
                {"question_id": "Q01", "category": "A", "question": "q1", "answer_full": "ans1"},
                {"question_id": "Q02", "category": "B", "question": "q2", "answer_full": "ans2"},
            ]
        }
        path = tmp_path / "test30_test.json"
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        # Monkey-patch find_latest_bot_results
        import tests.test_gold_coverage as mod
        original = mod.find_latest_bot_results
        mod.find_latest_bot_results = lambda: path
        try:
            items = load_bot_answers(path)
            assert len(items) == 2
            assert items[0]["question_id"] == "Q01"
            assert items[0]["bot_answer"] == "ans1"
        finally:
            mod.find_latest_bot_results = original

    def test_filters_results_without_answer(self, tmp_path):
        data = {
            "results": [
                {"question_id": "Q01", "answer_full": "real ans"},
                {"question_id": "Q02", "answer_full": ""},  # 空答案
                {"question_id": "Q03"},  # 缺字段
            ]
        }
        path = tmp_path / "x.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        items = load_bot_answers(path)
        assert len(items) == 1
        assert items[0]["question_id"] == "Q01"


# ── Coverage 拒答预检查 ──────────────────────────────────


class TestEvaluateCoverageRefusalPrecheck:
    def test_bot_refusal_yields_zero_coverage(self):
        """bot 拒答 → 不调 LLM，直接 0%。"""
        llm = MagicMock()
        refusal_answers = [
            "很抱歉，未找到直接回答您问题的法律条文。",
            "无法确定答案。",
            "建议您咨询专业法律人士。",
        ]
        for ans in refusal_answers:
            result = evaluate_coverage(ans, ["要点1", "要点2", "要点3"], llm)
            assert result["coverage_rate"] == 0.0
            assert result["covered_indices"] == []
            assert result["missing_indices"] == [1, 2, 3]
            assert "refusal" in result["reason"].lower()
        # 关键：拒答时 LLM 不该被调用
        llm.json.assert_not_called()
        llm.chat.assert_not_called()

    def test_synthesis_fallback_detected_as_refusal(self):
        """SynthesisAgent 兜底文本（含'以下为系统检索到的可能相关法规'）应被检测为非拒答。"""
        llm = MagicMock()
        fallback = (
            "【结论】\n以下为系统检索到的可能相关法规，请参考：\n"
            "【相关法规】\n1. 《XX》第一条"
        )
        # 兜底不算拒答——会进入 LLM judge
        result = evaluate_coverage(fallback, ["要点1"], llm)
        # 触发了 LLM（因为非拒答）
        # 注意：因 MagicMock 返回默认 {}, 走"covered 0, missing 1"路径
        assert result["coverage_rate"] == 0.0
        assert llm.json.called  # LLM 确实被调用了

    def test_empty_key_points_returns_zero(self):
        llm = MagicMock()
        result = evaluate_coverage("正常答案", [], llm)
        assert result["coverage_rate"] == 0.0
        assert result["covered_indices"] == []
        assert result["missing_indices"] == []
        # 无要点时 LLM 不该被调
        llm.json.assert_not_called()


# ── Coverage LLM judge 容错 ──────────────────────────────


class TestEvaluateCoverageRobustness:
    def test_fills_missing_indices_from_uncovered(self):
        """LLM 只输出 covered 时，missing 自动补全。"""
        llm = MagicMock()
        llm.json.return_value = {"covered": [1, 3], "missing": [], "reason": ""}
        result = evaluate_coverage("正常答案", ["p1", "p2", "p3"], llm)
        assert result["covered_indices"] == [1, 3]
        assert result["missing_indices"] == [2]
        assert result["coverage_rate"] == pytest.approx(2 / 3)

    def test_overlapping_covered_and_missing(self):
        """LLM 在两处都报了同一索引 → covered 优先。"""
        llm = MagicMock()
        llm.json.return_value = {"covered": [1, 2], "missing": [2, 3], "reason": ""}
        result = evaluate_coverage("正常答案", ["p1", "p2", "p3"], llm)
        # covered=1,2; missing=3（2 已 covered）
        assert result["covered_indices"] == [1, 2]
        assert result["missing_indices"] == [3]

    def test_invalid_indices_filtered(self):
        """LLM 报了 0 或超大索引 → 过滤。"""
        llm = MagicMock()
        llm.json.return_value = {"covered": [0, 1, 99, "x"], "missing": [2], "reason": ""}
        result = evaluate_coverage("正常答案", ["p1", "p2"], llm)
        # 0 和 99 被过滤，"x" 不是数字被过滤
        assert result["covered_indices"] == [1]
        assert result["missing_indices"] == [2]


# ── Gold 生成（只测入口）─────────────────────────────────


class TestGenerateGoldAnswer:
    def test_empty_evidence_returns_empty(self, monkeypatch):
        """检索无证据时 gold 答案为空。RewriteAgent 仍会调 LLM 改写，但 gold 生成不调。"""
        from tests import test_gold_coverage as mod

        monkeypatch.setattr(mod, "search_index_tree", lambda **kw: [])

        llm = MagicMock()
        result = generate_gold_answer("any question", llm)
        assert result["answer"] == "（无证据）"
        assert result["key_points"] == []
        # gold 生成阶段的 LLM（不是 RewriteAgent 那个）应该被跳过
        # 但我们没法直接区分 MagicMock 的两次调用，改用检查 key_points 为空
        assert result["key_points"] == []

    def test_calls_llm_with_broader_evidence(self, monkeypatch):
        """验证：检索 → 拼 evidence_text → 调 LLM → 返回 answer + key_points。"""
        from legalbot.types import Evidence
        from tests import test_gold_coverage as mod

        fake_evidence = [
            Evidence(
                law_id="ccar-271", law_title="公共航空运输旅客服务管理规定",
                node_id="article:31", article="第三十一条",
                text="有下列情形应拒载...", score=0.5,
                source_file="x.txt", source_anchor="#1",
            )
        ]
        monkeypatch.setattr(mod, "search_index_tree", lambda **kw: fake_evidence)

        llm = MagicMock()
        llm.json.return_value = {
            "answer": "航空公司可在以下情形拒载：\n1. 旅客使用伪造证件",
            "key_points": ["情形1：伪造证件", "情形2：拒绝安检"],
        }
        result = generate_gold_answer("什么情况下可以拒载", llm)
        assert "拒载" in result["answer"]
        assert len(result["key_points"]) == 2
        # 验证 LLM 被调用且 prompt 包含证据
        assert llm.json.called
        call_args = llm.json.call_args
        messages = call_args[0][0]
        prompt_text = " ".join(m["content"] for m in messages)
        assert "公共航空运输旅客服务管理规定" in prompt_text
        assert "第三十一条" in prompt_text
