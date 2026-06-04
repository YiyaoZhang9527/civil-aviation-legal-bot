"""LLMClient.json 防御性解析 + 渲染兜底测试（Q22 crash 修复）。"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import legalbot.config as cfg
from legalbot.llm import LLMClient, LLMError, parse_json_object
from legalbot.agents import _render_structured_answer
from legalbot.types import Evidence


# ── _render_structured_answer 兜底（防 "{}" 输出）───────


class TestRenderStructuredAnswerFallback:
    def test_empty_dict_with_evidence_returns_useful_fallback(self):
        """data 为空（LLM JSON 解析失败）但有 evidence → 列出 evidence，不输出 "{}"。"""
        evidence = [
            Evidence(law_id="ccar-xxx", law_title="测试法规", node_id="article:1",
                     article="第一条", text="测试内容", score=0.5,
                     source_file="x.txt", source_anchor="#1"),
        ]
        result = _render_structured_answer({}, evidence)
        # 不应是 "{}"
        assert result != "{}"
        # 应包含 evidence 信息
        assert "测试法规" in result
        assert "第一条" in result

    def test_empty_dict_no_evidence_returns_apology(self):
        """data 为空 + 无 evidence → 友好道歉。"""
        result = _render_structured_answer({}, [])
        assert result != "{}"
        assert "抱歉" in result or "未能" in result

    def test_normal_data_still_works(self):
        """正常 data 不受影响。"""
        data = {"conclusion": "测试结论", "claims": [{"text": "声明1", "source": "法规规定", "node_ids": []}]}
        result = _render_structured_answer(data, [])
        assert "测试结论" in result
        assert "声明1" in result


# ── parse_json_object 基础行为 ─────────────────────────────


class TestParseJsonObject:
    def test_valid_json(self):
        result = parse_json_object('{"a": 1, "b": "x"}')
        assert result == {"a": 1, "b": "x"}

    def test_json_in_markdown_fence(self):
        text = '```json\n{"a": 1}\n```'
        result = parse_json_object(text)
        assert result == {"a": 1}

    def test_json_with_surrounding_text(self):
        text = '这是说明文字\n{"a": 1}\n更多文字'
        result = parse_json_object(text)
        assert result == {"a": 1}

    def test_truncated_json_repaired(self):
        """缺闭合括号自动补。"""
        text = '{"a": 1, "b": 2'  # 缺一个 }
        result = parse_json_object(text)
        assert result == {"a": 1, "b": 2}

    def test_invalid_json_no_match_raises(self):
        with pytest.raises(LLMError):
            parse_json_object("没有 JSON 在这里")


# ── LLMClient.json 防御性行为 ─────────────────────────────


class TestLLMClientJsonFallback:
    def test_valid_json_first_try(self):
        llm = LLMClient.__new__(LLMClient)  # bypass __init__
        llm.config = MagicMock()
        llm.chat = MagicMock(return_value='{"a": 1}')
        result = llm.json([{"role": "user", "content": "test"}])
        assert result == {"a": 1}
        # 一次成功不重试
        assert llm.chat.call_count == 1

    def test_invalid_json_retries_with_hint(self):
        llm = LLMClient.__new__(LLMClient)
        llm.config = MagicMock()
        # 第一次返回真正无法修复的非法 JSON（缺逗号），第二次返回合法
        llm.chat = MagicMock(side_effect=[
            '{"a": 1\n"b": 2}',  # 缺逗号，_repair_truncated_json 修不了
            '{"a": 1, "b": 2}',
        ])
        result = llm.json([{"role": "system", "content": "sys"}, {"role": "user", "content": "test"}])
        assert result == {"a": 1, "b": 2}
        assert llm.chat.call_count == 2
        # 第二次调用的 system message 应包含格式提醒
        second_call_messages = llm.chat.call_args_list[1][0][0]
        assert "格式提醒" in second_call_messages[0]["content"]

    def test_invalid_json_all_retries_falls_back_to_empty(self):
        """所有重试都失败 → 返回 {}（不抛异常）。"""
        llm = LLMClient.__new__(LLMClient)
        llm.config = MagicMock()
        llm.chat = MagicMock(side_effect=ValueError("LLM says no"))  # 任何错误
        # 临时降低重试次数以加快测试
        original_max = cfg.MAX_RETRIES
        cfg.MAX_RETRIES = 1
        try:
            # LLMError 会让 json 重试并最终 fallback → {}
            # 但 MagicMock 抛 ValueError 而非 LLMError，我的 except 只接 LLMError
            # 调整：让 mock 返回非法 JSON 字符串
            llm.chat = MagicMock(return_value='{"broken": ')
            result = llm.json([{"role": "user", "content": "test"}])
            assert result == {}  # 兜底返回空 dict
        finally:
            cfg.MAX_RETRIES = original_max

    def test_invalid_json_does_not_propagate(self):
        """关键测试：Q22 风格的 JSONDecodeError 不应向外抛出。"""
        llm = LLMClient.__new__(LLMClient)
        llm.config = MagicMock()
        # 模拟返回的 JSON 缺逗号（json.loads 抛 JSONDecodeError）
        llm.chat = MagicMock(return_value='{"a": 1\n"b": 2}')  # 缺逗号
        original_max = cfg.MAX_RETRIES
        cfg.MAX_RETRIES = 1
        try:
            # 不应抛异常
            result = llm.json([{"role": "user", "content": "test"}])
            assert result == {}
        finally:
            cfg.MAX_RETRIES = original_max

    def test_retry_appends_hint_to_system_message(self):
        """重试时应在原 system message 末尾追加 JSON 格式提醒。"""
        llm = LLMClient.__new__(LLMClient)
        llm.config = MagicMock()
        llm.chat = MagicMock(side_effect=[
            '{"bad',  # 第一次：坏 JSON
            '{"a": 1}',  # 第二次：好 JSON
        ])
        original_max = cfg.MAX_RETRIES
        cfg.MAX_RETRIES = 2
        try:
            llm.json([{"role": "system", "content": "你是助手"}, {"role": "user", "content": "test"}])
            # 第二次的 system message 应包含"格式提醒"
            second_messages = llm.chat.call_args_list[1][0][0]
            assert second_messages[0]["role"] == "system"
            assert "你是助手" in second_messages[0]["content"]
            assert "格式提醒" in second_messages[0]["content"]
        finally:
            cfg.MAX_RETRIES = original_max
