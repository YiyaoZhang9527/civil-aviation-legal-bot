"""SynthesisAgent 拒答兜底测试。"""

import pytest

from legalbot.agents import (
    RetrievalPlan,
    SynthesisAgent,
)
from legalbot.types import CitationCheck, Conflict, Evidence
from legalbot.llm import LLMClient


# ── Fake LLM ──────────────────────────────────────────────


class FakeLLM:
    """返回预设 chat / json 响应的假 LLM。"""

    def __init__(self, chat_response: str = "", json_response: dict | None = None):
        self.chat_response = chat_response
        self.json_response = json_response or {}
        self.chat_calls = 0
        self.json_calls = 0

    def chat(self, messages, **kw):
        self.chat_calls += 1
        return self.chat_response

    def json(self, messages, **kw):
        self.json_calls += 1
        return self.json_response


# ── 拒答模式识别 ──────────────────────────────────────────


class TestLooksLikeRefusal:
    def test_empty_text_is_refusal(self):
        assert SynthesisAgent._looks_like_refusal("") is True

    def test_detects_未找到(self):
        assert SynthesisAgent._looks_like_refusal("很抱歉，未找到相关法律条文。") is True

    def test_detects_无法确定(self):
        assert SynthesisAgent._looks_like_refusal("抱歉，无法确定答案。") is True

    def test_detects_建议咨询(self):
        assert SynthesisAgent._looks_like_refusal("建议您咨询专业法律人士。") is True

    def test_detects_partial_pattern(self):
        assert SynthesisAgent._looks_like_refusal("暂无法回答该问题。") is True

    def test_normal_answer_not_refusal(self):
        text = (
            "【结论】\n航空公司可在以下情形拒载：\n"
            "1. 旅客使用伪造证件\n"
            "2. 旅客拒绝安检\n\n"
            "【法律依据】《公共航空运输旅客服务管理规定》第三十一条"
        )
        assert SynthesisAgent._looks_like_refusal(text) is False

    def test_refusal_pattern_in_middle(self):
        # 兜底应捕获任何位置的拒绝词
        text = "部分情形可参考《XXX》第二条。但其他细节暂无法确定。"
        assert SynthesisAgent._looks_like_refusal(text) is True


# ── 兜底渲染 ─────────────────────────────────────────────


def _make_evidence() -> list[Evidence]:
    return [
        Evidence(
            law_id="ccar-271",
            law_title="公共航空运输旅客服务管理规定",
            node_id="article:31",
            article="第三十一条",
            text="有下列情况之一的，承运人应当拒绝运输：\n（一）使用伪造证件的旅客；\n（二）拒绝安全检查的旅客；\n（三）法律法规规定的其他情形。",
            score=0.5,
            source_file="data/法律数据/ccar-271.txt",
            source_anchor="#article-31",
        ),
        Evidence(
            law_id="ccar-271",
            law_title="公共航空运输旅客服务管理规定",
            node_id="article:32",
            article="第三十二条",
            text="旅客因本规定第三十一条被拒绝运输而要求退票的，承运人应当办理退票。",
            score=0.4,
            source_file="data/法律数据/ccar-271.txt",
            source_anchor="#article-32",
        ),
    ]


class TestRenderEvidenceFallback:
    def test_lists_each_evidence_with_law_and_article(self):
        ev = _make_evidence()
        result = SynthesisAgent._render_evidence_fallback("什么情况下可以拒载", RetrievalPlan(intent="legal"), ev)
        assert "公共航空运输旅客服务管理规定" in result
        assert "第三十一条" in result
        assert "第三十二条" in result
        assert "【相关法规】" in result
        assert "【结论】" in result
        assert "【说明】" in result

    def test_truncates_long_excerpts(self):
        long_text = "测试文本" * 100
        ev = [Evidence(
            law_id="x", law_title="测试法", node_id="article:1",
            article="第一条", text=long_text, score=0.1,
            source_file="x.txt", source_anchor="#1",
        )]
        result = SynthesisAgent._render_evidence_fallback("q", RetrievalPlan(intent="legal"), ev)
        # 摘要应被截断（≤ 80 字符 + …）
        assert "…" in result
        # 整段原文不应完整出现
        assert long_text not in result

    def test_empty_evidence_returns_empty_string(self):
        result = SynthesisAgent._render_evidence_fallback("q", RetrievalPlan(intent="legal"), [])
        assert result == ""

    def test_respects_max_items(self, monkeypatch):
        from legalbot import config
        monkeypatch.setattr(config, "SYNTHESIS_FALLBACK_MAX_ITEMS", 1, raising=False)
        ev = _make_evidence()
        result = SynthesisAgent._render_evidence_fallback("q", RetrievalPlan(intent="legal"), ev)
        # 只列第一条
        assert "第三十一条" in result
        assert "第三十二条" not in result

    def test_handles_missing_law_title(self):
        ev = [Evidence(
            law_id="x", law_title="", node_id="article:1",
            article="第一条", text="x", score=0.1,
            source_file="x.txt", source_anchor="#1",
        )]
        result = SynthesisAgent._render_evidence_fallback("q", RetrievalPlan(intent="legal"), ev)
        # fallback to law_id
        assert "《x》" in result

    def test_handles_empty_text(self):
        ev = [Evidence(
            law_id="x", law_title="测试法", node_id="article:1",
            article="第一条", text="", score=0.1,
            source_file="x.txt", source_anchor="#1",
        )]
        result = SynthesisAgent._render_evidence_fallback("q", RetrievalPlan(intent="legal"), ev)
        # 不应有"摘录："行
        assert "摘录：" not in result
        assert "测试法" in result


# ── compose_answer 集成测试 ────────────────────────────────


def _empty_citations() -> list[CitationCheck]:
    return [
        CitationCheck(claim="拒载情形", law_id="ccar-271", node_id="article:31",
                      status="unsupported", reason="CE=0.005", confidence=0.05),
    ]


class TestComposeAnswerFallback:
    def test_fallback_triggers_on_llm_refusal_with_evidence(self, monkeypatch):
        # 关闭 JSON mode 和 relevance gate，直接走 free-text
        from legalbot import config
        monkeypatch.setattr(config, "SYNTHESIS_JSON_MODE", False, raising=False)
        monkeypatch.setattr(config, "RELEVANCE_GATE_ENABLED", False, raising=False)
        monkeypatch.setattr(config, "SYNTHESIS_REFUSAL_FALLBACK", True, raising=False)

        llm = FakeLLM(chat_response="很抱歉，未找到直接回答您问题的法律条文。")
        agent = SynthesisAgent(llm)  # type: ignore[arg-type]

        result = agent.compose_answer(
            question="什么情况下可以拒载",
            plan=RetrievalPlan(intent="legal"),
            evidence=_make_evidence(),
            citations=_empty_citations(),
            conflicts=[],
        )

        # 触发了兜底
        assert "【相关法规】" in result
        assert "公共航空运输旅客服务管理规定" in result
        assert "第三十一条" in result
        # 兜底措辞——不含拒绝词，避免被 P1-A 误判
        assert "以下为系统检索到的可能相关法规" in result
        assert "未找到" not in result

    def test_fallback_skipped_when_llm_provides_substantive_answer(self, monkeypatch):
        from legalbot import config
        monkeypatch.setattr(config, "SYNTHESIS_JSON_MODE", False, raising=False)
        monkeypatch.setattr(config, "RELEVANCE_GATE_ENABLED", False, raising=False)
        monkeypatch.setattr(config, "SYNTHESIS_REFUSAL_FALLBACK", True, raising=False)

        substantive = (
            "【结论】\n航空公司可在旅客使用伪造证件时拒载。\n\n"
            "【法律依据】《公共航空运输旅客服务管理规定》第三十一条"
        )
        llm = FakeLLM(chat_response=substantive)
        agent = SynthesisAgent(llm)  # type: ignore[arg-type]

        result = agent.compose_answer(
            question="什么情况下可以拒载",
            plan=RetrievalPlan(intent="legal"),
            evidence=_make_evidence(),
            citations=[],
            conflicts=[],
        )

        # 没有触发了兜底（应保留 LLM 原答案）
        assert result == substantive

    def test_fallback_disabled_via_config(self, monkeypatch):
        from legalbot import config
        monkeypatch.setattr(config, "SYNTHESIS_JSON_MODE", False, raising=False)
        monkeypatch.setattr(config, "RELEVANCE_GATE_ENABLED", False, raising=False)
        monkeypatch.setattr(config, "SYNTHESIS_REFUSAL_FALLBACK", False, raising=False)

        llm = FakeLLM(chat_response="很抱歉，未找到相关法律条文。")
        agent = SynthesisAgent(llm)  # type: ignore[arg-type]

        result = agent.compose_answer(
            question="什么情况下可以拒载",
            plan=RetrievalPlan(intent="legal"),
            evidence=_make_evidence(),
            citations=[],
            conflicts=[],
        )

        # 兜底关闭，LLM 原答案保留
        assert "未找到" in result
        assert "【相关法规】" not in result

    def test_fallback_skipped_when_evidence_empty(self, monkeypatch):
        from legalbot import config
        monkeypatch.setattr(config, "SYNTHESIS_JSON_MODE", False, raising=False)
        monkeypatch.setattr(config, "RELEVANCE_GATE_ENABLED", False, raising=False)
        monkeypatch.setattr(config, "SYNTHESIS_REFUSAL_FALLBACK", True, raising=False)

        llm = FakeLLM(chat_response="很抱歉，未找到相关法律条文。")
        agent = SynthesisAgent(llm)  # type: ignore[arg-type]

        result = agent.compose_answer(
            question="q",
            plan=RetrievalPlan(intent="legal"),
            evidence=[],
            citations=[],
            conflicts=[],
        )

        # 无 evidence 不触发兜底
        assert "【相关法规】" not in result
        assert "未找到" in result
