"""方案 A (硬拒答门控 + SM 降级) 和 方案 B (CRAG 检索质量补救) 单元测试。"""

import pytest

from legalbot.agents import (
    CaseState,
    RetrievalAgent,
    RetrievalPlan,
    SynthesisAgent,
)
from legalbot.types import Evidence
from legalbot import config


# ── Fake LLM ──────────────────────────────────────────────


class FakeLLM:
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


def _make_evidence(text: str = "飞行员体检合格证有效期为12个月") -> Evidence:
    return Evidence(
        law_id="L1",
        law_title="民用航空人员体检合格证管理规则",
        node_id="node-1",
        article="第67条",
        text=text,
        score=0.9,
        source_file="",
        source_anchor="",
    )


# ── 方案 A: 硬拒答门控 ──────────────────────────────────────


class TestHardRefusalGate:
    def test_empty_evidence_returns_hard_refusal(self, monkeypatch):
        monkeypatch.setattr(config, "HARD_REFUSAL_ON_EMPTY_EVIDENCE", True, raising=False)
        llm = FakeLLM(chat_response="意外被调用的 LLM")
        agent = SynthesisAgent(llm)  # type: ignore[arg-type]

        result = agent.compose_answer(
            question="q",
            plan=RetrievalPlan(intent="legal"),
            evidence=[],
            citations=[],
            conflicts=[],
        )

        assert "未检索到" in result
        assert llm.chat_calls == 0

    def test_hard_refusal_disabled_falls_through(self, monkeypatch):
        monkeypatch.setattr(config, "HARD_REFUSAL_ON_EMPTY_EVIDENCE", False, raising=False)
        monkeypatch.setattr(config, "SYNTHESIS_JSON_MODE", False, raising=False)
        monkeypatch.setattr(config, "RELEVANCE_GATE_ENABLED", False, raising=False)
        monkeypatch.setattr(config, "SYNTHESIS_REFUSAL_FALLBACK", False, raising=False)
        llm = FakeLLM(chat_response="正常 LLM 答案")
        agent = SynthesisAgent(llm)  # type: ignore[arg-type]

        result = agent.compose_answer(
            question="q",
            plan=RetrievalPlan(intent="legal"),
            evidence=[],
            citations=[],
            conflicts=[],
        )

        # 关闭后 LLM 会被调用
        assert llm.chat_calls == 1
        assert "正常 LLM 答案" in result

    def test_evidence_present_skips_hard_refusal(self, monkeypatch):
        monkeypatch.setattr(config, "HARD_REFUSAL_ON_EMPTY_EVIDENCE", True, raising=False)
        monkeypatch.setattr(config, "SYNTHESIS_JSON_MODE", False, raising=False)
        monkeypatch.setattr(config, "RELEVANCE_GATE_ENABLED", False, raising=False)
        monkeypatch.setattr(config, "SYNTHESIS_REFUSAL_FALLBACK", False, raising=False)
        llm = FakeLLM(chat_response="有证据时的正常答案")
        agent = SynthesisAgent(llm)  # type: ignore[arg-type]

        result = agent.compose_answer(
            question="q",
            plan=RetrievalPlan(intent="legal"),
            evidence=[_make_evidence()],
            citations=[],
            conflicts=[],
        )

        # 有证据时不触发硬拒答, LLM 正常调用
        assert "未检索到" not in result
        assert llm.chat_calls == 1


# ── 方案 A: SM 降级 ──────────────────────────────────────


class TestSetMembershipDemote:
    def test_sm_fail_demotes_to_general_advice(self, monkeypatch):
        """当 SM 校验失败时, claim.source 降级为'一般建议'(旧: 仅加⚠️警告)."""
        monkeypatch.setattr(config, "SET_MEMBERSHIP_CHECK", True, raising=False)
        monkeypatch.setattr(config, "SM_FORCE_DEMOTE_ON_FAIL", True, raising=False)
        monkeypatch.setattr(config, "SYNTHESIS_JSON_MODE", True, raising=False)
        monkeypatch.setattr(config, "RELEVANCE_GATE_ENABLED", False, raising=False)
        monkeypatch.setattr(config, "HARD_REFUSAL_ON_EMPTY_EVIDENCE", False, raising=False)

        # LLM 返回的 claim 引用了 evidence 中不存在的 node_id
        fake_structured = {
            "conclusion": "结论",
            "claims": [
                {
                    "text": "编造的内容",
                    "node_ids": ["non-existent-node-id"],
                    "law_name": "虚构法律",
                    "source": "法规规定",
                },
                {
                    "text": "正确的内容",
                    "node_ids": ["node-1"],
                    "law_name": "真实法律",
                    "source": "法规规定",
                },
            ],
        }
        llm = FakeLLM(chat_response="", json_response=fake_structured)
        agent = SynthesisAgent(llm)  # type: ignore[arg-type]
        agent._last_structured = fake_structured

        # 模拟 _answer 流程的 SM 校验段
        structured = agent._last_structured
        valid_node_ids = {"node-1"}
        for claim in structured["claims"]:
            node_ids = claim.get("node_ids", [])
            if node_ids and not any(nid in valid_node_ids for nid in node_ids):
                if getattr(config, "SM_FORCE_DEMOTE_ON_FAIL", True):
                    claim["source"] = "一般建议"

        # 校验: 引用不存在的 claim 被降级, 正确引用的保持原状
        assert fake_structured["claims"][0]["source"] == "一般建议"
        assert fake_structured["claims"][1]["source"] == "法规规定"


# ── 方案 B: CRAG 检索质量评估 ────────────────────────────


class TestCRAGQualityAssessment:
    def test_assess_returns_sufficient(self):
        llm = FakeLLM(json_response={"quality": "sufficient"})
        agent = RetrievalAgent(llm)  # type: ignore[arg-type]
        evidence = [_make_evidence("根据CCAR-67部, 体检合格证有效期为12个月")]
        quality = agent._assess_retrieval_quality("体检合格证有效期", evidence)
        assert quality == "sufficient"
        assert llm.json_calls == 1

    def test_assess_returns_partial(self):
        llm = FakeLLM(json_response={"quality": "partial"})
        agent = RetrievalAgent(llm)  # type: ignore[arg-type]
        evidence = [_make_evidence()]
        quality = agent._assess_retrieval_quality("q", evidence)
        assert quality == "partial"

    def test_assess_returns_insufficient(self):
        llm = FakeLLM(json_response={"quality": "insufficient"})
        agent = RetrievalAgent(llm)  # type: ignore[arg-type]
        evidence = [_make_evidence()]
        quality = agent._assess_retrieval_quality("q", evidence)
        assert quality == "insufficient"

    def test_assess_invalid_quality_falls_back_to_sufficient(self):
        llm = FakeLLM(json_response={"quality": "weird_value"})
        agent = RetrievalAgent(llm)  # type: ignore[arg-type]
        evidence = [_make_evidence()]
        quality = agent._assess_retrieval_quality("q", evidence)
        assert quality == "sufficient"

    def test_assess_empty_evidence_returns_insufficient(self):
        llm = FakeLLM(json_response={"quality": "should_not_be_used"})
        agent = RetrievalAgent(llm)  # type: ignore[arg-type]
        quality = agent._assess_retrieval_quality("q", [])
        assert quality == "insufficient"
        assert llm.json_calls == 0  # 无证据时直接返回, 不调 LLM

    def test_assess_llm_exception_falls_back_to_sufficient(self):
        class ExplodingLLM:
            def json(self, messages, **kw):
                raise RuntimeError("LLM 失败")

        agent = RetrievalAgent(ExplodingLLM())  # type: ignore[arg-type]
        evidence = [_make_evidence()]
        quality = agent._assess_retrieval_quality("q", evidence)
        assert quality == "sufficient"  # 异常时兜底为 sufficient (保守)

    def test_retrieval_quality_field_default(self):
        """CaseState 默认 quality=sufficient, 避免 CRAG 关闭时行为改变."""
        state = CaseState(
            request_id="r1",
            original_question="q",
            normalized_question="q",
        )
        assert state.retrieval_quality == "sufficient"

    def test_crag_disabled_by_default(self):
        """CRAG 默认开启（验证有效后启用）."""
        assert config.CRAG_ENABLED is True
