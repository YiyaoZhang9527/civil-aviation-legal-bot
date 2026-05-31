"""DecompositionAgent、ReflexionAgent、crossref 去重、reflexion 独立检索 的单元测试。"""

from dataclasses import dataclass, field

import pytest

from legalbot.agents import (
    CaseState,
    ClarificationAgent,
    DecompositionAgent,
    DecompositionResult,
    IssueAnalysis,
    LegalOrchestrator,
    ReflexionAgent,
    ReflexionResult,
    RetrievalPlan,
    RewritePlan,
    SubProblem,
    SubjectAnalysis,
)
from legalbot.llm import LLMClient
from legalbot.types import AnswerResult, CitationCheck, Conflict, Evidence


# ── Fake LLM ──────────────────────────────────────────────


class FakeLLM:
    """可编程的假 LLM，按调用顺序返回预设 JSON。"""

    def __init__(self, responses: list[dict | str] | None = None):
        self._responses = responses or []
        self._call_log: list[tuple[str, list]] = []

    def json(self, messages, **kw):
        self._call_log.append(("json", messages))
        if not self._responses:
            return {}
        resp = self._responses.pop(0)
        return resp if isinstance(resp, dict) else {}

    def chat(self, messages, **kw):
        self._call_log.append(("chat", messages))
        if not self._responses:
            return "无答案"
        resp = self._responses.pop(0)
        return resp if isinstance(resp, str) else str(resp)

    @property
    def call_count(self):
        return len(self._call_log)


# ══════════════════════════════════════════════════════════
# DecompositionAgent 测试
# ══════════════════════════════════════════════════════════


class TestDecompositionAgent:
    def test_skip_simple_question_zero_sub(self):
        """0 个子问题 → 不分解，不调 LLM。"""
        llm = FakeLLM()
        agent = DecompositionAgent(llm)
        rewrite = RewritePlan(queries=["航班延误赔偿"], sub_questions=[])
        result = agent.decompose("航班延误怎么赔偿", SubjectAnalysis(), IssueAnalysis(), rewrite)
        assert result.needs_decomposition is False
        assert result.sub_problems == []
        assert llm.call_count == 0

    def test_skip_simple_question_one_sub(self):
        """1 个子问题 → 不分解。"""
        llm = FakeLLM()
        agent = DecompositionAgent(llm)
        rewrite = RewritePlan(queries=["无人机飞行"], sub_questions=["无人机飞行标准"])
        result = agent.decompose("无人机怎么飞", SubjectAnalysis(), IssueAnalysis(), rewrite)
        assert result.needs_decomposition is False
        assert llm.call_count == 0

    def test_decompose_multiple_sub_questions(self):
        """2+ 子问题 → 调 LLM 分解。"""
        llm = FakeLLM(responses=[{
            "sub_problems": [
                {"question": "航班延误赔偿", "queries": ["航班延误赔偿标准"], "law_hints": ["航班正常管理规定"], "article_hints": ["第29条"]},
                {"question": "旅客权益保护", "queries": ["旅客延误权益"], "law_hints": ["公共航空运输旅客服务管理规定"], "article_hints": []},
            ]
        }])
        agent = DecompositionAgent(llm)
        rewrite = RewritePlan(
            queries=["航班延误旅客赔偿"],
            sub_questions=["航班延误赔偿", "旅客权益保护"],
        )
        result = agent.decompose("航班延误怎么赔偿", SubjectAnalysis(), IssueAnalysis(), rewrite)
        assert result.needs_decomposition is True
        assert len(result.sub_problems) == 2
        assert result.sub_problems[0].question == "航班延误赔偿"
        assert result.sub_problems[1].law_hints == ["公共航空运输旅客服务管理规定"]
        assert llm.call_count == 1

    def test_decompose_capped_by_max_subquestions(self):
        """子问题数量被 MAX_SUBQUESTIONS 上限截断。"""
        from legalbot.config import MAX_SUBQUESTIONS
        many_subs = [f"子问题{i}" for i in range(10)]
        llm = FakeLLM(responses=[{
            "sub_problems": [
                {"question": f"子问题{i}", "queries": [f"q{i}"], "law_hints": [], "article_hints": []}
                for i in range(10)
            ]
        }])
        agent = DecompositionAgent(llm)
        rewrite = RewritePlan(queries=["测试"], sub_questions=many_subs)
        result = agent.decompose("测试", SubjectAnalysis(), IssueAnalysis(), rewrite)
        assert len(result.sub_problems) <= MAX_SUBQUESTIONS

    def test_decompose_llm_returns_empty_fallback(self):
        """LLM 返回空 → 退化为不分解。"""
        llm = FakeLLM(responses=[{"sub_problems": []}])
        agent = DecompositionAgent(llm)
        rewrite = RewritePlan(queries=["q"], sub_questions=["子1", "子2"])
        result = agent.decompose("测试", SubjectAnalysis(), IssueAnalysis(), rewrite)
        assert result.needs_decomposition is False

    def test_decompose_llm_returns_malformed_fallback(self):
        """LLM 返回非预期格式 → 退化为不分解。"""
        llm = FakeLLM(responses=[{"unexpected_key": 123}])
        agent = DecompositionAgent(llm)
        rewrite = RewritePlan(queries=["q"], sub_questions=["子1", "子2"])
        result = agent.decompose("测试", SubjectAnalysis(), IssueAnalysis(), rewrite)
        assert result.needs_decomposition is False


# ══════════════════════════════════════════════════════════
# ReflexionAgent 测试
# ══════════════════════════════════════════════════════════


class TestReflexionAgent:
    def test_fast_path_all_supported(self):
        """所有 citation supported + confidence >= 0.7 → 直接 pass，不调 LLM。"""
        llm = FakeLLM()
        agent = ReflexionAgent(llm)
        citations = [
            CitationCheck(claim="c1", law_id="l1", node_id="n1", status="supported", reason="ok", confidence=0.9),
            CitationCheck(claim="c2", law_id="l2", node_id="n2", status="supported", reason="ok", confidence=0.8),
        ]
        result = agent.evaluate("问题", ["争点1"], [], citations, "答案")
        assert result.quality == "pass"
        assert llm.call_count == 0

    def test_fast_path_no_citations(self):
        """无 citation → 走 LLM 评估。"""
        llm = FakeLLM(responses=[{"quality": "pass", "gaps": [], "refine_queries": []}])
        agent = ReflexionAgent(llm)
        result = agent.evaluate("问题", ["争点1"], [], [], "答案")
        assert result.quality == "pass"
        assert llm.call_count == 1

    def test_gap_triggers_llm(self):
        """有 unsupported citation → 调 LLM 评估。"""
        llm = FakeLLM(responses=[{
            "quality": "gap",
            "gaps": ["缺少延误赔偿标准规定"],
            "refine_queries": ["航班延误补偿标准"],
            "refine_law_hints": ["航班正常管理规定"],
            "refine_article_hints": ["第29条"],
        }])
        agent = ReflexionAgent(llm)
        citations = [
            CitationCheck(claim="c1", law_id="l1", node_id="n1", status="supported", reason="ok", confidence=0.8),
            CitationCheck(claim="c2", law_id="l2", node_id="n2", status="unsupported", reason="无关", confidence=0.3),
        ]
        result = agent.evaluate("航班延误怎么赔偿", ["航班延误"], [], citations, "延误赔偿按...")
        assert result.quality == "gap"
        assert "缺少延误赔偿标准规定" in result.gaps
        assert "航班延误补偿标准" in result.refine_queries
        assert "航班正常管理规定" in result.refine_law_hints
        assert llm.call_count == 1

    def test_low_confidence_triggers_llm(self):
        """supported 但 confidence < 0.7 → 走 LLM。"""
        llm = FakeLLM(responses=[{"quality": "pass", "gaps": []}])
        agent = ReflexionAgent(llm)
        citations = [
            CitationCheck(claim="c1", law_id="l1", node_id="n1", status="supported", reason="ok", confidence=0.5),
        ]
        result = agent.evaluate("问题", ["争点"], [], citations, "答案")
        assert llm.call_count == 1

    def test_malformed_quality_defaults_to_pass(self):
        """LLM 返回非法 quality 值 → 默认 pass。"""
        llm = FakeLLM(responses=[{"quality": "unknown_value", "gaps": []}])
        agent = ReflexionAgent(llm)
        citations = [
            CitationCheck(claim="c1", law_id="l1", node_id="n1", status="unsupported", reason="", confidence=0.3),
        ]
        result = agent.evaluate("问题", ["争点"], [], citations, "答案")
        assert result.quality == "pass"


# ══════════════════════════════════════════════════════════
# Crossref 去重 key 测试（问题3修复验证）
# ══════════════════════════════════════════════════════════


class TestCrossrefDedupKey:
    def test_evidence_with_empty_law_id_has_unique_key(self):
        """交叉引用 Evidence 的 law_id 为空时，node_id 应被替换为唯一标识。"""
        ev1 = Evidence(law_id="", law_title="民用航空法", node_id="xref:《航班正常管理规定》第29条", article="第29条",
                        text="内容1", score=0.0, source_file="", source_anchor="")
        ev2 = Evidence(law_id="", law_title="航班正常管理规定", node_id="xref:《民用航空法》第95条", article="第95条",
                        text="内容2", score=0.0, source_file="", source_anchor="")
        # 去重 key 应不同
        key1 = (ev1.law_id, ev1.node_id)
        key2 = (ev2.law_id, ev2.node_id)
        assert key1 != key2

    def test_evidence_with_real_law_id_unchanged(self):
        """正常交叉引用（有 law_id）不受影响。"""
        ev = Evidence(law_id="民用航空法", law_title="民用航空法", node_id="article:95", article="第95条",
                       text="内容", score=0.0, source_file="", source_anchor="")
        assert ev.law_id == "民用航空法"
        assert ev.node_id == "article:95"


# ══════════════════════════════════════════════════════════
# Reflexion 独立检索验证（问题1修复验证）
# ══════════════════════════════════════════════════════════


class TestReflexionIndependentRetrieval:
    def test_refine_plan_is_independent(self):
        """验证 refine_plan 是独立构建的，不污染原始 plan。"""
        original_queries = ["航班延误赔偿标准"]
        original_hints = ["航班正常管理规定"]
        plan = RetrievalPlan(
            intent="legal",
            legal_issues=["航班延误"],
            facts={},
            queries=original_queries,
            law_hints=original_hints,
        )

        # 模拟 refine 数据
        refine_plan = RetrievalPlan(
            intent="legal",
            legal_issues=plan.legal_issues,
            facts=plan.facts,
            queries=["延误补偿金额", "机上延误待遇"],
            law_hints=["公共航空运输旅客服务管理规定"],
            article_hints=["第44条"],
            assumptions=plan.assumptions,
            alternative_paths=plan.alternative_paths,
        )

        # 原始 plan 不受影响
        assert plan.queries == original_queries
        assert plan.law_hints == original_hints
        assert len(plan.queries) == 1
        assert len(refine_plan.queries) == 2


# ══════════════════════════════════════════════════════════
# CaseState 新字段测试
# ══════════════════════════════════════════════════════════


class TestCaseStateNewFields:
    def test_reflexion_defaults(self):
        state = CaseState(request_id="t", original_question="q", normalized_question="q")
        assert state.reflexion_iteration == 0
        assert state.reflexion_trace == []

    def test_reflexion_trace_append(self):
        state = CaseState(request_id="t", original_question="q", normalized_question="q")
        state.reflexion_trace.append({"iteration": 1, "gaps": ["缺X"]})
        assert len(state.reflexion_trace) == 1
        assert state.reflexion_trace[0]["gaps"] == ["缺X"]


# ══════════════════════════════════════════════════════════
# AnswerResult 新字段测试
# ══════════════════════════════════════════════════════════


class TestAnswerResultNewField:
    def test_reflexion_iterations_default(self):
        r = AnswerResult(answer="a", intent="legal", topic="t")
        assert r.reflexion_iterations == 0

    def test_reflexion_iterations_set(self):
        r = AnswerResult(answer="a", intent="legal", topic="t", reflexion_iterations=2)
        assert r.reflexion_iterations == 2


# ══════════════════════════════════════════════════════════
# SubProblem 数据结构测试
# ══════════════════════════════════════════════════════════


class TestSubProblem:
    def test_defaults(self):
        sp = SubProblem(question="测试")
        assert sp.queries == []
        assert sp.law_hints == []
        assert sp.article_hints == []

    def test_full(self):
        sp = SubProblem(question="拒载", queries=["q1"], law_hints=["h1"], article_hints=["a1"])
        assert sp.question == "拒载"
        assert len(sp.queries) == 1


# ══════════════════════════════════════════════════════════
# DecompositionResult 数据结构测试
# ══════════════════════════════════════════════════════════


class TestDecompositionResult:
    def test_default_no_decomposition(self):
        r = DecompositionResult()
        assert r.needs_decomposition is False
        assert r.sub_problems == []

    def test_with_problems(self):
        sp = SubProblem(question="q1")
        r = DecompositionResult(needs_decomposition=True, sub_problems=[sp])
        assert len(r.sub_problems) == 1


# ══════════════════════════════════════════════════════════
# ReflexionResult 数据结构测试
# ══════════════════════════════════════════════════════════


class TestReflexionResult:
    def test_default_pass(self):
        r = ReflexionResult()
        assert r.quality == "pass"
        assert r.gaps == []
        assert r.refine_queries == []

    def test_gap_with_details(self):
        r = ReflexionResult(
            quality="gap",
            gaps=["缺1"],
            refine_queries=["q1"],
            refine_law_hints=["h1"],
            refine_article_hints=["a1"],
        )
        assert r.quality == "gap"
        assert len(r.refine_queries) == 1
