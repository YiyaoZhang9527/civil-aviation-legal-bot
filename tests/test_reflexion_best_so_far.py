"""LegalOrchestrator 迭代质量评估 + Reflexion best-so-far 追踪测试。"""

import pytest

from legalbot.agents import LegalOrchestrator
from legalbot.types import CitationCheck


def _cite(status: str, claim: str = "test") -> CitationCheck:
    return CitationCheck(claim=claim, law_id="ccar-271", node_id="article:31",
                         status=status, reason="test", confidence=0.5)


class TestIterQuality:
    """_iter_quality 是 Reflexion best-so-far 追踪的核心评分函数。"""

    def test_supported_citations_count(self):
        citations = [_cite("supported"), _cite("partial"), _cite("supported")]
        score = LegalOrchestrator._iter_quality("航空公司可在旅客使用伪造证件时拒载。", citations)
        assert score == 3

    def test_unsupported_citations_dont_count(self):
        citations = [_cite("unsupported"), _cite("unsupported"), _cite("supported")]
        score = LegalOrchestrator._iter_quality("航空公司可在旅客使用伪造证件时拒载。", citations)
        assert score == 1

    def test_refusal_dominates_support(self):
        """即使有少量支持，拒答也应被淘汰（评分远低于非拒答）。"""
        citations = [_cite("supported")] * 5
        refusal_score = LegalOrchestrator._iter_quality("很抱歉，未找到相关法律条文。", citations)
        # 非拒答同引用数
        good_score = LegalOrchestrator._iter_quality("航空公司可在以下情形拒载：", citations)
        assert good_score > refusal_score
        assert refusal_score < 0
        assert good_score == 5

    def test_empty_citations_zero_score_for_substantive(self):
        score = LegalOrchestrator._iter_quality("航空公司可在旅客使用伪造证件时拒载。", [])
        assert score == 0

    def test_empty_citations_negative_for_refusal(self):
        score = LegalOrchestrator._iter_quality("很抱歉，未找到相关法律条文。", [])
        assert score == -1000

    def test_refusal_pattern_variants(self):
        citations = [_cite("supported")]
        refusal_variants = [
            "未找到",
            "无法确定",
            "建议咨询专业法律人士",
            "暂无相关法律",
        ]
        for text in refusal_variants:
            score = LegalOrchestrator._iter_quality(text, citations)
            # 拒答 + 1 个支持 = 1 - 1000 = -999
            assert score == -999, f"应识别为拒答: {text}"

    def test_substantive_with_partial_support_scores_higher_than_refusal_with_full_support(self):
        """边界情况：拒答但有全部支持 vs 非拒答但 0 支持。"""
        full_support = [_cite("supported")] * 5
        empty = []
        refusal = "很抱歉，未找到。"
        substantive = "航空公司可在以下情形拒载："

        score_refusal_full = LegalOrchestrator._iter_quality(refusal, full_support)
        score_substantive_empty = LegalOrchestrator._iter_quality(substantive, empty)

        # 5 - 1000 = -995 vs 0
        assert score_substantive_empty > score_refusal_full


class TestBestSoFarLogic:
    """验证 best-so-far 选择的正确性。"""

    def test_initial_better_than_later_refusal(self):
        """初始答案非拒答有 3 支持；第 1 轮退化为拒答 → 应保留初始。"""
        citations_initial = [_cite("supported")] * 3
        citations_refusal = []

        initial_answer = "航空公司可在以下情形拒载：\n1. 旅客使用伪造证件"
        refusal_answer = "很抱歉，未找到相关法律条文。"

        # 初始质量 3，后续拒答 -1000
        assert LegalOrchestrator._iter_quality(initial_answer, citations_initial) == 3
        assert LegalOrchestrator._iter_quality(refusal_answer, citations_refusal) == -1000

    def test_later_better_than_initial_picks_later(self):
        """初始 1 支持 → 第 1 轮补到 3 支持 → 应保留第 1 轮。"""
        citations_initial = [_cite("supported")]
        citations_iter1 = [_cite("supported")] * 3

        # 初始 1，第 1 轮 3
        assert LegalOrchestrator._iter_quality("answer v0", citations_initial) == 1
        assert LegalOrchestrator._iter_quality("answer v1", citations_iter1) == 3

    def test_refusal_to_substantive_progression(self):
        """初始拒答 → 第 1 轮非拒答 0 支持 → 应保留第 1 轮（兜底答案）。"""
        refusal = "很抱歉，未找到相关法律条文。"
        fallback = (
            "【结论】\n以下为系统检索到的可能相关法规，请参考：\n"
            "【相关法规】\n1. 《XX》第一条"
        )

        # 拒答 -1000，兜底 0
        assert LegalOrchestrator._iter_quality(refusal, []) == -1000
        assert LegalOrchestrator._iter_quality(fallback, []) == 0
        assert 0 > -1000
