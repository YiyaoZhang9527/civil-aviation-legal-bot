"""Unit tests for evidence sorting by CE citation status (P1-C: synthesis LLM 优先 supported)."""
from unittest.mock import MagicMock
from legalbot.agents import SynthesisAgent
from legalbot.types import Evidence, CitationCheck


def _make_ev(nid, title="某规则", article="第一条"):
    return Evidence(node_id=nid, law_id="x", law_title=title, article=article, text="...",
                    source_file="", source_anchor=article, score=0.5)


def _make_cite(nid, status, conf=0.9):
    return CitationCheck(
        claim="", law_id="x", node_id=nid,
        status=status, reason="", quote="", confidence=conf,
    )


def test_evidence_sorted_supported_first():
    """supported 排前，unsupported 排后。"""
    a = SynthesisAgent(llm=MagicMock())
    ev1 = _make_ev("a")
    ev2 = _make_ev("b")
    ev3 = _make_ev("c")
    evs = [ev1, ev2, ev3]
    cites = [
        _make_cite("a", "unsupported"),
        _make_cite("b", "supported"),
        _make_cite("c", "partial"),
    ]
    sorted_evs = a._sort_evidence_by_status(evs, cites)
    assert [e.node_id for e in sorted_evs] == ["b", "c", "a"]


def test_evidence_unsupported_only():
    """全部 unsupported 时保持原顺序（不破坏）。"""
    a = SynthesisAgent(llm=MagicMock())
    evs = [_make_ev(f"e{i}") for i in range(3)]
    cites = [_make_cite(f"e{i}", "unsupported") for i in range(3)]
    sorted_evs = a._sort_evidence_by_status(evs, cites)
    assert [e.node_id for e in sorted_evs] == ["e0", "e1", "e2"]


def test_evidence_no_citations():
    """无 citations 时保持原顺序。"""
    a = SynthesisAgent(llm=MagicMock())
    evs = [_make_ev(f"e{i}") for i in range(3)]
    sorted_evs = a._sort_evidence_by_status(evs, [])
    assert [e.node_id for e in sorted_evs] == ["e0", "e1", "e2"]


def test_evidence_partial_between_supported_unsupported():
    """partial 应排在 supported 之后、unsupported 之前。"""
    a = SynthesisAgent(llm=MagicMock())
    evs = [_make_ev(nid) for nid in ["x", "y", "z"]]
    cites = [
        _make_cite("x", "unsupported"),
        _make_cite("y", "partial"),
        _make_cite("z", "supported"),
    ]
    sorted_evs = a._sort_evidence_by_status(evs, cites)
    assert [e.node_id for e in sorted_evs] == ["z", "y", "x"]


def test_evidence_low_confidence_supported_downgraded():
    """低置信度 supported (conf<0.5) 降级为 partial，避免低质 supported 污染排序。"""
    a = SynthesisAgent(llm=MagicMock())
    evs = [_make_ev(nid) for nid in ["high", "low", "none"]]
    cites = [
        _make_cite("high", "supported", conf=0.95),  # 真支持
        _make_cite("low", "supported", conf=0.17),   # 弱支持（应降级 partial）
        _make_cite("none", "unsupported", conf=0.01),
    ]
    sorted_evs = a._sort_evidence_by_status(evs, cites)
    # 期望: high > low(降级为partial) > none
    assert [e.node_id for e in sorted_evs] == ["high", "low", "none"]


def test_evidence_high_confidence_supported_first_by_score():
    """多个高置信度 supported 时，按 score 降序排（CE confidence 越高越前）。"""
    a = SynthesisAgent(llm=MagicMock())
    evs = [_make_ev(nid) for nid in ["a", "b", "c"]]
    cites = [
        _make_cite("a", "supported", conf=0.95),
        _make_cite("b", "supported", conf=0.99),
        _make_cite("c", "supported", conf=0.80),
    ]
    sorted_evs = a._sort_evidence_by_status(evs, cites)
    # 期望: b(0.99) > a(0.95) > c(0.80)
    assert [e.node_id for e in sorted_evs] == ["b", "a", "c"]
