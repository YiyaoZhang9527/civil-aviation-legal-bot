"""eval_quality 解析/聚合逻辑的单测（不依赖 LLM）。"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.eval_quality import (
    aggregate_results,
    build_evidence_summaries,
    inter_rater_mqs,
    lemaj_score,
    load_bot_answers,
    mqs_score,
    faithfulness_score,
)


# ── 数据加载 ─────────────────────────────────────────────


class TestLoadBotAnswers:
    def test_loads_all_rows(self, tmp_path):
        import csv
        path = tmp_path / "x.csv"
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["question_id", "category", "question", "answer_full", "evidence_articles"])
            w.writeheader()
            w.writerow({"question_id": "Q01", "category": "A", "question": "q", "answer_full": "ans", "evidence_articles": "art1 | art2"})
        items = load_bot_answers(path)
        assert len(items) == 1
        assert items[0]["evidence_summary"] == "art1 | art2"


class TestBuildEvidenceSummaries:
    def test_empty_input(self):
        assert build_evidence_summaries("") == "（无可用证据摘要）"

    def test_truncates_to_8(self):
        arts = " | ".join(f"art{i}" for i in range(20))
        result = build_evidence_summaries(arts)
        # 保留前 8 条（art0-art7）
        assert "art0" in result and "art7" in result
        # 截断 art8 之后
        assert "art8" not in result
        assert "art9" not in result


# ── MQS 评分 ─────────────────────────────────────────────


class TestMqsScore:
    def test_full_marks(self):
        llm = MagicMock()
        llm.json.return_value = {
            "q_match": 2, "law_correct": 2, "coverage": 2, "calibration": 2, "format": 2,
            "reasons": {"q_match": "good"},
        }
        result = mqs_score("q", "ans", "evid", llm)
        assert result["q_match"] == 2
        assert result["weighted_score"] == 100.0
        assert result["reasons"]["q_match"] == "good"

    def test_zero_marks(self):
        llm = MagicMock()
        llm.json.return_value = {
            "q_match": 0, "law_correct": 0, "coverage": 0, "calibration": 0, "format": 0,
        }
        result = mqs_score("q", "ans", "evid", llm)
        assert result["weighted_score"] == 0.0

    def test_weighted_calculation(self):
        """权重：Q-match 30% + Law-correct 30% + Coverage 20% + Calib 10% + Format 10%"""
        llm = MagicMock()
        # Q=2, Law=0, Coverage=2, Calib=2, Format=2
        # 加权和 = 0.3*2 + 0.3*0 + 0.2*2 + 0.1*2 + 0.1*2 = 0.6 + 0 + 0.4 + 0.2 + 0.2 = 1.4
        # 归一化到 0-100: 1.4/2*100 = 70
        llm.json.return_value = {
            "q_match": 2, "law_correct": 0, "coverage": 2, "calibration": 2, "format": 2,
        }
        result = mqs_score("q", "ans", "evid", llm)
        assert result["weighted_score"] == 70.0

    def test_robust_to_missing_dims(self):
        """LLM 漏掉某维度时，默认为 0。"""
        llm = MagicMock()
        llm.json.return_value = {"q_match": 2}  # 其他维度缺失
        result = mqs_score("q", "ans", "evid", llm)
        assert result["q_match"] == 2
        assert result["law_correct"] == 0
        # Q=2, 其他全 0: 0.3*2/2*100 = 30
        assert result["weighted_score"] == 30.0


# ── LeMAJ 评分 ───────────────────────────────────────────


class TestLemajScore:
    def test_aggregates_three_dimensions(self):
        llm = MagicMock()
        llm.json.return_value = {
            "ldps": [
                {"text": "LDP1", "supported": True, "correct": True, "relevant": True},
                {"text": "LDP2", "supported": True, "correct": False, "relevant": True},
                {"text": "LDP3", "supported": False, "correct": True, "relevant": True},
            ],
        }
        result = lemaj_score("q", "ans", "evid", llm)
        assert result["summary"]["total_ldps"] == 3
        # supported: 2/3
        assert result["summary"]["supported_pct"] == pytest.approx(0.667, abs=0.01)
        # correct: 2/3
        assert result["summary"]["correct_pct"] == pytest.approx(0.667, abs=0.01)
        # relevant: 3/3
        assert result["summary"]["relevant_pct"] == 1.0

    def test_empty_ldps_safe(self):
        llm = MagicMock()
        llm.json.return_value = {"ldps": []}
        result = lemaj_score("q", "ans", "evid", llm)
        assert result["summary"]["total_ldps"] == 0
        assert result["summary"]["supported_pct"] == 0

    def test_handles_non_list_response(self):
        llm = MagicMock()
        llm.json.return_value = {"ldps": "not a list"}  # LLM 乱输出
        result = lemaj_score("q", "ans", "evid", llm)
        assert result["summary"]["total_ldps"] == 0


# ── Faithfulness 评分 ───────────────────────────────────


class TestFaithfulnessScore:
    def test_4_tier_aggregation(self):
        llm = MagicMock()
        llm.json.return_value = {
            "claims": [
                {"text": "c1", "status": "faithful"},
                {"text": "c2", "status": "faithful"},
                {"text": "c3", "status": "partial"},
                {"text": "c4", "status": "unverifiable"},
                {"text": "c5", "status": "hallucinated"},
            ],
        }
        result = faithfulness_score("q", "ans", "evid", llm)
        assert result["summary"]["total_claims"] == 5
        assert result["summary"]["faithful_pct"] == 0.4
        assert result["summary"]["partial_pct"] == 0.2
        assert result["summary"]["unverifiable_pct"] == 0.2
        assert result["summary"]["hallucinated_pct"] == 0.2

    def test_unknown_status_treated_as_unverifiable(self):
        llm = MagicMock()
        llm.json.return_value = {
            "claims": [
                {"text": "c1", "status": "unknown_status"},
            ],
        }
        result = faithfulness_score("q", "ans", "evid", llm)
        assert result["summary"]["unverifiable_pct"] == 1.0

    def test_empty_claims_safe(self):
        llm = MagicMock()
        llm.json.return_value = {"claims": []}
        result = faithfulness_score("q", "ans", "evid", llm)
        assert result["summary"]["total_claims"] == 0


# ── Inter-rater ──────────────────────────────────────────


class TestInterRaterMqs:
    def test_agreement_within_1_considered_agree(self):
        """两个模型在 ±1 分内算一致。"""
        # 直接 mock 整个 make_judge_llm 路径
        import tests.eval_quality as mod
        original = mod.make_judge_llm
        try:
            llm_a = MagicMock()
            llm_a.json.return_value = {
                "q_match": 2, "law_correct": 2, "coverage": 2, "calibration": 2, "format": 2,
            }
            llm_b = MagicMock()
            llm_b.json.return_value = {
                "q_match": 2, "law_correct": 1, "coverage": 2, "calibration": 2, "format": 1,
            }
            mod.make_judge_llm = lambda **kw: llm_a if "Qwen" not in str(kw) else llm_b
            ir = inter_rater_mqs("q", "ans", "evid")
            assert ir["n_agree_dims"] >= 4  # 至少 4 个维度一致
        finally:
            mod.make_judge_llm = original

    def test_full_disagreement_detected(self):
        import tests.eval_quality as mod
        original = mod.make_judge_llm
        try:
            llm_a = MagicMock()
            llm_a.json.return_value = {
                "q_match": 2, "law_correct": 2, "coverage": 2, "calibration": 2, "format": 2,
            }
            llm_b = MagicMock()
            llm_b.json.return_value = {
                "q_match": 0, "law_correct": 0, "coverage": 0, "calibration": 0, "format": 0,
            }
            mod.make_judge_llm = lambda **kw: llm_a if "Qwen" not in str(kw) else llm_b
            ir = inter_rater_mqs("q", "ans", "evid")
            # 所有维度 delta=2, 都算 disagreement
            assert ir["n_agree_dims"] == 0
        finally:
            mod.make_judge_llm = original


# ── 汇总 ────────────────────────────────────────────────


class TestAggregateResults:
    def test_averages_all_dimensions(self):
        results = [
            {
                "mqs": {"q_match": 2, "law_correct": 2, "coverage": 2, "calibration": 2, "format": 2, "weighted_score": 100},
                "lemaj": {"summary": {"supported_pct": 0.8, "correct_pct": 0.9, "relevant_pct": 0.7}},
                "faith": {"summary": {"faithful_pct": 0.5, "partial_pct": 0.3, "unverifiable_pct": 0.1, "hallucinated_pct": 0.1}},
            },
            {
                "mqs": {"q_match": 1, "law_correct": 1, "coverage": 1, "calibration": 1, "format": 1, "weighted_score": 50},
                "lemaj": {"summary": {"supported_pct": 0.6, "correct_pct": 0.7, "relevant_pct": 0.8}},
                "faith": {"summary": {"faithful_pct": 0.3, "partial_pct": 0.2, "unverifiable_pct": 0.4, "hallucinated_pct": 0.1}},
            },
        ]
        summary = aggregate_results(results)
        assert summary["n_questions"] == 2
        # MQS weighted_score mean = (100+50)/2 = 75
        assert summary["mqs"]["weighted_score"] == 75.0
        # Faithful mean = (0.5+0.3)/2 = 0.4
        assert summary["faithfulness"]["faithful_pct"] == 0.4

    def test_empty_results(self):
        assert aggregate_results([]) == {}

    def test_skips_results_without_metrics(self):
        results = [
            {"mqs": {"weighted_score": 80}},
            {"mqs": {"weighted_score": 60}},
            {"lemaj": {"summary": {}}},  # 无 mqs，跳过
        ]
        summary = aggregate_results(results)
        assert summary["n_questions"] == 3
        assert summary["mqs"]["weighted_score"] == 70.0  # (80+60)/2，跳过第三个
