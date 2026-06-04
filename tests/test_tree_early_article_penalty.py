"""TreeRetriever 位置降权测试。"""

import pytest

from legalbot.tree_retrieval import TreeRetriever


def _make_fused(*node_ids_with_scores: tuple[str, float]) -> dict:
    """构造 (law_id, node_id) -> score 的 fused 字典。"""
    return {("law_x", nid): s for nid, s in node_ids_with_scores}


class TestApplyEarlyArticlePenalty:
    """_apply_early_article_penalty 对前 N 条法条按位置降权。"""

    def test_disabled_when_penalty_equals_one(self):
        fused = _make_fused(("article:1", 0.9), ("article:31", 0.5))
        original = dict(fused)
        TreeRetriever._apply_early_article_penalty(fused, early_penalty=1.0, early_threshold=5)
        assert fused == original

    def test_penalty_applied_to_early_articles(self):
        fused = _make_fused(
            ("article:1", 1.0),   # 总则候选
            ("article:2", 1.0),   # 总则候选
            ("article:5", 1.0),   # 边界
            ("article:6", 1.0),   # 不应被降权
            ("article:31", 1.0),  # 真正的具体条款
        )
        TreeRetriever._apply_early_article_penalty(fused, early_penalty=0.6, early_threshold=5)
        assert fused[("law_x", "article:1")] == pytest.approx(0.6)
        assert fused[("law_x", "article:2")] == pytest.approx(0.6)
        assert fused[("law_x", "article:5")] == pytest.approx(0.6)  # 边界包含
        assert fused[("law_x", "article:6")] == 1.0  # 阈值外不降权
        assert fused[("law_x", "article:31")] == 1.0  # 远在阈值外

    def test_threshold_configurable(self):
        fused = _make_fused(
            ("article:3", 1.0),
            ("article:7", 1.0),
            ("article:10", 1.0),
        )
        TreeRetriever._apply_early_article_penalty(fused, early_penalty=0.5, early_threshold=3)
        # threshold=3 → 只有 1, 2, 3 被降权
        assert fused[("law_x", "article:3")] == pytest.approx(0.5)
        assert fused[("law_x", "article:7")] == 1.0
        assert fused[("law_x", "article:10")] == 1.0

    def test_handles_missing_article_number(self):
        """非 article: 编号的 key 不应被错误降权。"""
        fused = {
            ("law_x", "chapter:1"): 1.0,
            ("law_x", "section:2"): 1.0,
            ("law_x", "article:1"): 1.0,
        }
        TreeRetriever._apply_early_article_penalty(fused, early_penalty=0.5, early_threshold=5)
        # 非 article: 节点不动
        assert fused[("law_x", "chapter:1")] == 1.0
        assert fused[("law_x", "section:2")] == 1.0
        # article:1 降权
        assert fused[("law_x", "article:1")] == pytest.approx(0.5)

    def test_in_place_mutation(self):
        """验证是原地修改（与 search() 中的使用一致）。"""
        fused = _make_fused(("article:1", 0.8))
        fused_id = id(fused)
        TreeRetriever._apply_early_article_penalty(fused, early_penalty=0.5, early_threshold=5)
        assert id(fused) == fused_id
        assert fused[("law_x", "article:1")] == pytest.approx(0.4)

    def test_empty_dict_safe(self):
        fused: dict = {}
        TreeRetriever._apply_early_article_penalty(fused, early_penalty=0.5, early_threshold=5)
        assert fused == {}

    def test_penalty_multiplicative_with_existing_score(self):
        """已在 fused 中较低的分数 × penalty 会更低（验证 RRF 分数被降权）。"""
        fused = _make_fused(
            ("article:1", 0.3),  # 已是低分
            ("article:31", 0.9),  # 高分具体条款
        )
        TreeRetriever._apply_early_article_penalty(fused, early_penalty=0.6, early_threshold=5)
        assert fused[("law_x", "article:1")] == pytest.approx(0.18)
        # 排序：article:31 (0.9) > article:1 (0.18) → 具体条款上升
        sorted_keys = sorted(fused.keys(), key=lambda k: fused[k], reverse=True)
        assert sorted_keys[0] == ("law_x", "article:31")


# ── 长度门控：text_lens 参数生效时 ──────────────────────


class TestApplyEarlyArticlePenaltyWithLengthGate:
    """长度门控是位置降权的"平衡"——长条款即使在前 5 条也不动。"""

    def test_short_early_article_penalized(self):
        """短文本（≤ short_limit）在前 5 条 → 降权。"""
        fused = _make_fused(("article:1", 1.0))
        text_lens = {("law_x", "article:1"): 50}  # 50 字，短总则
        TreeRetriever._apply_early_article_penalty(
            fused, early_penalty=0.6, early_threshold=5,
            text_lens=text_lens, short_limit=200,
        )
        assert fused[("law_x", "article:1")] == pytest.approx(0.6)

    def test_long_early_article_skipped(self):
        """长文本（> short_limit）即使在前 5 条 → 跳过降权。"""
        fused = _make_fused(
            ("article:1", 1.0),   # 假设是 1000 字的详细条款
            ("article:3", 1.0),   # 也是 800 字
        )
        text_lens = {
            ("law_x", "article:1"): 1000,
            ("law_x", "article:3"): 800,
        }
        TreeRetriever._apply_early_article_penalty(
            fused, early_penalty=0.6, early_threshold=5,
            text_lens=text_lens, short_limit=200,
        )
        # 长条款不被降权
        assert fused[("law_x", "article:1")] == 1.0
        assert fused[("law_x", "article:3")] == 1.0

    def test_short_limit_boundary(self):
        """边界：正好等于 short_limit 时视为短文本（≤）。"""
        fused = _make_fused(("article:1", 1.0))
        text_lens = {("law_x", "article:1"): 200}  # 正好 200
        TreeRetriever._apply_early_article_penalty(
            fused, early_penalty=0.6, early_threshold=5,
            text_lens=text_lens, short_limit=200,
        )
        assert fused[("law_x", "article:1")] == pytest.approx(0.6)

    def test_short_limit_just_above(self):
        """200 字以上跳过降权。"""
        fused = _make_fused(("article:1", 1.0))
        text_lens = {("law_x", "article:1"): 201}
        TreeRetriever._apply_early_article_penalty(
            fused, early_penalty=0.6, early_threshold=5,
            text_lens=text_lens, short_limit=200,
        )
        assert fused[("law_x", "article:1")] == 1.0

    def test_unknown_text_length_skipped(self):
        """text_lens 缺失的 key → 保守不降权（不报错）。"""
        fused = _make_fused(("article:1", 1.0))
        text_lens = {}  # 空字典
        TreeRetriever._apply_early_article_penalty(
            fused, early_penalty=0.6, early_threshold=5,
            text_lens=text_lens, short_limit=200,
        )
        assert fused[("law_x", "article:1")] == 1.0

    def test_zero_text_length_skipped(self):
        """文本长度 0（异常数据）→ 不降权。"""
        fused = _make_fused(("article:1", 1.0))
        text_lens = {("law_x", "article:1"): 0}
        TreeRetriever._apply_early_article_penalty(
            fused, early_penalty=0.6, early_threshold=5,
            text_lens=text_lens, short_limit=200,
        )
        assert fused[("law_x", "article:1")] == 1.0

    def test_late_position_with_short_text_not_penalized(self):
        """短文本但在第 10 条 → 位置阈值外，不动。"""
        fused = _make_fused(
            ("article:10", 1.0),  # 短但位置 10
            ("article:31", 1.0),  # 长具体条款
        )
        text_lens = {
            ("law_x", "article:10"): 50,
            ("law_x", "article:31"): 1500,
        }
        TreeRetriever._apply_early_article_penalty(
            fused, early_penalty=0.6, early_threshold=5,
            text_lens=text_lens, short_limit=200,
        )
        # 第 10 条不在阈值内（1-5），不动
        assert fused[("law_x", "article:10")] == 1.0
        assert fused[("law_x", "article:31")] == 1.0

    def test_realistic_scenario_rejection_question(self):
        """模拟拒载问题：article 1-5 是短总则，article 31 是长具体条款。

        期望：article 1-5 降权，article 31 不动。
        """
        # 直接构造 fused（绕过 _make_fused 的 law_x 硬编码）
        fused = {
            ("ccar-271", "article:1"): 0.10,   # 短总则
            ("ccar-271", "article:2"): 0.10,
            ("ccar-271", "article:3"): 0.10,
            ("ccar-271", "article:31"): 0.10,  # 长具体条款
        }
        # 真实拒载法规的文本长度（模拟）
        text_lens = {
            ("ccar-271", "article:1"): 150,   # 短：目的/依据
            ("ccar-271", "article:2"): 80,
            ("ccar-271", "article:3"): 120,
            ("ccar-271", "article:31"): 1500,  # 长：具体拒载情形
        }
        TreeRetriever._apply_early_article_penalty(
            fused, early_penalty=0.6, early_threshold=5,
            text_lens=text_lens, short_limit=200,
        )
        assert fused[("ccar-271", "article:1")] == pytest.approx(0.06)
        assert fused[("ccar-271", "article:2")] == pytest.approx(0.06)
        assert fused[("ccar-271", "article:3")] == pytest.approx(0.06)
        # 关键：article:31 不被降权！
        assert fused[("ccar-271", "article:31")] == 0.10

    def test_backward_compatible_without_text_lens(self):
        """不传 text_lens → 旧行为（纯位置降权）保留。"""
        fused = _make_fused(
            ("article:1", 1.0),   # 即使是 5000 字，只要在前 5 条
            ("article:31", 1.0),
        )
        TreeRetriever._apply_early_article_penalty(
            fused, early_penalty=0.6, early_threshold=5,
            text_lens=None,  # 旧路径
        )
        # 旧行为：前 5 条全部降权
        assert fused[("law_x", "article:1")] == pytest.approx(0.6)
        assert fused[("law_x", "article:31")] == 1.0
