"""数字幻觉检测工具的单测（D 类问题：D-class digital hallucination）。"""

import pytest

from legalbot.utils import (
    extract_number_tokens,
    find_unsupported_numbers,
    wrap_unsupported_numbers,
)


# ── extract_number_tokens ──────────────────────────────────


class TestExtractNumberTokens:
    def test_basic_digits(self):
        result = extract_number_tokens("值勤期 14 小时")
        nums = [r[0] for r in result]
        assert "14" in nums

    def test_skip_article_reference(self):
        """'第31条' 不应被提取为待核实数字。"""
        result = extract_number_tokens("依据第31条规定")
        assert "31" not in [r[0] for r in result]

    def test_skip_year(self):
        """4 位年份（2021年）及其后续的月/日数字不应被提取。"""
        result = extract_number_tokens("2021年7月1日起施行")
        # 2021 是 4 位年份 → skip
        assert "2021" not in [r[0] for r in result]
        # 7 和 1 紧跟 2021（日期上下文）→ skip
        assert "7" not in [r[0] for r in result]
        assert "1" not in [r[0] for r in result]

    def test_skip_chapter_section(self):
        """第X章/第X款 也不应被提取。"""
        for text in ["第3章 罚则", "第2款", "第5项", "第10条"]:
            result = extract_number_tokens(text)
            assert result == [], f"应跳过 '{text}'，但得到 {[r[0] for r in result]}"

    def test_skip_list_index(self):
        """列表编号 (1) 1、 1. 不应被提取。"""
        for text in ["(1) 旅客", "1) 应当", "1、承运人", "1. 起飞前"]:
            result = extract_number_tokens(text)
            assert result == [], f"应跳过 '{text}'"

    def test_range_separated(self):
        """范围 9-14 应被拆为 9 和 14。"""
        result = extract_number_tokens("单次值勤期 9-14 小时")
        nums = [r[0] for r in result]
        assert "9" in nums
        assert "14" in nums

    def test_decimal(self):
        result = extract_number_tokens("长度 1.5 米")
        nums = [r[0] for r in result]
        assert "1.5" in nums

    def test_empty(self):
        assert extract_number_tokens("") == []

    def test_no_numbers(self):
        assert extract_number_tokens("纯中文无数字") == []

    def test_skip_date_context(self):
        """日期 "2021年2月" 中 "2" 紧跟 4 位年份 → skip。"""
        # "2021" 已 skip（year）；"2" 紧跟 "2021" → skip
        result = extract_number_tokens("2021年7月1日起施行")
        assert "2" not in [r[0] for r in result]
        # "7" 紧跟 "2021" → skip
        assert "7" not in [r[0] for r in result]

    def test_dont_skip_date_with_separator(self):
        """"2021 年 7 月"（有空格）也 skip。"""
        result = extract_number_tokens("2021 年 7 月")
        assert "7" not in [r[0] for r in result]

    def test_dont_skip_independent_single_digit(self):
        """独立单字数字（不在年份后）保留。"""
        # "7" 不紧跟 4 位年份，保留
        result = extract_number_tokens("7 个工作日")
        assert "7" in [r[0] for r in result]


# ── find_unsupported_numbers ─────────────────────────────


class TestFindUnsupportedNumbers:
    def test_all_supported(self):
        answer = "值勤期 14 小时，连续 7 天不超过 60 小时"
        evidences = ["值勤期不超过 14 小时", "连续 7 日历日不超过 60 小时"]
        result = find_unsupported_numbers(answer, evidences)
        nums = [r[0] for r in result]
        assert nums == []  # 全部 supported

    def test_some_unsupported(self):
        answer = "值勤期 14 小时，月度上限 999 小时"
        evidences = ["值勤期不超过 14 小时"]  # 没有 "999"
        result = find_unsupported_numbers(answer, evidences)
        nums = [r[0] for r in result]
        assert "999" in nums
        assert "14" not in nums

    def test_no_evidence(self):
        answer = "9小时 60小时 7天"
        result = find_unsupported_numbers(answer, [])
        nums = [r[0] for r in result]
        assert "9" in nums
        assert "60" in nums
        assert "7" in nums

    def test_article_refs_not_flagged(self):
        answer = "依据第31条、第48条规定"
        evidences = ["无相关内容"]
        result = find_unsupported_numbers(answer, evidences)
        # 31 和 48 是法条引用，不应被 flag
        nums = [r[0] for r in result]
        assert "31" not in nums
        assert "48" not in nums

    def test_year_not_flagged(self):
        answer = "本规定自2021年7月1日起施行"
        evidences = []  # 即使没证据，年份也不应被 flag
        result = find_unsupported_numbers(answer, [])
        nums = [r[0] for r in result]
        assert "2021" not in nums


# ── wrap_unsupported_numbers ─────────────────────────────


class TestWrapUnsupportedNumbers:
    def test_wraps_unsupported(self):
        answer = "值勤期 14 小时，月度上限 999 小时"
        evidences = ["值勤期不超过 14 小时"]
        new_answer, unsupported = wrap_unsupported_numbers(answer, evidences)
        assert "999" in new_answer
        assert "[待核实]" in new_answer
        assert "999[待核实]" in new_answer  # 在 999 后插入
        # 14 不在 unsupported 中（evidence 有 14）
        assert "14" not in unsupported
        assert "999" in unsupported

    def test_no_unsupported_no_modification(self):
        answer = "依据第31条，值勤期 14 小时"
        evidences = ["第31条 飞行值勤期不超过 14 小时"]
        new_answer, unsupported = wrap_unsupported_numbers(answer, evidences)
        assert new_answer == answer  # 不修改
        assert unsupported == []

    def test_preserves_original_positioning(self):
        """数字后正确插入 [待核实]，不影响其他内容。"""
        answer = "A 是 100，B 是 200，C 是 300"
        evidences = ["A 是 100"]  # B/C 数字不在 evidence
        new_answer, unsupported = wrap_unsupported_numbers(answer, evidences)
        # 检查 B/C 后加了标记，A 没动
        assert "100，" in new_answer or "100," in new_answer
        assert "200[待核实]" in new_answer
        assert "300[待核实]" in new_answer
        assert "[待核实]" in new_answer
        assert unsupported == ["200", "300"]


# ── 端到端测试：D 类真实场景 ───────────────────────────


class TestDClassScenario:
    """D 类问题：bot 编造了 evidence 没有的具体数字。"""

    def test_q05_value_duty_hours_hallucination(self):
        """Q05 风格：bot 说 '14小时'、'60小时'、'210小时'，evidence 没有。"""
        # 模拟 bot 答案（含幻觉数字）
        bot_answer = (
            "单次飞行值勤期最大为14小时，连续7个日历日不超过60小时，"
            "任一日历月不超过210小时。休息期为10小时。"
        )
        # 模拟 evidence（不含这些数字）
        evidence = [
            "飞行机组的飞行值勤期应当符合本规则规定",
            "具体数值参见运行合格证持有人的运行规范",
        ]
        new_answer, unsupported = wrap_unsupported_numbers(bot_answer, evidence)
        # 多个数字被 flag
        assert "14" in unsupported
        assert "60" in unsupported
        assert "210" in unsupported
        assert "10" in unsupported
        # 7 可能在 evidence 中（"连续 7 个日历日"），不应被 flag
        # 但我们的 evidence 没有 7，所以 7 也会被 flag
        # 7 在"未支撑"，因为 evidence 没说"7"
        # 答案被加上 [待核实] 标记
        assert "[待核实]" in new_answer

    def test_evidence_provides_real_numbers_not_flagged(self):
        """当 evidence 给出具体数字时，答案引用它们不应被 flag。"""
        bot_answer = "单次值勤期 14 小时（CCAR-121 第121.485条）"
        evidence = ["第121.485条 飞行值勤期最大为 14 小时"]
        new_answer, unsupported = wrap_unsupported_numbers(bot_answer, evidence)
        # 14 在 evidence 中，不被 flag
        assert unsupported == []
        assert new_answer == bot_answer

    def test_d_class_wrong_number_引自不同条款(self):
        """D 类陷阱：bot 从 evidence 选错数字（"20小时" 但 evidence 只有 "9小时"）→ 应被 flag。"""
        bot_answer = "飞行值勤期最大为20小时"
        evidence = ["飞行值勤期 9小时"]  # 注意：evidence 是 9，不是 20
        new_answer, unsupported = wrap_unsupported_numbers(bot_answer, evidence)
        # 20 在 evidence 中不存在（"9" 在）
        # 但"20小时"作为组合也不在 evidence
        assert "20" in unsupported
        assert "[待核实]" in new_answer

    def test_d_class_digit_in_evidence_but_combination_mismatch(self):
        """D 类陷阱 2：evidence 有 "9小时" 和 "20公里" 两个数字，
        答案说"20小时"——20 在 evidence 但 20小时 组合不在 → flag。"""
        bot_answer = "飞机值勤 20小时"
        evidence = ["值勤 9小时", "里程 20公里"]
        new_answer, unsupported = wrap_unsupported_numbers(bot_answer, evidence)
        # "20" 在 evidence ("20公里"), 但 "20小时" 组合不在
        assert "20" in unsupported

    def test_unit_alias_passed(self):
        """单位别名不应被 flag（如 evidence 写 "7日历日"，答案写 "7天" 含义相同）。"""
        bot_answer = "连续 7 天不超过 60 小时"
        evidence = ["连续 7 日历日不超过 60 小时"]
        new_answer, unsupported = wrap_unsupported_numbers(bot_answer, evidence)
        # "7天" 组合不在 evidence，但 "7" 在 → 单字单位天，跳过组合检查
        # "60小时" 组合不在 evidence（evidence 用 "60 小时"），但 "60" 在 → 单字 "小时"...等等
        # 实际上 "小时" 是 2 字，应该被检查
        # evidence: "7 日历日不超过 60 小时"，有 "60 小时"（带空格）
        # 紧凑形式 "60小时" 是否在 all_evidence_compact? evidence 紧凑: "7日历日不超过60小时"
        # "60小时" in "7日历日不超过60小时" → True → pass
        assert "7" not in unsupported
        assert "60" not in unsupported
