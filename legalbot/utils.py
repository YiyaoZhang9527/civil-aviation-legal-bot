"""通用工具。"""

from __future__ import annotations

import re
from collections import Counter


CN_NUM_MAP = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}


def chinese_to_int(text: str) -> int | None:
    text = text.strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    if text in CN_NUM_MAP:
        return CN_NUM_MAP[text]
    if text == "十":
        return 10
    total = 0
    if "十" in text:
        head, tail = text.split("十", 1)
        total += CN_NUM_MAP.get(head, 1) * 10
        if tail:
            total += CN_NUM_MAP.get(tail, 0)
        return total
    if "百" in text:
        head, tail = text.split("百", 1)
        total += CN_NUM_MAP.get(head, 1) * 100
        if tail:
            if tail.startswith("零"):
                tail = tail[1:]
            tail_val = chinese_to_int(tail)
            if tail_val is not None:
                total += tail_val
        return total
    return None


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", "", text)
    text = text.replace("《", "").replace("》", "")
    text = text.replace("（", "(").replace("）", ")")
    return text


def strip_heading(text: str) -> str:
    return re.sub(r"^第[一二三四五六七八九十百零\d]+[章节条款项]\s*", "", text).strip()


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"[。！？；\n]+", text)
    return [p.strip() for p in parts if p.strip()]


def extract_phrases(text: str, min_len: int = 2) -> list[str]:
    phrases = re.findall(r"[\u4e00-\u9fff]{%d,}" % min_len, text)
    seen: set[str] = set()
    result: list[str] = []
    for p in phrases:
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result


def top_keywords(text: str, max_n: int = 6) -> list[str]:
    candidates = extract_phrases(text)
    counter: Counter[str] = Counter()
    for phrase in candidates:
        counter[phrase] += 1
    return [word for word, _ in counter.most_common(max_n)]


def safe_preview(text: str, max_len: int = 120) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= max_len else text[:max_len] + "..."


def article_number_to_label(num: int) -> str:
    digits = "零一二三四五六七八九"
    if num <= 10:
        return ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十"][num]
    if num < 20:
        return "十" + (digits[num - 10] if num > 10 else "")
    if num < 100:
        tens, ones = divmod(num, 10)
        return digits[tens] + "十" + (digits[ones] if ones else "")
    hundreds, rest = divmod(num, 100)
    if rest == 0:
        return digits[hundreds] + "百"
    return digits[hundreds] + "百" + article_number_to_label(rest)


def article_label(num: int) -> str:
    return f"第{article_number_to_label(num)}条"


# ── 数字幻觉检测（D 类问题：D-class digital hallucination）────


import re as _re

# 数字 token：1+ 数字，可选小数。不用 \b（中文文本里 \b 不工作）
_NUMBER_TOKEN_RE = _re.compile(r"\d+(?:\.\d+)?")

# 跳过：4 位年份（1900-2099），避免误报"2021年"
_YEAR_RE = _re.compile(r"^(19|20)\d{2}$")

# 跳过前文：第X条/章/款/项/年代
_SKIP_PREFIX_CHARS = "第章条款项"

# 中文/英文时间/数量单位 — 用于"数字+单位"组合检查（增强版检测）
_UNIT_CHARS = "小时天月日分秒年周刻元角分厘块公斤里米尺亩平方立升吨个条项款章节倍次种类人名位张台辆架"


def extract_number_tokens(text: str) -> list[tuple[str, int, int]]:
    """提取文本中可能是事实数字的 token（已过滤：法条引用、年份、列表编号）。

    返回 [(数字, start, end), ...]
    """
    if not text:
        return []
    candidates: list[tuple[str, int, int]] = []
    for m in _NUMBER_TOKEN_RE.finditer(text):
        num = m.group()
        start, end = m.span()
        # 跳过 4 位年份
        if _YEAR_RE.match(num):
            continue
        # 跳过法条引用（"第31条"）：检查前 2 字符
        prefix2 = text[max(0, start - 2):start]
        if any(c in prefix2 for c in _SKIP_PREFIX_CHARS):
            continue
        # 跳过列表编号：(1) 1) 1、 1. — 检查前/后 1 字符
        prev_ch = text[start - 1] if start > 0 else ""
        next_ch = text[end] if end < len(text) else ""
        # 注意：空字符串 "" in 任何字符串 都返回 True，所以要显式检查 truthy
        if prev_ch and prev_ch in "(['":  # (1)  [1]
            continue
        if next_ch and next_ch in ")、.】":  # 1) 1、 1.]
            continue
        # 跳过日期上下文中的短数字：紧跟在 4 位年份后面的 1-2 位数字（月份/日期）
        # 例：2021年2月 → "2" 在 evidence "2021" 中是子串，不应被 flag
        if len(num) <= 2 and num.isdigit():
            # 向前 10 字符内找 4 位年份（容忍"年/月/日"分隔符，最多 2 个分隔符）
            prefix_window = text[max(0, start - 10):start]
            year_match = _re.search(r"(19|20)\d{2}", prefix_window)
            if year_match and (start - 10 + year_match.end()) >= start - 6:
                # 年份在窗口内且距当前数字不超过 6 字符 → 可能是日期，skip
                continue
        candidates.append((num, start, end))
    return candidates


def _get_unit_value_pair(text: str, num_start: int, num_end: int) -> str | None:
    """提取数字后续的"单位"片段（最多 3 个字符），用于检测"14小时"等组合。"""
    # 跳过空格和标点
    unit_start = num_end
    while unit_start < len(text) and text[unit_start] in " \t　":
        unit_start += 1
    # 收集单位字符（汉字）
    unit_end = unit_start
    while unit_end < len(text) and unit_end < unit_start + 4 and text[unit_end] in _UNIT_CHARS:
        unit_end += 1
    if unit_end > unit_start:
        return text[num_start:unit_end]
    return None


def find_unsupported_numbers(answer: str, evidence_texts: list[str] | None) -> list[tuple[str, int, int]]:
    """找答案中所有 evidence 未支撑的数字 token。

    增强逻辑（多层检查）：
    1. 数字本身是否在 evidence（基础）
    2. 数字+多字单位（小时/分钟/公里）组合是否在 evidence（防"错引其他数字"）
       例：evidence 有 "9小时" 没有 "20小时" → 答案说 "20小时" 应被 flag
    3. 单字单位（天/月/年/日）跳过组合检查（防 false positive："7天" vs "7日历日"）
    """
    candidates = extract_number_tokens(answer)
    if not candidates:
        return []
    all_evidence = "\n".join(evidence_texts or [])
    all_evidence_compact = all_evidence.replace(" ", "").replace("\u3000", "")

    unsupported: list[tuple[str, int, int]] = []
    for num, start, end in candidates:
        # 检查 1：数字本身在 evidence？
        num_in_evidence = num in all_evidence

        if not num_in_evidence:
            # 数字本身就不在 evidence → 直接 flag
            unsupported.append((num, start, end))
            continue

        # 数字在 evidence。检查是否有"多字单位"组合不匹配。
        unit_pair = _get_unit_value_pair(answer, start, end)
        if not unit_pair:
            # 纯数字，无单位 → pass
            continue

        # 提取单位部分（去掉数字和空格）
        unit_chars = unit_pair[len(num):].replace(" ", "").replace("\u3000", "")
        if len(unit_chars) < 2:
            # 单字单位（天/月/年/日/时）→ 容忍差异，pass
            continue

        # 多字单位（小时/分钟/公里/百分比）→ 必须组合匹配
        compact = unit_pair.replace(" ", "").replace("\u3000", "")
        if compact not in all_evidence_compact:
            # evidence 没这个数字+单位组合（即使数字本身在）→ flag
            # 例：evidence 有 "9小时"，答案说 "20小时"（数字错引）
            unsupported.append((num, start, end))

    return unsupported


def wrap_unsupported_numbers(
    answer: str,
    evidence_texts: list[str] | None,
    flag: str = "[待核实]",
) -> tuple[str, list[str]]:
    """把 evidence 未支撑的数字加上 [待核实] 标记。

    返回 (新答案, 未支撑的数字列表)
    """
    unsupported = find_unsupported_numbers(answer, evidence_texts)
    if not unsupported:
        return answer, []
    # 从右往左插入，避免位置偏移
    answer_chars = list(answer)
    insertions = sorted(unsupported, key=lambda x: x[1], reverse=True)
    for _, start, end in insertions:
        answer_chars.insert(end, flag)
    return "".join(answer_chars), [n for n, _, _ in unsupported]

