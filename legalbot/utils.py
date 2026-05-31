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

