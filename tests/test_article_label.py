"""Unit tests for _short_article_label helper and article text rendering."""
from legalbot.agents import _short_article_label


def test_short_article_label_strips_title_tail():
    assert _short_article_label("第三十一条 有下列情况之一的,承运人应当拒绝运") == "第三十一条"


def test_short_article_label_pure_label():
    assert _short_article_label("第三十一条") == "第三十一条"


def test_short_article_label_decimal_article():
    assert _short_article_label("67.33 特殊运行") == "67.33"


def test_short_article_label_empty():
    assert _short_article_label("") == ""


def test_short_article_label_chapter():
    assert _short_article_label("第X章  罚则") == "第X章"


def test_short_article_label_english_article():
    # 异常格式：fallback to first word
    assert _short_article_label("Section 5") == "Section"


def test_short_article_label_dotted_decimal():
    """小数点编号（如 67.33）保留"""
    assert _short_article_label("121.657 燃油量要求") == "121.657"
