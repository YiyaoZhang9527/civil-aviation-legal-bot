"""数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class LawNode:
    node_id: str
    type: str
    title: str
    summary: str = ""
    keywords: list[str] = field(default_factory=list)
    source_file: str = ""
    source_anchor: str = ""
    text: str = ""
    line_start: int | None = None
    line_end: int | None = None
    children: list["LawNode"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LawDocument:
    law_id: str
    title: str
    source_file: str
    root: LawNode
    anchor_map: dict[str, dict[str, Any]] = field(default_factory=dict)

    def flatten(self) -> list[LawNode]:
        result: list[LawNode] = []

        def visit(node: LawNode) -> None:
            result.append(node)
            for child in node.children:
                visit(child)

        visit(self.root)
        return result


@dataclass
class Evidence:
    law_id: str
    law_title: str
    node_id: str
    article: str
    text: str
    score: float
    source_file: str
    source_anchor: str
    verified: bool = False


@dataclass
class CitationCheck:
    claim: str
    law_id: str
    node_id: str
    status: str
    reason: str
    quote: str = ""
    confidence: float = 0.0


@dataclass
class Conflict:
    law_titles: list[str] = field(default_factory=list)
    reason: str = ""
    priority_order: list[str] = field(default_factory=list)


@dataclass
class IntentResult:
    intent: str
    topic: str
    query: str
    law_hints: list[str] = field(default_factory=list)
    article_hints: list[str] = field(default_factory=list)
    need_clarification: bool = False
    clarification: str = ""
    options: list[str] = field(default_factory=list)
    sub_questions: list[str] = field(default_factory=list)


@dataclass
class AnswerResult:
    answer: str
    intent: str
    topic: str
    evidence: list[Evidence] = field(default_factory=list)
    citations: list[CitationCheck] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    status: str = "ok"
    pending_clarification: dict[str, Any] | None = None
    reflexion_iterations: int = 0
    structured_claims: list[dict] = field(default_factory=list)
    unsupported_claims_removed: int = 0
