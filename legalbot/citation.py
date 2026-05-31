"""LLM 引用与证据校验。"""

from __future__ import annotations

import json

from .llm import LLMClient
from .types import CitationCheck, Evidence


class CitationVerifier:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def extract_claims(self, question: str, legal_issues: list[str]) -> list[str]:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是法律主张抽取 Agent。只输出 JSON。"
                    "请从用户问题和法律争点中抽取需要被法条证据支持的待验证主张。"
                    "不要输出法律结论，不要编造。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"用户问题：{question}\n"
                    f"法律争点：{json.dumps(legal_issues, ensure_ascii=False)}\n\n"
                    "输出 JSON：{\"claims\": [\"...\"]}"
                ),
            },
        ]
        data = self.llm.json(messages)
        claims = [str(x).strip() for x in data.get("claims", []) if str(x).strip()]
        return claims or [question]

    def verify(self, question: str, legal_issues: list[str], evidence: list[Evidence]) -> list[CitationCheck]:
        if not evidence:
            return []
        claims = self.extract_claims(question, legal_issues)
        # 按批次校验，每批最多 BATCH_SIZE 条 evidence，确保每条 evidence 都被检查
        BATCH_SIZE = 15
        all_checks: list[CitationCheck] = []
        for batch_start in range(0, len(evidence), BATCH_SIZE):
            batch = evidence[batch_start:batch_start + BATCH_SIZE]
            evidence_payload = []
            for idx, item in enumerate(batch, 1):
                evidence_payload.append(
                    {
                        "id": idx,
                        "law_id": item.law_id,
                        "law_title": item.law_title,
                        "node_id": item.node_id,
                        "article": item.article,
                        "text": item.text[:1500],
                    }
                )
            messages = [
                {
                    "role": "system",
                    "content": (
                        "你是法律引用校验 Agent。只输出 JSON。\n"
                        "任务：对每一条 evidence，判断它是否支持了至少一个 claim。\n"
                        "你必须为每一条 evidence 都输出一条 check，不能遗漏任何 evidence。\n"
                        "status 只能是：supported（直接支持某个 claim）、partial（部分相关）、unsupported（与所有 claim 无关）。\n"
                        "quote 摘取最能支持判断的短句；没有则为空。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"用户问题：{question}\n"
                        f"claims：{json.dumps(claims, ensure_ascii=False)}\n\n"
                        f"evidence（共 {len(batch)} 条，必须每条都输出 check）：\n"
                        f"{json.dumps(evidence_payload, ensure_ascii=False)}\n\n"
                        "输出 JSON：\n"
                        "{\n"
                        '  "checks": [\n'
                        '    {"evidence_id": 1, "claim": "最相关的claim", "status": "supported|partial|unsupported", "reason": "...", "quote": "...", "confidence": 0.0}\n'
                        "  ]\n"
                        "}\n"
                        f"注意：checks 数组长度必须等于 {len(batch)}，即每条 evidence 恰好一条 check。"
                    ),
                },
            ]
            data = self.llm.json(messages)
            for item in data.get("checks", []):
                evidence_id = int(item.get("evidence_id", 0) or 0)
                ev = batch[evidence_id - 1] if 1 <= evidence_id <= len(batch) else None
                if ev is None:
                    all_checks.append(
                        CitationCheck(
                            claim=str(item.get("claim", "")),
                            law_id="",
                            node_id="",
                            status=str(item.get("status", "unsupported")),
                            reason=str(item.get("reason", "")),
                            quote=str(item.get("quote", "")),
                            confidence=float(item.get("confidence", 0.0) or 0.0),
                        )
                    )
                    continue
                status = str(item.get("status", "unsupported"))
                ev.verified = status == "supported"
                all_checks.append(
                    CitationCheck(
                        claim=str(item.get("claim", "")),
                        law_id=ev.law_id,
                        node_id=ev.node_id,
                        status=status,
                        reason=str(item.get("reason", "")),
                        quote=str(item.get("quote", "")),
                        confidence=float(item.get("confidence", 0.0) or 0.0),
                    )
                )
        return all_checks

