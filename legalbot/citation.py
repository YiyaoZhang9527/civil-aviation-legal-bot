"""LLM 引用与证据校验。"""

from __future__ import annotations

import json

from . import config
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

    def verify_with_cross_encoder(
        self, question: str, legal_issues: list[str], evidence: list[Evidence],
    ) -> list[CitationCheck]:
        """C1: 校验——开关控制 claim→evidence(新) 或 evidence→claim(旧) 方向。"""
        if not evidence:
            return []
        if getattr(config, 'CLAIM_LEVEL_CITATION', True):
            return self._verify_claim_nli(question, legal_issues, evidence)
        return self._verify_evidence_level(question, legal_issues, evidence)

    def _verify_claim_nli(
        self, question: str, legal_issues: list[str], evidence: list[Evidence],
    ) -> list[CitationCheck]:
        """Claim-NLI：对每条claim找最佳支持的evidence，claim分数过低时回退原始问题打分。"""
        claims = self.extract_claims(question, legal_issues)

        from .reranker import _load_model
        model = _load_model()
        if model is None:
            return self.verify(question, legal_issues, evidence)

        threshold = config.CROSS_ENCODER_CITATION_THRESHOLD
        partial_threshold = config.CROSS_ENCODER_PARTIAL_THRESHOLD
        max_chars = config.CROSS_ENCODER_MAX_CHARS
        batch_size = config.CROSS_ENCODER_BATCH

        # 批量构建所有 (claim, evidence) 对
        all_pairs = []
        pair_map = []  # (claim_idx, ev_idx)
        for ci, claim in enumerate(claims):
            for ei, ev in enumerate(evidence):
                text = ev.text[:max_chars] if ev.text else ""
                if text:
                    all_pairs.append((claim, text))
                    pair_map.append((ci, ei))

        if not all_pairs:
            return [CitationCheck(claim=question, law_id="", node_id="",
                                  status="unsupported", reason="no evidence text",
                                  confidence=0.0)]

        # 分批推理（GPU 显存保护）
        all_scores = []
        for start in range(0, len(all_pairs), batch_size):
            batch = all_pairs[start:start + batch_size]
            scores = model.predict(batch, show_progress_bar=False)
            all_scores.extend(float(s) for s in scores)

        # 聚合：每个 claim 取最高分 evidence
        claim_best = {}  # claim_idx -> (score, ev_idx)
        for idx, (ci, ei) in enumerate(pair_map):
            s = all_scores[idx]
            if ci not in claim_best or s > claim_best[ci][0]:
                claim_best[ci] = (s, ei)

        # 回退：claim分数全部过低时，用原始问题作为自然查询重新打分
        needs_fallback = all(
            claim_best.get(ci, (0.0, -1))[0] < partial_threshold
            for ci in range(len(claims))
        )
        if needs_fallback:
            fb_pairs = [(question, ev.text[:max_chars]) for ev in evidence if ev.text]
            fb_scores = []
            for start in range(0, len(fb_pairs), batch_size):
                batch = fb_pairs[start:start + batch_size]
                scores = model.predict(batch, show_progress_bar=False)
                fb_scores.extend(float(s) for s in scores)
            # 用问题分数更新每个claim的最佳匹配
            fb_ei = 0
            for ei, ev in enumerate(evidence):
                if not ev.text:
                    continue
                q_score = fb_scores[fb_ei] if fb_ei < len(fb_scores) else 0.0
                fb_ei += 1
                for ci in range(len(claims)):
                    if ci not in claim_best or q_score > claim_best[ci][0]:
                        claim_best[ci] = (q_score, ei)

        all_checks: list[CitationCheck] = []
        for ci, claim in enumerate(claims):
            if ci not in claim_best:
                all_checks.append(CitationCheck(
                    claim=claim, law_id="", node_id="",
                    status="unsupported", reason="no scorable evidence",
                    confidence=0.0))
                continue
            best_score, best_ei = claim_best[ci]
            ev = evidence[best_ei]
            if best_score >= threshold:
                status = "supported"
                ev.verified = True
            elif best_score >= partial_threshold:
                status = "partial"
            else:
                status = "unsupported"
            all_checks.append(CitationCheck(
                claim=claim, law_id=ev.law_id, node_id=ev.node_id,
                status=status, reason=f"claim-NLI score={best_score:.3f}",
                quote="", confidence=max(best_score, 0.0)))
        return all_checks

    def _verify_evidence_level(
        self, question: str, legal_issues: list[str], evidence: list[Evidence],
    ) -> list[CitationCheck]:
        """旧方向：对每条evidence找最匹配的claim，claim分数为0时回退原始问题打分。"""
        claims = self.extract_claims(question, legal_issues)
        from .reranker import _load_model
        model = _load_model()
        if model is None:
            return self.verify(question, legal_issues, evidence)
        threshold = config.CROSS_ENCODER_CITATION_THRESHOLD
        partial_threshold = config.CROSS_ENCODER_PARTIAL_THRESHOLD
        max_chars = config.CROSS_ENCODER_MAX_CHARS
        all_checks: list[CitationCheck] = []
        for ev in evidence:
            best_claim = claims[0] if claims else question
            best_score = -1.0
            text = ev.text[:max_chars] if ev.text else ""
            if not text:
                all_checks.append(CitationCheck(
                    claim=best_claim, law_id=ev.law_id, node_id=ev.node_id,
                    status="unsupported", reason="no evidence text",
                    quote="", confidence=0.0))
                continue
            for claim in claims:
                scores = model.predict([(claim, text)])
                s = float(scores[0])
                if s > best_score:
                    best_score = s
                    best_claim = claim
            # claim分数过低时，用原始问题作为自然查询回退打分
            if best_score < partial_threshold:
                q_scores = model.predict([(question, text)])
                q_s = float(q_scores[0])
                if q_s > best_score:
                    best_score = q_s
                    best_claim = question
            if best_score >= threshold:
                status = "supported"
                ev.verified = True
            elif best_score >= partial_threshold:
                status = "partial"
            else:
                status = "unsupported"
            all_checks.append(CitationCheck(
                claim=best_claim, law_id=ev.law_id, node_id=ev.node_id,
                status=status, reason=f"cross-encoder score={best_score:.3f}",
                quote="", confidence=max(best_score, 0.0)))
        return all_checks

    def verify(self, question: str, legal_issues: list[str], evidence: list[Evidence]) -> list[CitationCheck]:
        if not evidence:
            return []
        claims = self.extract_claims(question, legal_issues)
        batch_size = config.CROSS_ENCODER_BATCH
        truncate = config.CITATION_LLM_TRUNCATE
        all_checks: list[CitationCheck] = []
        for batch_start in range(0, len(evidence), batch_size):
            batch = evidence[batch_start:batch_start + batch_size]
            evidence_payload = []
            for idx, item in enumerate(batch, 1):
                evidence_payload.append(
                    {
                        "id": idx,
                        "law_id": item.law_id,
                        "law_title": item.law_title,
                        "node_id": item.node_id,
                        "article": item.article,
                        "text": item.text[:truncate],
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
