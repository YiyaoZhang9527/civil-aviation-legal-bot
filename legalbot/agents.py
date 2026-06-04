"""LLM 驱动的多 Agent 编排。"""

from __future__ import annotations

from dataclasses import dataclass, field
import json

from .citation import CitationVerifier
from . import config
from .llm import LLMClient
from .logger import TerminalLogger
from .retrieval import read_law_node, search_index_tree
from .types import AnswerResult, CitationCheck, Conflict, Evidence, IntentResult
from .utils import normalize_text, wrap_unsupported_numbers


@dataclass
class RetrievalPlan:
    intent: str
    legal_issues: list[str] = field(default_factory=list)
    facts: dict = field(default_factory=dict)
    queries: list[str] = field(default_factory=list)
    law_hints: list[str] = field(default_factory=list)
    article_hints: list[str] = field(default_factory=list)
    need_clarification: bool = False
    clarification: str = ""
    sub_questions: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    alternative_paths: list[str] = field(default_factory=list)


@dataclass
class SubjectAnalysis:
    subjects: dict = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    relationships: list[str] = field(default_factory=list)
    uncertain_facts: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    alternative_paths: list[str] = field(default_factory=list)
    clarification_decision: str = "answer"
    clarification_reason: str = ""
    need_clarification: bool = False
    clarification: str = ""


@dataclass
class IssueAnalysis:
    legal_issues: list[str] = field(default_factory=list)
    issue_types: list[str] = field(default_factory=list)
    applicable_law_domains: list[str] = field(default_factory=list)
    missing_facts: list[str] = field(default_factory=list)


@dataclass
class RewritePlan:
    queries: list[str] = field(default_factory=list)
    law_hints: list[str] = field(default_factory=list)
    article_hints: list[str] = field(default_factory=list)
    sub_questions: list[str] = field(default_factory=list)


@dataclass
class SubProblem:
    question: str
    queries: list[str] = field(default_factory=list)
    law_hints: list[str] = field(default_factory=list)
    article_hints: list[str] = field(default_factory=list)


@dataclass
class DecompositionResult:
    needs_decomposition: bool = False
    sub_problems: list[SubProblem] = field(default_factory=list)


@dataclass
class ReflexionResult:
    quality: str = "pass"  # "pass" | "gap"
    gaps: list[str] = field(default_factory=list)
    refine_queries: list[str] = field(default_factory=list)
    refine_law_hints: list[str] = field(default_factory=list)
    refine_article_hints: list[str] = field(default_factory=list)


@dataclass
class CaseState:
    request_id: str
    original_question: str
    normalized_question: str
    subject_analysis: SubjectAnalysis | None = None
    issue_analysis: IssueAnalysis | None = None
    rewrite_plan: RewritePlan | None = None
    plan: RetrievalPlan | None = None
    evidence: list[Evidence] = field(default_factory=list)
    citations: list[CitationCheck] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)
    final_answer: str = ""
    reflexion_iteration: int = 0
    reflexion_trace: list[dict] = field(default_factory=list)


@dataclass
class ClarificationResolution:
    resolved: bool = False
    is_new_question: bool = False
    filled_slots: dict = field(default_factory=dict)
    still_missing: list[str] = field(default_factory=list)
    enriched_question: str = ""
    clarification: str = ""
    reason: str = ""


@dataclass
class FollowupRewrite:
    is_followup: bool = False
    is_new_question: bool = True
    rewrite: str = ""
    reason: str = ""


_STOP_WORDS = frozenset(
    "的 了 是 在 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好 自己 这 那 个 他 她 它 们 把 被 让 给 对 与 但 而 或 如 因 所 以 之 于 则 已 还 又 再 请 可 能".split()
)


def _lexical_support_score(answer: str, evidence: list[Evidence]) -> float:
    """E1: 答案content tokens被证据文本覆盖的比例。确定性，零LLM。"""
    import jieba
    answer_tokens = {w for w in jieba.cut(answer) if len(w) >= 2 and w not in _STOP_WORDS}
    if not answer_tokens:
        return 1.0
    evidence_tokens: set[str] = set()
    for ev in evidence:
        if ev.text:
            evidence_tokens.update(w for w in jieba.cut(ev.text) if len(w) >= 2)
    covered = answer_tokens & evidence_tokens
    return len(covered) / len(answer_tokens)


def _build_citation_block(citations: list[CitationCheck]) -> str:
    """E1: 从citations构建标准引用列表附加到答案末尾。"""
    supported = [c for c in citations if c.status in {"supported", "partial"} and c.law_id]
    if not supported:
        return ""
    lines = ["\n\n【引用法条】"]
    for c in supported[:8]:
        lines.append(f"- {c.law_id} {c.node_id} [{c.status}]")
    return "\n".join(lines)


class SubjectAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def extract_subjects(self, question: str) -> SubjectAnalysis:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是民航法律主体判断 Agent。只输出 JSON。"
                    "你的任务是把用户口语问题解析为法律主体、事件链、合理假设和真正需要澄清的事实。"
                    "不要回答法律结论，不要检索法条。"
                    "不要因为中文省略主语就轻易追问。"
                    "能根据日常语义、叙事主语、代词继承合理补全的，应补全并记录 assumptions。"
                    "只有多个法律路径同等可能且没有合理默认主体时，才设置 need_clarification=true。"
                    "普通通用咨询即使缺少航班号、航空器型号、飞行阶段等细节，也应设置 need_clarification=false，把缺失事实放入 uncertain_facts。"
                    "澄清策略只能是 answer、answer_with_assumption、need_clarification。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "用户问题：\n"
                    f"{question}\n\n"
                    "主体消解规则：\n"
                    "1. 中文口语常省略'我'。如果问题以第一人称描述民航相关场景，默认叙事主体为用户本人。\n"
                    "2. 涉及飞行员/空管/机务等人员资质问题，默认当事人=用户本人，所在单位=航空公司或机场。\n"
                    "3. 涉及旅客权益问题（延误赔偿、行李丢失、拒载等），默认旅客=用户本人，承运人=航空公司。\n"
                    "4. 缺少航班号、航空器型号、机场名称、飞行阶段等事实，不是主体阻断条件，放入 uncertain_facts。\n"
                    "5. 如果直接回答依赖合理推定，必须在 assumptions 写明推定，并在 alternative_paths 写明另一种主体路径。\n\n"
                    "只输出 JSON，格式：\n"
                    "{\n"
                    '  "subjects": {"当事人": "...", "航空运营人": "...", "飞行员/空勤人员": "...", "旅客": "...", "机场": "...", "空管单位": "...", "其他主体": "..."},\n'
                    '  "events": [{"type": "flight_incident|safety_violation|certification|passenger_rights|airport_ops|air_traffic|other", "subject": "...", "actor": "...", "purpose": "...", "confidence": 0.0}],\n'
                    '  "relationships": ["..."],\n'
                    '  "uncertain_facts": ["..."],\n'
                    '  "assumptions": ["..."],\n'
                    '  "alternative_paths": ["..."],\n'
                    '  "clarification_policy": {"decision": "answer|answer_with_assumption|need_clarification", "reason": "..."},\n'
                    '  "need_clarification": false,\n'
                    '  "clarification": ""\n'
                    "}\n"
                    "need_clarification 必须与 clarification_policy.decision=need_clarification 保持一致。"
                ),
            },
        ]
        data = self.llm.json(messages)
        policy = data.get("clarification_policy", {})
        if not isinstance(policy, dict):
            policy = {}
        decision = str(policy.get("decision", "answer")).strip()
        if decision not in {"answer", "answer_with_assumption", "need_clarification"}:
            decision = "answer"
        return SubjectAnalysis(
            subjects=data.get("subjects", {}) if isinstance(data.get("subjects", {}), dict) else {},
            events=data.get("events", []) if isinstance(data.get("events", []), list) else [],
            relationships=[str(x) for x in data.get("relationships", []) if str(x).strip()],
            uncertain_facts=[str(x) for x in data.get("uncertain_facts", []) if str(x).strip()],
            assumptions=[str(x) for x in data.get("assumptions", []) if str(x).strip()],
            alternative_paths=[str(x) for x in data.get("alternative_paths", []) if str(x).strip()],
            clarification_decision=decision,
            clarification_reason=str(policy.get("reason", "")),
            need_clarification=bool(data.get("need_clarification", False)) or decision == "need_clarification",
            clarification=str(data.get("clarification", "")),
        )


class IssueAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def identify_issues(self, question: str, subject_analysis: SubjectAnalysis) -> IssueAnalysis:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是民航法律争点识别 Agent。只输出 JSON。"
                    "基于用户问题和主体分析，识别需要检索的法律问题。"
                    "不要回答法律结论。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"用户问题：{question}\n\n"
                    f"主体分析：{json.dumps(subject_analysis.__dict__, ensure_ascii=False)}\n\n"
                    "只输出 JSON，格式：\n"
                    "{\n"
                    '  "legal_issues": ["..."],\n'
                    '  "issue_types": ["适航认证|飞行标准|机场管理|安全保卫|航空运输|空中交通|旅客权益|无人机|其他"],\n'
                    '  "applicable_law_domains": ["..."],\n'
                    '  "missing_facts": ["..."]\n'
                    "}\n"
                ),
            },
        ]
        data = self.llm.json(messages)
        return IssueAnalysis(
            legal_issues=[str(x) for x in data.get("legal_issues", []) if str(x).strip()],
            issue_types=[str(x) for x in data.get("issue_types", []) if str(x).strip()],
            applicable_law_domains=[str(x) for x in data.get("applicable_law_domains", []) if str(x).strip()],
            missing_facts=[str(x) for x in data.get("missing_facts", []) if str(x).strip()],
        )


class ClarificationAgent:
    def should_clarify(self, subject_analysis: SubjectAnalysis, issue_analysis: IssueAnalysis) -> tuple[bool, str]:
        if subject_analysis.clarification_decision == "answer_with_assumption":
            return False, ""
        if subject_analysis.need_clarification:
            return True, subject_analysis.clarification or "请补充关键主体关系后再判断。"
        return False, ""


class ClarificationResolutionAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def resolve(self, pending: dict, user_reply: str, history: list[dict]) -> ClarificationResolution:
        history_text = _format_history_for_prompt(history[-config.HISTORY_TURNS:])
        messages = [
            {
                "role": "system",
                "content": (
                    "你是民航法律澄清回复解析 Agent。只输出 JSON。"
                    "你的任务是判断用户当前输入是在回答上一轮澄清问题，还是开启新问题。"
                    "如果是在回答澄清问题，必须把上一轮原始问题和用户回复合并为一个独立完整的法律问题。"
                    "不要回答法律结论，不要检索法条。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"对话历史：\n{history_text}\n\n"
                    f"待澄清状态：\n{json.dumps(pending, ensure_ascii=False, indent=2)}\n\n"
                    f"用户当前回复：{user_reply}\n\n"
                    "只输出 JSON，格式：\n"
                    "{\n"
                    '  "resolved": true,\n'
                    '  "is_new_question": false,\n'
                    '  "filled_slots": {"请假者": "...", "被辞退者": "..."},\n'
                    '  "still_missing": [],\n'
                    '  "enriched_question": "合并后的独立完整法律问题",\n'
                    '  "clarification": "如果仍缺信息，继续追问的问题",\n'
                    '  "reason": "判断理由"\n'
                    "}\n"
                    "规则：\n"
                    "1. 用户回复“我/本人/是我”通常表示上一轮追问中的相关主体为用户本人，但要结合澄清问题判断。\n"
                    "2. 用户回复“她/老婆/我老婆/妻子”通常表示相关主体为用户妻子。\n"
                    "3. 如果用户明显提出了新的法律问题，is_new_question=true，resolved=false。\n"
                    "4. resolved=true 时 enriched_question 必须让没看过历史的人也能理解。\n"
                    "5. 不能确定时 resolved=false，并在 clarification 中继续追问。\n"
                ),
            },
        ]
        data = self.llm.json(messages)
        return ClarificationResolution(
            resolved=bool(data.get("resolved", False)),
            is_new_question=bool(data.get("is_new_question", False)),
            filled_slots=data.get("filled_slots", {}) if isinstance(data.get("filled_slots", {}), dict) else {},
            still_missing=[str(x) for x in data.get("still_missing", []) if str(x).strip()],
            enriched_question=str(data.get("enriched_question", "")),
            clarification=str(data.get("clarification", "")),
            reason=str(data.get("reason", "")),
        )


class FollowupRewriteAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def rewrite(self, user_input: str, history: list[dict]) -> FollowupRewrite:
        history_text = _format_history_for_prompt(history[-config.HISTORY_TURNS:])
        messages = [
            {
                "role": "system",
                "content": (
                    "你是民航法律多轮对话改写 Agent。只输出 JSON。"
                    "判断用户当前输入是否是在补充上一轮事实、追问上一轮答案、纠正上一轮假设，"
                    "还是一个全新的法律问题。"
                    "如果是补充/追问/纠正，必须结合历史改写成一个独立完整的法律问题。"
                    "不要回答法律结论，不要检索法条。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"最近对话历史：\n{history_text}\n\n"
                    f"用户当前输入：{user_input}\n\n"
                    "只输出 JSON，格式：\n"
                    "{\n"
                    '  "is_followup": true,\n'
                    '  "is_new_question": false,\n'
                    '  "rewrite": "改写后的独立完整法律问题；如果是新问题则原样或轻微补全",\n'
                    '  "reason": "判断理由"\n'
                    "}\n\n"
                    "判断规则：\n"
                    "1. 当前输入是短语、数字、时间、金额、地点、证据、事实补充，例如“航班延误4小时”“在北京”“没批准”“有拒载通知”，通常是上一轮补充。\n"
                    "2. 当前输入追问上一轮答案，例如“那还能赔多少钱”“这样算违法吗”“怎么办”，通常是上一轮追问。\n"
                    "3. 当前输入明确切换到另一个主题，例如“无人机能飞多高”“飞行员执照怎么申请”，通常是新问题。\n"
                    "4. 改写必须保留上一轮用户问题的核心事实，不要只围绕当前短句回答。\n"
                    "5. 如果不确定，优先视为新问题，避免历史污染。\n"
                ),
            },
        ]
        data = self.llm.json(messages)
        rewrite = str(data.get("rewrite", "")).strip()
        is_followup = bool(data.get("is_followup", False))
        is_new_question = bool(data.get("is_new_question", not is_followup))
        return FollowupRewrite(
            is_followup=is_followup,
            is_new_question=is_new_question,
            rewrite=rewrite or user_input,
            reason=str(data.get("reason", "")),
        )


class RewriteAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def rewrite_queries(self, question: str, subject_analysis: SubjectAnalysis, issue_analysis: IssueAnalysis) -> RewritePlan:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是民航法律检索改写 Agent。只输出 JSON。"
                    "把口语问题改写成适合检索法律索引树的法律查询。"
                    "可以给出可能相关法律名称和条号，但不要编造法律结论。\n"
                    "改写规则：\n"
                    "1. 保持原始问题的核心关键词，不要过度泛化\n"
                    "2. 如果问题包含具体操作/动作（如'签派员放行检查'），query 必须包含这些动作词，不能只提取名词（如只写'签派员'）\n"
                    "3. 每个 query 必须是完整的具体检索短语，不是单个词\n"
                    "4. law_hints 应尽可能给出具体法规名称"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"用户问题：{question}\n\n"
                    f"主体分析：{json.dumps(subject_analysis.__dict__, ensure_ascii=False)}\n\n"
                    f"争点分析：{json.dumps(issue_analysis.__dict__, ensure_ascii=False)}\n\n"
                    "只输出 JSON，格式：\n"
                    "{\n"
                    '  "queries": ["..."],\n'
                    '  "law_hints": ["..."],\n'
                    '  "article_hints": ["..."],\n'
                    '  "sub_questions": ["..."]\n'
                    "}\n"
                ),
            },
        ]
        data = self.llm.json(messages)
        return RewritePlan(
            queries=[str(x) for x in data.get("queries", []) if str(x).strip()],
            law_hints=[str(x) for x in data.get("law_hints", []) if str(x).strip()],
            article_hints=[str(x) for x in data.get("article_hints", []) if str(x).strip()],
            sub_questions=[str(x) for x in data.get("sub_questions", []) if str(x).strip()],
        )


class DecompositionAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def decompose(self, question: str, subject_analysis: SubjectAnalysis,
                  issue_analysis: IssueAnalysis,
                  rewrite_plan: RewritePlan) -> DecompositionResult:
        sub_questions = rewrite_plan.sub_questions
        if len(sub_questions) <= 1:
            return DecompositionResult(needs_decomposition=False)
        from .config import MAX_SUBQUESTIONS
        capped = sub_questions[:MAX_SUBQUESTIONS]
        messages = [
            {
                "role": "system",
                "content": (
                    "你是民航法律问题拆分 Agent。只输出 JSON。"
                    "你收到一组子问题，为每个子问题生成独立的检索策略。"
                    "如果问题涉及多个不同法规领域（如航班延误+餐饮服务、飞行员+体检标准），"
                    "必须确保每个子问题指向不同的法规，生成对应的law_hints。"
                    "不要回答法律结论，不要检索法条。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"原始问题：{question}\n\n"
                    f"主体分析：{json.dumps(subject_analysis.__dict__, ensure_ascii=False)}\n\n"
                    f"争点分析：{json.dumps(issue_analysis.__dict__, ensure_ascii=False)}\n\n"
                    f"子问题列表：{json.dumps(capped, ensure_ascii=False)}\n\n"
                    "只输出 JSON，格式：\n"
                    "{\n"
                    '  "sub_problems": [\n'
                    "    {\n"
                    '      "question": "子问题",\n'
                    '      "queries": ["检索query1", "检索query2"],\n'
                    '      "law_hints": ["可能相关法律名"],\n'
                    '      "article_hints": ["可能相关条号"]\n'
                    "    }\n"
                    "  ]\n"
                    "}\n"
                ),
            },
        ]
        data = self.llm.json(messages)
        raw = data.get("sub_problems", [])
        if not isinstance(raw, list):
            raw = []
        problems = []
        for item in raw[:MAX_SUBQUESTIONS]:
            if not isinstance(item, dict):
                continue
            q = str(item.get("question", "")).strip()
            if not q:
                continue
            problems.append(SubProblem(
                question=q,
                queries=[str(x) for x in item.get("queries", []) if str(x).strip()],
                law_hints=[str(x) for x in item.get("law_hints", []) if str(x).strip()],
                article_hints=[str(x) for x in item.get("article_hints", []) if str(x).strip()],
            ))
        if not problems:
            return DecompositionResult(needs_decomposition=False)
        return DecompositionResult(needs_decomposition=True, sub_problems=problems)


class RetrievalAgent:
    def retrieve(self, state: CaseState, logger: TerminalLogger | None = None) -> list[Evidence]:
        if state.plan is None:
            return []
        queries = state.plan.queries or [state.normalized_question]
        if logger:
            logger.info("retrieval/检索", f"开始检索: {len(queries)}个查询",
                         f"查询: {queries[:2]}... | 法律hint: {state.plan.law_hints} | 条号hint: {state.plan.article_hints}")

        # 每个 query 独立检索
        query_results: list[list[Evidence]] = []
        for query in queries:
            hits = search_index_tree(
                query=query,
                law_hints=state.plan.law_hints,
                article_hints=state.plan.article_hints,
                top_k=config.RETRIEVAL_TOP_K,
                _logger=logger,
            )
            query_results.append(hits)

        # 合并策略：WRRF（前向兼容，开关关闭时走等权合并）
        if getattr(config, 'WRRF_ENABLED', False):
            # 保存原始query top-4作为保底
            primary_top4 = query_results[0][:config.WRRF_PRIMARY_TOP] if query_results else []
            all_hits = self._wrrf_merge(query_results)
            all_hits = self._cap_per_law(all_hits)
            # 保底插入：原始query top-4 不在结果中的强制加入
            final_keys = {(h.law_id, h.node_id) for h in all_hits}
            for hit in primary_top4:
                if (hit.law_id, hit.node_id) not in final_keys:
                    all_hits.append(hit)
            all_hits.sort(key=lambda h: h.score, reverse=True)
        else:
            all_hits: list[Evidence] = []
            seen: set[tuple[str, str]] = set()
            for hits in query_results:
                for hit in hits:
                    key = (hit.law_id, hit.node_id)
                    if key in seen:
                        continue
                    seen.add(key)
                    all_hits.append(hit)
            all_hits.sort(key=lambda item: item.score, reverse=True)

        loaded: list[Evidence] = []
        for hit in all_hits[:config.EVIDENCE_LOAD_LIMIT]:
            node_result = read_law_node(hit.law_id, hit.node_id, include_context=True)
            if node_result.get("found"):
                hit.text = node_result["text"]
                hit.source_file = node_result["source_file"]
                hit.source_anchor = node_result["source_anchor"]
                loaded.append(hit)
        if logger:
            logger.success("retrieval/检索", f"回读法条原文: 加载{len(loaded)}条证据", "")
            for hit in loaded[:5]:
                logger.debug("retrieval/证据详情", f"  {hit.law_title} {hit.article} (score={hit.score:.2f})",
                             f"node={hit.node_id}")

        # ── 法条交叉引用补全 ──
        if loaded:
            from .crossref import expand_evidence_references
            xrefs = expand_evidence_references(loaded, max_items=5)
            if xrefs and logger:
                logger.info("retrieval/交叉引用", f"从证据文本中提取对其他法条的引用，自动补全",
                             f"发现{len(xrefs)}处交叉引用: {', '.join(xr['article'] + '(' + xr['law_title'] + ')' for xr in xrefs[:3])}")
            for xr in xrefs:
                from .types import Evidence as _Ev
                xr_law_id = xr.get("law_id", "") or ""
                xr_node_id = xr.get("node_id", "") or ""
                # 交叉引用无法解析到具体法条时，用 raw 文本作为唯一标识
                if not xr_law_id or not xr_node_id:
                    xr_node_id = f"xref:{xr.get('raw', '')}"
                loaded.append(_Ev(
                    law_id=xr_law_id,
                    law_title=xr.get("law_title", ""),
                    node_id=xr_node_id,
                    article=xr.get("article", ""),
                    text=xr.get("text", ""),
                    score=0.0,
                    source_file="",
                    source_anchor=xr.get("raw", ""),
                    verified=False,
                ))

        return loaded

    @staticmethod
    def _wrrf_merge(query_results: list[list[Evidence]],
                    decay: float = 0.3) -> list[Evidence]:
        """Weighted RRF: 第一个query权重1.0，后续按decay衰减。"""
        k = config.RRF_K
        scores: dict[tuple[str, str], float] = {}
        ev_map: dict[tuple[str, str], Evidence] = {}
        for qi, hits in enumerate(query_results):
            w = max(config.WRRF_QUERY_DECAY, 1.0 - decay * qi) if qi > 0 else 1.0
            for rank, hit in enumerate(hits, 1):
                key = (hit.law_id, hit.node_id)
                scores[key] = scores.get(key, 0.0) + w / (k + rank)
                ev_map.setdefault(key, hit)
        sorted_keys = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [ev_map[k_] for k_, _ in sorted_keys]

    @staticmethod
    def _cap_per_law(hits: list[Evidence],
                     max_per_law: int | None = None) -> list[Evidence]:
        """每部法规最多保留 max_per_law 篇证据，防止单法规垄断。"""
        if max_per_law is None:
            max_per_law = getattr(config, 'MAX_EVIDENCE_PER_LAW', 3)
        law_counts: dict[str, int] = {}
        kept: list[Evidence] = []
        for hit in hits:
            cnt = law_counts.get(hit.law_id, 0)
            if cnt < max_per_law:
                kept.append(hit)
                law_counts[hit.law_id] = cnt + 1
        return kept


class CitationAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.verifier = CitationVerifier(llm)

    def verify(self, question: str, legal_issues: list[str], evidence: list[Evidence], logger: TerminalLogger | None = None) -> list[CitationCheck]:
        if not evidence:
            return []
        # C1: Cross-Encoder确定性校验分支
        if config.CROSS_ENCODER_CITATION:
            if logger:
                logger.info("citation/引用校验", "Cross-Encoder语义校验模式", f"待校验证据={len(evidence)}条")
            checks = self.verifier.verify_with_cross_encoder(question, legal_issues, evidence)
        else:
            if logger:
                logger.info("citation/引用校验", "LLM从问题中抽取法律主张，逐条判断证据是否支持", f"待校验证据={len(evidence)}条")
            checks = self.verifier.verify(question, legal_issues, evidence)
        for check in checks:
            if logger:
                status_zh = {"supported": "支持", "partial": "部分支持", "unsupported": "不支持"}.get(check.status, check.status)
                level = "SUCCESS" if check.status == "supported" else "WARNING"
                logger.log("citation/引用校验", f"{check.node_id} {status_zh}(置信度={check.confidence:.1f}): {check.reason[:80]}", level)
        return checks


class ConflictAgent:
    """检测证据中不同法律条文对同一争点的适用优先级差异。

    真正的法条冲突（同一事实有矛盾法律后果）需要 LLM 判断，这里只做
    结构化优先级标注：上位法优于下位法，特别法优于一般法，新法优于旧法。
    """

    # 简化的法律层级优先级（序号越小优先级越高）
    _LAW_PRIORITY = {
        "中华人民共和国民用航空法": 1,
        "民用航空器国籍登记规定": 2,
        "民用航空安全检查规则": 2,
        "公共航空运输旅客服务管理规定": 2,
        "航班正常管理规定": 2,
        "民用航空安全管理规定": 2,
        "一般运行和飞行规则": 2,
        "民用航空人员体检合格证管理规则": 3,
        "民用航空器驾驶员学校合格审定规则": 3,
        "民用航空空中交通管理规则": 3,
        "运输机场运行安全管理规定": 3,
        "通用航空经营许可管理规定": 3,
        "民用航空行政处罚实施办法": 3,
    }

    def check(self, evidence: list[Evidence], logger: TerminalLogger | None = None) -> list[Conflict]:
        by_law: dict[str, list[Evidence]] = {}
        for item in evidence:
            if item.law_title:
                by_law.setdefault(item.law_title, []).append(item)
        if len(by_law) <= 1:
            if logger:
                logger.success("conflict/冲突检测", "证据仅来自单部法律，无适用层级问题", "")
            return []
        # 按优先级排序，标注适用层级
        laws = sorted(
            by_law.keys(),
            key=lambda t: self._LAW_PRIORITY.get(t, 99),
        )
        if logger:
            logger.info("conflict/冲突检测", "证据涉及多部法律，按上位法>下位法、特别法>一般法排序",
                         "适用优先级: " + " > ".join(laws))
        return [Conflict(
            law_titles=laws,
            reason="multi_law_applicable",
            priority_order=laws,
        )]


def _short_article_label(article: str) -> str:
    """把 ev.article 切成纯"第X条"形式（去掉后面的标题尾巴）。"""
    if not article:
        return ""
    import re
    m = re.match(r"^第[\d.]+条", article)
    return m.group(0) if m else article.split()[0] if article else ""


def _render_structured_answer(data: dict, evidence: list[Evidence]) -> str:
    """将结构化 JSON 渲染为用户可读的答案文本。按 source 分区渲染。"""
    node_map = {ev.node_id: ev for ev in evidence}
    parts = []
    if data.get("conclusion"):
        parts.append(f"【结论】\n{data['conclusion']}")
    claims = data.get("claims", [])

    # 按 source 分组
    evidence_claims = []
    suggestion_claims = []
    for c in claims:
        src = c.get("source", "法规规定")
        if src == "法规规定":
            evidence_claims.append(c)
        else:
            suggestion_claims.append(c)

    # 法规规定区
    if evidence_claims:
        parts.append("【法律依据】")
        for c in evidence_claims:
            text = c.get("text", "")
            node_ids = c.get("node_ids", [])
            law_name = c.get("law_name", "")
            refs = []
            primary_ev = None
            for nid in node_ids:
                ev = node_map.get(nid)
                if ev:
                    label = _short_article_label(ev.article)
                    refs.append(f"《{ev.law_title}》{label}")
                    if primary_ev is None:
                        primary_ev = ev
                elif law_name:
                    refs.append(f"《{law_name}》")
            ref_str = f"（{', '.join(refs)}）" if refs else ""
            warn = " ⚠️待核实" if c.get("_sm_warning") else ""
            parts.append(f"- {text}{ref_str}{warn}")
            if primary_ev is not None and primary_ev.text:
                excerpt = primary_ev.text.strip().replace("\n", " ")
                if len(excerpt) > config.ANSWER_EXCERPT_MAX_CHARS:
                    excerpt = excerpt[:config.ANSWER_EXCERPT_MAX_CHARS] + "…"
                parts.append(f"  > 原文摘录：{excerpt}")

    # 补充建议区
    if suggestion_claims:
        parts.append("【补充说明】（以下为一般性建议，非法规直接规定）")
        for c in suggestion_claims:
            text = c.get("text", "")
            src = c.get("source", "一般建议")
            tag = "合理推断" if src == "合理推断" else "建议"
            node_ids = c.get("node_ids", [])
            law_name = c.get("law_name", "")
            refs = []
            for nid in node_ids:
                ev = node_map.get(nid)
                if ev:
                    refs.append(f"《{ev.law_title}》{ev.article}")
                elif law_name:
                    refs.append(f"《{law_name}》")
            ref_str = f"（{', '.join(refs)}）" if refs else ""
            parts.append(f"- [{tag}] {text}{ref_str}")

    for key, label in [("legal_basis", "法律依据详情"), ("conditions", "适用条件"), ("risks", "风险提示")]:
        if data.get(key):
            parts.append(f"【{label}】\n{data[key]}")
    if not parts:
        # 数据为空（LLM JSON 解析失败兜底返回 {}）→ 不要输出 str({})，
        # 改用 evidence 摘要做最简兜底
        if evidence:
            lines = ["【结论】\n根据检索到的相关法规，请参考以下条文："]
            for i, ev in enumerate(evidence[:5], 1):
                lines.append(f"{i}. 《{ev.law_title}》{_short_article_label(ev.article)}")
            return "\n".join(lines)
        return "抱歉，未能生成答案。请重新提问或提供更多细节。"
    return "\n\n".join(parts)


_REFUSAL_PATTERNS = (
    "未找到", "无法找到", "未检索到", "找不到",
    "无法确定", "无法回答", "无法判断", "无法精确",
    "建议咨询", "建议您咨询", "暂无", "不明确",
)


class SynthesisAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def compose_answer(self, question: str, plan: RetrievalPlan, evidence: list[Evidence], citations: list[CitationCheck], conflicts: list[Conflict]) -> str:
        # 把 CE 验证通过的证据排到前面，避免 LLM 被大量未支持的总则类证据带偏
        # （典型场景：12 条里 3 条 supported 是真正的"答案"条款，9 条 unsupported 是
        # 91.1 目的、121.3 适用范围等占位总则。LLM 看到 75% 是"目的/依据"会判"证据不足"。）
        if getattr(config, 'EVIDENCE_SORT_BY_CE', True):
            evidence = self._sort_evidence_by_status(evidence, citations)
        evidence_text = self._build_evidence_text(evidence, citations)
        citation_text = [
            {"claim": c.claim, "law_id": c.law_id, "node_id": c.node_id,
             "status": c.status, "reason": c.reason, "quote": c.quote,
             "confidence": c.confidence}
            for c in citations
        ]

        # 证据相关性门控：独立评估调用，通过后才生成答案
        refusal_due_to_no_relevance = False
        if getattr(config, 'RELEVANCE_GATE_ENABLED', False) and evidence:
            assessment = self._assess_relevance(question, evidence_text)
            if assessment == "none":
                refusal_due_to_no_relevance = True

        if refusal_due_to_no_relevance:
            answer = (
                "【结论】\n很抱歉，未找到直接回答您问题的法律条文。\n\n"
                "【建议】\n建议您咨询专业法律人士，或提供更多细节以便重新检索相关法规。"
            )
        elif getattr(config, 'SYNTHESIS_JSON_MODE', False):
            answer = self._compose_structured(question, plan, evidence, evidence_text, citation_text, conflicts)
        else:
            answer = self._compose_free_text(question, plan, evidence_text, citation_text, conflicts)

        # 拒答兜底：LLM 给出"未找到/无法判断"等拒绝话术时，至少把检索到的法规列出来
        if (getattr(config, 'SYNTHESIS_REFUSAL_FALLBACK', True)
                and evidence
                and self._looks_like_refusal(answer)):
            fallback = self._render_evidence_fallback(question, plan, evidence)
            if fallback:
                return fallback
        return answer

    @staticmethod
    def _looks_like_refusal(text: str) -> bool:
        if not text:
            return True
        return any(p in text for p in _REFUSAL_PATTERNS)

    @staticmethod
    def _sort_evidence_by_status(evidence: list, citations: list) -> list:
        """把 CE 验证通过的证据排到前面（supported > partial > unsupported > no_check）。
        避免 LLM 被大量未支持的总则类证据带偏而判"证据不足"。

        关键：低置信度 supported（CE 分数 < EVIDENCE_SORT_MIN_SUPPORTED_CONF）不算"真支持"，
        按 partial 处理，否则会把 score=0.17 的"勉强通过"证据挤掉 score=0.99 的"高质量支持"证据。
        """
        if not citations:
            return list(evidence)
        cite_map = {c.node_id: c for c in citations}
        known = {"supported", "partial", "unsupported"}
        if not any(c.status in known for c in citations):
            return list(evidence)
        order = {"supported": 0, "partial": 1, "unsupported": 2}
        min_conf = getattr(config, "EVIDENCE_SORT_MIN_SUPPORTED_CONF", 0.5)

        def sort_key(ev):
            c = cite_map.get(ev.node_id)
            if c is None:
                return (3, 0.0)
            status = c.status
            conf = getattr(c, "confidence", 0.0) or 0.0
            # 低置信度 supported 降级为 partial，避免污染排序
            if status == "supported" and conf < min_conf:
                effective = "partial"
            else:
                effective = status
            return (order.get(effective, 3), -conf)

        return sorted(evidence, key=sort_key)

    @staticmethod
    def _render_evidence_fallback(question: str, plan: RetrievalPlan, evidence: list[Evidence]) -> str:
        limit = getattr(config, 'SYNTHESIS_FALLBACK_MAX_ITEMS', 5)
        items = evidence[:limit]
        if not items:
            return ""
        lines = [
            "【结论】",
            "以下为系统检索到的可能相关法规，请参考：",
            "",
            "【相关法规】",
        ]
        for i, ev in enumerate(items, 1):
            law = ev.law_title or ev.law_id or "未知法规"
            article = _short_article_label(ev.article)
            excerpt = (ev.text or "").replace("\n", " ")
            if len(excerpt) > config.ANSWER_FALLBACK_EXCERPT_MAX_CHARS:
                excerpt = excerpt[:config.ANSWER_FALLBACK_EXCERPT_MAX_CHARS] + "…"
            lines.append(f"{i}. 《{law}》{article}")
            if excerpt:
                lines.append(f"   摘录：{excerpt}")
        lines.extend([
            "",
            "【说明】",
            "以上条目由系统按相关度自动排序，可能与您的具体情形有偏差。",
            "如需进一步确认，建议补充更多细节（如具体情形、适用法规名）或咨询专业法律人士。",
        ])
        return "\n".join(lines)

    def _assess_relevance(self, question: str, evidence_text: list[str]) -> str:
        evidence_summary = "\n".join(f"- {ev[:200]}" for ev in evidence_text[:8])
        messages = [
            {"role": "system", "content": (
                "你是证据相关性评估器。判断以下证据是否与用户问题属于同一法律领域。\n"
                "只输出一个 JSON：{\"overall\": \"sufficient\" | \"partial\" | \"none\", \"reason\": \"...\"}\n\n"
                "判定标准：\n"
                "- \"sufficient\"：证据涉及与问题相同的法规或同一法律领域，能部分或全部回答问题\n"
                "- \"partial\"：证据与问题相关但覆盖不完整\n"
                "- \"none\"：仅当所有证据都与用户问题属于完全不同的法律领域。"
                "例如问\"无人机管理\"但证据是\"外国民用航空器\"，问\"旅客行李赔偿\"但证据是\"行政机关国家赔偿\"。"
                "只要证据和问题涉及同一部法规或同一法律主题，就不能标 none。"
            )},
            {"role": "user", "content": f"用户问题：{question}\n\n证据摘要：\n{evidence_summary}"},
        ]
        try:
            data = self.llm.json(messages)
            return data.get("overall", "sufficient")
        except Exception:
            return "sufficient"

    def _compose_structured(self, question, plan, evidence, evidence_text, citation_text, conflicts):
        # 把 CE 验证状态嵌在 evidence 索引里，让 LLM 知道哪条被交叉验证过（不屏蔽）
        nid_status = {c["node_id"]: c.get("status", "no_check") for c in citation_text}
        evidence_index = "\n".join(
            f"  [{i+1}] [CE={nid_status.get(ev.node_id, 'no_check')}] {ev.law_title} {ev.article} (node_id={ev.node_id})"
            for i, ev in enumerate(evidence[:config.SYNTHESIS_EVIDENCE_LIMIT])
        )
        assumptions_note = ""
        if plan.assumptions:
            assumptions_note = "“以下按" + "、".join(plan.assumptions) + "理解。"
        alt_note = ""
        if plan.alternative_paths:
            alt_note = "提示用户实际主体不同会导致适用路径不同。"
        json_schema = (
            '{"conclusion": "结论", "claims": [{"text": "1-3句声明", '
            '"node_ids": ["证据的node_id"], "law_name": "法规名", '
            '"source": "法规规定|合理推断|一般建议"}], '
            '"legal_basis": "法律依据", "conditions": "适用条件", "risks": "风险提示"}'
        )
        messages = [
            {"role": "system", "content": (
                "你是民航法律问答 Agent。\n\n"
                "## 严格规则\n"
                "1. 每条法律主张必须引用证据编号(node_id)，node_id只能从提供的证据列表中选择。\n"
                "2. 每条声明必须标注source字段：\n"
                "   - \"法规规定\"：该声明的核心内容在证据原文中有直接支撑，且引用了正确的node_id\n"
                "   - \"合理推断\"：基于证据中的原则性条款做出的合理延伸，但证据未给出具体细节\n"
                "   - \"一般建议\"：来自你的通用法律知识，证据中没有直接依据\n"
                "3. 不得编造不存在的法规名或条号。\n"
                "4. 宁可多标\"合理推断\"或\"一般建议\"，也不要把没有证据直接支撑的内容标为\"法规规定\"。\n"
                "5. evidence 文本外的具体数字（小时、天数、金额、比例）必须标注 \"[待核实]\" 后缀或归入 source=\"一般建议\"，禁止凭印象给出精确数字。\n\n"
                "## 覆盖度要求（重要）\n"
                "6. 先读完整【证据全文】再下结论，不要只看前几条 evidence 标题。\n"
                "7. 如果用户问题包含多个并列子项（如\"哪些情况下...\"列举 N 种情形），"
                "每种子情况应有对应 claim。\n"
                "8. evidence 文本中能直接读到的数字、列举、分类等具体信息（如 5 种情形、3 类情形），"
                "不要概括成\"主要包括\"，应分别列条。\n\n"
                "输出JSON格式：\n" + json_schema + "\n"
                + assumptions_note + alt_note
                + "语气柔和口语化，让普通人也能理解。"
            )},
            {"role": "user", "content": (
                f"用户问题：{question}\n\n"
                f"可用证据（[CE=xxx] 为交叉验证状态，CE=supported 表示已被验证相关）：\n{evidence_index}\n\n"
                + "证据全文：\n" + "\n\n".join(evidence_text)
                + "\n\n请先通读所有【证据全文】，再独立判断哪些内容支持用户问题。请输出JSON格式答案。"
            )},
        ]
        try:
            data = self.llm.json(messages)
        except Exception:
            return self._compose_free_text(question, plan, evidence_text, citation_text, conflicts)
        self._last_structured = data
        return _render_structured_answer(data, evidence)

    def _compose_free_text(self, question, plan, evidence_text, citation_text, conflicts):
        assumptions_note = ""
        if plan.assumptions:
            assumptions_note = "“回答开头必须先说明”" + "、".join(f"“以下按{a}”理解" for a in plan.assumptions) + "。"
        alt_note = ""
        if plan.alternative_paths:
            alt_note = "必须提示用户实际主体不同会导致适用路径不同。"
        messages = [
            {"role": "system", "content": (
                "你是民航法律问答 Agent。必须只基于给定证据回答。"
                "不得凭记忆补充法条，不得编造。"
                "如果证据不足或事实不清，必须说明需要补充哪些事实。"
                "evidence 文本外的具体数字（小时、天数、金额、比例）必须标注 \"[待核实]\" 后缀，或归入【风险提示】部分，禁止凭印象给出精确数字。"
                + assumptions_note + alt_note
                + "回答格式：【结论】、【法律依据（必须引用原文标出出处）】、【适用条件】、【风险提示】。"
                + "语气柔和口语化，让普通人也能理解。"
            )},
            {"role": "user", "content": (
                f"用户问题：{question}\n\n"
                f"事实抽取与检索计划：\n{json.dumps(plan.__dict__, ensure_ascii=False, indent=2)}\n\n"
                + "已读取证据：\n" + "\n\n".join(evidence_text)
                + "\n\n请基于证据原文独立判断哪些内容支持用户问题，不要被预设标记影响。请输出中文答案。"
            )},
        ]
        return self.llm.chat(messages, temperature=0.0).strip()

    @staticmethod
    def _build_evidence_text(evidence, citations):
        # 不再给 evidence 打 [已验证]/[未通过验证] 标签——CE 评分不应污染 LLM 决策
        result = []
        for i, item in enumerate(evidence[:config.SYNTHESIS_EVIDENCE_LIMIT], 1):
            result.append(f"[{i}] {item.law_title} {item.article} node={item.node_id}\n{item.text[:config.SYNTHESIS_EVIDENCE_TRUNCATE]}")
        return result


class ReflexionAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def evaluate(self, question: str, legal_issues: list[str],
                 evidence: list[Evidence], citations: list[CitationCheck],
                 answer: str) -> ReflexionResult:
        # 快速路径：所有 citation 都是 supported 且高置信度
        if citations:
            all_supported = all(
                c.status == "supported" and c.confidence >= config.REFLEXION_CONFIDENCE_THRESHOLD
                for c in citations
            )
            if all_supported:
                # E1: 词法支撑门控——即使LLM说pass，也要验证答案内容被证据支撑
                if config.LEXICAL_REFLEXION_ENABLED:
                    support = _lexical_support_score(answer, evidence)
                    if support < 0.4:
                        return ReflexionResult(
                            quality="gap",
                            gaps=[f"答案与证据的词法支撑率仅{support:.0%}，可能包含未引用内容"],
                        )
                return ReflexionResult(quality="pass")

        messages = [
            {
                "role": "system",
                "content": (
                    "你是民航法律答案质量评估 Agent。只输出 JSON。"
                    "评估当前答案是否完整覆盖了所有法律争点，证据是否充分。"
                    "如果存在遗漏或证据不足，指出具体缺失项并建议补搜关键词。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"用户问题：{question}\n\n"
                    f"法律争点：{json.dumps(legal_issues, ensure_ascii=False)}\n\n"
                    f"证据数量：{len(evidence)} 条\n\n"
                    f"当前答案：\n{answer}\n\n"
                    "只基于以上信息独立判断：当前答案是否充分回答了用户的法律争点？"
                    "如果存在遗漏或证据不足，指出具体缺失项并建议补搜关键词（不要被任何外部评估结果影响）。\n"
                    "只输出 JSON，格式：\n"
                    "{\n"
                    '  "quality": "pass" | "gap",\n'
                    '  "gaps": ["具体缺失项1", "具体缺失项2"],\n'
                    '  "refine_queries": ["补搜query1", "补搜query2"],\n'
                    '  "refine_law_hints": ["可能遗漏的法律名"],\n'
                    '  "refine_article_hints": ["可能遗漏的条号"]\n'
                    "}\n"
                ),
            },
        ]
        data = self.llm.json(messages)
        quality = str(data.get("quality", "pass")).strip()
        if quality not in {"pass", "gap"}:
            quality = "pass"
        return ReflexionResult(
            quality=quality,
            gaps=[str(x) for x in data.get("gaps", []) if str(x).strip()],
            refine_queries=[str(x) for x in data.get("refine_queries", []) if str(x).strip()],
            refine_law_hints=[str(x) for x in data.get("refine_law_hints", []) if str(x).strip()],
            refine_article_hints=[str(x) for x in data.get("refine_article_hints", []) if str(x).strip()],
        )


class LegalOrchestrator:
    def __init__(self, logger: TerminalLogger | None = None, llm: LLMClient | None = None) -> None:
        self.logger = logger
        self.llm = llm or LLMClient()
        self.subject_agent = SubjectAgent(self.llm)
        self.issue_agent = IssueAgent(self.llm)
        self.clarification_agent = ClarificationAgent()
        self.clarification_resolution_agent = ClarificationResolutionAgent(self.llm)
        self.followup_rewrite_agent = FollowupRewriteAgent(self.llm)
        self.rewrite_agent = RewriteAgent(self.llm)
        self.decomposition_agent = DecompositionAgent(self.llm)
        self.retrieval_agent = RetrievalAgent()
        self.citation_agent = CitationAgent(self.llm)
        self.conflict_agent = ConflictAgent()
        self.synthesis_agent = SynthesisAgent(self.llm)
        self.reflexion_agent = ReflexionAgent(self.llm)

    def answer(self, question: str) -> AnswerResult:
        if self.logger:
            self.logger.info("orchestrator", "收到问题 / request", question)
        try:
            return self._answer(question)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            if self.logger:
                self.logger.error("orchestrator", "异常中断 / error", str(exc))
            return AnswerResult(
                answer="抱歉，处理您的问题时遇到了异常，请稀后重试。",
                intent="error",
                topic="",
                status="error",
            )

    def _answer(self, question: str) -> AnswerResult:
        state = CaseState(
            request_id="local",
            original_question=question,
            normalized_question=normalize_text(question),
        )
        # Step 1: SubjectAgent — LLM从问题中抽取法律主体、事件链、假设
        try:
            subject_analysis = self.subject_agent.extract_subjects(question)
        except Exception as exc:
            if self.logger:
                self.logger.warning("subject/主体提取", f"LLM失败，使用默认值: {exc}", "")
            subject_analysis = SubjectAnalysis()
        state.subject_analysis = subject_analysis
        if self.logger:
            subs = ", ".join(f"{k}={v}" for k, v in subject_analysis.subjects.items() if v and v != "不适用")
            self.logger.info("subject/主体提取", "LLM抽取法律主体和事件链",
                             f"主体: {subs} | 决策: {subject_analysis.clarification_decision}")

        # Step 2: IssueAgent — LLM基于主体分析识别法律争点
        try:
            issue_analysis = self.issue_agent.identify_issues(question, subject_analysis)
        except Exception as exc:
            if self.logger:
                self.logger.warning("issue/争点识别", f"LLM失败，使用默认值: {exc}", "")
            issue_analysis = IssueAnalysis()
        state.issue_analysis = issue_analysis
        if self.logger:
            self.logger.info("issue/争点识别", "LLM识别法律争点",
                             f"争点({len(issue_analysis.legal_issues)}个): {' | '.join(issue_analysis.legal_issues[:3])}")

        # Step 3: ClarificationAgent — 判断关键事实是否缺失需要追问
        need_clarification, clarification = self.clarification_agent.should_clarify(subject_analysis, issue_analysis)
        if need_clarification:
            if self.logger:
                self.logger.warning("clarification/澄清判断", "关键事实缺失，需要追问用户", clarification)
            return AnswerResult(
                answer=clarification,
                intent="clarify",
                topic=",".join(issue_analysis.legal_issues),
                status="need_clarification",
                pending_clarification={
                    "type": "subject_clarification",
                    "original_question": question,
                    "normalized_question": state.normalized_question,
                    "subject_analysis": subject_analysis.__dict__,
                    "issue_analysis": issue_analysis.__dict__,
                    "missing_slots": subject_analysis.uncertain_facts + issue_analysis.missing_facts,
                    "clarification": clarification,
                    "attempts": 0,
                },
            )
        if self.logger:
            self.logger.info("clarification/澄清判断", "无需追问，直接进入检索阶段", "")

        # Step 4: RewriteAgent — LLM将口语问题改写为法律检索查询
        try:
            rewrite_plan = self.rewrite_agent.rewrite_queries(question, subject_analysis, issue_analysis)
        except Exception as exc:
            if self.logger:
                self.logger.warning("rewrite/查询改写", f"LLM失败，使用原始问题: {exc}", "")
            rewrite_plan = RewritePlan(queries=[question])
        state.rewrite_plan = rewrite_plan
        if self.logger:
            self.logger.info("rewrite/查询改写", "LLM将口语问题改写为法律检索查询",
                             f"查询({len(rewrite_plan.queries)}个): {' | '.join(rewrite_plan.queries[:3])}")
            if rewrite_plan.law_hints:
                self.logger.info("rewrite/查询改写", "LLM推荐的可能相关法律", " | ".join(rewrite_plan.law_hints))
            if rewrite_plan.article_hints:
                self.logger.info("rewrite/查询改写", "LLM推荐的可能相关法条", " | ".join(rewrite_plan.article_hints))

        # ── A1: 查询确定性门控 ──
        if config.QUERY_GATE_ENABLED:
            normalized_q = state.normalized_question
            gate_queries = [normalized_q]
            for q in rewrite_plan.queries:
                q_n = normalize_text(q)
                if q_n != normalized_q and q_n not in gate_queries:
                    gate_queries.append(q)
            gate_queries = gate_queries[:5]
            if self.logger:
                self.logger.info("query_gate/确定性门控",
                                 "原始query确保参与检索，LLM改写仅做扩展",
                                 f"gate_queries({len(gate_queries)}): {gate_queries[:3]}")
        else:
            gate_queries = rewrite_plan.queries or [question]

        plan = RetrievalPlan(
            intent="legal",
            legal_issues=issue_analysis.legal_issues,
            facts=subject_analysis.subjects,
            queries=gate_queries,
            law_hints=rewrite_plan.law_hints,
            article_hints=rewrite_plan.article_hints,
            need_clarification=False,
            clarification="",
            sub_questions=rewrite_plan.sub_questions,
            assumptions=subject_analysis.assumptions,
            alternative_paths=subject_analysis.alternative_paths,
        )
        state.plan = plan

        # ── Step 5: DecompositionAgent — 复杂问题拆分为子问题 ──
        try:
            decomposition = self.decomposition_agent.decompose(
                question, subject_analysis, issue_analysis, rewrite_plan)
        except Exception as exc:
            if self.logger:
                self.logger.warning("decomposition/问题拆分", f"LLM失败，跳过拆分: {exc}", "")
            decomposition = DecompositionResult(needs_decomposition=False)
        if self.logger:
            if decomposition.needs_decomposition:
                self.logger.info("decomposition/问题拆分", "LLM判断为复杂问题，拆分为独立子问题分别检索",
                                 f"拆分为{len(decomposition.sub_problems)}个子问题")
            else:
                self.logger.info("decomposition/问题拆分", "单问题，无需拆分", "")

        if decomposition.needs_decomposition:
            evidence = self._retrieve_decomposed(state, decomposition)
        else:
            evidence = self.retrieval_agent.retrieve(state, self.logger)

        # Step 7: CitationAgent — LLM抽取法律主张并逐条校验证据是否支持
        citations = self.citation_agent.verify(question, plan.legal_issues, evidence, self.logger)

        # Step 8: ConflictAgent — 检测证据涉及多部法律时的适用优先级
        conflicts = self.conflict_agent.check(evidence, self.logger)
        if self.logger:
            supported = sum(1 for c in citations if c.status in {"supported", "partial"})
            self.logger.info("synthesis/答案生成", "汇总证据和校验结果，LLM生成最终答案",
                             f"证据={len(evidence)}条, 引用校验={supported}/{len(citations)}通过")
        answer = self.synthesis_agent.compose_answer(question, plan, evidence, citations, conflicts)

        # ── Set-Membership 校验：标记引用不存在 node_id 的 claim（警告而非删除） ──
        if getattr(config, 'SET_MEMBERSHIP_CHECK', False):
            structured = getattr(self.synthesis_agent, '_last_structured', None)
            if structured and structured.get('claims'):
                valid_node_ids = {ev.node_id for ev in evidence}
                warned = 0
                for claim in structured['claims']:
                    node_ids = claim.get('node_ids', [])
                    if node_ids and not any(nid in valid_node_ids for nid in node_ids):
                        claim['_sm_warning'] = True
                        warned += 1
                if warned > 0:
                    if self.logger:
                        self.logger.warning("set_membership", f"标记{warned}条引用不匹配证据的声明",
                                           f"共{len(structured['claims'])}条")
                    answer = _render_structured_answer(structured, evidence)
                    state._sm_removed = warned

        # ── Step 9: Reflexion 自检循环 — LLM评估答案质量，不足则补搜重试 ──
        state.evidence = evidence
        state.citations = citations
        state.conflicts = conflicts
        state.final_answer = answer

        # 追踪最佳迭代结果，防止 Reflexion 越补越差
        best_quality = self._iter_quality(state.final_answer, state.citations)
        best_answer = state.final_answer
        best_citations = list(state.citations)
        best_evidence = list(state.evidence)

        reflexion_iterations = 0
        from .config import MAX_REFLEXION_ITERATIONS
        for _ in range(MAX_REFLEXION_ITERATIONS):
            reflexion = self.reflexion_agent.evaluate(
                question, plan.legal_issues, state.evidence, state.citations, state.final_answer)
            if reflexion.quality == "pass":
                if self.logger:
                    self.logger.success("reflexion/质量自检", "LLM判定答案质量合格，通过", "")
                break
            reflexion_iterations += 1
            if self.logger:
                self.logger.warning("reflexion/质量自检",
                                    f"LLM判定答案有缺陷(第{reflexion_iterations}轮)，将补搜证据后重试",
                                    f"缺失项: {'; '.join(reflexion.gaps[:3])}")
            # 用补搜词构建独立检索计划，不污染原始 plan
            # 跨法规补搜：清空 law_hints 限制，在全部 129 法内重新检索
            refine_plan = RetrievalPlan(
                intent="legal",
                legal_issues=plan.legal_issues,
                facts=plan.facts,
                queries=reflexion.refine_queries,
                law_hints=plan.law_hints,  # 保留原始法规限制，避免补搜引入无关法规
                article_hints=reflexion.refine_article_hints,
                assumptions=plan.assumptions,
                alternative_paths=plan.alternative_paths,
            )
            refine_state = CaseState(
                request_id=state.request_id,
                original_question=question,
                normalized_question=state.normalized_question,
                plan=refine_plan,
            )
            new_evidence = self.retrieval_agent.retrieve(refine_state, self.logger)
            seen = {(e.law_id, e.node_id) for e in state.evidence}
            for e in new_evidence:
                if (e.law_id, e.node_id) not in seen:
                    state.evidence.append(e)
                    seen.add((e.law_id, e.node_id))
            # 按分数排序后截断，避免低分补搜证据挤掉高分初始证据
            state.evidence.sort(key=lambda e: e.score, reverse=True)
            state.evidence = state.evidence[:config.EVIDENCE_LOAD_LIMIT]
            state.citations = self.citation_agent.verify(question, plan.legal_issues, state.evidence, self.logger)
            state.conflicts = self.conflict_agent.check(state.evidence, self.logger)
            state.final_answer = self.synthesis_agent.compose_answer(
                question, plan, state.evidence, state.citations, state.conflicts)
            state.reflexion_trace.append({
                "iteration": reflexion_iterations,
                "gaps": reflexion.gaps,
                "refine_queries": reflexion.refine_queries,
            })

            # 跟踪本轮质量，若优于历史最佳则记录
            q = self._iter_quality(state.final_answer, state.citations)
            if q > best_quality:
                best_quality = q
                best_answer = state.final_answer
                best_citations = list(state.citations)
                best_evidence = list(state.evidence)

        # 还原最佳迭代结果——避免 Reflexion 越补越差
        state.final_answer = best_answer
        state.citations = best_citations
        state.evidence = best_evidence

        # D 类防御：标记 evidence 未支撑的数字（防 LLM 编造具体数字）
        if getattr(config, 'NUMBER_GUARD_ENABLED', True) and state.evidence:
            evidence_texts = [e.text for e in state.evidence if e.text]
            wrapped, unsupported_nums = wrap_unsupported_numbers(state.final_answer, evidence_texts)
            if unsupported_nums and self.logger:
                self.logger.warning("number_guard/数字幻觉",
                                    f"标记 {len(unsupported_nums)} 个未支撑数字: {unsupported_nums[:5]}",
                                    f"已自动加 [待核实] 后缀")
            state.final_answer = wrapped

        if self.logger:
            self.logger.success("orchestrator/完成",
                                f"回答生成完毕(自检{reflexion_iterations}轮)",
                                f"答案长度={len(state.final_answer)}字, 最终证据={len(state.evidence)}条")

        # E1: 答案格式后处理——附加标准引用列表
        if config.LEXICAL_REFLEXION_ENABLED and state.citations:
            citation_block = _build_citation_block(state.citations)
            if citation_block:
                state.final_answer = state.final_answer.rstrip() + citation_block

        return AnswerResult(
            answer=state.final_answer,
            intent=plan.intent,
            topic=",".join(plan.legal_issues),
            evidence=state.evidence,
            citations=state.citations,
            conflicts=state.conflicts,
            reflexion_iterations=reflexion_iterations,
            structured_claims=getattr(self.synthesis_agent, '_last_structured', {}).get('claims', []),
            unsupported_claims_removed=getattr(state, '_sm_removed', 0),
        )

    @staticmethod
    def _iter_quality(answer: str, citations: list[CitationCheck]) -> int:
        """评估一次 Reflexion 迭代的答案质量。

        评分 = 支持/部分支持的引用数 - 拒答惩罚(1000)。
        拒答（"未找到"等）永远不如非拒答，即使有少量引用。
        """
        support = sum(1 for c in citations if c.status in {"supported", "partial"})
        if SynthesisAgent._looks_like_refusal(answer):
            return support - 1000
        return support

    def _retrieve_decomposed(self, state: CaseState, decomposition: DecompositionResult) -> list[Evidence]:
        per_sub: list[list[Evidence]] = []
        for sp in decomposition.sub_problems:
            sub_plan = RetrievalPlan(
                intent="legal",
                legal_issues=state.plan.legal_issues if state.plan else [],
                facts=state.plan.facts if state.plan else {},
                queries=sp.queries or [sp.question],
                law_hints=sp.law_hints,
                article_hints=sp.article_hints,
                sub_questions=[sp.question],
                assumptions=state.plan.assumptions if state.plan else [],
                alternative_paths=state.plan.alternative_paths if state.plan else [],
            )
            sub_state = CaseState(
                request_id=state.request_id,
                original_question=sp.question,
                normalized_question=normalize_text(sp.question),
                plan=sub_plan,
            )
            hits = self.retrieval_agent.retrieve(sub_state, self.logger)
            per_sub.append(hits)

        # Round-Robin 合并：每个子问题最多贡献4条，保证跨法规覆盖
        all_evidence: list[Evidence] = []
        seen: set[tuple[str, str]] = set()
        max_per_sub = 4
        for hits in per_sub:
            count = 0
            for hit in hits:
                key = (hit.law_id, hit.node_id)
                if key not in seen and count < max_per_sub:
                    seen.add(key)
                    all_evidence.append(hit)
                    count += 1
        if self.logger:
            self.logger.info("decomposition/问题拆分", f"所有子问题检索完成，Round-Robin合并",
                             f"共{len(all_evidence)}条证据来自{len(per_sub)}个子问题")
        return all_evidence


def _format_history_for_prompt(history: list[dict]) -> str:
    lines = []
    for item in history:
        role = "用户" if item.get("role") == "user" else "助手"
        content = str(item.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "无"
