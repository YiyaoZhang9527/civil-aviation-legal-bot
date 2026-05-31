from pathlib import Path

from legalbot.conversation import ConversationManager
from legalbot.agents import LegalOrchestrator, RetrievalPlan, SubjectAnalysis, IssueAnalysis, ClarificationAgent
from legalbot.llm import LLMError, parse_json_object
from legalbot.session_store import SessionStore
from legalbot.types import AnswerResult


CASES = [
    "航班延误怎么赔偿？",
    "无人机需要什么证才能飞？",
    "飞行员执照怎么申请？",
    "行李丢失航空公司怎么赔？",
    "机场噪音超标怎么办？",
    "航空安全员有什么职责？",
    "通用航空经营许可怎么办理？",
    "民航旅客被拒载怎么维权？",
    "民用航空器适航证怎么取得？",
    "空中交通管制员执照怎么考？",
]


def test_10_colloquial_cases():
    try:
        bot = LegalOrchestrator()
    except LLMError:
        return
    for question in CASES:
        try:
            result = bot.answer(question)
        except LLMError:
            return
        assert result.answer
        assert result.intent in {"legal", "clarify", "chitchat"}


def test_no_old_hardcoded_topic_router():
    retrieval = Path("legalbot/retrieval.py").read_text(encoding="utf-8")
    agents = Path("legalbot/agents.py").read_text(encoding="utf-8")
    assert "TOPIC_HINTS" not in retrieval
    assert "trial_period" not in agents
    assert "试用期上限取决于合同期限" not in agents
    assert "加班费" not in agents
    assert "劳动合同" not in agents


def test_parse_json_object_repairs_missing_closing_brace():
    parsed = parse_json_object('{"legal_issues": ["未签合同"], "missing_facts": []')
    assert parsed["legal_issues"] == ["未签合同"]


def test_answer_with_assumption_does_not_block_clarification():
    subject = SubjectAnalysis(
        subjects={"旅客": "用户本人", "承运人": "航空公司"},
        assumptions=["按中文口语省略主语，理解为用户本人作为旅客被拒载"],
        alternative_paths=["如果实际为他人被拒载，需确认具体旅客身份"],
        clarification_decision="answer_with_assumption",
        need_clarification=True,
        clarification="请明确是被谁拒载",
    )
    issue = IssueAnalysis(missing_facts=["地区", "请假天数"])
    need, clarification = ClarificationAgent().should_clarify(subject, issue)

    assert need is False
    assert clarification == ""


def test_retrieval_plan_carries_subject_assumptions():
    plan = RetrievalPlan(
        intent="legal",
        assumptions=["以下按用户本人作为旅客理解"],
        alternative_paths=["如果是货运纠纷则适用货物运输规定"],
    )

    assert plan.assumptions
    assert plan.alternative_paths


class FakeResolution:
    resolved = True
    is_new_question = False
    filled_slots = {"旅客": "用户本人", "承运人": "航空公司"}
    still_missing = []
    enriched_question = "用户本人乘坐航班被航空公司拒载，是否合法？"
    clarification = ""
    reason = "用户回复确认自己是旅客"


class FakeResolutionAgent:
    def resolve(self, pending, user_reply, history):
        assert pending["original_question"] == "航班超售被拒载怎么办"
        assert user_reply == "是我"
        assert history
        return FakeResolution()


class FakeBot:
    def __init__(self):
        self.clarification_resolution_agent = FakeResolutionAgent()
        self.followup_rewrite_agent = None
        self.questions = []

    def answer(self, question):
        self.questions.append(question)
        return AnswerResult(answer="已进入法律分析", intent="legal", topic="拒载")


def test_conversation_resolves_pending_clarification(tmp_path):
    store = SessionStore(tmp_path)
    store.save_history(
        "case1",
        [
            {"role": "user", "content": "航班超售被拒载怎么办"},
            {"role": "assistant", "content": "请明确被拒载的是您本人还是他人？"},
        ],
    )
    store.save_pending(
        "case1",
        {
            "type": "subject_clarification",
            "original_question": "航班超售被拒载怎么办",
            "clarification": "请明确被拒载的是您本人还是他人？",
            "missing_slots": ["旅客", "承运人"],
        },
    )
    fake_bot = FakeBot()
    conversation = ConversationManager(session_id="case1", store=store, bot=fake_bot)

    result = conversation.handle("是我")

    assert result.status == "ok"
    assert store.load_pending("case1") is None
    assert fake_bot.questions == ["用户本人乘坐航班被航空公司拒载，是否合法？"]


class FakeFollowup:
    is_followup = True
    is_new_question = False
    rewrite = "用户本人航班延误4小时，补充事实：延误原因为航空公司自身原因。请判断是否可以获得赔偿。"
    reason = "用户补充上一轮延误原因"


class FakeFollowupAgent:
    def rewrite(self, user_input, history):
        assert user_input == "航空公司自己的问题"
        assert history
        return FakeFollowup()


class FakeFollowupBot:
    def __init__(self):
        self.followup_rewrite_agent = FakeFollowupAgent()
        self.questions = []

    def answer(self, question):
        self.questions.append(question)
        return AnswerResult(answer="已基于补充事实重新分析", intent="legal", topic="航班延误")


def test_conversation_rewrites_non_pending_followup(tmp_path):
    store = SessionStore(tmp_path)
    store.save_history(
        "case2",
        [
            {"role": "user", "content": "航班延误4小时怎么办"},
            {"role": "assistant", "content": "以下按您本人遭遇航班延误来分析。"},
        ],
    )
    fake_bot = FakeFollowupBot()
    conversation = ConversationManager(session_id="case2", store=store, bot=fake_bot)

    result = conversation.handle("航空公司自己的问题")

    assert result.status == "ok"
    assert fake_bot.questions == [FakeFollowup.rewrite]
