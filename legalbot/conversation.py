"""Conversation manager for multi-turn CLI sessions."""

from __future__ import annotations

from .agents import LegalOrchestrator
from . import config
from .logger import TerminalLogger
from .session_store import SessionStore
from .types import AnswerResult


class ConversationManager:
    def __init__(
        self,
        session_id: str = "default",
        logger: TerminalLogger | None = None,
        store: SessionStore | None = None,
        bot: LegalOrchestrator | None = None,
    ) -> None:
        self.session_id = session_id
        self.logger = logger
        self.store = store or SessionStore()
        self.bot = bot or LegalOrchestrator(logger=logger)

    def handle(self, user_input: str) -> AnswerResult:
        history = self.store.load_history(self.session_id)
        pending = self.store.load_pending(self.session_id)
        question = user_input

        if pending:
            if self.logger:
                self.logger.info("conversation/会话管理", "检测到上一轮的澄清等待用户回复", pending.get("clarification", ""))
            resolution = self.bot.clarification_resolution_agent.resolve(pending, user_input, history)
            if self.logger:
                self.logger.info("conversation/会话管理", "LLM判断用户回复是否回答了澄清问题", resolution.reason)
            if resolution.resolved and resolution.enriched_question.strip():
                question = resolution.enriched_question.strip()
                self.store.clear_pending(self.session_id)
                if self.logger:
                    self.logger.info("conversation/会话管理", "澄清已解决，合并为独立完整问题", question)
            elif resolution.is_new_question:
                self.store.clear_pending(self.session_id)
                if self.logger:
                    self.logger.info("conversation/会话管理", "LLM判断用户开启了新问题，清空澄清状态", "")
            else:
                pending["attempts"] = int(pending.get("attempts", 0)) + 1
                if pending["attempts"] >= config.MAX_CLARIFICATION_ATTEMPTS:
                    # 超过最大澄清次数，用原始问题强制回答
                    self.store.clear_pending(self.session_id)
                    if self.logger:
                        self.logger.warning("conversation/会话管理", f"已追问{pending['attempts']}次仍未解决，用原问题强制回答", "")
                    question = pending.get("original_question", user_input)
                else:
                    if resolution.still_missing:
                        pending["missing_slots"] = resolution.still_missing
                    self.store.save_pending(self.session_id, pending)
                    answer = resolution.clarification or pending.get("clarification") or "请继续补充关键信息。"
                    result = AnswerResult(
                        answer=answer,
                        intent="clarify",
                        topic="",
                        status="need_clarification",
                        pending_clarification=pending,
                    )
                    self._append_history(history, user_input, answer)
                    return result
        elif history:
            followup = self.bot.followup_rewrite_agent.rewrite(user_input, history)
            if self.logger:
                self.logger.info("conversation/会话管理",
                                 "LLM判断当前输入是追问/补充还是全新问题",
                                 f"追问={followup.is_followup}, 新问题={followup.is_new_question}, 理由: {followup.reason}")
            if followup.is_followup and not followup.is_new_question:
                question = followup.rewrite
                if self.logger:
                    self.logger.info("conversation/会话管理", "检测到追问，结合历史改写为独立完整问题", question)

        result = self.bot.answer(question)
        if result.status == "need_clarification" and result.pending_clarification:
            self.store.save_pending(self.session_id, result.pending_clarification)
        else:
            self.store.clear_pending(self.session_id)
        self._append_history(history, user_input, result.answer)
        return result

    def clear(self) -> None:
        self.store.clear(self.session_id)

    def _append_history(self, history: list[dict], user_input: str, answer: str) -> None:
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": answer})
        del history[:-20]
        self.store.save_history(self.session_id, history)
