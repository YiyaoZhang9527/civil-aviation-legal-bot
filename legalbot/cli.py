"""命令行入口。"""

from __future__ import annotations

import argparse
import sys

from .index_builder import build_all_documents
from .agents import LegalOrchestrator
from .conversation import ConversationManager
from .logger import TerminalLogger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="legalbot")
    parser.add_argument("question", nargs="*", help="法律问题")
    parser.add_argument("--build-index", action="store_true", help="生成索引树")
    parser.add_argument("--chat", action="store_true", help="进入终端连续对话模式")
    parser.add_argument("--quiet", action="store_true", help="隐藏分阶段日志")
    args = parser.parse_args(argv)

    if args.build_index:
        build_all_documents()
        print("index built")
        return 0

    if args.chat:
        return chat_loop(show_logs=not args.quiet)

    question = " ".join(args.question).strip()
    if not question:
        question = sys.stdin.read().strip()
    if not question:
        print("请提供问题")
        return 1

    bot = LegalOrchestrator(logger=TerminalLogger(enabled=not args.quiet))
    result = bot.answer(question)
    print(result.answer)
    return 0


def chat_loop(show_logs: bool = True) -> int:
    build_all_documents()
    logger = TerminalLogger(enabled=show_logs)
    conversation = ConversationManager(session_id="default", logger=logger)
    print("民航法律机器人 CLI 对话模式")
    print("输入问题后回车。输入 /exit 退出，/quiet 关闭日志，/logs 开启日志，/clear 清空上下文。")
    logs_enabled = show_logs
    while True:
        try:
            question = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出")
            return 0
        if not question:
            continue
        if question in {"/exit", "exit", "退出", "q"}:
            print("已退出")
            return 0
        if question == "/quiet":
            logs_enabled = False
            logger.enabled = False
            print("已关闭日志")
            continue
        if question == "/logs":
            logs_enabled = True
            logger.enabled = True
            print("已开启日志")
            continue
        if question == "/clear":
            conversation.clear()
            print("已清空上下文")
            continue

        result = conversation.handle(question)
        print("\n机器人>")
        print(result.answer)


if __name__ == "__main__":
    raise SystemExit(main())
