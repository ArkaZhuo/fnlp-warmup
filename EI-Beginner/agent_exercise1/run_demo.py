from __future__ import annotations

from pathlib import Path

from agent import PersonalScheduleAgent, format_decision
from schemas import PendingAction


def main() -> None:
    root = Path(__file__).resolve().parent
    agent = PersonalScheduleAgent(root)
    pending_action: PendingAction | None = None

    print("Personal Schedule Agent 已启动。输入 exit 退出。")
    while True:
        try:
            user_input = input("\n用户> ").strip()
        except EOFError:
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        turn_result = agent.handle(user_input, pending_action=pending_action)
        pending_action = turn_result.pending_action
        print("\nAgent Decision:")
        print(format_decision(turn_result.decision, pending_action))


if __name__ == "__main__":
    main()
