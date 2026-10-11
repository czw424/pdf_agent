import logging
from collections import defaultdict
from typing import Dict, List

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from app.config import settings

logger = logging.getLogger(__name__)


class SessionMemory:
    """
    进程内多轮对话历史管理（按 session_id 隔离）。
    仅用于demo/单实例部署；生产环境应替换为Redis等外部存储，
    否则多进程/多副本部署时同一会话可能落到不同实例上导致历史丢失。
    """

    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self._store: Dict[str, List[BaseMessage]] = defaultdict(list)

    def get_history(self, session_id: str) -> List[BaseMessage]:
        return list(self._store.get(session_id, []))

    def add_turn(self, session_id: str, question: str, answer: str) -> None:
        history = self._store[session_id]
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=answer))
        max_messages = self.max_turns * 2
        if len(history) > max_messages:
            self._store[session_id] = history[-max_messages:]

    def clear(self, session_id: str) -> None:
        self._store.pop(session_id, None)
        logger.info(f"会话 {session_id} 历史已清空")


session_memory = SessionMemory(max_turns=settings.SESSION_HISTORY_MAX_TURNS)
