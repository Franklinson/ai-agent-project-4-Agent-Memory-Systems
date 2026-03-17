"""Conversation memory management with multi-turn context tracking."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
import uuid


class ConversationMemoryError(Exception):
    """Base exception for conversation memory operations."""


class ConversationNotFoundError(ConversationMemoryError):
    """Raised when a conversation is not found."""


@dataclass
class Message:
    """A single message in a conversation."""
    id: str
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    references: List[str] = field(default_factory=list)  # IDs of referenced messages


@dataclass
class Conversation:
    """A conversation with ordered messages, topics, and context."""
    id: str
    user_id: str
    messages: List[Message] = field(default_factory=list)
    topics: Set[str] = field(default_factory=set)
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def last_message(self) -> Optional[Message]:
        return self.messages[-1] if self.messages else None


class ConversationMemory:
    """Stores and manages multi-turn conversations with context tracking."""

    def __init__(self):
        self._conversations: Dict[str, Conversation] = {}

    @property
    def count(self) -> int:
        return len(self._conversations)

    # --- Conversation lifecycle ---

    def create_conversation(self, user_id: str, context: Optional[Dict[str, Any]] = None,
                            conversation_id: Optional[str] = None) -> Conversation:
        cid = conversation_id or str(uuid.uuid4())
        if cid in self._conversations:
            raise ConversationMemoryError(f"Conversation '{cid}' already exists")
        conv = Conversation(id=cid, user_id=user_id, context=context or {})
        self._conversations[cid] = conv
        return conv

    def get_conversation(self, conversation_id: str) -> Conversation:
        if conversation_id not in self._conversations:
            raise ConversationNotFoundError(f"Conversation '{conversation_id}' not found")
        return self._conversations[conversation_id]

    def delete_conversation(self, conversation_id: str) -> bool:
        self.get_conversation(conversation_id)  # raises if missing
        del self._conversations[conversation_id]
        return True

    # --- Message handling ---

    def add_message(self, conversation_id: str, role: str, content: str,
                    metadata: Optional[Dict[str, Any]] = None,
                    references: Optional[List[str]] = None,
                    topics: Optional[List[str]] = None) -> Message:
        conv = self.get_conversation(conversation_id)
        msg = Message(
            id=str(uuid.uuid4()),
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {},
            references=references or [],
        )
        conv.messages.append(msg)
        conv.updated_at = msg.timestamp
        if topics:
            conv.topics.update(topics)
        return msg

    def get_messages(self, conversation_id: str, last_n: Optional[int] = None) -> List[Message]:
        conv = self.get_conversation(conversation_id)
        if last_n is None:
            return list(conv.messages)
        return conv.messages[-last_n:]

    def get_message_by_id(self, conversation_id: str, message_id: str) -> Message:
        conv = self.get_conversation(conversation_id)
        for msg in conv.messages:
            if msg.id == message_id:
                return msg
        raise ConversationMemoryError(f"Message '{message_id}' not found")

    # --- Context maintenance ---

    def update_context(self, conversation_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        conv = self.get_conversation(conversation_id)
        conv.context.update(context)
        conv.updated_at = datetime.now()
        return conv.context

    def get_context(self, conversation_id: str) -> Dict[str, Any]:
        return self.get_conversation(conversation_id).context

    def add_topics(self, conversation_id: str, topics: List[str]) -> Set[str]:
        conv = self.get_conversation(conversation_id)
        conv.topics.update(topics)
        return conv.topics

    def get_topics(self, conversation_id: str) -> Set[str]:
        return self.get_conversation(conversation_id).topics

    # --- Retrieval ---

    def by_user(self, user_id: str) -> List[Conversation]:
        return sorted(
            [c for c in self._conversations.values() if c.user_id == user_id],
            key=lambda c: c.created_at,
        )

    def by_time_range(self, start: datetime, end: datetime) -> List[Conversation]:
        return sorted(
            [c for c in self._conversations.values() if start <= c.created_at <= end],
            key=lambda c: c.created_at,
        )

    def by_topic(self, topic: str) -> List[Conversation]:
        return sorted(
            [c for c in self._conversations.values() if topic in c.topics],
            key=lambda c: c.created_at,
        )

    def search_content(self, query: str) -> List[Conversation]:
        """Find conversations containing query substring in any message."""
        q = query.lower()
        results = []
        for conv in self._conversations.values():
            if any(q in msg.content.lower() for msg in conv.messages):
                results.append(conv)
        return sorted(results, key=lambda c: c.created_at)

    # --- Multi-turn helpers ---

    def get_turn_pairs(self, conversation_id: str) -> List[Dict[str, str]]:
        """Return user/assistant turn pairs for the conversation."""
        msgs = self.get_conversation(conversation_id).messages
        pairs = []
        i = 0
        while i < len(msgs):
            pair = {}
            if msgs[i].role == "user":
                pair["user"] = msgs[i].content
                if i + 1 < len(msgs) and msgs[i + 1].role == "assistant":
                    pair["assistant"] = msgs[i + 1].content
                    i += 2
                else:
                    i += 1
            else:
                pair["assistant"] = msgs[i].content
                i += 1
            pairs.append(pair)
        return pairs

    def get_referenced_messages(self, conversation_id: str, message_id: str) -> List[Message]:
        """Get all messages referenced by a given message."""
        msg = self.get_message_by_id(conversation_id, message_id)
        conv = self.get_conversation(conversation_id)
        msg_map = {m.id: m for m in conv.messages}
        return [msg_map[rid] for rid in msg.references if rid in msg_map]

    def build_context_window(self, conversation_id: str, last_n: Optional[int] = None) -> str:
        """Build a formatted context string from recent messages."""
        msgs = self.get_messages(conversation_id, last_n=last_n)
        return "\n".join(f"{m.role}: {m.content}" for m in msgs)
