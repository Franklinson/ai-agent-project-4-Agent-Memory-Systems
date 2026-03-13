"""Conversation history management with truncation strategies."""

from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum


class TruncationStrategy(Enum):
    """Available truncation strategies."""
    SLIDING_WINDOW = "sliding_window"
    SUMMARY_BASED = "summary_based"
    SELECTIVE = "selective"


class Message:
    """Represents a conversation message."""
    
    def __init__(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        self.role = role
        self.content = content
        self.timestamp = datetime.now()
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    def format_for_context(self) -> str:
        return f"{self.role}: {self.content}"


class ConversationManager:
    """Manages conversation history with truncation strategies."""
    
    def __init__(self, max_messages: int = 10, strategy: TruncationStrategy = TruncationStrategy.SLIDING_WINDOW):
        self.messages: List[Message] = []
        self.max_messages = max_messages
        self.strategy = strategy
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Store a message with metadata."""
        message = Message(role, content, metadata)
        self.messages.append(message)
        self._apply_truncation()
    
    def _apply_truncation(self) -> None:
        """Apply the configured truncation strategy."""
        if self.strategy == TruncationStrategy.SLIDING_WINDOW:
            self._sliding_window_truncation()
        elif self.strategy == TruncationStrategy.SUMMARY_BASED:
            self._summary_based_truncation()
        elif self.strategy == TruncationStrategy.SELECTIVE:
            self._selective_truncation()
    
    def _sliding_window_truncation(self) -> None:
        """Keep last N messages, remove oldest when full."""
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def _summary_based_truncation(self) -> None:
        """Basic summary-based truncation."""
        if len(self.messages) > self.max_messages:
            # Keep first message (system prompt) and last N-1 messages
            if self.messages[0].role == "system":
                self.messages = [self.messages[0]] + self.messages[-(self.max_messages - 1):]
            else:
                self.messages = self.messages[-self.max_messages:]
    
    def _selective_truncation(self) -> None:
        """Keep important messages and recent context."""
        if len(self.messages) > self.max_messages:
            important = [m for m in self.messages if m.metadata.get("important", False)]
            recent = self.messages[-(self.max_messages - len(important)):]
            
            # Combine and deduplicate
            seen_ids = set()
            result = []
            for msg in important + recent:
                msg_id = id(msg)
                if msg_id not in seen_ids:
                    seen_ids.add(msg_id)
                    result.append(msg)
            
            self.messages = sorted(result, key=lambda m: m.timestamp)[:self.max_messages]
    
    def get_recent_messages(self, count: Optional[int] = None) -> List[Message]:
        """Get recent messages."""
        if count is None:
            return self.messages.copy()
        return self.messages[-count:]
    
    def get_messages_by_role(self, role: str) -> List[Message]:
        """Filter messages by role."""
        return [m for m in self.messages if m.role == role]
    
    def format_for_context(self, count: Optional[int] = None) -> str:
        """Format messages for context."""
        messages = self.get_recent_messages(count)
        return "\n".join(m.format_for_context() for m in messages)
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get full history as dictionaries."""
        return [m.to_dict() for m in self.messages]
    
    def clear(self) -> None:
        """Clear all messages."""
        self.messages.clear()
    
    def set_strategy(self, strategy: TruncationStrategy) -> None:
        """Change truncation strategy."""
        self.strategy = strategy
        self._apply_truncation()
    
    def get_message_count(self) -> int:
        """Get total message count."""
        return len(self.messages)
