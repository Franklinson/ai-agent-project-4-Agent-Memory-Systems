"""Tests for conversation_manager.py"""

import pytest
from datetime import datetime
from conversation_manager import ConversationManager, Message, TruncationStrategy


class TestMessage:
    """Test Message class."""
    
    def test_message_creation(self):
        msg = Message("user", "Hello", {"key": "value"})
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.metadata == {"key": "value"}
        assert isinstance(msg.timestamp, datetime)
    
    def test_message_to_dict(self):
        msg = Message("assistant", "Hi there")
        result = msg.to_dict()
        assert result["role"] == "assistant"
        assert result["content"] == "Hi there"
        assert "timestamp" in result
    
    def test_message_format_for_context(self):
        msg = Message("user", "Test message")
        assert msg.format_for_context() == "user: Test message"


class TestConversationManager:
    """Test ConversationManager class."""
    
    def test_initialization(self):
        manager = ConversationManager(max_messages=5)
        assert manager.max_messages == 5
        assert manager.strategy == TruncationStrategy.SLIDING_WINDOW
        assert len(manager.messages) == 0
    
    def test_add_message(self):
        manager = ConversationManager()
        manager.add_message("user", "Hello")
        assert manager.get_message_count() == 1
        assert manager.messages[0].role == "user"
        assert manager.messages[0].content == "Hello"
    
    def test_sliding_window_truncation(self):
        manager = ConversationManager(max_messages=3, strategy=TruncationStrategy.SLIDING_WINDOW)
        
        for i in range(5):
            manager.add_message("user", f"Message {i}")
        
        assert manager.get_message_count() == 3
        assert manager.messages[0].content == "Message 2"
        assert manager.messages[-1].content == "Message 4"
    
    def test_summary_based_truncation(self):
        manager = ConversationManager(max_messages=4, strategy=TruncationStrategy.SUMMARY_BASED)
        
        manager.add_message("system", "System prompt")
        for i in range(5):
            manager.add_message("user", f"Message {i}")
        
        assert manager.get_message_count() == 4
        assert manager.messages[0].role == "system"
        assert manager.messages[0].content == "System prompt"
    
    def test_selective_truncation(self):
        manager = ConversationManager(max_messages=4, strategy=TruncationStrategy.SELECTIVE)
        
        manager.add_message("user", "Important", {"important": True})
        manager.add_message("user", "Message 1")
        manager.add_message("user", "Message 2")
        manager.add_message("user", "Message 3")
        manager.add_message("user", "Message 4")
        
        assert manager.get_message_count() == 4
        # Important message should be retained
        important_msgs = [m for m in manager.messages if m.metadata.get("important")]
        assert len(important_msgs) == 1
    
    def test_get_recent_messages(self):
        manager = ConversationManager()
        for i in range(5):
            manager.add_message("user", f"Message {i}")
        
        recent = manager.get_recent_messages(3)
        assert len(recent) == 3
        assert recent[0].content == "Message 2"
    
    def test_get_messages_by_role(self):
        manager = ConversationManager()
        manager.add_message("user", "User message 1")
        manager.add_message("assistant", "Assistant message")
        manager.add_message("user", "User message 2")
        
        user_msgs = manager.get_messages_by_role("user")
        assert len(user_msgs) == 2
        assert all(m.role == "user" for m in user_msgs)
    
    def test_format_for_context(self):
        manager = ConversationManager()
        manager.add_message("user", "Hello")
        manager.add_message("assistant", "Hi")
        
        context = manager.format_for_context()
        assert "user: Hello" in context
        assert "assistant: Hi" in context
    
    def test_get_history(self):
        manager = ConversationManager()
        manager.add_message("user", "Test")
        
        history = manager.get_history()
        assert len(history) == 1
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Test"
    
    def test_clear(self):
        manager = ConversationManager()
        manager.add_message("user", "Test")
        manager.clear()
        assert manager.get_message_count() == 0
    
    def test_set_strategy(self):
        manager = ConversationManager(max_messages=2, strategy=TruncationStrategy.SLIDING_WINDOW)
        
        for i in range(5):
            manager.add_message("user", f"Message {i}")
        
        assert manager.get_message_count() == 2
        
        # Change strategy
        manager.set_strategy(TruncationStrategy.SUMMARY_BASED)
        assert manager.strategy == TruncationStrategy.SUMMARY_BASED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
