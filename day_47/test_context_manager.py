"""Tests for context_manager.py"""

import pytest
from context_manager import ContextManager, ContextComponent, Priority
from conversation_manager import ConversationManager
from token_counter import TokenBudgetManager


class TestContextComponent:
    """Test ContextComponent class."""
    
    def test_creation(self):
        component = ContextComponent(
            name="test",
            content="Test content",
            priority=Priority.HIGH,
            tokens=10
        )
        assert component.name == "test"
        assert component.content == "Test content"
        assert component.priority == Priority.HIGH
        assert component.tokens == 10
    
    def test_to_dict(self):
        component = ContextComponent(
            name="test",
            content="content",
            priority=Priority.MEDIUM,
            tokens=5
        )
        data = component.to_dict()
        assert data["name"] == "test"
        assert data["priority"] == "MEDIUM"
        assert data["tokens"] == 5


class TestContextManager:
    """Test ContextManager class."""
    
    def test_initialization(self):
        token_manager = TokenBudgetManager(total_budget=2000)
        manager = ContextManager(token_manager)
        assert manager.token_manager == token_manager
        assert manager.conversation_manager is not None
        assert len(manager.components) == 0
    
    def test_add_component(self):
        token_manager = TokenBudgetManager(total_budget=2000)
        manager = ContextManager(token_manager)
        
        component = manager.add_component(
            "test",
            "Test content",
            Priority.HIGH
        )
        
        assert len(manager.components) == 1
        assert component.name == "test"
        assert component.tokens > 0
    
    def test_add_system_prompt(self):
        token_manager = TokenBudgetManager(total_budget=2000)
        manager = ContextManager(token_manager)
        
        component = manager.add_system_prompt("You are helpful")
        assert component.priority == Priority.CRITICAL
        assert component.name == "system_prompt"
    
    def test_add_conversation_history(self):
        token_manager = TokenBudgetManager(total_budget=2000)
        conv_manager = ConversationManager()
        conv_manager.add_message("user", "Hello")
        conv_manager.add_message("assistant", "Hi")
        
        manager = ContextManager(token_manager, conv_manager)
        component = manager.add_conversation_history()
        
        assert component.priority == Priority.HIGH
        assert "Hello" in component.content
    
    def test_add_tool_definitions(self):
        token_manager = TokenBudgetManager(total_budget=2000)
        manager = ContextManager(token_manager)
        
        tools = [
            {"name": "search", "description": "Search the web"},
            {"name": "calculate", "description": "Perform calculations"}
        ]
        
        component = manager.add_tool_definitions(tools)
        assert component.priority == Priority.MEDIUM
        assert "search" in component.content
    
    def test_calculate_importance_score(self):
        token_manager = TokenBudgetManager(total_budget=2000)
        manager = ContextManager(token_manager)
        
        critical = ContextComponent("test", "content", Priority.CRITICAL, 10)
        high = ContextComponent("test", "content", Priority.HIGH, 10)
        
        critical_score = manager.calculate_importance_score(critical)
        high_score = manager.calculate_importance_score(high)
        
        assert critical_score > high_score
    
    def test_importance_score_with_metadata(self):
        token_manager = TokenBudgetManager(total_budget=2000)
        manager = ContextManager(token_manager)
        
        base = ContextComponent("test", "content", Priority.HIGH, 10)
        required = ContextComponent(
            "test", "content", Priority.HIGH, 10,
            metadata={"required": True}
        )
        
        base_score = manager.calculate_importance_score(base)
        required_score = manager.calculate_importance_score(required)
        
        assert required_score > base_score
    
    def test_select_components_no_overflow(self):
        token_manager = TokenBudgetManager(total_budget=2000)
        manager = ContextManager(token_manager, max_context_tokens=1000)
        
        manager.add_component("comp1", "Short text", Priority.HIGH)
        manager.add_component("comp2", "Another short text", Priority.MEDIUM)
        
        selected = manager.select_components()
        assert len(selected) == 2
    
    def test_select_components_with_overflow(self):
        token_manager = TokenBudgetManager(total_budget=2000)
        manager = ContextManager(token_manager, max_context_tokens=50)
        
        manager.add_component("critical", "Critical content", Priority.CRITICAL)
        manager.add_component("high", "High priority content", Priority.HIGH)
        manager.add_component("low", "Low priority content " * 50, Priority.LOW)
        
        selected = manager.select_components()
        
        # Should prioritize critical and high over low
        priorities = [c.priority for c in selected]
        assert Priority.CRITICAL in priorities
        assert len(selected) < len(manager.components)
    
    def test_detect_overflow(self):
        token_manager = TokenBudgetManager(total_budget=2000)
        manager = ContextManager(token_manager, max_context_tokens=20)
        
        manager.add_component("test", "This is a longer text " * 10, Priority.HIGH)
        
        assert manager.detect_overflow()
    
    def test_no_overflow(self):
        token_manager = TokenBudgetManager(total_budget=2000)
        manager = ContextManager(token_manager, max_context_tokens=1000)
        
        manager.add_component("test", "Short text", Priority.HIGH)
        
        assert not manager.detect_overflow()
    
    def test_handle_overflow(self):
        token_manager = TokenBudgetManager(total_budget=2000)
        manager = ContextManager(token_manager, max_context_tokens=50)
        
        manager.add_component("critical", "Critical", Priority.CRITICAL)
        manager.add_component("low", "Low priority " * 100, Priority.LOW)
        
        result = manager.handle_overflow()
        
        # Should keep critical, drop or truncate low
        assert any(c.priority == Priority.CRITICAL for c in result)
        total_tokens = sum(c.tokens for c in result)
        assert total_tokens <= manager.max_context_tokens
    
    def test_build_context(self):
        token_manager = TokenBudgetManager(total_budget=2000)
        manager = ContextManager(token_manager)
        
        manager.add_system_prompt("You are helpful")
        manager.add_component("context", "Some context", Priority.HIGH)
        
        context = manager.build_context()
        
        assert "system_prompt" in context
        assert "You are helpful" in context
    
    def test_build_messages(self):
        token_manager = TokenBudgetManager(total_budget=2000)
        conv_manager = ConversationManager()
        conv_manager.add_message("user", "Hello")
        
        manager = ContextManager(token_manager, conv_manager)
        manager.add_system_prompt("You are helpful")
        manager.add_conversation_history()
        
        messages = manager.build_messages()
        
        assert len(messages) > 0
        assert any(m["role"] == "system" for m in messages)
    
    def test_get_context_summary(self):
        token_manager = TokenBudgetManager(total_budget=2000)
        manager = ContextManager(token_manager)
        
        manager.add_component("comp1", "Text", Priority.HIGH)
        manager.add_component("comp2", "More text", Priority.MEDIUM)
        
        summary = manager.get_context_summary()
        
        assert "total_components" in summary
        assert "total_tokens" in summary
        assert "by_priority" in summary
        assert summary["total_components"] == 2
    
    def test_clear(self):
        token_manager = TokenBudgetManager(total_budget=2000)
        manager = ContextManager(token_manager)
        
        manager.add_component("test", "content", Priority.HIGH)
        manager.clear()
        
        assert len(manager.components) == 0
    
    def test_remove_component(self):
        token_manager = TokenBudgetManager(total_budget=2000)
        manager = ContextManager(token_manager)
        
        manager.add_component("comp1", "Text", Priority.HIGH)
        manager.add_component("comp2", "More", Priority.MEDIUM)
        
        removed = manager.remove_component("comp1")
        
        assert removed
        assert len(manager.components) == 1
        assert manager.components[0].name == "comp2"
    
    def test_priority_ordering(self):
        token_manager = TokenBudgetManager(total_budget=2000)
        manager = ContextManager(token_manager, max_context_tokens=100)
        
        manager.add_component("low", "Low priority", Priority.LOW)
        manager.add_component("critical", "Critical", Priority.CRITICAL)
        manager.add_component("high", "High priority", Priority.HIGH)
        manager.add_component("medium", "Medium", Priority.MEDIUM)
        
        selected = manager.select_components()
        
        # Critical should always be first
        critical_components = [c for c in selected if c.priority == Priority.CRITICAL]
        assert len(critical_components) > 0
    
    def test_integration_with_managers(self):
        # Full integration test
        token_manager = TokenBudgetManager(total_budget=4000, response_reserve=1000)
        conv_manager = ConversationManager(max_messages=5)
        
        # Add conversation
        conv_manager.add_message("user", "What is AI?")
        conv_manager.add_message("assistant", "AI is artificial intelligence")
        
        # Create context manager
        manager = ContextManager(token_manager, conv_manager)
        
        # Build context
        manager.add_system_prompt("You are an AI assistant")
        manager.add_conversation_history()
        manager.add_examples("Example: User asks, AI responds")
        
        # Get summary
        summary = manager.get_context_summary()
        
        assert summary["total_components"] == 3
        assert not summary["overflow"]
        
        # Build final context
        context = manager.build_context()
        assert len(context) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
