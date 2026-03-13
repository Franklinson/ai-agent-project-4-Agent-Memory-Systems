"""Tests for token_counter.py"""

import pytest
from token_counter import TokenCounter, BudgetTracker, TokenBudgetManager, TokenUsage


class TestTokenCounter:
    """Test TokenCounter class."""
    
    def test_initialization(self):
        counter = TokenCounter("gpt-4")
        assert counter.model == "gpt-4"
        assert counter.encoding is not None
    
    def test_count_text(self):
        counter = TokenCounter("gpt-4")
        text = "Hello, world!"
        tokens = counter.count_text(text)
        assert tokens > 0
        assert isinstance(tokens, int)
    
    def test_count_empty_text(self):
        counter = TokenCounter("gpt-4")
        assert counter.count_text("") == 0
    
    def test_count_messages(self):
        counter = TokenCounter("gpt-4")
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        tokens = counter.count_messages(messages)
        assert tokens > 0
        # Should include message overhead
        assert tokens > counter.count_text("Hello") + counter.count_text("Hi there!")
    
    def test_count_context(self):
        counter = TokenCounter("gpt-4")
        context = {
            "system": "You are a helpful assistant",
            "history": "Previous conversation",
            "query": "What is AI?"
        }
        counts = counter.count_context(context)
        assert len(counts) == 3
        assert all(v > 0 for v in counts.values())


class TestBudgetTracker:
    """Test BudgetTracker class."""
    
    def test_initialization(self):
        tracker = BudgetTracker(1000)
        assert tracker.total_budget == 1000
        assert tracker.get_total_usage() == 0
    
    def test_track_usage(self):
        tracker = BudgetTracker(1000)
        tracker.track_usage("component1", 100)
        assert tracker.get_total_usage() == 100
        assert tracker.get_component_usage("component1") == 100
    
    def test_multiple_components(self):
        tracker = BudgetTracker(1000)
        tracker.track_usage("component1", 100)
        tracker.track_usage("component2", 200)
        tracker.track_usage("component1", 50)
        
        assert tracker.get_total_usage() == 350
        assert tracker.get_component_usage("component1") == 150
        assert tracker.get_component_usage("component2") == 200
    
    def test_remaining_budget(self):
        tracker = BudgetTracker(1000)
        tracker.track_usage("component1", 300)
        assert tracker.get_remaining_budget() == 700
    
    def test_budget_exceeded_alert(self):
        tracker = BudgetTracker(100)
        tracker.track_usage("component1", 150)
        assert tracker.has_alerts()
        alerts = tracker.get_alerts()
        assert len(alerts) > 0
        assert "exceeded" in alerts[0].lower()
    
    def test_allocate_budget(self):
        tracker = BudgetTracker(1000)
        success = tracker.allocate_budget("component1", 300)
        assert success
        assert tracker.component_budgets["component1"] == 300
    
    def test_allocate_budget_overflow(self):
        tracker = BudgetTracker(1000)
        tracker.allocate_budget("component1", 600)
        success = tracker.allocate_budget("component2", 500)
        assert not success
    
    def test_component_budget_alert(self):
        tracker = BudgetTracker(1000)
        tracker.allocate_budget("component1", 100)
        tracker.track_usage("component1", 150)
        
        assert tracker.has_alerts()
        alerts = tracker.get_alerts()
        assert any("component1" in alert for alert in alerts)
    
    def test_get_usage_by_component(self):
        tracker = BudgetTracker(1000)
        tracker.track_usage("comp1", 100)
        tracker.track_usage("comp2", 200)
        tracker.track_usage("comp1", 50)
        
        usage = tracker.get_usage_by_component()
        assert usage["comp1"] == 150
        assert usage["comp2"] == 200
    
    def test_clear_alerts(self):
        tracker = BudgetTracker(100)
        tracker.track_usage("component1", 150)
        assert tracker.has_alerts()
        tracker.clear_alerts()
        assert not tracker.has_alerts()
    
    def test_reset(self):
        tracker = BudgetTracker(1000)
        tracker.track_usage("component1", 100)
        tracker.reset()
        assert tracker.get_total_usage() == 0
        assert len(tracker.usage_history) == 0


class TestTokenBudgetManager:
    """Test TokenBudgetManager class."""
    
    def test_initialization(self):
        manager = TokenBudgetManager(total_budget=1000, response_reserve=200)
        assert manager.tracker.total_budget == 1000
        assert manager.response_reserve == 200
    
    def test_count_and_track(self):
        manager = TokenBudgetManager(1000)
        tokens = manager.count_and_track("test", "Hello, world!")
        assert tokens > 0
        assert manager.tracker.get_component_usage("test") == tokens
    
    def test_count_messages_and_track(self):
        manager = TokenBudgetManager(1000)
        messages = [{"role": "user", "content": "Hello"}]
        tokens = manager.count_messages_and_track("messages", messages)
        assert tokens > 0
        assert manager.tracker.get_component_usage("messages") == tokens
    
    def test_allocate_component_budget(self):
        manager = TokenBudgetManager(1000)
        success = manager.allocate_component_budget("context", 300)
        assert success
        assert "context" in manager.allocations
    
    def test_get_available_for_context(self):
        manager = TokenBudgetManager(1000, response_reserve=200)
        manager.count_and_track("used", "Some text")
        available = manager.get_available_for_context()
        # Should be total - used - reserve
        assert available < 1000 - 200
    
    def test_can_fit_in_budget(self):
        manager = TokenBudgetManager(1000, response_reserve=200)
        manager.count_and_track("used", "Some text")
        
        # Should fit without response
        assert manager.can_fit_in_budget(100, include_response=False)
        
        # Large amount shouldn't fit
        assert not manager.can_fit_in_budget(2000, include_response=True)
    
    def test_get_budget_summary(self):
        manager = TokenBudgetManager(1000, response_reserve=200)
        manager.count_and_track("component1", "Test text")
        
        summary = manager.get_budget_summary()
        assert "total_budget" in summary
        assert "total_used" in summary
        assert "remaining" in summary
        assert "response_reserve" in summary
        assert "usage_by_component" in summary
        assert summary["total_budget"] == 1000
        assert summary["response_reserve"] == 200
    
    def test_optimize_context(self):
        manager = TokenBudgetManager(1000)
        context_items = [
            ("item1", "Short text"),
            ("item2", "Another short text"),
            ("item3", "Yet another text")
        ]
        
        # Optimize with very small budget
        optimized = manager.optimize_context(context_items, max_tokens=5)
        assert len(optimized) < len(context_items)
    
    def test_optimize_context_all_fit(self):
        manager = TokenBudgetManager(10000)
        context_items = [
            ("item1", "Short"),
            ("item2", "Text")
        ]
        
        optimized = manager.optimize_context(context_items, max_tokens=1000)
        assert len(optimized) == len(context_items)
    
    def test_reset_tracking(self):
        manager = TokenBudgetManager(1000)
        manager.count_and_track("test", "Some text")
        manager.reset_tracking()
        assert manager.tracker.get_total_usage() == 0


class TestTokenUsage:
    """Test TokenUsage dataclass."""
    
    def test_creation(self):
        usage = TokenUsage(component="test", tokens_used=100)
        assert usage.component == "test"
        assert usage.tokens_used == 100
        assert usage.timestamp is not None
    
    def test_to_dict(self):
        usage = TokenUsage(component="test", tokens_used=100)
        data = usage.to_dict()
        assert data["component"] == "test"
        assert data["tokens_used"] == 100
        assert "timestamp" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
