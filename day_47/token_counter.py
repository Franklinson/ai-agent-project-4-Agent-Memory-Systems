"""Token counting and budgeting tools for LLM applications."""

import tiktoken
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TokenUsage:
    """Tracks token usage for a component."""
    component: str
    tokens_used: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "tokens_used": self.tokens_used,
            "timestamp": self.timestamp.isoformat()
        }


class TokenCounter:
    """Counts tokens using tiktoken."""
    
    def __init__(self, model: str = "gpt-4"):
        self.encoding = tiktoken.encoding_for_model(model)
        self.model = model
    
    def count_text(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    def count_messages(self, messages: List[Dict[str, str]]) -> int:
        """Count tokens in message list (OpenAI format)."""
        tokens = 0
        for message in messages:
            tokens += 4  # Message formatting overhead
            for key, value in message.items():
                tokens += self.count_text(str(value))
        tokens += 2  # Conversation formatting
        return tokens
    
    def count_context(self, context: Dict[str, Any]) -> Dict[str, int]:
        """Count tokens in different context components."""
        counts = {}
        for key, value in context.items():
            if isinstance(value, str):
                counts[key] = self.count_text(value)
            elif isinstance(value, list):
                counts[key] = self.count_messages(value) if value and isinstance(value[0], dict) else sum(self.count_text(str(v)) for v in value)
            else:
                counts[key] = self.count_text(str(value))
        return counts


class BudgetTracker:
    """Tracks token usage against budget."""
    
    def __init__(self, total_budget: int):
        self.total_budget = total_budget
        self.usage_history: List[TokenUsage] = []
        self.component_budgets: Dict[str, int] = {}
        self.alerts: List[str] = []
    
    def track_usage(self, component: str, tokens: int) -> None:
        """Track token usage for a component."""
        usage = TokenUsage(component=component, tokens_used=tokens)
        self.usage_history.append(usage)
        
        # Check budget limits
        if self.get_total_usage() > self.total_budget:
            self.alerts.append(f"Total budget exceeded: {self.get_total_usage()}/{self.total_budget}")
        
        if component in self.component_budgets:
            component_usage = self.get_component_usage(component)
            if component_usage > self.component_budgets[component]:
                self.alerts.append(f"{component} budget exceeded: {component_usage}/{self.component_budgets[component]}")
    
    def get_total_usage(self) -> int:
        """Get total tokens used."""
        return sum(u.tokens_used for u in self.usage_history)
    
    def get_component_usage(self, component: str) -> int:
        """Get tokens used by specific component."""
        return sum(u.tokens_used for u in self.usage_history if u.component == component)
    
    def get_remaining_budget(self) -> int:
        """Get remaining tokens in budget."""
        return max(0, self.total_budget - self.get_total_usage())
    
    def allocate_budget(self, component: str, tokens: int) -> bool:
        """Allocate budget to a component."""
        total_allocated = sum(self.component_budgets.values())
        if total_allocated + tokens > self.total_budget:
            return False
        self.component_budgets[component] = tokens
        return True
    
    def get_usage_by_component(self) -> Dict[str, int]:
        """Get usage breakdown by component."""
        components = {}
        for usage in self.usage_history:
            components[usage.component] = components.get(usage.component, 0) + usage.tokens_used
        return components
    
    def has_alerts(self) -> bool:
        """Check if there are budget alerts."""
        return len(self.alerts) > 0
    
    def get_alerts(self) -> List[str]:
        """Get all budget alerts."""
        return self.alerts.copy()
    
    def clear_alerts(self) -> None:
        """Clear all alerts."""
        self.alerts.clear()
    
    def reset(self) -> None:
        """Reset all usage tracking."""
        self.usage_history.clear()
        self.alerts.clear()


class TokenBudgetManager:
    """Manages token budgets with allocation and monitoring."""
    
    def __init__(self, total_budget: int, model: str = "gpt-4", response_reserve: int = 1000):
        self.counter = TokenCounter(model)
        self.tracker = BudgetTracker(total_budget)
        self.response_reserve = response_reserve
        self.allocations: Dict[str, int] = {}
    
    def count_and_track(self, component: str, text: str) -> int:
        """Count tokens and track usage."""
        tokens = self.counter.count_text(text)
        self.tracker.track_usage(component, tokens)
        return tokens
    
    def count_messages_and_track(self, component: str, messages: List[Dict[str, str]]) -> int:
        """Count message tokens and track usage."""
        tokens = self.counter.count_messages(messages)
        self.tracker.track_usage(component, tokens)
        return tokens
    
    def allocate_component_budget(self, component: str, tokens: int) -> bool:
        """Allocate budget to a component."""
        if self.tracker.allocate_budget(component, tokens):
            self.allocations[component] = tokens
            return True
        return False
    
    def get_available_for_context(self) -> int:
        """Get tokens available for context (excluding response reserve)."""
        return max(0, self.tracker.get_remaining_budget() - self.response_reserve)
    
    def can_fit_in_budget(self, tokens: int, include_response: bool = True) -> bool:
        """Check if tokens fit in remaining budget."""
        required = tokens + (self.response_reserve if include_response else 0)
        return required <= self.tracker.get_remaining_budget()
    
    def get_budget_summary(self) -> Dict[str, Any]:
        """Get comprehensive budget summary."""
        return {
            "total_budget": self.tracker.total_budget,
            "total_used": self.tracker.get_total_usage(),
            "remaining": self.tracker.get_remaining_budget(),
            "response_reserve": self.response_reserve,
            "available_for_context": self.get_available_for_context(),
            "usage_by_component": self.tracker.get_usage_by_component(),
            "component_budgets": self.tracker.component_budgets.copy(),
            "alerts": self.tracker.get_alerts()
        }
    
    def optimize_context(self, context_items: List[tuple[str, str]], max_tokens: Optional[int] = None) -> List[tuple[str, str]]:
        """Optimize context to fit within token budget."""
        if max_tokens is None:
            max_tokens = self.get_available_for_context()
        
        result = []
        total_tokens = 0
        
        for name, content in context_items:
            tokens = self.counter.count_text(content)
            if total_tokens + tokens <= max_tokens:
                result.append((name, content))
                total_tokens += tokens
            else:
                break
        
        return result
    
    def reset_tracking(self) -> None:
        """Reset usage tracking."""
        self.tracker.reset()
