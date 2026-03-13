"""Context window management with prioritization and overflow handling."""

from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
from conversation_manager import ConversationManager, TruncationStrategy
from token_counter import TokenBudgetManager


class Priority(Enum):
    """Priority levels for context components."""
    CRITICAL = 1    # Must include (system prompts)
    HIGH = 2        # Very important (recent messages, key context)
    MEDIUM = 3      # Important (tools, examples)
    LOW = 4         # Nice to have (older history, extra context)


@dataclass
class ContextComponent:
    """Represents a component of the context."""
    name: str
    content: str
    priority: Priority
    tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "content": self.content,
            "priority": self.priority.name,
            "tokens": self.tokens,
            "metadata": self.metadata
        }


class ContextManager:
    """Manages context window with prioritization and overflow handling."""
    
    def __init__(
        self,
        token_manager: TokenBudgetManager,
        conversation_manager: Optional[ConversationManager] = None,
        max_context_tokens: Optional[int] = None
    ):
        self.token_manager = token_manager
        self.conversation_manager = conversation_manager or ConversationManager()
        self.max_context_tokens = max_context_tokens or token_manager.get_available_for_context()
        self.components: List[ContextComponent] = []
    
    def add_component(
        self,
        name: str,
        content: str,
        priority: Priority,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ContextComponent:
        """Add a context component."""
        tokens = self.token_manager.counter.count_text(content)
        component = ContextComponent(
            name=name,
            content=content,
            priority=priority,
            tokens=tokens,
            metadata=metadata or {}
        )
        self.components.append(component)
        return component
    
    def add_system_prompt(self, prompt: str) -> ContextComponent:
        """Add system prompt (critical priority)."""
        return self.add_component("system_prompt", prompt, Priority.CRITICAL)
    
    def add_conversation_history(self, max_messages: Optional[int] = None) -> ContextComponent:
        """Add conversation history from conversation manager."""
        history = self.conversation_manager.format_for_context(max_messages)
        return self.add_component("conversation_history", history, Priority.HIGH)
    
    def add_tool_definitions(self, tools: List[Dict[str, Any]]) -> ContextComponent:
        """Add tool definitions."""
        content = self._format_tools(tools)
        return self.add_component("tool_definitions", content, Priority.MEDIUM)
    
    def add_examples(self, examples: str) -> ContextComponent:
        """Add examples."""
        return self.add_component("examples", examples, Priority.MEDIUM)
    
    def add_retrieved_context(self, context: str, priority: Priority = Priority.HIGH) -> ContextComponent:
        """Add retrieved context (e.g., from RAG)."""
        return self.add_component("retrieved_context", context, priority)
    
    def _format_tools(self, tools: List[Dict[str, Any]]) -> str:
        """Format tool definitions."""
        formatted = []
        for tool in tools:
            formatted.append(f"Tool: {tool.get('name', 'unknown')}")
            if 'description' in tool:
                formatted.append(f"  Description: {tool['description']}")
        return "\n".join(formatted)
    
    def calculate_importance_score(self, component: ContextComponent) -> float:
        """Calculate importance score for a component."""
        # Base score from priority
        priority_scores = {
            Priority.CRITICAL: 100.0,
            Priority.HIGH: 75.0,
            Priority.MEDIUM: 50.0,
            Priority.LOW: 25.0
        }
        score = priority_scores[component.priority]
        
        # Adjust based on metadata
        if component.metadata.get("required", False):
            score += 20.0
        if component.metadata.get("recent", False):
            score += 10.0
        if component.metadata.get("user_specified", False):
            score += 15.0
        
        # Penalize very large components
        if component.tokens > 1000:
            score -= 10.0
        
        return score
    
    def select_components(self, max_tokens: Optional[int] = None) -> List[ContextComponent]:
        """Select components based on priority and token budget."""
        if max_tokens is None:
            max_tokens = self.max_context_tokens
        
        # Sort by importance score (descending)
        sorted_components = sorted(
            self.components,
            key=lambda c: (self.calculate_importance_score(c), -c.tokens),
            reverse=True
        )
        
        selected = []
        total_tokens = 0
        
        # First pass: include all CRITICAL components
        for component in sorted_components:
            if component.priority == Priority.CRITICAL:
                selected.append(component)
                total_tokens += component.tokens
        
        # Second pass: add other components by priority
        for component in sorted_components:
            if component.priority != Priority.CRITICAL:
                if total_tokens + component.tokens <= max_tokens:
                    selected.append(component)
                    total_tokens += component.tokens
        
        return selected
    
    def detect_overflow(self, components: Optional[List[ContextComponent]] = None) -> bool:
        """Detect if context exceeds token budget."""
        if components is None:
            components = self.components
        total_tokens = sum(c.tokens for c in components)
        return total_tokens > self.max_context_tokens
    
    def handle_overflow(self) -> List[ContextComponent]:
        """Handle context overflow with graceful degradation."""
        if not self.detect_overflow():
            return self.components
        
        # Try selecting components by priority
        selected = self.select_components()
        
        if not self.detect_overflow(selected):
            return selected
        
        # If still overflowing, truncate lower priority components
        return self._truncate_components(selected)
    
    def _truncate_components(self, components: List[ContextComponent]) -> List[ContextComponent]:
        """Truncate components to fit within budget."""
        result = []
        total_tokens = 0
        
        # Sort by priority
        sorted_components = sorted(components, key=lambda c: c.priority.value)
        
        for component in sorted_components:
            if total_tokens + component.tokens <= self.max_context_tokens:
                result.append(component)
                total_tokens += component.tokens
            elif component.priority == Priority.CRITICAL:
                # Must include critical components, even if truncated
                available = self.max_context_tokens - total_tokens
                if available > 0:
                    truncated = self._truncate_content(component, available)
                    result.append(truncated)
                    total_tokens += truncated.tokens
        
        return result
    
    def _truncate_content(self, component: ContextComponent, max_tokens: int) -> ContextComponent:
        """Truncate component content to fit token budget."""
        if component.tokens <= max_tokens:
            return component
        
        # Simple truncation: take first portion
        words = component.content.split()
        truncated_words = []
        current_tokens = 0
        
        for word in words:
            word_tokens = self.token_manager.counter.count_text(word + " ")
            if current_tokens + word_tokens <= max_tokens:
                truncated_words.append(word)
                current_tokens += word_tokens
            else:
                break
        
        truncated_content = " ".join(truncated_words) + "..."
        return ContextComponent(
            name=component.name,
            content=truncated_content,
            priority=component.priority,
            tokens=self.token_manager.counter.count_text(truncated_content),
            metadata={**component.metadata, "truncated": True}
        )
    
    def build_context(self, handle_overflow: bool = True) -> str:
        """Build final context string."""
        components = self.handle_overflow() if handle_overflow else self.components
        
        # Sort by priority for output
        sorted_components = sorted(components, key=lambda c: c.priority.value)
        
        context_parts = []
        for component in sorted_components:
            if component.content.strip():
                context_parts.append(f"# {component.name}")
                context_parts.append(component.content)
                context_parts.append("")  # Empty line
        
        return "\n".join(context_parts)
    
    def build_messages(self, handle_overflow: bool = True) -> List[Dict[str, str]]:
        """Build context as message list (OpenAI format)."""
        components = self.handle_overflow() if handle_overflow else self.components
        
        messages = []
        
        # Add system message
        system_components = [c for c in components if c.priority == Priority.CRITICAL]
        if system_components:
            system_content = "\n\n".join(c.content for c in system_components)
            messages.append({"role": "system", "content": system_content})
        
        # Add conversation history
        history_components = [c for c in components if c.name == "conversation_history"]
        if history_components and self.conversation_manager.get_message_count() > 0:
            for msg in self.conversation_manager.get_recent_messages():
                messages.append({"role": msg.role, "content": msg.content})
        
        # Add other context as system messages
        other_components = [
            c for c in components 
            if c.priority != Priority.CRITICAL and c.name != "conversation_history"
        ]
        if other_components:
            other_content = "\n\n".join(f"{c.name}:\n{c.content}" for c in other_components)
            messages.append({"role": "system", "content": other_content})
        
        return messages
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of context state."""
        total_tokens = sum(c.tokens for c in self.components)
        selected = self.select_components()
        selected_tokens = sum(c.tokens for c in selected)
        
        by_priority = {}
        for priority in Priority:
            components = [c for c in self.components if c.priority == priority]
            by_priority[priority.name] = {
                "count": len(components),
                "tokens": sum(c.tokens for c in components)
            }
        
        return {
            "total_components": len(self.components),
            "total_tokens": total_tokens,
            "max_tokens": self.max_context_tokens,
            "selected_components": len(selected),
            "selected_tokens": selected_tokens,
            "overflow": self.detect_overflow(),
            "by_priority": by_priority
        }
    
    def clear(self) -> None:
        """Clear all components."""
        self.components.clear()
    
    def remove_component(self, name: str) -> bool:
        """Remove component by name."""
        original_length = len(self.components)
        self.components = [c for c in self.components if c.name != name]
        return len(self.components) < original_length
