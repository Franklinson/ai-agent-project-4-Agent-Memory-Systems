# Conversation History Management System

A lightweight conversation history manager with multiple truncation strategies for managing agent memory.

## Features

- **Message Storage**: Store messages with metadata and timestamps
- **Sliding Window Truncation**: Keep last N messages
- **Summary-Based Truncation**: Preserve system prompts and recent context
- **Selective Truncation**: Retain important messages
- **History Retrieval**: Get recent messages with filtering
- **Message Formatting**: Format messages for context injection

## Installation

```bash
pip install pytest  # For running tests
```

## Usage

### Basic Usage

```python
from conversation_manager import ConversationManager, TruncationStrategy

# Create manager with sliding window (default)
manager = ConversationManager(max_messages=10)

# Add messages
manager.add_message("user", "Hello, how are you?")
manager.add_message("assistant", "I'm doing well, thank you!")

# Get recent messages
recent = manager.get_recent_messages(5)

# Format for context
context = manager.format_for_context()
print(context)
```

### Truncation Strategies

#### 1. Sliding Window (Default)
Keeps the last N messages, removing oldest when full.

```python
manager = ConversationManager(
    max_messages=10,
    strategy=TruncationStrategy.SLIDING_WINDOW
)

for i in range(15):
    manager.add_message("user", f"Message {i}")

# Only last 10 messages retained
print(manager.get_message_count())  # Output: 10
```

#### 2. Summary-Based
Preserves system prompts and keeps recent messages.

```python
manager = ConversationManager(
    max_messages=10,
    strategy=TruncationStrategy.SUMMARY_BASED
)

manager.add_message("system", "You are a helpful assistant")
for i in range(15):
    manager.add_message("user", f"Message {i}")

# System prompt + last 9 messages
messages = manager.get_recent_messages()
print(messages[0].role)  # Output: system
```

#### 3. Selective
Retains messages marked as important plus recent context.

```python
manager = ConversationManager(
    max_messages=10,
    strategy=TruncationStrategy.SELECTIVE
)

# Mark important messages
manager.add_message("user", "Critical info", {"important": True})

for i in range(15):
    manager.add_message("user", f"Message {i}")

# Important message + recent messages retained
```

### Filtering and Retrieval

```python
# Get messages by role
user_messages = manager.get_messages_by_role("user")
assistant_messages = manager.get_messages_by_role("assistant")

# Get specific count of recent messages
last_5 = manager.get_recent_messages(5)

# Get full history as dictionaries
history = manager.get_history()
```

### Dynamic Strategy Changes

```python
manager = ConversationManager(max_messages=10)

# Start with sliding window
manager.add_message("user", "Hello")

# Switch to summary-based
manager.set_strategy(TruncationStrategy.SUMMARY_BASED)
```

### Message Metadata

```python
# Add metadata to messages
manager.add_message(
    "user",
    "Important question",
    metadata={
        "important": True,
        "category": "technical",
        "priority": "high"
    }
)

# Access metadata
for msg in manager.messages:
    if msg.metadata.get("priority") == "high":
        print(f"High priority: {msg.content}")
```

## API Reference

### ConversationManager

#### Constructor
```python
ConversationManager(max_messages=10, strategy=TruncationStrategy.SLIDING_WINDOW)
```

#### Methods

- `add_message(role, content, metadata=None)`: Add a message
- `get_recent_messages(count=None)`: Get recent messages
- `get_messages_by_role(role)`: Filter by role
- `format_for_context(count=None)`: Format as string
- `get_history()`: Get all messages as dicts
- `clear()`: Remove all messages
- `set_strategy(strategy)`: Change truncation strategy
- `get_message_count()`: Get total message count

### Message

#### Constructor
```python
Message(role, content, metadata=None)
```

#### Methods

- `to_dict()`: Convert to dictionary
- `format_for_context()`: Format as "role: content"

## Running Tests

```bash
cd day_47
python -m pytest test_conversation_manager.py -v
```

## Use Cases

1. **Chatbot Context Management**: Maintain conversation history within token limits
2. **Agent Memory**: Store recent interactions for context-aware responses
3. **Session Management**: Track user conversations with automatic cleanup
4. **Multi-turn Dialogues**: Preserve important context while managing memory

## Performance

- **Time Complexity**: O(1) for add, O(n) for truncation
- **Space Complexity**: O(max_messages)
- **Memory Efficient**: Automatic cleanup prevents unbounded growth

## Best Practices

1. Set `max_messages` based on your token budget
2. Use `SELECTIVE` strategy for long conversations with key information
3. Mark critical messages with `important: True` metadata
4. Use `SUMMARY_BASED` when system prompts must be preserved
5. Clear history between sessions with `clear()`


---

# Token Counting and Budgeting System

A comprehensive token counting and budget management system for LLM applications using tiktoken.

## Features

- **Token Counting**: Count tokens in text, messages, and context
- **Budget Tracking**: Track usage by component with alerts
- **Budget Allocation**: Allocate budgets to different components
- **Usage Monitoring**: Monitor token usage and limits
- **Context Optimization**: Automatically fit context within budget
- **Response Reserve**: Reserve tokens for model responses

## Installation

```bash
pip install tiktoken
```

## Usage

### Basic Token Counting

```python
from token_counter import TokenCounter

counter = TokenCounter("gpt-4")

# Count text
tokens = counter.count_text("Hello, world!")
print(f"Tokens: {tokens}")

# Count messages (OpenAI format)
messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
]
tokens = counter.count_messages(messages)

# Count context components
context = {
    "system": "You are helpful",
    "query": "What is AI?"
}
counts = counter.count_context(context)
```

### Budget Tracking

```python
from token_counter import BudgetTracker

tracker = BudgetTracker(total_budget=4000)

# Track usage
tracker.track_usage("system_prompt", 100)
tracker.track_usage("history", 500)

# Check usage
print(f"Total used: {tracker.get_total_usage()}")
print(f"Remaining: {tracker.get_remaining_budget()}")

# Get component breakdown
usage = tracker.get_usage_by_component()
```

### Budget Allocation

```python
# Allocate budgets to components
tracker.allocate_budget("system", 200)
tracker.allocate_budget("history", 1500)
tracker.allocate_budget("context", 1000)

# Track usage and get alerts
tracker.track_usage("history", 1600)  # Exceeds allocation

if tracker.has_alerts():
    for alert in tracker.get_alerts():
        print(f"Alert: {alert}")
```

### Complete Budget Management

```python
from token_counter import TokenBudgetManager

manager = TokenBudgetManager(
    total_budget=8000,
    model="gpt-4",
    response_reserve=2000
)

# Allocate component budgets
manager.allocate_component_budget("system", 200)
manager.allocate_component_budget("history", 2000)

# Count and track automatically
tokens = manager.count_and_track("system", "You are helpful")

# Check if content fits
if manager.can_fit_in_budget(500):
    print("Content fits in budget")

# Get comprehensive summary
summary = manager.get_budget_summary()
print(f"Used: {summary['total_used']}/{summary['total_budget']}")
```

### Context Optimization

```python
# Optimize context to fit within budget
context_items = [
    ("profile", "User profile information..."),
    ("history", "Conversation history..."),
    ("knowledge", "Retrieved knowledge...")
]

# Automatically select items that fit
optimized = manager.optimize_context(
    context_items,
    max_tokens=manager.get_available_for_context()
)

print(f"Optimized from {len(context_items)} to {len(optimized)} items")
```

## API Reference

### TokenCounter

**Constructor:**
```python
TokenCounter(model="gpt-4")
```

**Methods:**
- `count_text(text: str) -> int`: Count tokens in text
- `count_messages(messages: List[Dict]) -> int`: Count tokens in message list
- `count_context(context: Dict) -> Dict[str, int]`: Count tokens by component

### BudgetTracker

**Constructor:**
```python
BudgetTracker(total_budget: int)
```

**Methods:**
- `track_usage(component: str, tokens: int)`: Track token usage
- `get_total_usage() -> int`: Get total tokens used
- `get_component_usage(component: str) -> int`: Get usage for component
- `get_remaining_budget() -> int`: Get remaining tokens
- `allocate_budget(component: str, tokens: int) -> bool`: Allocate budget
- `get_usage_by_component() -> Dict[str, int]`: Get usage breakdown
- `has_alerts() -> bool`: Check for budget alerts
- `get_alerts() -> List[str]`: Get all alerts
- `reset()`: Reset all tracking

### TokenBudgetManager

**Constructor:**
```python
TokenBudgetManager(total_budget: int, model="gpt-4", response_reserve=1000)
```

**Methods:**
- `count_and_track(component: str, text: str) -> int`: Count and track
- `count_messages_and_track(component: str, messages: List) -> int`: Count messages and track
- `allocate_component_budget(component: str, tokens: int) -> bool`: Allocate budget
- `get_available_for_context() -> int`: Get tokens available for context
- `can_fit_in_budget(tokens: int, include_response=True) -> bool`: Check if fits
- `get_budget_summary() -> Dict`: Get comprehensive summary
- `optimize_context(items: List[tuple], max_tokens: int) -> List[tuple]`: Optimize context
- `reset_tracking()`: Reset usage tracking

## Running Tests

```bash
cd day_47
python -m pytest test_token_counter.py -v
```

## Running Examples

```bash
cd day_47
python examples_token_counter.py
```

## Use Cases

1. **LLM Context Management**: Ensure prompts fit within model context windows
2. **Cost Control**: Track and limit token usage for cost management
3. **Agent Systems**: Allocate budgets to different agent components
4. **Context Optimization**: Automatically fit context within token limits
5. **Usage Analytics**: Monitor token usage patterns by component

## Best Practices

1. **Set Response Reserve**: Always reserve tokens for model responses
2. **Allocate by Component**: Allocate budgets to system, history, context, etc.
3. **Monitor Alerts**: Check for budget alerts before making API calls
4. **Optimize Context**: Use context optimization for large contexts
5. **Track by Component**: Track usage by component for better insights
6. **Reset Between Sessions**: Reset tracking for new conversations

## Performance

- **Token Counting**: ~0.1ms per 1K characters
- **Budget Tracking**: O(1) for tracking, O(n) for component queries
- **Context Optimization**: O(n) where n is number of context items
- **Memory Efficient**: Minimal overhead for tracking

## Integration Example

```python
from conversation_manager import ConversationManager
from token_counter import TokenBudgetManager

# Initialize both systems
conv_manager = ConversationManager(max_messages=20)
token_manager = TokenBudgetManager(total_budget=8000, response_reserve=2000)

# Add messages and track tokens
conv_manager.add_message("user", "Hello")
messages = [m.to_dict() for m in conv_manager.get_recent_messages()]
tokens = token_manager.count_messages_and_track("history", messages)

# Check budget before API call
if token_manager.can_fit_in_budget(0):  # Just checking reserve
    # Make API call
    print("Budget OK, proceeding with API call")
else:
    print("Budget exceeded, truncating context")
```


---

# Context Window Management with Prioritization

A sophisticated context window manager that handles component prioritization, overflow detection, and graceful degradation.

## Features

- **Context Building**: Build context from multiple components
- **Priority Levels**: 4-level priority system (Critical, High, Medium, Low)
- **Overflow Handling**: Automatic detection and resolution
- **Dynamic Selection**: Select components based on token budget
- **Importance Scoring**: Score components based on priority and metadata
- **Graceful Degradation**: Intelligently truncate when necessary
- **Integration**: Works with conversation and token managers

## Priority Levels

- **CRITICAL**: Must include (system prompts, core instructions)
- **HIGH**: Very important (recent messages, key context)
- **MEDIUM**: Important (tools, examples)
- **LOW**: Nice to have (older history, extra context)

## Usage

### Basic Context Building

```python
from context_manager import ContextManager, Priority
from token_counter import TokenBudgetManager

token_manager = TokenBudgetManager(total_budget=4000, response_reserve=1000)
manager = ContextManager(token_manager)

# Add components
manager.add_system_prompt("You are a helpful AI assistant")
manager.add_component("user_query", "What is AI?", Priority.HIGH)
manager.add_examples("Example interactions...")

# Build context
context = manager.build_context()
```

### Integration with Conversation Manager

```python
from conversation_manager import ConversationManager

# Initialize managers
token_manager = TokenBudgetManager(total_budget=4000)
conv_manager = ConversationManager(max_messages=10)

# Add conversation
conv_manager.add_message("user", "Hello")
conv_manager.add_message("assistant", "Hi there!")

# Create context manager
manager = ContextManager(token_manager, conv_manager)
manager.add_system_prompt("You are helpful")
manager.add_conversation_history()

# Build as messages
messages = manager.build_messages()
```

### Handling Overflow

```python
# Small context window
manager = ContextManager(token_manager, max_context_tokens=200)

# Add more than fits
manager.add_system_prompt("System instructions")
manager.add_component("context1", "Long text...", Priority.HIGH)
manager.add_component("context2", "More text...", Priority.LOW)

# Detect overflow
if manager.detect_overflow():
    print("Context exceeds budget")

# Handle automatically
selected = manager.handle_overflow()
# Returns prioritized components that fit
```

### Priority-Based Selection

```python
# Add components with different priorities
manager.add_component("critical", "Must include", Priority.CRITICAL)
manager.add_component("important", "Should include", Priority.HIGH)
manager.add_component("optional", "Nice to have", Priority.LOW)

# Select based on budget
selected = manager.select_components(max_tokens=500)
# Returns highest priority components that fit
```

### Importance Scoring with Metadata

```python
# Add metadata to influence selection
manager.add_component(
    "required_context",
    "Important information",
    Priority.HIGH,
    metadata={
        "required": True,      # +20 to score
        "recent": True,        # +10 to score
        "user_specified": True # +15 to score
    }
)

# Calculate score
score = manager.calculate_importance_score(component)
```

### Tool Definitions

```python
tools = [
    {"name": "search", "description": "Search the web"},
    {"name": "calculate", "description": "Perform calculations"}
]

manager.add_tool_definitions(tools)
```

### Context Summary

```python
summary = manager.get_context_summary()

print(f"Total components: {summary['total_components']}")
print(f"Total tokens: {summary['total_tokens']}")
print(f"Overflow: {summary['overflow']}")
print(f"By priority: {summary['by_priority']}")
```

## API Reference

### ContextManager

**Constructor:**
```python
ContextManager(
    token_manager: TokenBudgetManager,
    conversation_manager: Optional[ConversationManager] = None,
    max_context_tokens: Optional[int] = None
)
```

**Methods:**

- `add_component(name, content, priority, metadata=None) -> ContextComponent`: Add component
- `add_system_prompt(prompt) -> ContextComponent`: Add system prompt (critical)
- `add_conversation_history(max_messages=None) -> ContextComponent`: Add conversation
- `add_tool_definitions(tools) -> ContextComponent`: Add tool definitions
- `add_examples(examples) -> ContextComponent`: Add examples
- `add_retrieved_context(context, priority=HIGH) -> ContextComponent`: Add retrieved context
- `calculate_importance_score(component) -> float`: Calculate importance score
- `select_components(max_tokens=None) -> List[ContextComponent]`: Select components
- `detect_overflow(components=None) -> bool`: Detect overflow
- `handle_overflow() -> List[ContextComponent]`: Handle overflow
- `build_context(handle_overflow=True) -> str`: Build context string
- `build_messages(handle_overflow=True) -> List[Dict]`: Build message list
- `get_context_summary() -> Dict`: Get context summary
- `clear()`: Clear all components
- `remove_component(name) -> bool`: Remove component by name

### ContextComponent

**Attributes:**
- `name: str`: Component name
- `content: str`: Component content
- `priority: Priority`: Priority level
- `tokens: int`: Token count
- `metadata: Dict`: Additional metadata

### Priority Enum

- `Priority.CRITICAL`: Must include (value: 1)
- `Priority.HIGH`: Very important (value: 2)
- `Priority.MEDIUM`: Important (value: 3)
- `Priority.LOW`: Nice to have (value: 4)

## Running Tests

```bash
cd day_47
python -m pytest test_context_manager.py -v
```

## Running Examples

```bash
cd day_47
python examples_context_manager.py
```

## Use Cases

1. **Agent Context Management**: Build context for AI agents with multiple components
2. **Token Budget Optimization**: Fit context within model limits
3. **Priority-Based Selection**: Ensure critical information is always included
4. **Overflow Handling**: Gracefully handle context that exceeds limits
5. **Multi-Component Systems**: Coordinate system prompts, history, tools, and context

## Best Practices

1. **Use Priority Levels**: Assign appropriate priorities to components
2. **Mark Critical Components**: System prompts should be CRITICAL
3. **Add Metadata**: Use metadata to influence importance scoring
4. **Monitor Overflow**: Check for overflow before API calls
5. **Handle Gracefully**: Use handle_overflow() for automatic optimization
6. **Integrate Managers**: Use with conversation and token managers
7. **Clear Between Sessions**: Clear components for new conversations

## Performance

- **Selection**: O(n log n) where n is number of components
- **Overflow Detection**: O(n) for token counting
- **Context Building**: O(n) for concatenation
- **Memory Efficient**: Only stores selected components

## Integration Example

```python
from context_manager import ContextManager, Priority
from conversation_manager import ConversationManager
from token_counter import TokenBudgetManager

# Initialize all managers
token_manager = TokenBudgetManager(total_budget=8000, response_reserve=2000)
conv_manager = ConversationManager(max_messages=20)
context_manager = ContextManager(token_manager, conv_manager)

# Add conversation
conv_manager.add_message("user", "Help me with Python")
conv_manager.add_message("assistant", "I'd be happy to help!")

# Build context
context_manager.add_system_prompt("You are a Python expert")
context_manager.add_conversation_history()
context_manager.add_retrieved_context("Python is a programming language...")

# Check and build
summary = context_manager.get_context_summary()
if not summary['overflow']:
    messages = context_manager.build_messages()
    # Make API call with messages
else:
    # Overflow handled automatically
    messages = context_manager.build_messages(handle_overflow=True)
```

## Advanced Features

### Custom Importance Scoring

Components are scored based on:
- Base priority (25-100 points)
- `required` metadata (+20 points)
- `recent` metadata (+10 points)
- `user_specified` metadata (+15 points)
- Size penalty for large components (-10 points)

### Graceful Degradation

When overflow occurs:
1. Select components by priority
2. If still overflowing, truncate lower priority components
3. Always preserve CRITICAL components
4. Mark truncated components in metadata

### Dynamic Selection

Adjust selection based on available tokens:
```python
# Try different budgets
for budget in [1000, 500, 200]:
    selected = manager.select_components(max_tokens=budget)
    print(f"Budget {budget}: {len(selected)} components")
```
