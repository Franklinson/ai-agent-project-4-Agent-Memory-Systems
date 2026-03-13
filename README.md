# AI Agent Memory Systems

A comprehensive collection of memory management systems and tools for AI agents, including conversation history management, token counting, budgeting, and context window management with prioritization.

## Project Overview

This project implements various memory architectures and management systems for AI agents, focusing on practical implementations of conversation management, token budgeting, context prioritization, and memory optimization strategies.

## Structure

### Day 46: Memory Architecture Documentation
- **memory_architecture.md**: Comprehensive memory architecture for agent systems
  - Storage layers (working, short-term, long-term memory)
  - Retrieval mechanisms (exact match, similarity search, query-based, hybrid)
  - Update strategies and management patterns
- **memory_types_comparison.md**: Comparison of different memory types
- **use_cases_analysis.md**: Real-world use cases and applications

### Day 47: Conversation, Token & Context Management
- **conversation_manager.py**: Conversation history management with truncation strategies
  - Message storage with metadata and timestamps
  - Sliding window truncation
  - Summary-based truncation
  - Selective truncation (importance-based)
  - History retrieval and formatting
  
- **token_counter.py**: Token counting and budget management
  - Token counting using tiktoken
  - Budget tracking by component
  - Usage monitoring and alerts
  - Budget allocation
  - Context optimization
  - Response reserve management

- **context_manager.py**: Context window management with prioritization
  - Context building from multiple components
  - 4-level priority system (Critical, High, Medium, Low)
  - Overflow detection and handling
  - Dynamic component selection
  - Importance scoring
  - Graceful degradation
  - Integration with conversation and token managers

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Conversation Management

```python
from day_47.conversation_manager import ConversationManager, TruncationStrategy

# Create manager with sliding window
manager = ConversationManager(max_messages=10)

# Add messages
manager.add_message("user", "Hello!")
manager.add_message("assistant", "Hi there!")

# Get formatted context
context = manager.format_for_context()
```

### Token Counting & Budgeting

```python
from day_47.token_counter import TokenBudgetManager

# Initialize with budget
manager = TokenBudgetManager(
    total_budget=8000,
    response_reserve=2000
)

# Count and track tokens
tokens = manager.count_and_track("system", "You are helpful")

# Check budget
if manager.can_fit_in_budget(500):
    print("Content fits in budget")

# Get summary
summary = manager.get_budget_summary()
```

### Context Window Management

```python
from day_47.context_manager import ContextManager, Priority

# Initialize
context_manager = ContextManager(token_manager, conv_manager)

# Add components with priorities
context_manager.add_system_prompt("You are helpful")
context_manager.add_conversation_history()
context_manager.add_component("context", "Important info", Priority.HIGH)

# Build context with automatic overflow handling
context = context_manager.build_context()
messages = context_manager.build_messages()
```

## Features

### Conversation Management
- ✅ Message storage with metadata
- ✅ Multiple truncation strategies
- ✅ Role-based filtering
- ✅ Context formatting
- ✅ Timestamp tracking

### Token Management
- ✅ Accurate token counting (tiktoken)
- ✅ Component-based budget tracking
- ✅ Usage alerts and monitoring
- ✅ Budget allocation
- ✅ Context optimization
- ✅ Response reserve

### Context Window Management
- ✅ Priority-based component selection
- ✅ Overflow detection and handling
- ✅ Importance scoring
- ✅ Graceful degradation
- ✅ Dynamic selection
- ✅ Integration with conversation and token managers

## Running Tests

```bash
# Activate virtual environment
source venv/bin/activate

# Run all tests
cd day_47
python -m pytest test_conversation_manager.py -v
python -m pytest test_token_counter.py -v
python -m pytest test_context_manager.py -v

# Run specific test file
python -m pytest test_context_manager.py::TestContextManager -v
```

## Running Examples

```bash
source venv/bin/activate
cd day_47
python examples_token_counter.py
python examples_context_manager.py
```

## Use Cases

1. **Chatbot Context Management**: Maintain conversation history within token limits
2. **Agent Memory Systems**: Store and retrieve relevant context for AI agents
3. **Cost Control**: Track and limit token usage for API cost management
4. **Multi-turn Dialogues**: Preserve important context while managing memory
5. **LLM Application Development**: Build token-aware applications
6. **Priority-Based Context**: Ensure critical information is always included
7. **Overflow Handling**: Gracefully handle context that exceeds model limits
8. **Multi-Component Systems**: Coordinate system prompts, history, tools, and context

## Best Practices

1. **Set appropriate budgets**: Allocate tokens based on model context windows
2. **Use truncation strategies**: Choose the right strategy for your use case
3. **Monitor usage**: Track token usage by component for optimization
4. **Reserve for responses**: Always reserve tokens for model responses
5. **Optimize context**: Use context optimization for large contexts
6. **Reset between sessions**: Clear history and tracking for new conversations
7. **Use priority levels**: Assign appropriate priorities to context components
8. **Handle overflow gracefully**: Use automatic overflow handling
9. **Integrate managers**: Use conversation, token, and context managers together

## Dependencies

- Python 3.8+
- tiktoken: Token counting for OpenAI models
- pytest: Testing framework

## Project Structure

```
ai-agent-project-4 Agent Memory Systems/
├── day_46/                          # Memory architecture documentation
│   ├── memory_architecture.md
│   ├── memory_types_comparison.md
│   └── use_cases_analysis.md
├── day_47/                          # Implementations
│   ├── conversation_manager.py      # Conversation history management
│   ├── token_counter.py             # Token counting & budgeting
│   ├── context_manager.py           # Context window management
│   ├── test_conversation_manager.py # Tests
│   ├── test_token_counter.py        # Tests
│   ├── test_context_manager.py      # Tests
│   ├── examples_token_counter.py    # Usage examples
│   ├── examples_context_manager.py  # Usage examples
│   └── README.md                    # Detailed documentation
├── venv/                            # Virtual environment
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Contributing

When adding new features:
1. Check for venv before running environment commands
2. Update this README with new functionality
3. Add tests for new implementations
4. Document usage with examples

## License

This project is for educational and research purposes.

## Future Enhancements

- [ ] Vector-based memory retrieval
- [ ] Persistent storage backends
- [ ] Memory consolidation strategies
- [ ] Multi-agent memory sharing
- [ ] Advanced context compression
- [ ] Memory importance scoring
