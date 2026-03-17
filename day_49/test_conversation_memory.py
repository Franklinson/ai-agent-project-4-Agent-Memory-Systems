"""Tests for conversation memory management."""

import pytest
from datetime import datetime, timedelta
from conversation_memory import (
    ConversationMemory, Conversation, Message,
    ConversationMemoryError, ConversationNotFoundError,
)


@pytest.fixture
def mem():
    return ConversationMemory()


@pytest.fixture
def populated(mem):
    """Memory with 2 conversations, messages, and topics."""
    c1 = mem.create_conversation("alice", conversation_id="c1")
    m1 = mem.add_message("c1", "user", "Hello", topics=["greeting"])
    m2 = mem.add_message("c1", "assistant", "Hi there!")
    mem.add_message("c1", "user", "Tell me about Python", topics=["python"])
    mem.add_message("c1", "assistant", "Python is a programming language.")

    c2 = mem.create_conversation("bob", conversation_id="c2")
    mem.add_message("c2", "user", "What is AWS?", topics=["aws"])
    mem.add_message("c2", "assistant", "AWS is a cloud platform.")
    return mem, m1, m2


# --- Conversation lifecycle ---

class TestConversationLifecycle:
    def test_create_conversation(self, mem):
        conv = mem.create_conversation("alice")
        assert isinstance(conv, Conversation)
        assert conv.user_id == "alice"
        assert mem.count == 1

    def test_create_with_custom_id(self, mem):
        conv = mem.create_conversation("alice", conversation_id="custom")
        assert conv.id == "custom"

    def test_create_with_context(self, mem):
        conv = mem.create_conversation("alice", context={"lang": "en"})
        assert conv.context == {"lang": "en"}

    def test_create_duplicate_raises(self, mem):
        mem.create_conversation("alice", conversation_id="dup")
        with pytest.raises(ConversationMemoryError, match="already exists"):
            mem.create_conversation("alice", conversation_id="dup")

    def test_get_conversation(self, populated):
        mem, _, _ = populated
        conv = mem.get_conversation("c1")
        assert conv.user_id == "alice"

    def test_get_nonexistent_raises(self, mem):
        with pytest.raises(ConversationNotFoundError):
            mem.get_conversation("nope")

    def test_delete_conversation(self, populated):
        mem, _, _ = populated
        assert mem.delete_conversation("c2")
        assert mem.count == 1
        with pytest.raises(ConversationNotFoundError):
            mem.get_conversation("c2")

    def test_delete_nonexistent_raises(self, mem):
        with pytest.raises(ConversationNotFoundError):
            mem.delete_conversation("nope")


# --- Message handling ---

class TestMessages:
    def test_add_message(self, mem):
        mem.create_conversation("alice", conversation_id="c")
        msg = mem.add_message("c", "user", "Hello")
        assert isinstance(msg, Message)
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_message_order_preserved(self, populated):
        mem, _, _ = populated
        msgs = mem.get_messages("c1")
        assert [m.content for m in msgs] == [
            "Hello", "Hi there!", "Tell me about Python",
            "Python is a programming language.",
        ]

    def test_get_last_n_messages(self, populated):
        mem, _, _ = populated
        msgs = mem.get_messages("c1", last_n=2)
        assert len(msgs) == 2
        assert msgs[0].content == "Tell me about Python"

    def test_message_metadata(self, mem):
        mem.create_conversation("alice", conversation_id="c")
        msg = mem.add_message("c", "user", "Hi", metadata={"source": "web"})
        assert msg.metadata == {"source": "web"}

    def test_get_message_by_id(self, populated):
        mem, m1, _ = populated
        found = mem.get_message_by_id("c1", m1.id)
        assert found.content == "Hello"

    def test_get_message_by_id_not_found(self, mem):
        mem.create_conversation("alice", conversation_id="c")
        with pytest.raises(ConversationMemoryError, match="not found"):
            mem.get_message_by_id("c", "bad-id")

    def test_add_message_updates_timestamp(self, mem):
        conv = mem.create_conversation("alice", conversation_id="c")
        before = conv.updated_at
        mem.add_message("c", "user", "Hi")
        assert conv.updated_at >= before

    def test_conversation_properties(self, populated):
        mem, _, _ = populated
        conv = mem.get_conversation("c1")
        assert conv.message_count == 4
        assert conv.last_message.content == "Python is a programming language."

    def test_last_message_empty(self, mem):
        conv = mem.create_conversation("alice", conversation_id="c")
        assert conv.last_message is None


# --- Context maintenance ---

class TestContext:
    def test_update_context(self, mem):
        mem.create_conversation("alice", conversation_id="c")
        ctx = mem.update_context("c", {"mood": "happy"})
        assert ctx == {"mood": "happy"}

    def test_update_context_merges(self, mem):
        mem.create_conversation("alice", conversation_id="c", context={"a": 1})
        ctx = mem.update_context("c", {"b": 2})
        assert ctx == {"a": 1, "b": 2}

    def test_get_context(self, mem):
        mem.create_conversation("alice", conversation_id="c", context={"k": "v"})
        assert mem.get_context("c") == {"k": "v"}

    def test_add_topics(self, mem):
        mem.create_conversation("alice", conversation_id="c")
        topics = mem.add_topics("c", ["python", "aws"])
        assert topics == {"python", "aws"}

    def test_add_topics_via_message(self, populated):
        mem, _, _ = populated
        assert "greeting" in mem.get_topics("c1")
        assert "python" in mem.get_topics("c1")

    def test_get_topics(self, populated):
        mem, _, _ = populated
        assert mem.get_topics("c2") == {"aws"}


# --- Retrieval ---

class TestRetrieval:
    def test_by_user(self, populated):
        mem, _, _ = populated
        convs = mem.by_user("alice")
        assert len(convs) == 1
        assert convs[0].id == "c1"

    def test_by_user_empty(self, mem):
        assert mem.by_user("nobody") == []

    def test_by_time_range(self, mem):
        now = datetime.now()
        mem.create_conversation("alice", conversation_id="c1")
        convs = mem.by_time_range(now - timedelta(seconds=1), now + timedelta(seconds=1))
        assert len(convs) == 1

    def test_by_topic(self, populated):
        mem, _, _ = populated
        convs = mem.by_topic("python")
        assert len(convs) == 1
        assert convs[0].id == "c1"

    def test_by_topic_empty(self, populated):
        mem, _, _ = populated
        assert mem.by_topic("nonexistent") == []

    def test_search_content(self, populated):
        mem, _, _ = populated
        results = mem.search_content("AWS")
        assert len(results) == 1
        assert results[0].id == "c2"

    def test_search_content_case_insensitive(self, populated):
        mem, _, _ = populated
        results = mem.search_content("python")
        assert len(results) == 1
        assert results[0].id == "c1"

    def test_search_content_no_match(self, populated):
        mem, _, _ = populated
        assert mem.search_content("nonexistent") == []


# --- Multi-turn handling ---

class TestMultiTurn:
    def test_turn_pairs(self, populated):
        mem, _, _ = populated
        pairs = mem.get_turn_pairs("c1")
        assert len(pairs) == 2
        assert pairs[0] == {"user": "Hello", "assistant": "Hi there!"}
        assert pairs[1] == {"user": "Tell me about Python",
                            "assistant": "Python is a programming language."}

    def test_turn_pairs_incomplete(self, mem):
        mem.create_conversation("alice", conversation_id="c")
        mem.add_message("c", "user", "Hello")
        pairs = mem.get_turn_pairs("c")
        assert pairs == [{"user": "Hello"}]

    def test_turn_pairs_assistant_only(self, mem):
        mem.create_conversation("alice", conversation_id="c")
        mem.add_message("c", "assistant", "Welcome!")
        pairs = mem.get_turn_pairs("c")
        assert pairs == [{"assistant": "Welcome!"}]

    def test_references(self, populated):
        mem, m1, _ = populated
        m3 = mem.add_message("c1", "user", "Back to that greeting", references=[m1.id])
        refs = mem.get_referenced_messages("c1", m3.id)
        assert len(refs) == 1
        assert refs[0].content == "Hello"

    def test_references_missing_id_skipped(self, mem):
        mem.create_conversation("alice", conversation_id="c")
        msg = mem.add_message("c", "user", "Hi", references=["bad-id"])
        refs = mem.get_referenced_messages("c", msg.id)
        assert refs == []

    def test_build_context_window(self, populated):
        mem, _, _ = populated
        ctx = mem.build_context_window("c1", last_n=2)
        assert ctx == "user: Tell me about Python\nassistant: Python is a programming language."

    def test_build_context_window_full(self, populated):
        mem, _, _ = populated
        ctx = mem.build_context_window("c1")
        lines = ctx.split("\n")
        assert len(lines) == 4
        assert lines[0] == "user: Hello"
