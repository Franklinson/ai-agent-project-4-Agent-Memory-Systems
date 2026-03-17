"""Tests for experience tracker."""

import pytest
from datetime import datetime, timedelta
from experience_tracker import (
    ExperienceTracker, Experience, Lesson, Outcome,
    ExperienceTrackerError, ExperienceNotFoundError,
)


@pytest.fixture
def tracker():
    return ExperienceTracker()


@pytest.fixture
def populated(tracker):
    """Tracker with mixed experiences across two actions."""
    base = datetime(2025, 1, 1)
    tracker.log("search", Outcome.SUCCESS, score=0.6, tags=["web"], timestamp=base, experience_id="s1")
    tracker.log("search", Outcome.FAILURE, score=0.2, tags=["web", "api"], timestamp=base + timedelta(hours=1), experience_id="s2")
    tracker.log("search", Outcome.SUCCESS, score=0.8, tags=["web"], timestamp=base + timedelta(hours=2), experience_id="s3")
    tracker.log("search", Outcome.SUCCESS, score=0.9, tags=["api"], timestamp=base + timedelta(hours=3), experience_id="s4")
    tracker.log("summarize", Outcome.PARTIAL, score=0.5, tags=["text"], timestamp=base + timedelta(hours=4), experience_id="m1")
    tracker.log("summarize", Outcome.SUCCESS, score=0.7, tags=["text"], timestamp=base + timedelta(hours=5), experience_id="m2")
    return tracker


# --- Logging ---

class TestLogging:
    def test_log_experience(self, tracker):
        exp = tracker.log("search", Outcome.SUCCESS, context={"q": "test"})
        assert isinstance(exp, Experience)
        assert exp.action == "search"
        assert exp.outcome == Outcome.SUCCESS
        assert tracker.count == 1

    def test_log_with_all_fields(self, tracker):
        exp = tracker.log(
            "search", Outcome.PARTIAL,
            context={"q": "x"}, feedback="ok", score=0.5,
            tags=["web"], metadata={"model": "v1"},
        )
        assert exp.feedback == "ok"
        assert exp.score == 0.5
        assert exp.tags == ["web"]
        assert exp.metadata == {"model": "v1"}

    def test_log_custom_id(self, tracker):
        exp = tracker.log("a", Outcome.SUCCESS, experience_id="custom")
        assert exp.id == "custom"

    def test_log_duplicate_id_raises(self, tracker):
        tracker.log("a", Outcome.SUCCESS, experience_id="dup")
        with pytest.raises(ExperienceTrackerError, match="already exists"):
            tracker.log("a", Outcome.SUCCESS, experience_id="dup")

    def test_log_invalid_score_raises(self, tracker):
        with pytest.raises(ExperienceTrackerError, match="Score"):
            tracker.log("a", Outcome.SUCCESS, score=1.5)

    def test_log_negative_score_raises(self, tracker):
        with pytest.raises(ExperienceTrackerError, match="Score"):
            tracker.log("a", Outcome.SUCCESS, score=-0.1)

    def test_log_defaults(self, tracker):
        exp = tracker.log("a", Outcome.UNKNOWN)
        assert exp.context == {}
        assert exp.feedback is None
        assert exp.score is None
        assert exp.tags == []

    def test_get_experience(self, populated):
        exp = populated.get("s1")
        assert exp.action == "search"

    def test_get_not_found_raises(self, tracker):
        with pytest.raises(ExperienceNotFoundError):
            tracker.get("nope")

    def test_add_feedback(self, tracker):
        exp = tracker.log("a", Outcome.SUCCESS, experience_id="e1")
        updated = tracker.add_feedback("e1", "great", score=0.9)
        assert updated.feedback == "great"
        assert updated.score == 0.9

    def test_add_feedback_invalid_score_raises(self, tracker):
        tracker.log("a", Outcome.SUCCESS, experience_id="e1")
        with pytest.raises(ExperienceTrackerError, match="Score"):
            tracker.add_feedback("e1", "bad", score=2.0)


# --- Retrieval ---

class TestRetrieval:
    def test_by_action(self, populated):
        results = populated.by_action("search")
        assert len(results) == 4
        assert all(e.action == "search" for e in results)

    def test_by_outcome(self, populated):
        results = populated.by_outcome(Outcome.SUCCESS)
        assert len(results) == 4

    def test_by_tag(self, populated):
        results = populated.by_tag("api")
        assert len(results) == 2

    def test_by_time_range(self, populated):
        base = datetime(2025, 1, 1)
        results = populated.by_time_range(base, base + timedelta(hours=2))
        assert len(results) == 3

    def test_recent(self, populated):
        results = populated.recent(2)
        assert len(results) == 2
        assert results[0].id == "m1"
        assert results[1].id == "m2"

    def test_by_action_empty(self, tracker):
        assert tracker.by_action("nope") == []

    def test_by_tag_empty(self, tracker):
        assert tracker.by_tag("nope") == []


# --- Pattern recognition ---

class TestPatterns:
    def test_success_rate_action(self, populated):
        rate = populated.success_rate("search")
        assert rate == pytest.approx(0.75)

    def test_success_rate_overall(self, populated):
        rate = populated.success_rate()
        assert rate == pytest.approx(4 / 6)

    def test_success_rate_empty(self, tracker):
        assert tracker.success_rate("nope") == 0.0

    def test_average_score_action(self, populated):
        avg = populated.average_score("search")
        assert avg == pytest.approx((0.6 + 0.2 + 0.8 + 0.9) / 4)

    def test_average_score_overall(self, populated):
        avg = populated.average_score()
        assert avg == pytest.approx((0.6 + 0.2 + 0.8 + 0.9 + 0.5 + 0.7) / 6)

    def test_average_score_none_when_no_scores(self, tracker):
        tracker.log("a", Outcome.SUCCESS)
        assert tracker.average_score() is None

    def test_outcome_distribution(self, populated):
        dist = populated.outcome_distribution("search")
        assert dist == {"success": 3, "failure": 1}

    def test_outcome_distribution_overall(self, populated):
        dist = populated.outcome_distribution()
        assert dist == {"success": 4, "failure": 1, "partial": 1}

    def test_common_tags(self, populated):
        tags = populated.common_tags("search")
        assert tags[0] == "web"  # web appears 3 times

    def test_common_tags_top_n(self, populated):
        tags = populated.common_tags("search", top_n=1)
        assert len(tags) == 1

    def test_score_trend(self, populated):
        trend = populated.score_trend("search")
        assert trend == [0.6, 0.2, 0.8, 0.9]


# --- Learning ---

class TestLearning:
    def test_extract_lesson_high_success(self, populated):
        lesson = populated.extract_lesson("search")
        assert isinstance(lesson, Lesson)
        assert lesson.action == "search"
        assert lesson.total == 4
        assert lesson.success_rate == pytest.approx(0.75)
        assert lesson.avg_score is not None

    def test_extract_lesson_recommendation_high(self, tracker):
        for i in range(5):
            tracker.log("a", Outcome.SUCCESS)
        lesson = tracker.extract_lesson("a")
        assert lesson.recommendation == "Continue current approach"

    def test_extract_lesson_recommendation_mid(self, tracker):
        tracker.log("a", Outcome.SUCCESS)
        tracker.log("a", Outcome.FAILURE)
        lesson = tracker.extract_lesson("a")
        assert lesson.recommendation == "Review partial failures for improvement"

    def test_extract_lesson_recommendation_low(self, tracker):
        for i in range(5):
            tracker.log("a", Outcome.FAILURE)
        lesson = tracker.extract_lesson("a")
        assert lesson.recommendation == "Significant changes needed"

    def test_extract_lesson_no_experiences_raises(self, tracker):
        with pytest.raises(ExperienceTrackerError, match="No experiences"):
            tracker.extract_lesson("nope")

    def test_compare_actions(self, populated):
        lessons = populated.compare_actions(["search", "summarize"])
        assert len(lessons) == 2
        assert lessons[0].success_rate >= lessons[1].success_rate

    def test_is_improving_true(self, tracker):
        scores = [0.3, 0.3, 0.3, 0.7, 0.8, 0.9]
        for i, s in enumerate(scores):
            tracker.log("a", Outcome.SUCCESS, score=s,
                        timestamp=datetime(2025, 1, 1, i))
        assert tracker.is_improving("a", window=3) is True

    def test_is_improving_false(self, tracker):
        scores = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
        for i, s in enumerate(scores):
            tracker.log("a", Outcome.SUCCESS, score=s,
                        timestamp=datetime(2025, 1, 1, i))
        assert tracker.is_improving("a", window=3) is False

    def test_is_improving_insufficient_data(self, tracker):
        tracker.log("a", Outcome.SUCCESS, score=0.5)
        assert tracker.is_improving("a", window=3) is None

    def test_is_improving_overall(self, tracker):
        scores = [0.2, 0.3, 0.4, 0.6, 0.7, 0.8]
        for i, s in enumerate(scores):
            tracker.log("a", Outcome.SUCCESS, score=s,
                        timestamp=datetime(2025, 1, 1, i))
        assert tracker.is_improving(window=3) is True


# --- Outcome enum ---

class TestOutcome:
    def test_all_outcomes(self, tracker):
        for outcome in Outcome:
            exp = tracker.log("test", outcome)
            assert exp.outcome == outcome

    def test_outcome_values(self):
        assert Outcome.SUCCESS.value == "success"
        assert Outcome.FAILURE.value == "failure"
        assert Outcome.PARTIAL.value == "partial"
        assert Outcome.UNKNOWN.value == "unknown"
