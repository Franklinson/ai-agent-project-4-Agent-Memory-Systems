"""Experience tracking for agent learning with pattern recognition and adaptation."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


class ExperienceTrackerError(Exception):
    """Base exception for experience tracker operations."""


class ExperienceNotFoundError(ExperienceTrackerError):
    """Raised when an experience is not found."""


class Outcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


@dataclass
class Experience:
    """A single logged experience with outcome and feedback."""
    id: str
    action: str
    context: Dict[str, Any]
    outcome: Outcome
    timestamp: datetime
    feedback: Optional[str] = None
    score: Optional[float] = None  # 0.0–1.0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Lesson:
    """A lesson extracted from a group of similar experiences."""
    action: str
    total: int
    success_rate: float
    avg_score: Optional[float]
    common_tags: List[str]
    recommendation: str


class ExperienceTracker:
    """Logs experiences, recognises patterns, and supports learning."""

    def __init__(self):
        self._experiences: Dict[str, Experience] = {}
        self._timeline: List[str] = []

    @property
    def count(self) -> int:
        return len(self._experiences)

    # --- Logging ---

    def log(
        self,
        action: str,
        outcome: Outcome,
        context: Optional[Dict[str, Any]] = None,
        feedback: Optional[str] = None,
        score: Optional[float] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
        experience_id: Optional[str] = None,
    ) -> Experience:
        eid = experience_id or str(uuid.uuid4())
        if eid in self._experiences:
            raise ExperienceTrackerError(f"Experience '{eid}' already exists")
        if score is not None and not (0.0 <= score <= 1.0):
            raise ExperienceTrackerError("Score must be between 0.0 and 1.0")

        exp = Experience(
            id=eid,
            action=action,
            context=context or {},
            outcome=outcome,
            timestamp=timestamp or datetime.now(),
            feedback=feedback,
            score=score,
            tags=tags or [],
            metadata=metadata or {},
        )
        self._experiences[eid] = exp
        self._timeline.append(eid)
        return exp

    def get(self, experience_id: str) -> Experience:
        if experience_id not in self._experiences:
            raise ExperienceNotFoundError(f"Experience '{experience_id}' not found")
        return self._experiences[experience_id]

    def add_feedback(self, experience_id: str, feedback: str, score: Optional[float] = None) -> Experience:
        exp = self.get(experience_id)
        exp.feedback = feedback
        if score is not None:
            if not (0.0 <= score <= 1.0):
                raise ExperienceTrackerError("Score must be between 0.0 and 1.0")
            exp.score = score
        return exp

    # --- Retrieval ---

    def by_action(self, action: str) -> List[Experience]:
        return [self._experiences[eid] for eid in self._timeline
                if self._experiences[eid].action == action]

    def by_outcome(self, outcome: Outcome) -> List[Experience]:
        return [self._experiences[eid] for eid in self._timeline
                if self._experiences[eid].outcome == outcome]

    def by_tag(self, tag: str) -> List[Experience]:
        return [self._experiences[eid] for eid in self._timeline
                if tag in self._experiences[eid].tags]

    def by_time_range(self, start: datetime, end: datetime) -> List[Experience]:
        return [self._experiences[eid] for eid in self._timeline
                if start <= self._experiences[eid].timestamp <= end]

    def recent(self, n: int) -> List[Experience]:
        return [self._experiences[eid] for eid in self._timeline[-n:]]

    # --- Pattern recognition ---

    def success_rate(self, action: Optional[str] = None) -> float:
        """Success rate (0.0–1.0) for an action or overall."""
        exps = self.by_action(action) if action else [self._experiences[eid] for eid in self._timeline]
        if not exps:
            return 0.0
        successes = sum(1 for e in exps if e.outcome == Outcome.SUCCESS)
        return successes / len(exps)

    def average_score(self, action: Optional[str] = None) -> Optional[float]:
        """Average score for an action or overall. None if no scores."""
        exps = self.by_action(action) if action else [self._experiences[eid] for eid in self._timeline]
        scored = [e.score for e in exps if e.score is not None]
        if not scored:
            return None
        return sum(scored) / len(scored)

    def outcome_distribution(self, action: Optional[str] = None) -> Dict[str, int]:
        """Count of each outcome for an action or overall."""
        exps = self.by_action(action) if action else [self._experiences[eid] for eid in self._timeline]
        dist: Dict[str, int] = {}
        for e in exps:
            dist[e.outcome.value] = dist.get(e.outcome.value, 0) + 1
        return dist

    def common_tags(self, action: Optional[str] = None, top_n: int = 5) -> List[str]:
        """Most frequent tags for an action or overall."""
        exps = self.by_action(action) if action else [self._experiences[eid] for eid in self._timeline]
        counts: Dict[str, int] = {}
        for e in exps:
            for t in e.tags:
                counts[t] = counts.get(t, 0) + 1
        return sorted(counts, key=counts.get, reverse=True)[:top_n]

    def score_trend(self, action: Optional[str] = None) -> List[float]:
        """Chronological list of scores for trend analysis."""
        exps = self.by_action(action) if action else [self._experiences[eid] for eid in self._timeline]
        return [e.score for e in exps if e.score is not None]

    # --- Learning ---

    def extract_lesson(self, action: str) -> Lesson:
        """Extract a lesson from all experiences of a given action."""
        exps = self.by_action(action)
        if not exps:
            raise ExperienceTrackerError(f"No experiences for action '{action}'")

        sr = self.success_rate(action)
        avg = self.average_score(action)
        tags = self.common_tags(action)

        if sr >= 0.8:
            rec = "Continue current approach"
        elif sr >= 0.5:
            rec = "Review partial failures for improvement"
        else:
            rec = "Significant changes needed"

        return Lesson(
            action=action,
            total=len(exps),
            success_rate=sr,
            avg_score=avg,
            common_tags=tags,
            recommendation=rec,
        )

    def compare_actions(self, actions: List[str]) -> List[Lesson]:
        """Extract and return lessons for multiple actions, sorted by success rate."""
        lessons = [self.extract_lesson(a) for a in actions]
        return sorted(lessons, key=lambda l: l.success_rate, reverse=True)

    def is_improving(self, action: Optional[str] = None, window: int = 3) -> Optional[bool]:
        """Check if recent scores trend upward vs earlier scores.

        Compares the average of the last `window` scores against the
        average of the preceding `window` scores. Returns None if
        insufficient data.
        """
        scores = self.score_trend(action)
        if len(scores) < window * 2:
            return None
        recent_avg = sum(scores[-window:]) / window
        earlier_avg = sum(scores[-window * 2:-window]) / window
        return recent_avg > earlier_avg
