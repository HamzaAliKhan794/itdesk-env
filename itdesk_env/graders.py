"""
Deterministic evaluators for ITDesk-Env tasks.
Checks if a given trajectory or final state meets the task objectives.
All scores are guaranteed to be strictly between 0 and 1 (exclusive).
"""


def _clamp(value: float) -> float:
    """Ensure the score is strictly inside (0, 1) exclusive."""
    lo = 0.01
    hi = 0.99
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _to_state(state) -> dict:
    """Normalize state to a plain dict regardless of input type."""
    if isinstance(state, dict):
        inner = state.get("internal_state")
        if isinstance(inner, dict):
            return inner
        return state
    # Pydantic model or any object with __dict__
    if hasattr(state, "model_dump"):
        return state.model_dump()
    if hasattr(state, "__dict__"):
        return state.__dict__
    return {}


def grade_easy(state) -> float:
    """
    Evaluate the easy task: single user login/access issue.
    Correct flow: classify as access_issue, priority low, team access_support, resolve.
    """
    d = _to_state(state)
    score = 0.05

    if d.get("classification") == "access_issue":
        score += 0.2
    if d.get("priority") == "low":
        score += 0.2
    if d.get("assigned_team") == "access_support":
        score += 0.35
    if d.get("resolved") is True:
        score += 0.15

    if d.get("escalated") is True:
        score -= 0.25
    if d.get("duplicate_flag") is True:
        score -= 0.1
    if d.get("assigned_team") in ["security_ops", "network_team"]:
        score -= 0.2

    return _clamp(score)


def grade_medium(state) -> float:
    """
    Evaluate the medium task: VPN/network outage cluster.
    Correct flow: classify as network_issue, priority high, team network_team, flag duplicate, resolve.
    """
    d = _to_state(state)
    score = 0.05

    if d.get("classification") == "network_issue":
        score += 0.2
    if d.get("priority") == "high":
        score += 0.2
    if d.get("assigned_team") == "network_team":
        score += 0.2
    if d.get("duplicate_flag") is True:
        score += 0.2
    if d.get("resolved") is True:
        score += 0.1

    if d.get("escalated") is True:
        score -= 0.25
    if d.get("assigned_team") == "security_ops":
        score -= 0.25
    if d.get("assigned_team") == "access_support":
        score -= 0.15

    return _clamp(score)


def grade_hard(state) -> float:
    """
    Evaluate the hard task: ransomware/phishing security escalation.
    Correct flow: classify security_issue, critical priority, security_ops, escalate, resolve.
    """
    d = _to_state(state)
    score = 0.05

    if d.get("classification") == "security_issue":
        score += 0.2
    if d.get("priority") == "critical":
        score += 0.2
    if d.get("assigned_team") == "security_ops":
        score += 0.2
    if d.get("escalated") is True:
        score += 0.2
    if d.get("resolved") is True:
        score += 0.1

    if d.get("escalated") is not True:
        score -= 0.35
    if d.get("assigned_team") in ["access_support", "network_team"]:
        score -= 0.35
    if d.get("priority") in ["low", "medium"]:
        score -= 0.25

    return _clamp(score)


def grade_task(task_name: str, state) -> float:
    """Route to the correct task grader by name."""
    name = str(task_name).lower().strip()
    if name == "easy":
        return grade_easy(state)
    elif name == "medium":
        return grade_medium(state)
    elif name == "hard":
        return grade_hard(state)
    # Unknown task name: return a safe default score
    return 0.05


class Grader:
    """Evaluation class compatible with OpenEnv workflows."""

    def evaluate(self, state, task_name: str) -> float:
        return grade_task(task_name, state)


# Module-level hooks for OpenEnv dynamic reflection
def evaluate(state, task_name: str) -> float:
    return grade_task(task_name, state)


def score(state, task_name: str) -> float:
    return grade_task(task_name, state)
