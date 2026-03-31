"""
Deterministic evaluators for ITDesk-Env tasks.
Checks if a given trajectory or final state meets the task objectives successfully.
"""

from itdesk_env.models import State

def grade_easy(state: State) -> float:
    """
    Evaluate the easy task (Single user login issue).
    Max score 1.0. Deductions for unsafe escalation or wrong routing.
    """
    score = 0.0
    
    # Positive points (Max 1.0)
    if state.classification == "access_issue":
        score += 0.2
    if state.priority == "low":
        score += 0.2
    if state.assigned_team == "access_support":
        score += 0.4
    if state.resolved:
        score += 0.2
        
    # Negative penalties bounds
    if state.escalated:
        score -= 0.3  # Penalty for unnecessarily escalating a simple password reset
    if state.duplicate_flag:
        score -= 0.1  # Not a duplicate cluster
    if state.assigned_team in ["security_ops", "network_team"]:
        score -= 0.2  # Wrong team
        
    return max(0.0, min(1.0, score))


def grade_medium(state: State) -> float:
    """
    Evaluate the medium task (VPN/network outage cluster).
    Requires duplicate detection and high priority network routing.
    """
    score = 0.0
    
    # Positive points (Max 1.0)
    if state.classification == "network_issue":
        score += 0.2
    if state.priority == "high":
        score += 0.2
    if state.assigned_team == "network_team":
        score += 0.2
    if state.duplicate_flag:
        score += 0.2
    if state.resolved:
        score += 0.2
        
    # Negative penalties bounds
    if state.escalated:
        score -= 0.3  # Penalty: Outages are high priority but rarely security escalations
    if state.assigned_team == "security_ops":
        score -= 0.3  # Penalty: Wasting security team's time
    if state.assigned_team == "access_support":
        score -= 0.2  # Penalty: Standard tier 1 cannot handle VPN router drops
        
    return max(0.0, min(1.0, score))


def grade_hard(state: State) -> float:
    """
    Evaluate the hard task (Phishing/Ransomware security escalation).
    Strictly requires critical priority, security ops team, and immediate escalation.
    """
    score = 0.0
    
    # Positive points (Max 1.0)
    if state.classification == "security_issue":
        score += 0.2
    if state.priority == "critical":
        score += 0.2
    if state.assigned_team == "security_ops":
        score += 0.2
    if state.escalated:
        score += 0.2
    if state.resolved:
        score += 0.2
        
    # Negative penalties bounds
    if not state.escalated:
        score -= 0.5  # Massive penalty for failing to escalate a cyber breach
    if state.assigned_team in ["access_support", "network_team"]:
        score -= 0.4  # Dangerous to send a live breach to standard IT operations
    if state.priority in ["low", "medium"]:
        score -= 0.3  # Dangerous underscoping of digital threat
        
    return max(0.0, min(1.0, score))


def grade_task(task_name: str, state: State) -> float:
    """Route to the correct task grader deterministically."""
    task_name = task_name.lower().strip()
    if task_name == "easy":
         return grade_easy(state)
    elif task_name == "medium":
         return grade_medium(state)
    elif task_name == "hard":
         return grade_hard(state)
    return 0.0


class Grader:
    """Evaluation envelope natively compatible with OpenEnv workflows."""
    def evaluate(self, state: State, task_name: str) -> float:
        return grade_task(task_name, state)
