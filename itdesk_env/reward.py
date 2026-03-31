"""
Dense reward shaping module for ITDesk-Env.
Provides step-by-step reinforcement signals to guide agent workflows.
"""

from itdesk_env.models import State, Action

def compute_reward(state: State, action: Action, previous_state: State) -> float:
    """
    Computes a deterministic, bounded reward [-1.0, 1.0] for the transition.
    Values clean workflow adherence heavily, penalizing dangerous short-circuits natively.
    """
    reward = 0.0
    flags = state.flags
    
    # 1. Negative Penalties (Workflow Violations)
    if flags.get("premature_resolution", False):
        reward -= 0.5
    if flags.get("wrong_team", False):
        reward -= 0.3
    if flags.get("unnecessary_escalation", False):
        reward -= 0.3
    if flags.get("missing_security_escalation", False):
        reward -= 0.5
    if flags.get("missing_duplicate_handling", False):
        reward -= 0.3
        
    # Generalized error checking (e.g. flagging duplicate on non-cluster, or generic errors)
    errors = flags.get("errors", [])
    if errors:
        reward -= min(0.5, (0.1 * len(errors)))
        
    # 2. Positive Progress Signals
    
    # Reward classification if it was newly set to a valid category
    if state.classification not in [None, "unclassified"] and previous_state.classification in [None, "unclassified"]:
        reward += 0.2
        
    # Reward priority if newly set cleanly
    if state.priority not in [None, "unassigned"] and previous_state.priority in [None, "unassigned"]:
        reward += 0.2
        
    # Reward team assignment if newly set and no "wrong_team" flag triggered
    if state.assigned_team not in [None, "unassigned"] and previous_state.assigned_team in [None, "unassigned"]:
        if not flags.get("wrong_team", False):
            reward += 0.2
            
    # Reward context requested natively
    if flags.get("context_requested", False) and not previous_state.flags.get("context_requested", False):
        reward += 0.1
        
    # Reward duplicate handling if actually useful
    if state.duplicate_flag and not previous_state.duplicate_flag:
        if not errors and not flags.get("premature_resolution", False):
            reward += 0.2
            
    # Reward escalation if newly set and NOT unnecessary
    if state.escalated and not previous_state.escalated:
        if not flags.get("unnecessary_escalation", False):
            reward += 0.2
            
    # 3. Final Resolution Check
    if action.action_type in ["resolve_ticket", "close_as_duplicate"] and state.resolved and not previous_state.resolved:
        # Give a substantial completion cap ONLY if the workflow wasn't corrupted
        if not flags.get("premature_resolution") and not flags.get("missing_security_escalation") and not errors:
            reward += 0.5
        else:
            reward -= 0.5 # Severe hit for closing a corrupted trace
            
    # Apply small time penalty for any step to encourage efficiency over looping
    if action.action_type != "noop":
         reward -= 0.02
    else:
         reward -= 0.1 # Large penalty for doing absolutely nothing
            
    return max(-1.0, min(1.0, float(reward)))
