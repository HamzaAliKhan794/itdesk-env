"""
Internal simulator mechanics.
Handles backing state changes, external stubbing, or simulated subsystems.
"""

from typing import Dict, Any
from itdesk_env.models import State, Action, Observation

class Simulator:
    """Handles deep simulation mechanics underneath the environment envelope."""
    
    @staticmethod
    def apply_action(state: State, action: Action, observation: Observation) -> State:
        """Apply a deterministic rule-based action to the ticket state."""
        
        new_state = state.model_copy(deep=True)
        if "errors" not in new_state.flags:
            new_state.flags["errors"] = []
            
        params = action.parameters
        
        # Scenario identification from observation
        desc = str(observation.issue_description).lower()
        signals = " ".join([str(sig).lower() for sig in observation.detected_signals])
        systems = " ".join([str(sys).lower() for sys in observation.affected_systems])
        combined_text = f"{desc} {signals} {systems}"
        
        is_security = any(kw in combined_text for kw in ["suspicious", "phishing", "unknown login", "foreign login", "weird email", "security alert", "ransomware", "impossible_travel", "malicious"])
        is_network = observation.similar_ticket_count > 2 or any(kw in combined_text for kw in ["vpn", "network", "cannot connect", "outage", "packet_loss"])
        is_access = any(kw in combined_text for kw in ["password", "login", "account locked"])
        is_duplicate_scenario = observation.similar_ticket_count > 2
        
        action_type = action.action_type
        
        if action_type == "classify_issue":
            new_state.classification = params.get("category", "unclassified")
            
        elif action_type == "set_priority":
            new_state.priority = params.get("severity", "low")
            
        elif action_type == "assign_team":
            team = params.get("team", "unassigned")
            new_state.assigned_team = team
            
            if is_security and team != "security_ops":
                new_state.flags["wrong_team"] = True
                new_state.flags["errors"].append(f"Incorrect team assignment: assigned {team}, but security cases require security_ops.")
            elif is_network and team not in ["network_team", "security_ops"]:
                new_state.flags["wrong_team"] = True
                new_state.flags["errors"].append(f"Incorrect team assignment: assigned {team}, but VPN issues require network_team.")
                
        elif action_type == "escalate_incident":
            new_state.escalated = True
            if not is_security:
                new_state.flags["unnecessary_escalation"] = True
            
        elif action_type == "flag_duplicate_cluster":
            new_state.duplicate_flag = True
            if not is_duplicate_scenario:
                new_state.flags["errors"].append("Flagged duplicate on a non-cluster issue.")
                
        elif action_type == "request_missing_context":
            new_state.flags["context_requested"] = True
            
        elif action_type == "close_as_duplicate":
            if is_duplicate_scenario:
                new_state.duplicate_flag = True
                new_state.resolved = True
            else:
                new_state.flags["premature_resolution"] = True
                new_state.flags["errors"].append("Dangerous action: Closed as duplicate for a unique/critical ticket.")
                
        elif action_type == "resolve_ticket":
            if new_state.classification in [None, "unclassified", "unassigned"] or \
               new_state.assigned_team in [None, "unassigned"] or \
               new_state.priority in [None, "unassigned"]:
                new_state.flags["premature_resolution"] = True
                new_state.flags["errors"].append("Resolving before classification, assignment, or priority is set.")
            new_state.resolved = True
            
        # Additional end-of-step validation maps constraints natively into state evaluation
        if new_state.resolved:
            if is_security and not new_state.escalated:
                new_state.flags["missing_security_escalation"] = True
                new_state.flags["errors"].append("Security cases MUST be escalated before resolution.")
            if is_duplicate_scenario and not new_state.duplicate_flag:
                new_state.flags["missing_duplicate_handling"] = True
                new_state.flags["errors"].append("Duplicate detection required for this issue cluster.")
                
        return new_state
