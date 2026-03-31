"""
Deterministic task generation module for ITDesk-Env.
Contains heavily benchmarked configurations for evaluation scenarios.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from itdesk_env.models import Observation, State

class TaskDefinition(BaseModel):
    """Configuration for a specific scenario and its win conditions."""
    task_id: str
    difficulty: str
    initial_observation: Observation
    initial_state: State
    
    # Core expectations
    expected_classification: str
    expected_team: str
    expected_priority: str
    expected_escalation: bool
    expected_duplicate: bool
    
    # Optional constraints mapped for grader precision
    avoid_teams: List[str] = []
    avoid_escalation: Optional[bool] = None


class TaskManager:
    """Provides deterministic baseline tasks for OpenEnv simulation."""
    
    @staticmethod
    def get_task(task_name: str) -> TaskDefinition:
        """Fetch a specific evaluation task definition based on its string key."""
        task_name = task_name.lower().strip()
        if task_name == "easy":
            return TaskManager._get_easy_task()
        elif task_name == "medium":
            return TaskManager._get_medium_task()
        elif task_name == "hard":
            return TaskManager._get_hard_task()
        else:
            raise ValueError(f"Unknown task configuration: {task_name}")

    @staticmethod
    def _get_easy_task() -> TaskDefinition:
        obs = Observation(
            ticket_id="TKT-EASY-001",
            user_role="contractor_staff",
            issue_description="I can't log into the customer portal. It keeps saying 'Incorrect Password' even though I just reset it yesterday. Can someone unlock my account so I can finish my timesheet?",
            detected_signals=["failed_login_attempts: 3", "account_status: locked", "ip_location: standard_office"],
            affected_systems=["active_directory_auth"],
            similar_ticket_count=0,
            allowed_teams=["access_support", "network_team", "security_ops", "hardware_support"]
        )
        
        state = State(
            ticket_id="TKT-EASY-001",
            classification="unclassified",
            priority="unassigned",
            assigned_team="unassigned",
            escalated=False,
            duplicate_flag=False,
            resolved=False,
            current_status="open",
            step_count=0,
            flags={}
        )
        
        return TaskDefinition(
            task_id="EASY-001",
            difficulty="easy",
            initial_observation=obs,
            initial_state=state,
            expected_classification="access_issue",
            expected_team="access_support",
            expected_priority="low",
            expected_escalation=False,
            expected_duplicate=False,
            avoid_teams=["security_ops", "network_team"],
            avoid_escalation=True
        )

    @staticmethod
    def _get_medium_task() -> TaskDefinition:
        obs = Observation(
            ticket_id="TKT-MED-001",
            user_role="regional_director",
            issue_description="The central VPN just dropped all active connections in our building. My team cannot connect to any internal shared drives and there's a complete outage on our floor.",
            detected_signals=["router_timeout: US-EAST-04", "packet_loss_spike: 100%", "bgp_route_flap_detected"],
            affected_systems=["corporate_vpn_gateway", "internal_dfs_shares"],
            similar_ticket_count=5,
            allowed_teams=["access_support", "network_team", "security_ops", "hardware_support"]
        )
        
        state = State(
            ticket_id="TKT-MED-001",
            classification="unclassified",
            priority="unassigned",
            assigned_team="unassigned",
            escalated=False,
            duplicate_flag=False,
            resolved=False,
            current_status="open",
            step_count=0,
            flags={}
        )
        
        return TaskDefinition(
            task_id="MED-001",
            difficulty="medium",
            initial_observation=obs,
            initial_state=state,
            expected_classification="network_issue",
            expected_team="network_team",
            expected_priority="high",
            expected_escalation=False,
            expected_duplicate=True,
            avoid_teams=["security_ops", "access_support", "hardware_support"],
            avoid_escalation=True
        )

    @staticmethod
    def _get_hard_task() -> TaskDefinition:
        obs = Observation(
            ticket_id="TKT-HRD-001",
            user_role="chief_financial_officer",
            issue_description="I clicked a tracking link in an urgent invoice email. My screen flashed black for a second, and several of my local Excel files have changed to '.locked' extensions. I also just got an automated text saying a login from Russia was approved?",
            detected_signals=["ransomware_heuristic_flagged", "impossible_travel_detected: RU", "mass_file_modification_alert"],
            affected_systems=["cfo_local_machine", "financial_subnet"],
            similar_ticket_count=0,
            allowed_teams=["access_support", "network_team", "security_ops", "hardware_support"]
        )
        
        state = State(
            ticket_id="TKT-HRD-001",
            classification="unclassified",
            priority="unassigned",
            assigned_team="unassigned",
            escalated=False,
            duplicate_flag=False,
            resolved=False,
            current_status="open",
            step_count=0,
            flags={}
        )
        
        return TaskDefinition(
            task_id="HRD-001",
            difficulty="hard",
            initial_observation=obs,
            initial_state=state,
            expected_classification="security_issue",
            expected_team="security_ops",
            expected_priority="critical",
            expected_escalation=True,
            expected_duplicate=False,
            avoid_teams=["access_support", "network_team", "hardware_support"],
            avoid_escalation=False
        )
