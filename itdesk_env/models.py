"""
Pydantic data models for typed states, actions, and observations.
Ensures strong typing across the environment API.
"""

from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel, Field

class Observation(BaseModel):
    """Model representing an agent's observation of the environment at a specific step."""
    ticket_id: str = Field(..., description="Unique identifier for the ticket.")
    user_role: str = Field(..., description="Role of the user who submitted the ticket.")
    issue_description: str = Field(..., description="Detailed description of the issue.")
    detected_signals: List[str] = Field(default_factory=list, description="List of detected signals or alerts associated with the ticket.")
    affected_systems: List[str] = Field(default_factory=list, description="List of systems affected by the issue.")
    similar_ticket_count: int = Field(default=0, description="Number of similar previous tickets.")
    possible_duplicate: bool = Field(default=False, description="Indicates if this ticket might be a duplicate.")
    current_status: str = Field(default="open", description="Current workflow status of the ticket.")
    allowed_teams: List[str] = Field(default_factory=list, description="List of teams the ticket can be assigned to.")
    allowed_priorities: List[str] = Field(default_factory=list, description="List of allowed priority levels.")
    step_count: int = Field(default=0, description="Current number of steps taken in the environment.")
    max_steps: int = Field(default=15, description="Maximum allowed steps in the environment.")
    history: List[str] = Field(default_factory=list, description="Historical log of actions and states for this ticket.")
    goal: str = Field(default="Resolve the IT ticket accurately.", description="The final goal or objective the agent needs to achieve.")

class Action(BaseModel):
    """Model representing an action taken by an agent to progress the environment state."""
    action_type: Literal[
        "classify_issue",
        "set_priority",
        "assign_team",
        "flag_duplicate_cluster",
        "request_missing_context",
        "escalate_incident",
        "resolve_ticket",
        "close_as_duplicate",
        "noop"
    ] = Field(..., description="The type of action to perform.")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the action.")

class StepResult(BaseModel):
    """Model representing the result of a single action step."""
    observation: Observation = Field(..., description="The new observation after the action is taken.")
    reward: float = Field(..., description="The reward achieved from the action.")
    done: bool = Field(..., description="Indicates if the episode has terminated.")
    info: Dict[str, Any] = Field(default_factory=dict, description="Additional debugging or environment-specific information.")

class State(BaseModel):
    """Internal representation of the ticket workflow state."""
    classification: Optional[str] = Field(default=None, description="The assigned classification/category of the ticket.")
    priority: Optional[str] = Field(default=None, description="The assigned priority level.")
    assigned_team: Optional[str] = Field(default=None, description="The team currently assigned to handle the ticket.")
    resolved: bool = Field(default=False, description="Flag indicating if the ticket has been resolved.")
    escalated: bool = Field(default=False, description="Flag indicating if the ticket has been escalated.")
    duplicate_flag: bool = Field(default=False, description="Flag indicating if the ticket has been marked as a duplicate.")
    flags: Dict[str, Any] = Field(default_factory=dict, description="Custom simulator tracking flags for correctness/errors.")
    task_name: Optional[str] = Field(default=None, description="Task scenario identifier for debugging and grading traceability.")
