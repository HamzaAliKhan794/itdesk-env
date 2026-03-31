"""
Core Environment class for ITDesk-Env.
Handles state transitions, observation updates, and deterministic routing.
"""

import random
from typing import Dict, Any, Optional

from itdesk_env.models import Action, Observation, StepResult, State
from itdesk_env.simulator import Simulator
from itdesk_env.tasks import TaskManager
from itdesk_env.reward import compute_reward

class ITDeskEnv:
    """Core environment for the IT Desk simulator."""
    
    def __init__(self, max_steps: int = 15):
        self.max_steps = max_steps
        self.current_state: Optional[State] = None
        self.current_observation: Optional[Observation] = None
        self.step_count = 0
        self.rng = random.Random()
        
    def reset(self, seed: Optional[int] = None, difficulty: Optional[str] = None) -> StepResult:
        """
        Reset environment to a new deterministic task scenario.
        Optionally override the deterministic randomizer by explicitly defining the difficulty.
        """
        if seed is not None:
            self.rng.seed(seed)
            
        # Select exact task or deterministic random selection
        if difficulty:
            task_def = TaskManager.get_task(difficulty)
        else:
            task_name = self.rng.choice(["easy", "medium", "hard"])
            task_def = TaskManager.get_task(task_name)
            
        self.current_state = task_def.initial_state.model_copy(deep=True)
        self.current_observation = task_def.initial_observation.model_copy(deep=True)
        
        # Sync environment step tracking dynamically to the task constraints
        self.step_count = 0
        
        # Check if the Observation schema natively dictates a max_steps bound natively
        if hasattr(self.current_observation, "max_steps") and getattr(self.current_observation, "max_steps") is not None:
            self.max_steps = getattr(self.current_observation, "max_steps")
        else:
            self.max_steps = 15 # Default fallback bound
            
        # Cleanly wipe status trackers specifically for the new run
        self.current_observation.history = []
        self.current_observation.current_status = "open"
        self.current_observation.step_count = 0
        
        return StepResult(
            observation=self.current_observation,
            reward=0.0,
            done=False,
            info={}
        )
        
    def step(self, action: Action) -> StepResult:
        """Perform an action natively and return the next state progression and reward."""
        if self.current_state is None or self.current_observation is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
            
        reward = 0.0
        done = False
        info: Dict[str, Any] = {}
        
        old_state = self.current_state.model_copy(deep=True)
        
        try:
            # Action execution logging natively appending parameters intuitively
            action_log = str(action.action_type)
            if action.parameters:
                action_log += f" | Params: {action.parameters}"
            self.current_observation.history.append(action_log)
            
            # Defer complex pipeline execution to the Deep Simulator
            self.current_state = Simulator.apply_action(self.current_state, action, self.current_observation)
            
            if self.current_state.flags.get("errors") and len(self.current_state.flags["errors"]) > 0:
                reward -= 0.05
                self.current_observation.history[-1] += f" [ERROR: {self.current_state.flags['errors'][-1]}]"
                # Clear errors locally so they don't compound spam history each loop
                self.current_state.flags["errors"] = []
            else:
                # Dynamically update the visual status tracker reflecting standard incident workflows
                if action.action_type == "resolve_ticket":
                    self.current_observation.current_status = "resolved"
                    done = True
                elif action.action_type == "escalate_incident":
                    self.current_observation.current_status = "escalated"
                elif action.action_type == "assign_team":
                    self.current_observation.current_status = "routed"
                elif action.action_type == "classify_issue" and self.current_observation.current_status == "open":
                    self.current_observation.current_status = "triage_in_progress"
                    
            # Compute real dense transition evaluations natively mapping OpenEnv specs
            step_reward = compute_reward(self.current_state, action, old_state)
            reward += step_reward
            
        except Exception as e:
            reward -= 0.05
            self.current_observation.history.append(f"EXCEPTION: {str(e)}")
            
        self.step_count += 1
        self.current_observation.step_count = self.step_count
        
        # Determine global environment timeouts locally
        if self.step_count >= self.max_steps:
            done = True
            
        return StepResult(
            observation=self.current_observation,
            reward=reward,
            done=done,
            info=info
        )
        
    def state(self) -> Dict[str, Any]:
        """Return the complete underlying state securely avoiding mutations."""
        if self.current_state is None:
            return {}
        return {
            "internal_state": self.current_state.model_dump(),
            "observation": self.current_observation.model_dump() if self.current_observation else {}
        }
