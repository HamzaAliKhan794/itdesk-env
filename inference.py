"""
Inference script for standard evaluation.
Handles running the agent against the local FastAPI environment to generate predictions/actions.
"""

import os
import json
import time
import requests
from openai import OpenAI
from itdesk_env.graders import Grader

# --- Configuration ---
ENV_URL = "http://127.0.0.1:8000"

# Defaults optimized for Hugging Face Router
DEFAULT_API_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

API_BASE_URL = os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL)
MODEL_NAME = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
API_KEY = os.getenv("API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

grader = Grader()
TASKS = ["easy", "medium", "hard"]

def get_api_key() -> str:
    """Return the prioritized API key."""
    if API_KEY:
        return API_KEY
    elif HF_TOKEN:
        return HF_TOKEN
    elif OPENAI_API_KEY:
        return OPENAI_API_KEY
    return ""

def is_llm_enabled() -> bool:
    """Determine if we actually have an API key to execute live LLM prediction calls."""
    return bool(API_KEY or HF_TOKEN or OPENAI_API_KEY)

def print_execution_mode_summary() -> None:
    """Print transparency logs about the agent inference configuration securely."""
    print("========================================")
    print(" ITDesk-Env Evaluation Initialization")
    print("========================================")
    print(f"Target Environment : {ENV_URL}")
    print(f"API Base URL       : {API_BASE_URL}")
    print(f"Target Model       : {MODEL_NAME}")
    
    if API_KEY:
        print("Execution Mode     : OpenEnv Proxy Mode (API_KEY detected)")
    elif HF_TOKEN:
        print("Execution Mode     : Hugging Face Router Mode (HF_TOKEN detected)")
    elif OPENAI_API_KEY:
        print("Execution Mode     : OpenAI Fallback Mode (OPENAI_API_KEY detected)")
    else:
        print("Execution Mode     : Fallback Heuristic Mode (No API keys mapped. Using rule-based fallback agent.)")
    print("========================================\n")

# Initialize client globally but robustly if keys exist
client = None
if is_llm_enabled():
    client = OpenAI(
        api_key=get_api_key(),
        base_url=API_BASE_URL
    )

def call_reset() -> dict:
    """Call the reset endpoint to spin up a new ticket."""
    response = requests.post(f"{ENV_URL}/reset", json={})
    response.raise_for_status()
    return response.json()

def call_step(action_dict: dict) -> dict:
    """Call the step endpoint with the formatted action."""
    response = requests.post(f"{ENV_URL}/step", json=action_dict)
    response.raise_for_status()
    return response.json()

def call_state() -> dict:
    """Retrieve full internal state for grading."""
    response = requests.get(f"{ENV_URL}/state")
    response.raise_for_status()
    return response.json()

def format_prompt(observation: dict) -> tuple[str, str]:
    """Format the observation securely into a prompt for the LLM."""
    sys_prompt = (
        "You are an IT Helpdesk Triage Agent. You must output exactly one JSON object representing your action.\n"
        "Valid action_types:\n"
        " - classify_issue (requires 'category' inside parameters. Must be one of: 'security_issue', 'network_issue', 'access_issue')\n"
        " - set_priority (requires 'severity' inside parameters. Must select exactly from 'allowed_priorities' in the observation)\n"
        " - assign_team (requires 'team' inside parameters. Must select exactly from 'allowed_teams' in the observation)\n"
        " - escalate_incident\n"
        " - flag_duplicate_cluster\n"
        " - resolve_ticket\n"
        " - noop\n\n"
        "Example output:\n"
        "{\n"
        '  "thought": "This ticket mentions critical data access loss, indicating a high severity access issue.",\n'
        '  "action_type": "classify_issue",\n'
        '  "parameters": {"category": "access_issue"}\n'
        "}\n\n"
        "CRITICAL RULES:\n"
        "1. You MUST include a 'thought' key explaining your reasoning step-by-step BEFORE the 'action_type' key.\n"
        "2. If an action requires no parameters, you MUST output an empty object {} for parameters, NEVER null.\n"
        "3. Review the 'history' in your observation. DO NOT repeat an action you have already performed. Always progress the workflow incrementally.\n"
        "4. Only output pure JSON. Do not write markdown, code blocks, or explanations."
    )
    
    user_prompt = f"Current Observation:\n{json.dumps(observation, indent=2)}"
    return sys_prompt, user_prompt

def parse_llm_response(response_text: str) -> dict:
    """Safely parse the LLM action response. Throws exception if parsing fails."""
    text = response_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
        
    parsed_json = json.loads(text.strip())
    
    thought = parsed_json.get("thought", "")
    if thought:
        print(f"\n[Agent Thought]: {thought}")
        
    if "parameters" in parsed_json and parsed_json["parameters"] is None:
        parsed_json["parameters"] = {}
        
    # Strip thought so it doesn't break strict pydantic if extra=forbid
    clean_action = {
        "action_type": parsed_json.get("action_type", "noop"),
        "parameters": parsed_json.get("parameters", {})
    }
    
    # Pass thought forward for centralized logging
    clean_action["thought"] = thought
        
    return clean_action

def choose_fallback_action(observation: dict) -> dict:
    """
    Deterministic rule-based fallback agent that progresses tickets sequentially 
    when the LLM is unavailable or fails.
    """
    history = [str(log).lower() for log in observation.get("history", [])]
    desc = str(observation.get("issue_description", "")).lower()
    signals = " ".join([str(sig).lower() for sig in observation.get("detected_signals", [])])
    systems = " ".join([str(sys).lower() for sys in observation.get("affected_systems", [])])
    
    combined_text = f"{desc} {signals} {systems}"
    similar_count = int(observation.get("similar_ticket_count", 0))
    
    # Analyze the scenario type
    is_security = any(kw in combined_text for kw in ["suspicious", "phishing", "unknown login", "foreign login", "weird email", "security alert", "ransomware", "impossible_travel", "malicious"])
    is_network = similar_count > 2 or any(kw in combined_text for kw in ["vpn", "network", "cannot connect", "outage", "packet_loss"])
    is_access = any(kw in combined_text for kw in ["password", "login", "account locked"])
    
    # State tracking using the history log
    history_str = " ".join(history)
    has_classified = "classify_issue" in history_str
    has_priority = "set_priority" in history_str
    has_team = "assign_team" in history_str
    has_escalated = "escalate_incident" in history_str
    has_flagged_dup = "flag_duplicate_cluster" in history_str
    has_resolved = "resolve_ticket" in history_str
    
    # Sequential Pipeline Steps:
    
    # 1. Classify Issue
    if not has_classified:
        category = "unclassified"
        if is_security:
            category = "security_issue"
        elif is_network:
            category = "network_issue"
        elif is_access:
            category = "access_issue"
        return {"action_type": "classify_issue", "parameters": {"category": category}}
        
    # 2. Set Priority
    if not has_priority:
        severity = "low"
        if is_security:
            severity = "critical"
        elif is_network:
            severity = "high"
        return {"action_type": "set_priority", "parameters": {"severity": severity}}
        
    # 3. Assign Correct Team
    if not has_team:
        team = "access_support" # Default
        if is_security:
            team = "security_ops"
        elif is_network:
            team = "network_team"
        return {"action_type": "assign_team", "parameters": {"team": team}}
        
    # 4. Escalate Security Issues
    if is_security and not has_escalated:
        return {"action_type": "escalate_incident", "parameters": {}}
        
    # 5. Flag Duplicate Outages
    if is_network and similar_count > 2 and not has_flagged_dup:
        return {"action_type": "flag_duplicate_cluster", "parameters": {}}
        
    # 6. Resolve Finally
    if not has_resolved:
        return {"action_type": "resolve_ticket", "parameters": {}}
        
    # Loop fallback catch
    return {"action_type": "noop", "parameters": {}}

def run_task(task_name: str) -> float:
    """Run an entire trajectory for a specific task and evaluate the final score."""
    print(f"[START] Task: {task_name.upper()}")
    
    try:
        observation = None
        # Loop reset until the randomly selected scenario matches the requested task
        for _ in range(50):
            obs = call_reset()
            task_tag_map = {"easy": "-E", "medium": "-M", "hard": "-H"}
            if task_tag_map[task_name] in obs.get("ticket_id", ""):
                observation = obs
                break
                
        if not observation:
             print(f"Failed to cleanly initialize {task_name} task.")
             return 0.0
             
        # Initialize an empty log file for dashboard when task starts
        try:
            with open("dashboard_logs.jsonl", "w") as f:
                pass
        except Exception:
            pass
             
        done = False
        step_count = 0
        max_steps = observation.get("max_steps", 15)
        
        while not done and step_count < max_steps:
            sys_prompt, user_prompt = format_prompt(observation)
            action_dict = None
            
            if is_llm_enabled():
                try:
                    completion = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.0, # Deterministic reasoning
                    )
                    response_text = completion.choices[0].message.content
                    action_dict = parse_llm_response(response_text)
                except Exception as e:
                    print(f"LLM Call Failed: {e}. Defaulting to rule-based fallback agent.")
                    action_dict = choose_fallback_action(observation)
            else:
                # LLM bypass mechanism natively triggers the fallback logic
                action_dict = choose_fallback_action(observation)
                
            print(f"[STEP] Action: {action_dict.get('action_type')} | Parameters: {action_dict.get('parameters')}")
            
            # Centralized Dashboard Telemetry Logging
            thought = action_dict.get("thought", "Determined via deterministic fallback heuristics.")
            clean_action = {
                "action_type": action_dict.get("action_type", "noop"),
                "parameters": action_dict.get("parameters", {})
            }
            dash_log = {"thought": thought, "action": clean_action}
            try:
                with open("dashboard_logs.jsonl", "a") as f:
                    f.write(json.dumps(dash_log) + "\n")
            except Exception:
                pass
            
            step_res = call_step(action_dict)
            if step_res:
                observation = step_res.get("observation", {})
                done = step_res.get("done", True)
            else:
                done = True
            step_count += 1
            time.sleep(0.1) # Prevent local thread flooding
            
        # Get final internal state for grading
        full_state = call_state()
        internal_state_dict = full_state.get("internal_state", {})
        
        # Convert dictionary recursively to Pydantic model for grader
        from itdesk_env.models import State
        final_state = State(**internal_state_dict)
        
        score = grader.evaluate(final_state, task_name)
        print(f"[END] Task {task_name.upper()} Score: {score}")
        return score
        
    except Exception as e:
        print(f"Error during execution of {task_name}: {e}")
        return 0.0

def main():
    print_execution_mode_summary()
    total_score = 0.0
    scores = {}
    
    for task in TASKS:
        score = run_task(task)
        scores[task] = score
        total_score += score
        
    print("\n====================")
    print("EVALUATION COMPLETE")
    print("====================")
    for task, score in scores.items():
        print(f" - {task.capitalize()}: {score:.2f} / 1.00")
        
    print(f"Total Cumulative Score: {total_score:.2f} / {len(TASKS)}.00")

if __name__ == "__main__":
    main()
