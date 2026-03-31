# ITDesk-Env

ITDesk-Env is a production-ready OpenEnv simulation framework that evaluates Large Language Models (LLMs) and autonomous agents on realistic IT Helpdesk triage and incident management tasks.

## Problem Description
As organizations scale, IT support teams are frequently overwhelmed by a massive influx of daily tickets ranging from simple password resets to critical security breaches. Autonomous AI agents present a promising solution to triage, classify, and resolve these incidents rapidly, but they require rigorous, structured environments to evaluate their decision-making accuracy before being deployed into live corporate systems.

## Real-World Motivation
Deploying untrained AI models into an IT support pipeline risks severe consequences, such as misclassifying a ransomware attack as a routine network drop or assigning a critical VPN failure to a lower-tier hardware team. ITDesk-Env bridges this gap by offering a fully deterministic, strictly evaluated sandbox for agents to safely practice complex diagnostic logic, duplicate detection, and correct escalation chains.

## Observation Space
The environment provides the agent with a deeply structured, JSON-serialized `Observation` object designed to mirror realistic IT ticketing systems. It includes:
* **Metadata**: `ticket_id`, `user_role`, `issue_description`
* **Telemetry**: `detected_signals`, `affected_systems`, `similar_ticket_count`
* **Agent Constraints**: `allowed_teams`, `allowed_priorities`, `max_steps`
* **State Trackers**: `possible_duplicate`, `history`, `current_status`, `step_count`, `goal`

## Action Space
The agent can act upon the environment uniquely using a strict `Action` JSON schema. The `action_type` parameter is firmly constrained by `Literal` type expectations to prevent hallucinations:
* `classify_issue`
* `set_priority`
* `assign_team`
* `flag_duplicate_cluster`
* `request_missing_context`
* `escalate_incident`
* `resolve_ticket`
* `close_as_duplicate`
* `noop`

## Reward Design
Rewards are bounded strictly between `-1.0` and `1.0` progressively at each step to incentivize precision and speed:
* **Positive Progress**: Correct classification (`+0.2`), correct priority (`+0.2`), correct team assignment (`+0.25`), successful valid resolution (`+0.15`).
* **Negative Penalties**: Wasting steps takes a mandatory `-0.05` toll. Severe logic failures, such as assigning an incorrect team based on ticket heuristics (`-0.2`) or failing to escalate a critical security incident (`-0.4`), incur heavy point deductions.

## Task Scenarios
The environment stubs three deterministic tasks matching `easy`, `medium`, and `hard` designations.
1. **Easy (Routine Triage)**: A single low-complexity user complains of a login failure. The agent must successfully recognize it and assign it to `access_support` without unnecessary convolution.
2. **Medium (Duplicate Cluster)**: A department director complains of VPN dropouts along with a telemetry alert showing `similar_ticket_count = 5`. The agent must detect the duplicate cluster, assign it high priority, and dispatch to the `network_team`.
3. **Hard (Severe Security Breach)**: A CFO triggers ransomware heuristics via an invoice phishing link. The agent must flawlessly identify the critical telemetry, enforce `escalate_incident`, and route directly to `security_ops`.

## Grading Methodology
Beyond the cumulative fractional rewards within the loop, the absolute performance of an agent's trajectory is evaluated deterministically by the `Grader`. Each task variant carries additive internal checks validating the correct pipeline flags (`classification`, `priority`, `assigned_team`, `escalation`, and terminal `resolution`). Final grading scores perfectly cap at `1.0` if no components were missed or wrongly typed.

---

## Setup Instructions
Clone the repository, create a virtual environment, and install the base components needed:
```bash
git clone https://github.com/yourusername/itdesk-env.git
cd itdesk-env
pip install -r requirements.txt
```

## Running Locally
Spool up the native FastAPI simulation instance using `uvicorn`:
```bash
uvicorn app:app --reload
```
You can then run the built-in evaluation wrapper `inference.py` to test an OpenAI model against the tasks!

## Docker Instructions
ITDesk-Env is fully containerized for scalable architecture. You can easily build and spin it up:
```bash
docker build -t itdesk-env .
docker run -p 7860:7860 itdesk-env
```

## HuggingFace Space Deployment
The environment natively binds to HuggingFace port standard (`7860`).
1. Create a new HuggingFace Space and select **Docker** as your Space SDK.
2. Upload this repository straight to your space (or sync it via GitHub).
3. The Space will automatically build the `Dockerfile` and expose the API natively! You can then point your autonomous agents at the remote HuggingFace Space URL (e.g. `https://username-itdesk-env.hf.space`).
