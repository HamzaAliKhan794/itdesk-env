"""
Main entry point for the FastAPI application.
Handles API routing and server setup to interact with the OpenEnv environment.
"""

from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from itdesk_env.env import ITDeskEnv
from itdesk_env.models import Action, Observation, StepResult

app = FastAPI(
    title="IT Desk OpenEnv Environment",
    description="API for the itdesk-env OpenEnv environment",
    version="1.0.0"
)

# Global environment instance
env = ITDeskEnv()

class ResetRequest(BaseModel):
    seed: Optional[int] = None
    difficulty: Optional[str] = None

@app.get("/")
def read_root():
    """Health check endpoint."""
    return {"status": "ok", "environment": "itdesk-env"}

@app.post("/reset", response_model=Observation)
def reset_environment(req: Optional[ResetRequest] = None):
    """Resets the environment and returns the initial observation."""
    seed = None
    difficulty = None

    if req:
        seed = req.seed
        if req.difficulty:
            difficulty = req.difficulty.lower().strip()
            if difficulty not in ["easy", "medium", "hard"]:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid difficulty: '{difficulty}'. Must be 'easy', 'medium', or 'hard'."
                )
                
    try:
        step_result = env.reset(seed=seed, difficulty=difficulty)
        return step_result.observation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step", response_model=StepResult)
def step_environment(action: Action):
    """Takes an action in the environment and returns the StepResult."""
    try:
        if env.current_state is None:
            raise HTTPException(status_code=400, detail="Environment not correctly initialized. Call /reset first.")
        step_result = env.step(action)
        return step_result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/state")
def get_state() -> Dict[str, Any]:
    """Returns the full internal environment state dictionary."""
    try:
        return env.state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/score")
def get_score() -> float:
    """Returns the final grader score dynamically for the OpenEnv validator framework."""
    try:
        return env.score()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == '__main__':
    main()
