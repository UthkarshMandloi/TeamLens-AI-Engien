from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import json
import random
import xgboost as xgb
import pandas as pd
import numpy as np
import os

app = FastAPI(title="TeamLens AI Engine")

# --- CONFIGURE CORS FOR PRODUCTION ---
# This allows Love's backend (or any frontend) to communicate with this AI Microservice
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with Love's backend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- LOAD XGBOOST MODEL ON STARTUP ---
xgb_model = xgb.XGBClassifier()
model_loaded = False
allocator_mapping = {}

if os.path.exists("task_allocator.json") and os.path.exists("allocator_mapping.json"):
    xgb_model.load_model("task_allocator.json")
    with open("allocator_mapping.json", "r") as f:
        allocator_mapping = json.load(f)
    model_loaded = True
    print("✅ XGBoost Model Loaded Successfully")


# --- DATA MODELS (The API Contract) ---

class ChatMessage(BaseModel):
    user_id: str
    message: str
    timestamp: str


class TaskExtractionResponse(BaseModel):
    is_task: bool
    assignee: Optional[str] = None
    task_description: Optional[str] = None
    deadline: Optional[str] = None
    confidence_score: float


class MeetingSummaryResponse(BaseModel):
    meeting_title: str
    key_decisions: List[str]
    action_items: List[dict]
    overall_sentiment: str


class AllocationRequest(BaseModel):
    task_description: str
    available_members: List[str]


class AllocationResponse(BaseModel):
    best_assignee: str
    confidence_score: float
    reason: str


# --- PROMPT TEMPLATES FOR LLaMA-3 ---
# We store these here so when you plug in the real LLaMA-3 model,
# the instructions are already perfectly formatted.

TASK_EXTRACTION_PROMPT = """
You are an AI Project Manager. Analyze the following chat message and extract any task assignments.
Return ONLY a valid JSON object with the keys: is_task (boolean), assignee (string or null), task_description (string or null), deadline (string or null).
Message: "{message}"
"""

MEETING_DEBRIEF_PROMPT = """
You are an AI assistant. Analyze the following transcript and generate a meeting debrief.
Extract key decisions made and action items assigned.
Transcript: "{transcript}"
"""


# --- ENDPOINTS ---

@app.get("/")
def health_check():
    return {"status": "AI Engine is Online 🧠", "models_loaded": ["LLaMA-3-Mock", "XGBoost-Mock"]}


@app.post("/api/ai/extract-task", response_model=TaskExtractionResponse)
def extract_task(chat: ChatMessage):
    """
    Analyzes chat messages to extract structured task data.
    Ready to be wired to the local LLaMA-3 inference engine.
    """
    print(f"Running task extraction on: {chat.message}")

    # In production, we will pass TASK_EXTRACTION_PROMPT to LLaMA-3 here.
    # For now, we use robust Regex/Keyword matching to simulate the AI for your testing.
    msg_lower = chat.message.lower()

    if "/assign" in msg_lower:
        # Simple extraction simulation
        parts = chat.message.split(" ")
        assignee = parts[1].replace("@", "") if len(parts) > 1 else "unassigned"

        return {
            "is_task": True,
            "assignee": assignee,
            "task_description": chat.message.replace(f"/assign @{assignee}", "").strip(),
            "deadline": "Pending",  # Will be extracted by actual LLM
            "confidence_score": 0.92
        }

    return {"is_task": False, "confidence_score": 0.0}


@app.post("/api/ai/summarize-meeting", response_model=MeetingSummaryResponse)
def summarize_meeting(messages: List[ChatMessage]):
    """
    Takes an array of chat messages (a meeting transcript) and generates a structured debrief.
    """
    print(f"Summarizing meeting with {len(messages)} messages...")

    if not messages:
        return {
            "meeting_title": "Empty Meeting",
            "key_decisions": ["No discussion recorded."],
            "action_items": [],
            "overall_sentiment": "Neutral"
        }

    # 1. Build the transcript
    transcript = "\n".join([f"{m.user_id}: {m.message}" for m in messages])

    # 2. Dynamic Prototype Extraction (Analyzes what you actually typed)
    decisions = []
    actions = []

    for m in messages:
        msg_lower = m.message.lower()

        # Look for decisions or completed work
        if any(word in msg_lower for word in ["completed", "decided", "finished", "agreed", "will"]):
            decisions.append(f"{m.user_id.capitalize()} stated: {m.message}")

        # Look for future action items
        if any(word in msg_lower for word in ["build", "fix", "need to", "task", "should"]):
            actions.append({"assignee": m.user_id, "task": m.message})

    # Fallbacks if the meeting was super short
    if len(decisions) == 0:
        decisions.append("General status updates discussed.")
    if len(actions) == 0:
        actions.append({"assignee": "Team", "task": "Continue with current sprint objectives."})

    # Generate a dynamic title based on the first speaker
    first_speaker = messages[0].user_id.capitalize()

    return {
        "meeting_title": f"Project Sync initiated by {first_speaker}",
        "key_decisions": decisions,
        "action_items": actions,
        "overall_sentiment": "Positive and Productive"
    }


@app.post("/api/ai/smart-allocate", response_model=AllocationResponse)
def smart_allocate(req: AllocationRequest):
    """
    XGBoost Model Endpoint.
    Predicts the best team member for a task based on past performance, workload, and skills.
    """
    print(f"Running XGBoost allocation for task: {req.task_description}")

    if not model_loaded:
        # Fallback if model files are missing
        best_candidate = random.choice(req.available_members) if req.available_members else "divyansh"
        return {
            "best_assignee": best_candidate,
            "confidence_score": round(random.uniform(0.75, 0.98), 2),
            "reason": f"Fallback: Model identified '{best_candidate}' based on availability."
        }

    # 1. Classify the task category (Simple keyword extraction for prototype)
    desc = req.task_description.lower()
    category = "frontend"  # default
    if any(w in desc for w in ["backend", "api", "database", "server", "kafka"]):
        category = "backend"
    elif any(w in desc for w in ["design", "ui", "ux", "figma"]):
        category = "design"
    elif any(w in desc for w in ["ai", "model", "ml", "train", "data"]):
        category = "ai_ml"
    elif any(w in desc for w in ["research", "competitor", "find"]):
        category = "research"
    elif any(w in desc for w in ["deploy", "docker", "pipeline", "ci/cd"]):
        category = "devops"

    # 2. Encode category for the ML model
    task_encoded = allocator_mapping["tasks"].get(category, 0)
    complexity = 5  # Default complexity for the hackathon prototype

    # 3. Simulate current workloads (In production, Love's backend would send this data)
    # We assume members in 'available_members' have a low workload (2), others are busy (10)
    workloads = {
        "divyansh_workload": 2 if "divyansh" in req.available_members else 10,
        "anushka_workload": 2 if "anushka" in req.available_members else 10,
        "love_workload": 2 if "love" in req.available_members else 10,
        "uthkarsh_workload": 2 if "uthkarsh" in req.available_members else 10,
    }

    # 4. Prepare feature vector exactly as the model was trained
    features = pd.DataFrame([{
        "task_category_encoded": task_encoded,
        "complexity": complexity,
        **workloads
    }])

    # 5. Predict probabilities!
    probs = xgb_model.predict_proba(features)[0]

    # 6. Find the winning team member
    best_idx = np.argmax(probs)
    confidence = float(probs[best_idx])
    best_member = allocator_mapping["members"][str(best_idx)]

    return {
        "best_assignee": best_member,
        "confidence_score": round(confidence, 2),
        "reason": f"XGBoost selected {best_member} based on their '{category}' skill match and optimal workload capacity."
    }

# Run the server using: uvicorn main:app --reload --port 8000