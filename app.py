from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import json
import os

# -------------------------------------------------
# App initialization
# -------------------------------------------------
app = FastAPI(title="RBRE API", version="1.0")


# -------------------------------------------------
# Load cost data at startup
# -------------------------------------------------
DATA_FILE = "cost_data_metro.json"

if not os.path.exists(DATA_FILE):
    raise RuntimeError("cost_data_metro.json not found. Make sure it is in the repo root.")

with open(DATA_FILE, "r", encoding="utf-8") as f:
    COST_DATA = json.load(f)

# Build label -> CBSA lookup for Tally submissions
LABEL_TO_CBSA = {}
for cbsa, prof in COST_DATA.items():
    label = prof.get("metro_name")
    if label:
        LABEL_TO_CBSA[label.strip()] = cbsa


# -------------------------------------------------
# Health check (Render + sanity test)
# -------------------------------------------------
@app.get("/")
def health_check():
    return {
        "status": "alive",
        "metros_loaded": len(COST_DATA)
    }


# -------------------------------------------------
# Tally webhook endpoint (JSON first)
# -------------------------------------------------
@app.post("/v1/report.json")
async def generate_report(request: Request):
    """
    Receives a Tally webhook payload, maps it to engine input,
    and returns a structured JSON response.
    """

    payload = await request.json()

    # Tally wraps answers inside "data"
    data = payload.get("data")
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="Invalid Tally payload")

    # ---- Extract fields by QUESTION LABEL ----
    try:
        household_type = data.get("Household type")
        downsizing = data.get("Are you downsizing?")
        net_income = float(data.get("Net monthly income"))
        fixed_costs = float(data.get("Fixed monthly obligations"))
        savings = float(data.get("Liquid savings available"))
        timeline = data.get("Timeline")
        risk = data.get("Risk tolerance")
        current_metro_label = data.get("Current metro area")
        target_labels = data.get("Metro areas you are considering (optional)", [])
except Exception:
    raise HTTPException(status_code=400, detail="Missing or invalid fields in Tally submission")


