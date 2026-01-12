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
    payload = await request.json()

    data = payload.get("data")
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="Invalid Tally payload")

    try:
        household_type = data.get("Household Type?")
        downsizing_raw = data.get("Are You Downsizing?")
        net_income = float(data.get("Net Monthly Income"))
        fixed_costs = float(data.get("Fixed Monthly Obligations"))
        savings = float(data.get("Liquid Savings Available"))
        timeline = data.get("Timeline (How soon do you want to relocate?)")
        risk = data.get("Risk Tolerance")
        current_metro_label = data.get("Current Metro Area")
        target_labels = data.get("Metro Areas You Are Considering (Optional)", [])
        email = data.get("Email Address? (So we can share our insight!)")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Missing or invalid fields in Tally submission"
        )

    downsizing = str(downsizing_raw).lower() == "yes"

    if isinstance(target_labels, str):
        target_labels = [target_labels]
    if not isinstance(target_labels, list):
        target_labels = []

    if current_metro_label not in LABEL_TO_CBSA:
        raise HTTPException(status_code=400, detail="Unknown current metro")

    current_cbsa = LABEL_TO_CBSA[current_metro_label]

    target_cbsas = [
        LABEL_TO_CBSA[label]
        for label in target_labels
        if label in LABEL_TO_CBSA
    ]

    result = {
        "household_type": household_type,
        "downsizing": downsizing,
        "net_monthly_income": net_income,
        "fixed_monthly_obligations": fixed_costs,
        "liquid_savings": savings,
        "timeline": timeline,
        "risk_tolerance": risk,
        "current_cbsa": current_cbsa,
        "target_cbsas": target_cbsas,
        "email": email,
        "status": "processed"
    }

    return JSONResponse(content=result)
