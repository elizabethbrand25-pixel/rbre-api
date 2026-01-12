"""
RBRE API (FastAPI) - Tally webhook friendly

- Loads cost_data_metro.json at startup
- Accepts Tally webhook payloads without requiring exact question wording
- Maps metro LABEL -> CBSA (or accepts CBSA directly)
- Returns JSON (MVP). Later you can add PDF/DocRaptor endpoints.

Render start command:
    uvicorn app:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional, List

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse


# -------------------------------------------------
# App initialization
# -------------------------------------------------
app = FastAPI(title="RBRE API", version="1.0")


# -------------------------------------------------
# Helpers: tolerant field mapping from Tally
# -------------------------------------------------
def _norm_key(s: str) -> str:
    """Normalize keys so small differences in punctuation/case/spacing don't matter."""
    s = str(s).strip().lower()
    s = re.sub(r"[\?\(\)\[\]\:\-]", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in {"yes", "y", "true", "1", "on", "affirmative"}


def _as_float(v: Any) -> float:
    # Handles "6,500" or "$6500"
    s = str(v).strip().replace(",", "")
    s = re.sub(r"[^0-9.\-]", "", s)
    if s in {"", "-", ".", "-."}:
        raise ValueError(f"Cannot parse numeric value from {v!r}")
    return float(s)


def _build_norm_map(data: Dict[str, Any]) -> Dict[str, str]:
    """Map normalized keys -> original keys (first occurrence wins)."""
    out: Dict[str, str] = {}
    for k in data.keys():
        nk = _norm_key(k)
        if nk and nk not in out:
            out[nk] = k
    return out


def pick(data: Dict[str, Any], aliases: List[str], required: bool = True) -> Optional[Any]:
    """
    Fetch a value from Tally 'data' dict by trying multiple aliases.
    Matching is case/punctuation/space-insensitive.
    """
    if not isinstance(data, dict):
        if required:
            raise ValueError("Payload 'data' is not a dict.")
        return None

    norm_map = _build_norm_map(data)
    for a in aliases:
        nk = _norm_key(a)
        if nk in norm_map:
            return data[norm_map[nk]]

    if required:
        raise KeyError(f"Missing field. Tried aliases={aliases}")
    return None


# -------------------------------------------------
# Load cost data at startup
# -------------------------------------------------
DATA_FILE = "cost_data_metro.json"

if not os.path.exists(DATA_FILE):
    # Raising here will prevent the app from starting (and Render will show import/start errors)
    raise RuntimeError(
        "cost_data_metro.json not found in the repo root. "
        "Make sure it's committed to GitHub at the top level."
    )

with open(DATA_FILE, "r", encoding="utf-8") as f:
    COST_DATA: Dict[str, Dict[str, Any]] = json.load(f)

# Build label -> CBSA lookup for Tally submissions
LABEL_TO_CBSA: Dict[str, str] = {}
for cbsa, prof in COST_DATA.items():
    label = prof.get("metro_name")
    if isinstance(label, str) and label.strip():
        LABEL_TO_CBSA[label.strip()] = cbsa


def label_or_cbsa_to_cbsa(x: Any) -> Optional[str]:
    """Accept CBSA directly or map a metro label to CBSA."""
    if x is None:
        return None
    s = str(x).strip()
    if s in COST_DATA:
        return s
    return LABEL_TO_CBSA.get(s)


# -------------------------------------------------
# Health check
# -------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "alive", "metros_loaded": len(COST_DATA)}


# -------------------------------------------------
# Tally webhook endpoint (JSON MVP)
# -------------------------------------------------
@app.post("/v1/report.json")
async def generate_report(request: Request):
    """
    Receives a Tally webhook payload.
    Tally typically sends: {"data": { "<question title>": <answer>, ... }, ...}

    This endpoint maps various question title variants to the required fields.
    """
    payload = await request.json()
    data = payload.get("data")

    if not isinstance(data, dict):
        raise HTTPException(
            status_code=400,
            detail={"error": "Invalid Tally payload: expected payload.data dict", "payload_keys": list(payload.keys())},
        )

    # ---- Pull fields using tolerant alias lists ----
    try:
        household_type = pick(
            data,
            aliases=[
                "Household type",
                "household",
                "household type (individual/couple)",
                "individual or couple",
                "household_type",
            ],
            required=True,
        )

        downsizing_raw = pick(
            data,
            aliases=[
                "Are you downsizing?",
                "downsizing",
                "are you downsizing",
                "downsizing?",
                "downsizing status",
                "downsizing_flag",
            ],
            required=True,
        )

        net_income_raw = pick(
            data,
            aliases=[
                "Net monthly income",
                "monthly income",
                "net income",
                "income (monthly)",
                "net_monthly_income",
            ],
            required=True,
        )

        fixed_costs_raw = pick(
            data,
            aliases=[
                "Fixed monthly obligations",
                "fixed obligations",
                "monthly obligations",
                "monthly bills",
                "fixed_monthly_obligations",
            ],
            required=True,
        )

        savings_raw = pick(
            data,
            aliases=[
                "Liquid savings available",
                "liquid savings",
                "savings available",
                "cash savings",
                "liquid_savings",
            ],
            required=True,
        )

        timeline = pick(
            data,
            aliases=[
                "Timeline",
                "move timeline",
                "how soon",
                "timeline (tight/moderate/flexible)",
            ],
            required=True,
        )

        risk_tolerance = pick(
            data,
            aliases=[
                "Risk tolerance",
                "risk",
                "risk level",
                "risk tolerance (low/medium/high)",
            ],
            required=True,
        )

        current_metro_label = pick(
            data,
            aliases=[
                "Current metro area",
                "current metro",
                "current location",
                "current cbsa",
                "current_cbsa",
            ],
            required=True,
        )

        target_labels = pick(
            data,
            aliases=[
                "Metro areas you are considering (optional)",
                "metros considering",
                "target metros",
                "target metro areas",
                "target_cbsas",
            ],
            required=False,
        )

        email = pick(
            data,
            aliases=[
                "Email address",
                "email",
                "e-mail",
            ],
            required=True,
        )

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Could not map Tally fields to API inputs",
                "reason": str(e),
                "received_keys": sorted(list(data.keys())),
            },
        )

    # ---- Normalize types ----
    try:
        downsizing = _as_bool(downsizing_raw)
        net_income = _as_float(net_income_raw)
        fixed_costs = _as_float(fixed_costs_raw)
        savings = _as_float(savings_raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail={"error": "Invalid numeric/boolean value", "reason": str(e)})

    # Targets can arrive as a single string, list, null, etc.
    if target_labels is None:
        target_labels_list: List[str] = []
    elif isinstance(target_labels, str):
        target_labels_list = [target_labels]
    elif isinstance(target_labels, list):
        target_labels_list = [str(x) for x in target_labels if x is not None]
    else:
        target_labels_list = []

    # ---- Map metros to CBSA ----
    current_cbsa = label_or_cbsa_to_cbsa(current_metro_label)
    if not current_cbsa:
        raise HTTPException(
            status_code=400,
            detail={"error": "Unknown current metro (label not found and not a CBSA code)", "value": current_metro_label},
        )

    target_cbsas: List[str] = []
    for t in target_labels_list:
        cbsa = label_or_cbsa_to_cbsa(t)
        if cbsa:
            target_cbsas.append(cbsa)

    # -------------------------------------------------
    # MVP RESPONSE (replace with your real engine later)
    # -------------------------------------------------
    result = {
        "status": "processed",
        "email": str(email).strip(),
        "household_type": household_type,
        "downsizing": downsizing,
        "net_monthly_income": net_income,
        "fixed_monthly_obligations": fixed_costs,
        "liquid_savings": savings,
        "timeline": timeline,
        "risk_tolerance": risk_tolerance,
        "current_cbsa": current_cbsa,
        "target_cbsas": target_cbsas,
    }

    return JSONResponse(content=result)
