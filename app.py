"""
RBRE API – Tally-safe FastAPI app

- Accepts Tally webhook payloads (data / fields / answers)
- Logs exact keys received from Tally
- Tolerant field mapping (no strict question text dependency)
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse


# -------------------------------------------------
# App initialization
# -------------------------------------------------
app = FastAPI(title="RBRE API", version="1.0")


# -------------------------------------------------
# Helper functions
# -------------------------------------------------
def _norm_key(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"yes", "y", "true", "1", "on"}


def _as_float(v: Any) -> float:
    s = str(v).replace(",", "")
    s = re.sub(r"[^0-9.\-]", "", s)
    if s == "":
        raise ValueError("Empty numeric value")
    return float(s)


def pick(data: Dict[str, Any], aliases: List[str], required: bool = True) -> Optional[Any]:
    norm_map = {_norm_key(k): k for k in data.keys()}

    for alias in aliases:
        nk = _norm_key(alias)
        if nk in norm_map:
            return data[norm_map[nk]]

    if required:
        raise KeyError(f"Missing field (aliases tried: {aliases})")
    return None


# -------------------------------------------------
# Load metro cost data
# -------------------------------------------------
DATA_FILE = "cost_data_metro.json"

if not os.path.exists(DATA_FILE):
    raise RuntimeError("cost_data_metro.json not found in repo root")

with open(DATA_FILE, "r", encoding="utf-8") as f:
    COST_DATA: Dict[str, Dict[str, Any]] = json.load(f)

LABEL_TO_CBSA: Dict[str, str] = {}
for cbsa, prof in COST_DATA.items():
    label = prof.get("metro_name")
    if isinstance(label, str):
        LABEL_TO_CBSA[label.strip()] = cbsa


def label_or_cbsa_to_cbsa(x: Any) -> Optional[str]:
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
def health():
    return {
        "status": "alive",
        "metros_loaded": len(COST_DATA),
    }


# -------------------------------------------------
# Tally webhook endpoint
# -------------------------------------------------
@app.post("/v1/report.json")
async def generate_report(request: Request):
    payload = await request.json()

    # 🔑 IMPORTANT CHANGE:
    # Tally may send answers under different keys
    data = payload.get("data") or payload.get("fields") or payload.get("answers")

    if not isinstance(data, dict):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid Tally payload",
                "payload_keys": list(payload.keys()),
            },
        )

    # 🔍 DEBUG: log exactly what Tally sent
    print("TALLY RECEIVED KEYS:", sorted(list(data.keys())))

    try:
        household_type = pick(
            data,
            ["household type", "household", "individual or couple"],
        )

        downsizing_raw = pick(
            data,
            ["are you downsizing", "downsizing"],
        )

        net_income_raw = pick(
            data,
            ["net monthly income", "monthly income", "income"],
        )

        fixed_costs_raw = pick(
            data,
            ["fixed monthly obligations", "monthly obligations", "fixed costs"],
        )

        savings_raw = pick(
            data,
            ["liquid savings available", "savings", "cash savings"],
        )

        timeline = pick(
            data,
            ["timeline", "move timeline"],
        )

        risk = pick(
            data,
            ["risk tolerance", "risk"],
        )

        current_metro_label = pick(
            data,
            ["current metro area", "current metro", "current location"],
        )

        target_labels = pick(
            data,
            ["metro areas you are considering", "target metros"],
            required=False,
        )

        email = pick(
            data,
            ["email address", "email"],
        )

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Field mapping failed",
                "reason": str(e),
                "received_keys": sorted(list(data.keys())),
            },
        )

    # Normalize values
    downsizing = _as_bool(downsizing_raw)
    net_income = _as_float(net_income_raw)
    fixed_costs = _as_float(fixed_costs_raw)
    savings = _as_float(savings_raw)

    if isinstance(target_labels, str):
        target_labels = [target_labels]
    if not isinstance(target_labels, list):
        target_labels = []

    current_cbsa = label_or_cbsa_to_cbsa(current_metro_label)
    if not current_cbsa:
        raise HTTPException(
            status_code=400,
            detail={"error": "Unknown current metro", "value": current_metro_label},
        )

    target_cbsas: List[str] = []
    for t in target_labels:
        cbsa = label_or_cbsa_to_cbsa(t)
        if cbsa:
            target_cbsas.append(cbsa)

    return JSONResponse(
        content={
            "status": "processed",
            "email": email,
            "household_type": household_type,
            "downsizing": downsizing,
            "net_monthly_income": net_income,
            "fixed_monthly_obligations": fixed_costs,
            "liquid_savings": savings,
            "timeline": timeline,
            "risk_tolerance": risk,
            "current_cbsa": current_cbsa,
            "target_cbsas": target_cbsas,
        }
    )
