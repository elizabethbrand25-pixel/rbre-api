"""
RBRE API – Tally-safe FastAPI app

Fixes:
- Tally webhook payloads may NOT be {"data": {...}}.
- Answers often arrive in payload["fields"] as a LIST of objects.
- This app extracts answers into a flat dict: {question_label: value}
- Logs:
  - payload keys
  - parsed answer keys
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

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


def _extract_tally_answers(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns (answers_dict, debug_info)

    Supports several common webhook shapes:
    1) payload["data"] is a dict of answers
    2) payload["data"]["fields"] is a list of objects
    3) payload["fields"] is a list of objects  (your case)
    4) payload["answers"] is a dict

    A "field object" might look like:
      {"label":"Net monthly income","value":6500}
    or sometimes:
      {"title":"Net monthly income","answer":6500}
    or nested:
      {"field": {"label":"..."}, "value": ...}
    """
    debug = {"payload_keys": sorted(list(payload.keys()))}

    # Candidate containers in priority order
    candidates: List[Any] = []
    if "data" in payload:
        candidates.append(payload.get("data"))
        if isinstance(payload.get("data"), dict) and "fields" in payload["data"]:
            candidates.append(payload["data"].get("fields"))
    if "fields" in payload:
        candidates.append(payload.get("fields"))
    if "answers" in payload:
        candidates.append(payload.get("answers"))

    # Case A: direct dict of answers
    for c in candidates:
        if isinstance(c, dict):
            # If it looks like answers (not just metadata), accept it.
            # If it contains "fields" and other metadata, we'll parse fields separately.
            if any(k for k in c.keys() if _norm_key(k) in {"net monthly income", "email", "household type"}):
                debug["shape"] = "dict_answers"
                return c, debug

    # Case B: list of field objects (most common for modern Tally webhooks)
    fields_list = None
    for c in candidates:
        if isinstance(c, list):
            fields_list = c
            break

    answers: Dict[str, Any] = {}
    if isinstance(fields_list, list):
        for item in fields_list:
            if not isinstance(item, dict):
                continue

            # Try a few common label/value shapes
            label = None
            value = None

            if "label" in item:
                label = item.get("label")
            elif "title" in item:
                label = item.get("title")
            elif isinstance(item.get("field"), dict) and "label" in item["field"]:
                label = item["field"].get("label")
            elif isinstance(item.get("field"), dict) and "title" in item["field"]:
                label = item["field"].get("title")

            if "value" in item:
                value = item.get("value")
            elif "answer" in item:
                value = item.get("answer")
            elif "response" in item:
                value = item.get("response")

            if isinstance(label, str) and label.strip():
                answers[label.strip()] = value

        debug["shape"] = "fields_list"
        debug["fields_count"] = len(fields_list)
        return answers, debug

    # Nothing matched
    debug["shape"] = "unknown"
    return {}, debug


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
    return {"status": "alive", "metros_loaded": len(COST_DATA)}


# -------------------------------------------------
# Tally webhook endpoint
# -------------------------------------------------
@app.post("/v1/report.json")
async def generate_report(request: Request):
    payload = await request.json()

    # Extract answers robustly
    answers, dbg = _extract_tally_answers(payload)

    # Log payload keys + parsed answer keys (Render logs)
    print("TALLY PAYLOAD KEYS:", dbg.get("payload_keys"))
    print("TALLY SHAPE:", dbg.get("shape"), "FIELDS_COUNT:", dbg.get("fields_count"))
    print("TALLY ANSWER KEYS:", sorted(list(answers.keys())))

    if not answers:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Could not extract answers from Tally payload",
                "debug": dbg,
                "payload_keys": dbg.get("payload_keys"),
            },
        )

    try:
        household_type = pick(
            answers,
            ["household type", "household", "individual or couple"],
        )

        downsizing_raw = pick(
            answers,
            ["are you downsizing", "downsizing"],
        )

        net_income_raw = pick(
            answers,
            ["net monthly income", "monthly income", "income"],
        )

        fixed_costs_raw = pick(
            answers,
            ["fixed monthly obligations", "monthly obligations", "fixed costs", "obligations"],
        )

        savings_raw = pick(
            answers,
            ["liquid savings available", "savings", "cash savings"],
        )

        timeline = pick(
            answers,
            ["timeline", "move timeline"],
        )

        risk = pick(
            answers,
            ["risk tolerance", "risk"],
        )

        current_metro_label = pick(
            answers,
            ["current metro area", "current metro", "current location"],
        )

        target_labels = pick(
            answers,
            ["metro areas you are considering", "target metros", "metros considering"],
            required=False,
        )

        email = pick(
            answers,
            ["email address", "email", "e mail"],
        )

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Field mapping failed",
                "reason": str(e),
                "received_answer_keys": sorted(list(answers.keys())),
            },
        )

    # Normalize
    downsizing = _as_bool(downsizing_raw)
    net_income = _as_float(net_income_raw)
    fixed_costs = _as_float(fixed_costs_raw)
    savings = _as_float(savings_raw)

    # targets normalization
    if isinstance(target_labels, str):
        target_labels = [target_labels]
    if not isinstance(target_labels, list):
        target_labels = []

    # Map metro label -> CBSA
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
