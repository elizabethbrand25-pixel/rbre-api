"""
RBRE API – Tally-safe FastAPI app (FULL FILE)

What this version does:
- Extracts answers from Tally payloads where answers arrive as payload["data"]["fields"] (list)
- Logs:
  - TALLY PAYLOAD KEYS
  - TALLY SHAPE + FIELDS_COUNT
  - TALLY ANSWER KEYS (flattened question labels)
- Maps your ACTUAL Tally question labels (including punctuation + extra wording)
- Returns JSON (MVP)

Render start command:
  uvicorn app:app --host 0.0.0.0 --port $PORT
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
# Normalization helpers
# -------------------------------------------------
def _norm_key(s: str) -> str:
    """Normalize keys so casing/punctuation/extra spaces don't matter."""
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"yes", "y", "true", "1", "on"}


def _as_float(v: Any) -> float:
    # Accepts "6,500", "$6500", etc.
    s = str(v).strip().replace(",", "")
    s = re.sub(r"[^0-9.\-]", "", s)
    if s == "":
        raise ValueError(f"Empty numeric value from {v!r}")
    return float(s)


def pick(data: Dict[str, Any], aliases: List[str], required: bool = True) -> Optional[Any]:
    """
    Fetch a value from dict using multiple aliases.
    Matching is case/punctuation/space-insensitive.
    """
    norm_map = {_norm_key(k): k for k in data.keys()}

    for alias in aliases:
        nk = _norm_key(alias)
        if nk in norm_map:
            return data[norm_map[nk]]

    if required:
        raise KeyError(f"Missing field. Tried aliases={aliases}")
    return None


# -------------------------------------------------
# Tally answer extraction
# -------------------------------------------------
def _extract_tally_answers(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns (answers_dict, debug_info)

    Supports common shapes:
    - payload["data"] is dict answers (rare)
    - payload["data"]["fields"] is list of objects (common)
    - payload["fields"] is list (sometimes)
    - payload["answers"] is dict (sometimes)

    We flatten into: {"Question Label": value, ...}
    """
    debug: Dict[str, Any] = {"payload_keys": sorted(list(payload.keys()))}

    # Gather candidates
    candidates: List[Any] = []
    if "data" in payload:
        candidates.append(payload.get("data"))
        if isinstance(payload.get("data"), dict) and "fields" in payload["data"]:
            candidates.append(payload["data"].get("fields"))
    if "fields" in payload:
        candidates.append(payload.get("fields"))
    if "answers" in payload:
        candidates.append(payload.get("answers"))

    # If any candidate is a dict that already looks like answers, use it
    for c in candidates:
        if isinstance(c, dict):
            # Heuristic: if it contains at least one expected-ish key, treat as answers dict
            likely = {"net monthly income", "email address", "household type", "current metro area"}
            if any(_norm_key(k) in likely for k in c.keys()):
                debug["shape"] = "dict_answers"
                return c, debug

    # Otherwise, look for a list of field objects
    fields_list: Optional[List[Any]] = None
    for c in candidates:
        if isinstance(c, list):
            fields_list = c
            break

    answers: Dict[str, Any] = {}
    if isinstance(fields_list, list):
        for item in fields_list:
            if not isinstance(item, dict):
                continue

            # Find label
            label = None
            if "label" in item:
                label = item.get("label")
            elif "title" in item:
                label = item.get("title")
            elif isinstance(item.get("field"), dict) and "label" in item["field"]:
                label = item["field"].get("label")
            elif isinstance(item.get("field"), dict) and "title" in item["field"]:
                label = item["field"].get("title")

            # Find value
            value = None
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

    debug["shape"] = "unknown"
    return {}, debug


# -------------------------------------------------
# Load metro cost data
# -------------------------------------------------
DATA_FILE = "cost_data_metro.json"

if not os.path.exists(DATA_FILE):
    raise RuntimeError("cost_data_metro.json not found in repo root (commit it to GitHub).")

with open(DATA_FILE, "r", encoding="utf-8") as f:
    COST_DATA: Dict[str, Dict[str, Any]] = json.load(f)

LABEL_TO_CBSA: Dict[str, str] = {}
for cbsa, prof in COST_DATA.items():
    label = prof.get("metro_name")
    if isinstance(label, str) and label.strip():
        LABEL_TO_CBSA[label.strip()] = cbsa


def label_or_cbsa_to_cbsa(x: Any) -> Optional[str]:
    """Accept CBSA code directly or map metro label to CBSA."""
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

    answers, dbg = _extract_tally_answers(payload)

    # Logs (Render -> Logs)
    print("TALLY PAYLOAD KEYS:", dbg.get("payload_keys"))
    print("TALLY SHAPE:", dbg.get("shape"), "FIELDS_COUNT:", dbg.get("fields_count"))
    print("TALLY ANSWER KEYS:", sorted(list(answers.keys())))

    if not answers:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Could not extract answers from Tally payload",
                "debug": dbg,
            },
        )

    # ---- Field mapping (matches your actual labels + flexible fallbacks) ----
    try:
        household_type = pick(
            answers,
            [
                "Household Type?",
                "Household Type",
                "household type",
                "household",
                "individual or couple",
            ],
            required=True,
        )

        downsizing_raw = pick(
            answers,
            [
                "Are You Downsizing?",
                "Are you downsizing?",
                "are you downsizing",
                "downsizing",
            ],
            required=True,
        )

        net_income_raw = pick(
            answers,
            [
                "Net Monthly Income",
                "Net monthly income",
                "monthly income",
                "income",
            ],
            required=True,
        )

        fixed_costs_raw = pick(
            answers,
            [
                "Fixed Monthly Obligations",
                "Fixed monthly obligations",
                "monthly obligations",
                "fixed costs",
                "obligations",
            ],
            required=True,
        )

        savings_raw = pick(
            answers,
            [
                "Liquid Savings Available",
                "Liquid savings available",
                "savings",
                "cash savings",
            ],
            required=True,
        )

        timeline = pick(
            answers,
            [
                "Timeline (How soon do you want to relocate?)",
                "Timeline",
                "move timeline",
                "how soon",
            ],
            required=True,
        )

        risk = pick(
            answers,
            [
                "Risk Tolerance",
                "Risk tolerance",
                "risk",
            ],
            required=True,
        )

        current_metro_label = pick(
            answers,
            [
                "Current Metro Area",
                "Current metro area",
                "current metro",
                "current location",
            ],
            required=True,
        )

        target_labels = pick(
            answers,
            [
                "Metro Areas You Are Considering (Optional)",
                "Metro areas you are considering (optional)",
                "Metro areas you are considering",
                "target metros",
                "metros considering",
            ],
            required=False,
        )

        email = pick(
            answers,
            [
                "Email Address? (So we can share our insight!)",
                "Email Address",
                "Email address",
                "email",
            ],
            required=True,
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

    # ---- Normalize values ----
    try:
        downsizing = _as_bool(downsizing_raw)
        net_income = _as_float(net_income_raw)
        fixed_costs = _as_float(fixed_costs_raw)
        savings = _as_float(savings_raw)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={"error": "Invalid numeric/boolean value", "reason": str(e)},
        )

    # Normalize targets into a list[str]
    if target_labels is None:
        target_labels_list: List[str] = []
    elif isinstance(target_labels, str):
        target_labels_list = [target_labels]
    elif isinstance(target_labels, list):
        target_labels_list = [str(x) for x in target_labels if x is not None]
    else:
        target_labels_list = []

    # ---- Map metro to CBSA ----
    current_cbsa = label_or_cbsa_to_cbsa(current_metro_label)
    if not current_cbsa:
        raise HTTPException(
            status_code=400,
            detail={"error": "Unknown current metro", "value": current_metro_label},
        )

    target_cbsas: List[str] = []
    for t in target_labels_list:
        cbsa = label_or_cbsa_to_cbsa(t)
        if cbsa:
            target_cbsas.append(cbsa)

    # MVP JSON output (next step is plugging in your real recommendation engine + PDF)
    return JSONResponse(
        content={
            "status": "processed",
            "email": str(email).strip(),
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
