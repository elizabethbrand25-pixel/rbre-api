from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(title="RBRE API", version="1.0")


# -------------------------------------------------
# Key normalization + picking
# -------------------------------------------------
def _norm_key(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def pick(data: Dict[str, Any], aliases: List[str], required: bool = True) -> Optional[Any]:
    norm_map = {_norm_key(k): k for k in data.keys()}
    for alias in aliases:
        nk = _norm_key(alias)
        if nk in norm_map:
            return data[norm_map[nk]]
    if required:
        raise KeyError(f"Missing field. Tried aliases={aliases}")
    return None


# -------------------------------------------------
# Tally value unwrapping + parsing
# -------------------------------------------------
def _unwrap_value(v: Any) -> Any:
    if v is None:
        return None

    if isinstance(v, list):
        return [_unwrap_value(x) for x in v]

    if isinstance(v, dict):
        for k in ("value", "label", "text", "name", "title"):
            if k in v and v[k] is not None:
                return _unwrap_value(v[k])
        return str(v)

    return v


def _as_bool(v: Any) -> bool:
    v = _unwrap_value(v)
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in {"yes", "y", "true", "1", "on"}


def _as_float(v: Any) -> float:
    v = _unwrap_value(v)
    if v is None:
        raise ValueError("Empty numeric value (None)")

    if isinstance(v, list):
        if not v:
            raise ValueError("Empty numeric list")
        v = v[0]

    s = str(v).strip().replace(",", "")
    s = re.sub(r"[^0-9.\-]", "", s)
    if s == "":
        raise ValueError(f"Could not parse numeric value from {v!r}")
    return float(s)


# -------------------------------------------------
# Tally answer extraction
# -------------------------------------------------
def _extract_tally_answers(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    debug: Dict[str, Any] = {"payload_keys": sorted(list(payload.keys()))}

    data = payload.get("data")
    fields_list = None

    if isinstance(data, dict) and isinstance(data.get("fields"), list):
        fields_list = data.get("fields")
        debug["shape"] = "fields_list"
        debug["fields_count"] = len(fields_list)
    elif isinstance(payload.get("fields"), list):
        fields_list = payload.get("fields")
        debug["shape"] = "fields_list"
        debug["fields_count"] = len(fields_list)
    else:
        debug["shape"] = "unknown"
        debug["fields_count"] = 0
        return {}, debug

    answers: Dict[str, Any] = {}
    for item in fields_list:
        if not isinstance(item, dict):
            continue

        label = item.get("label") or item.get("title")
        if label is None and isinstance(item.get("field"), dict):
            label = item["field"].get("label") or item["field"].get("title")

        value = None
        if "value" in item:
            value = item.get("value")
        elif "answer" in item:
            value = item.get("answer")
        elif "response" in item:
            value = item.get("response")

        if isinstance(label, str) and label.strip():
            answers[label.strip()] = value

    return answers, debug


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


# -------------------------------------------------
# ✅ OPTION 1: Tally option UUID -> CBSA mapping
# -------------------------------------------------
# IMPORTANT: fill this in with your real mapping.
TALLY_METRO_OPTION_TO_CBSA: Dict[str, str] = {
    # "8d7dc6df-eead-4c54-87a0-a76e139387b3": "31080",
    # "fb1e6396-0983-4cb7-929c-b2825f6e2924": "16980",
}


def label_or_cbsa_to_cbsa(x: Any) -> Optional[str]:
    x = _unwrap_value(x)
    if x is None:
        return None

    s = str(x).strip()

    # Option UUID -> CBSA
    if s in TALLY_METRO_OPTION_TO_CBSA:
        mapped = TALLY_METRO_OPTION_TO_CBSA[s]
        return mapped if mapped in COST_DATA else None

    # CBSA directly
    if s in COST_DATA:
        return s

    # Metro label -> CBSA
    return LABEL_TO_CBSA.get(s)


@app.get("/")
def health():
    return {"status": "alive", "metros_loaded": len(COST_DATA)}


@app.post("/v1/report.json")
async def generate_report(request: Request):
    payload = await request.json()

    answers, dbg = _extract_tally_answers(payload)

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
        household_type_raw = pick(
            answers,
            ["Household Type?", "Household Type", "household type", "household", "individual or couple"],
            required=True,
        )
        downsizing_raw = pick(
            answers,
            ["Are You Downsizing?", "Are you downsizing?", "are you downsizing", "downsizing"],
            required=True,
        )
        net_income_raw = pick(
            answers,
            ["Net Monthly Income", "Net monthly income", "monthly income", "income"],
            required=True,
        )
        fixed_costs_raw = pick(
            answers,
            ["Fixed Monthly Obligations", "Fixed monthly obligations", "monthly obligations", "fixed costs", "obligations"],
            required=True,
        )
        savings_raw = pick(
            answers,
            ["Liquid Savings Available", "Liquid savings available", "savings", "cash savings"],
            required=True,
        )
        timeline_raw = pick(
            answers,
            ["Timeline (How soon do you want to relocate?)", "Timeline", "move timeline", "how soon"],
            required=True,
        )
        risk_raw = pick(
            answers,
            ["Risk Tolerance", "Risk tolerance", "risk"],
            required=True,
        )
        current_metro_raw = pick(
            answers,
            ["Current Metro Area", "Current metro area", "current metro", "current location"],
            required=True,
        )
        target_labels_raw = pick(
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
        email_raw = pick(
            answers,
            ["Email Address? (So we can share our insight!)", "Email Address", "Email address", "email"],
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

    print(
        "RAW VALUES:",
        {
            "household_type": household_type_raw,
            "downsizing": downsizing_raw,
            "net_income": net_income_raw,
            "fixed_costs": fixed_costs_raw,
            "savings": savings_raw,
            "timeline": timeline_raw,
            "risk": risk_raw,
            "current_metro": current_metro_raw,
            "targets": target_labels_raw,
            "email": email_raw,
        },
    )

    try:
        household_type = str(_unwrap_value(household_type_raw)).strip()
        downsizing = _as_bool(downsizing_raw)
        net_income = _as_float(net_income_raw)
        fixed_costs = _as_float(fixed_costs_raw)
        savings = _as_float(savings_raw)
        timeline = str(_unwrap_value(timeline_raw)).strip()
        risk_tolerance = str(_unwrap_value(risk_raw)).strip()
        email = str(_unwrap_value(email_raw)).strip()
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Normalization failed",
                "reason": str(e),
            },
        )

    targets_unwrapped = _unwrap_value(target_labels_raw)
    if targets_unwrapped is None:
        target_labels: List[str] = []
    elif isinstance(targets_unwrapped, list):
        target_labels = [str(x).strip() for x in targets_unwrapped if x is not None and str(x).strip()]
    else:
        target_labels = [str(targets_unwrapped).strip()] if str(targets_unwrapped).strip() else []

    current_cbsa = label_or_cbsa_to_cbsa(current_metro_raw)
    if not current_cbsa:
        unwrapped = _unwrap_value(current_metro_raw)
        # If list, show first element since your payload wraps in a list
        if isinstance(unwrapped, list) and unwrapped:
            unwrapped_first = unwrapped[0]
        else:
            unwrapped_first = unwrapped

        metro_names_sample = list(LABEL_TO_CBSA.keys())[:8]

        raise HTTPException(
            status_code=400,
            detail={
                "error": "Unknown current metro (not CBSA, not metro_name label, not mapped option UUID).",
                "missing_option_id": str(unwrapped_first).strip() if unwrapped_first is not None else None,
                "hint": "Add missing_option_id to TALLY_METRO_OPTION_TO_CBSA with correct CBSA code.",
                "metro_names_sample": metro_names_sample,
            },
        )

    target_cbsas: List[str] = []
    unmapped_targets: List[str] = []
    for t in target_labels:
        cbsa = label_or_cbsa_to_cbsa(t)
        if cbsa:
            target_cbsas.append(cbsa)
        else:
            unmapped_targets.append(str(_unwrap_value(t)).strip())

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
            "risk_tolerance": risk_tolerance,
            "current_cbsa": current_cbsa,
            "target_cbsas": target_cbsas,
            "unmapped_targets": unmapped_targets,
        }
    )
