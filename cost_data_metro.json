"""
RBRE API – Tally-safe FastAPI app (FULL FILE, hardened + UUID-agnostic)

This rewrite removes brittle UUID->CBSA manual mapping by:
- Building a metro-label -> CBSA lookup from cost_data_metro.json (all CBSAs supported)
- Normalizing metro labels so minor formatting differences still match
- Extracting answers from Tally payload["data"]["fields"] list
- Logging:
  - TALLY PAYLOAD KEYS
  - TALLY SHAPE + FIELDS_COUNT
  - TALLY ANSWER KEYS
  - RAW mapped values (before parsing)
- Unwrapping Tally values (dict/list shapes) into primitives
- Robust numeric/boolean parsing with clear 400 errors
- Helpful 400 if metro values look like UUIDs (tells you to configure Tally to send labels)
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
app = FastAPI(title="RBRE API", version="1.1")


# -------------------------------------------------
# Key normalization + picking
# -------------------------------------------------
def _norm_key(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def pick(data: Dict[str, Any], aliases: List[str], required: bool = True) -> Optional[Any]:
    """Fetch a value using aliases with normalized matching."""
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
    """
    Tally "value" can be:
      - primitive (str/int/float/bool)
      - dict like {"label": "...", "value": "..."} or {"text": "..."}
      - list of dicts for multi-select
    Convert to a usable primitive or list of primitives.
    """
    if v is None:
        return None

    if isinstance(v, list):
        return [_unwrap_value(x) for x in v]

    if isinstance(v, dict):
        for k in ("label", "text", "value", "name", "title"):
            if k in v and v[k] is not None:
                return _unwrap_value(v[k])
        return str(v)

    return v


def _as_bool(v: Any) -> bool:
    """
    Conservative boolean parsing.
    If Tally sends labels like "Yes"/"No" this works.
    If Tally sends UUIDs, this will treat them as False (so fix Tally config).
    """
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
    """
    Returns (answers_dict, debug_info).

    Handles:
      payload["data"]["fields"] -> list of objects (your case)
    Flattens into: {"Question Label": value, ...}
    """
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
# Load metro cost data + build label->CBSA lookup (ALL CBSAs)
# -------------------------------------------------
DATA_FILE = "cost_data_metro.json"

if not os.path.exists(DATA_FILE):
    raise RuntimeError("cost_data_metro.json not found in repo root (commit it to GitHub).")

with open(DATA_FILE, "r", encoding="utf-8") as f:
    COST_DATA: Dict[str, Dict[str, Any]] = json.load(f)


def normalize_metro_name(s: str) -> str:
    """
    Make metro labels resilient to minor formatting differences.
    Examples this helps with:
      - removing "MSA", "HUD Metro FMR Area"
      - removing "(part)"
      - removing punctuation
      - normalizing whitespace
    """
    s = s.lower().strip()
    s = re.sub(r"\s*\(.*?\)\s*", " ", s)  # remove parenthetical notes
    s = s.replace("hud metro fmr area", " ")
    s = s.replace(" msa", " ")
    s = s.replace("msa", " ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


METRO_LOOKUP: Dict[str, str] = {}
for cbsa, prof in COST_DATA.items():
    name = prof.get("metro_name")
    if isinstance(name, str) and name.strip():
        METRO_LOOKUP[normalize_metro_name(name)] = cbsa


_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)


def label_or_cbsa_to_cbsa(x: Any) -> Optional[str]:
    """
    Accept CBSA code directly OR map metro label -> CBSA using COST_DATA-derived lookup.

    IMPORTANT:
    - If Tally sends option UUIDs instead of labels, this will not match.
      In that case, configure Tally to send choice labels.
    """
    x = _unwrap_value(x)
    if x is None:
        return None

    # If list, use first (single-selects sometimes come in lists)
    if isinstance(x, list):
        if not x:
            return None
        x = x[0]

    s = str(x).strip()

    # CBSA direct
    if s in COST_DATA:
        return s

    # Label match
    key = normalize_metro_name(s)
    return METRO_LOOKUP.get(key)


# -------------------------------------------------
# Health check
# -------------------------------------------------
@app.get("/")
def health():
    return {
        "status": "alive",
        "metros_loaded": len(COST_DATA),
        "metro_lookup_loaded": len(METRO_LOOKUP),
        "version": "1.1",
    }


# -------------------------------------------------
# Tally webhook endpoint
# -------------------------------------------------
@app.post("/v1/report.json")
async def generate_report(request: Request):
    payload = await request.json()

    answers, dbg = _extract_tally_answers(payload)

    # Logs (Render -> Logs)
    print("TALLY PAYLOAD KEYS:", dbg.get("payload_keys"), flush=True)
    print("TALLY SHAPE:", dbg.get("shape"), "FIELDS_COUNT:", dbg.get("fields_count"), flush=True)
    print("TALLY ANSWER KEYS:", sorted(list(answers.keys())), flush=True)

    if not answers:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Could not extract answers from Tally payload",
                "debug": dbg,
                "payload_keys": dbg.get("payload_keys"),
            },
        )

    # ---- Field mapping ----
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
        target_metros_raw = pick(
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
            "targets": target_metros_raw,
            "email": email_raw,
        },
        flush=True,
    )

    # ---- Normalize values ----
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
                "error": "Normalization failed (value shape likely dict/list)",
                "reason": str(e),
                "raw_values": {
                    "household_type": household_type_raw,
                    "downsizing": downsizing_raw,
                    "net_income": net_income_raw,
                    "fixed_costs": fixed_costs_raw,
                    "savings": savings_raw,
                    "timeline": timeline_raw,
                    "risk": risk_raw,
                    "current_metro": current_metro_raw,
                    "targets": target_metros_raw,
                    "email": email_raw,
                },
            },
        )

    # Targets normalization into list[str]
    targets_unwrapped = _unwrap_value(target_metros_raw)
    if targets_unwrapped is None:
        target_labels: List[str] = []
    elif isinstance(targets_unwrapped, list):
        target_labels = [str(x).strip() for x in targets_unwrapped if x is not None and str(x).strip()]
    else:
        target_labels = [str(targets_unwrapped).strip()] if str(targets_unwrapped).strip() else []

    # ---- Metro mapping ----
    current_cbsa = label_or_cbsa_to_cbsa(current_metro_raw)
    if not current_cbsa:
        unwrapped = _unwrap_value(current_metro_raw)
        first = unwrapped[0] if isinstance(unwrapped, list) and unwrapped else unwrapped
        first_s = str(first).strip() if first is not None else ""

        raise HTTPException(
            status_code=400,
            detail={
                "error": "Unknown current metro. Expected CBSA code or metro label matching cost_data_metro.json.",
                "received_value": first_s,
                "looks_like_uuid": bool(_UUID_RE.match(first_s)),
                "fix": (
                    "If this is a Tally option UUID, configure Tally to send the CHOICE LABEL (not the ID) "
                    "for Current Metro Area / Target Metros."
                ),
                "example_expected_label": next(iter(METRO_LOOKUP.keys()), None),
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

    # MVP JSON output (next step: plug in recommendation engine + PDF)
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
