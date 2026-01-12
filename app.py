"""
RBRE API – Tally-safe FastAPI app (FULL FILE, hardened)

✅ Includes Option 3 (guaranteed logging):
- Middleware that logs EVERY request:
  - HIT <METHOD> <PATH>
  - DONE <METHOD> <PATH> -> <STATUS_CODE>
  - Flushes stdout so Render Live Logs show it immediately

Also includes:
- Extract answers from Tally payload["data"]["fields"] list
- Resolves select/multi-select option UUIDs -> human LABELS via field options metadata
- Logs:
  - TALLY PAYLOAD KEYS
  - TALLY SHAPE + FIELDS_COUNT
  - TALLY ANSWER KEYS
  - RAW mapped values (before parsing)
- Unwrap Tally values (dict/list shapes) into primitives
- Robust numeric/boolean parsing with clear 400 errors
- Metro label -> CBSA mapping from cost_data_metro.json (all CBSAs supported)
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
app = FastAPI(title="RBRE API", version="1.3")


# -------------------------------------------------
# ✅ OPTION 3: Log every request (Render Live Logs)
# -------------------------------------------------
@app.middleware("http")
async def log_every_request(request: Request, call_next):
    try:
        print(f"HIT {request.method} {request.url.path}", flush=True)
    except Exception:
        # Never break requests due to logging
        pass

    response = await call_next(request)

    try:
        print(f"DONE {request.method} {request.url.path} -> {response.status_code}", flush=True)
    except Exception:
        pass

    return response


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
    """Conservative boolean parsing (expects labels like Yes/No after UUID->label resolution)."""
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
# Tally extraction helpers: UUID -> LABEL using options metadata
# -------------------------------------------------
def _collect_options(field_item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Options may appear in different places depending on Tally:
      - item["options"]
      - item["field"]["options"]
      - item["field"]["choices"]
      - item["choices"]
    """
    candidates: List[Any] = []
    for path in (
        ("options",),
        ("choices",),
        ("field", "options"),
        ("field", "choices"),
    ):
        cur: Any = field_item
        ok = True
        for k in path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok and isinstance(cur, list):
            candidates.append(cur)

    for c in candidates:
        if isinstance(c, list) and len(c) > 0:
            return [x for x in c if isinstance(x, dict)]
    return []


def _option_id(opt: Dict[str, Any]) -> Optional[str]:
    for k in ("id", "value", "uuid", "key"):
        if k in opt and opt[k] is not None:
            return str(opt[k]).strip()
    return None


def _option_label(opt: Dict[str, Any]) -> Optional[str]:
    for k in ("label", "text", "name", "title"):
        if k in opt and opt[k] is not None:
            return str(opt[k]).strip()
    return None


def _resolve_value_via_options(field_item: Dict[str, Any], raw_value: Any) -> Any:
    """
    If raw_value is an option UUID (or list of UUIDs) and options metadata is present,
    map UUID(s) -> human labels.
    """
    options = _collect_options(field_item)
    if not options:
        return raw_value

    id_to_label: Dict[str, str] = {}
    for opt in options:
        oid = _option_id(opt)
        olab = _option_label(opt)
        if oid and olab:
            id_to_label[oid] = olab

    if not id_to_label:
        return raw_value

    v = _unwrap_value(raw_value)

    if isinstance(v, list):
        out: List[Any] = []
        for item in v:
            s = str(item).strip() if item is not None else ""
            out.append(id_to_label.get(s, item))
        return out

    s = str(v).strip() if v is not None else ""
    return id_to_label.get(s, raw_value)


# -------------------------------------------------
# Tally answer extraction (with UUID->label resolution)
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
            resolved = _resolve_value_via_options(item, value)
            answers[label.strip()] = resolved

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
    s = s.lower().strip()
    s = re.sub(r"\s*\(.*?\)\s*", " ", s)
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


def label_or_cbsa_to_cbsa(x: Any) -> Optional[str]:
    x = _unwrap_value(x)
    if x is None:
        return None

    if isinstance(x, list):
        if not x:
            return None
        x = x[0]

    s = str(x).strip()
    if s in COST_DATA:
        return s
    return METRO_LOOKUP.get(normalize_metro_name(s))


# -------------------------------------------------
# Health check
# -------------------------------------------------
@app.get("/")
def health():
    return {
        "status": "alive",
        "version": "1.3",
        "metros_loaded": len(COST_DATA),
        "metro_lookup_loaded": len(METRO_LOOKUP),
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

    # Log raw mapped values (after UUID->LABEL resolution)
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
                "error": "Normalization failed",
                "reason": str(e),
            },
        )

    # Targets normalization into list[str]
    targets_unwrapped = _unwrap_value(target_labels_raw)
    if targets_unwrapped is None:
        target_labels: List[str] = []
    elif isinstance(targets_unwrapped, list):
        target_labels = [str(x).strip() for x in targets_unwrapped if x is not None and str(x).strip()]
    else:
        target_labels = [str(targets_unwrapped).strip()] if str(targets_unwrapped).strip() else []

    # Map metro labels -> CBSA
    current_cbsa = label_or_cbsa_to_cbsa(current_metro_raw)
    if not current_cbsa:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Unknown current metro after UUID->label resolution. Label did not match cost_data_metro.json.",
                "received_value": _unwrap_value(current_metro_raw),
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

    # ✅ Helpful concise result log (no huge payload dump)
    try:
        masked_email = email
        if "@" in masked_email:
            local, domain = masked_email.split("@", 1)
            masked_email = (local[:2] + "***@" + domain) if len(local) > 2 else "***@" + domain

        print(
            "FINAL:",
            {
                "eventId": payload.get("eventId"),
                "email": masked_email,
                "current_cbsa": current_cbsa,
                "targets_count": len(target_cbsas),
                "unmapped_targets": unmapped_targets,
            },
            flush=True,
        )
    except Exception:
        pass

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
