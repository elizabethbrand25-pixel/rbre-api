from __future__ import annotations

import json
import os
import re
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Literal

import boto3
import psycopg2
import resend
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    PlainTextResponse,
)
from fastapi.templating import Jinja2Templates
from jinja2 import TemplateNotFound
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


# -------------------------------------------------
# ✅ BOOT CONFIRM
# -------------------------------------------------
print("RBRE API BOOT CONFIRM: module imported (app.py)", flush=True)

app = FastAPI(title="RBRE API", version="2.4")

# Jinja templates (expects templates/results.html)
templates = Jinja2Templates(directory="templates")


# -------------------------------------------------
# ✅ Global exception handler (prints traceback)
# -------------------------------------------------
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    print("UNHANDLED ERROR:", repr(exc), flush=True)
    print(traceback.format_exc(), flush=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "reason": str(exc)},
    )


@app.on_event("startup")
async def _startup_log():
    print("RBRE API BOOT CONFIRM: startup event fired", flush=True)


# -------------------------------------------------
# ✅ Request logging middleware
# -------------------------------------------------
@app.middleware("http")
async def log_every_request(request: Request, call_next):
    print(f"HIT {request.method} {request.url.path}", flush=True)
    resp = await call_next(request)
    print(f"DONE {request.method} {request.url.path} -> {resp.status_code}", flush=True)
    return resp


# -------------------------------------------------
# ✅ Debug routes (for blank-page diagnosis)
# -------------------------------------------------
@app.get("/_debug/ping", response_class=PlainTextResponse)
def debug_ping():
    return PlainTextResponse("pong")


@app.get("/_debug/html", response_class=HTMLResponse)
def debug_html():
    return HTMLResponse(
        "<h1>RBRE debug HTML works</h1>"
        "<p>If you see this, HTML responses are rendering correctly.</p>"
    )


# -------------------------------------------------
# Environment / Config
# -------------------------------------------------
DATABASE_URL = (os.getenv("DATABASE_URL") or "").strip()

S3_ENDPOINT_URL = (os.getenv("S3_ENDPOINT_URL") or "").strip() or None
S3_REGION = (os.getenv("S3_REGION") or "auto").strip()
S3_ACCESS_KEY_ID = (os.getenv("S3_ACCESS_KEY_ID") or "").strip()
S3_SECRET_ACCESS_KEY = (os.getenv("S3_SECRET_ACCESS_KEY") or "").strip()
S3_BUCKET = (os.getenv("S3_BUCKET") or "").strip()
S3_PREFIX = (os.getenv("S3_PREFIX") or "reports/").strip()

PUBLIC_BASE_URL = (os.getenv("PUBLIC_BASE_URL") or "https://rbre-api.onrender.com").strip()

# Email (Resend)
RESEND_API_KEY = (os.getenv("RESEND_API_KEY") or "").strip()
EMAIL_FROM = (os.getenv("EMAIL_FROM") or "").strip()


def _require_env_core():
    missing: List[str] = []
    if not DATABASE_URL:
        missing.append("DATABASE_URL")
    if not S3_ACCESS_KEY_ID:
        missing.append("S3_ACCESS_KEY_ID")
    if not S3_SECRET_ACCESS_KEY:
        missing.append("S3_SECRET_ACCESS_KEY")
    if not S3_BUCKET:
        missing.append("S3_BUCKET")
    if not S3_ENDPOINT_URL:
        missing.append("S3_ENDPOINT_URL")  # required for R2
    if missing:
        raise RuntimeError(f"Missing required environment variables: {missing}")


def _email_enabled() -> bool:
    return bool(RESEND_API_KEY and EMAIL_FROM)


def db_conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not configured")
    return psycopg2.connect(DATABASE_URL)


def s3_client():
    if not (S3_ACCESS_KEY_ID and S3_SECRET_ACCESS_KEY):
        raise RuntimeError("S3 credentials not configured")
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL,
        region_name=S3_REGION,
        aws_access_key_id=S3_ACCESS_KEY_ID,
        aws_secret_access_key=S3_SECRET_ACCESS_KEY,
    )


# -------------------------------------------------
# Email helper (Resend)
# -------------------------------------------------
def send_results_email(to_email: str, submission_id: str, results_url: str) -> None:
    """
    Sends the results link to the user. Never raises (caller should treat as best-effort).
    """
    if not _email_enabled():
        print("EMAIL: Skipped (RESEND_API_KEY/EMAIL_FROM not set)", flush=True)
        return

    try:
        resend.api_key = RESEND_API_KEY
        pdf_url = f"{PUBLIC_BASE_URL.rstrip('/')}/results/{submission_id}/report.pdf"

        subject = "Your RBRE relocation report is ready"
        html = f"""
        <div style="font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; line-height: 1.5;">
          <h2>Your RBRE report is ready</h2>
          <p>Thanks for submitting your info. Here are your links:</p>
          <ul>
            <li><a href="{results_url}">View your results</a></li>
            <li><a href="{pdf_url}">Download your PDF</a> (link valid ~1 hour)</li>
          </ul>
          <p><small>Submission ID: {submission_id}</small></p>
        </div>
        """

        resp = resend.Emails.send(
            {
                "from": EMAIL_FROM,
                "to": [to_email],
                "subject": subject,
                "html": html,
            }
        )
        print("EMAIL: Sent via Resend", {"to": to_email, "resp": resp}, flush=True)
    except Exception as e:
        print("EMAIL: Failed to send", {"to": to_email, "err": repr(e)}, flush=True)


# -------------------------------------------------
# Helpers: normalization + picking
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


def _unwrap_value(v: Any) -> Any:
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


def _money(n: float) -> str:
    try:
        return f"{float(n):,.0f}"
    except Exception:
        return "0"


# -------------------------------------------------
# UUID -> LABEL resolution via field options metadata
# -------------------------------------------------
def _collect_options(field_item: Dict[str, Any]) -> List[Dict[str, Any]]:
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
            answers[label.strip()] = _resolve_value_via_options(item, value)

    return answers, debug


# -------------------------------------------------
# Metro data: label -> CBSA lookup
# -------------------------------------------------
DATA_FILE = "cost_data_metro.json"
if not os.path.exists(DATA_FILE):
    raise RuntimeError("cost_data_metro.json not found in repo root (commit it).")

with open(DATA_FILE, "r", encoding="utf-8") as f:
    COST_DATA: Dict[str, Dict[str, Any]] = json.load(f)


def normalize_metro_name(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s*\(.*?\)\s*", " ", s)
    s = s.replace(" hud metro fmr area", " ")
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


def _get_metro_name(cbsa: Optional[str], fallback: str = "Your selected metro") -> str:
    if cbsa and cbsa in COST_DATA:
        name = COST_DATA[cbsa].get("metro_name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    return fallback


def _extract_numeric_from_profile(profile: Dict[str, Any], candidate_keys: List[str]) -> Optional[float]:
    for k in candidate_keys:
        if k in profile and profile[k] is not None:
            try:
                return float(str(profile[k]).replace(",", "").strip())
            except Exception:
                continue
    return None


def estimate_housing_cost_for_cbsa(cbsa: str) -> float:
    """
    Best-effort estimate. Tries common key names.
    If you want this 100% correct, tell me your JSON schema and we’ll pin the exact field.
    """
    profile = COST_DATA.get(cbsa) or {}

    candidates = [
        "monthly_housing_cost",
        "housing_cost",
        "median_housing_cost",
        "avg_housing_cost",
        "median_rent",
        "avg_rent",
        "rent",
        "fmr_2br",
        "fmr2br",
        "two_bed_fmr",
        "rent_2br",
        "two_bed_rent",
    ]

    v = _extract_numeric_from_profile(profile, candidates)
    if v is None:
        return 0.0
    return max(v, 0.0)


# -------------------------------------------------
# Verdict logic (Comfortable / Tight / High Risk)
# -------------------------------------------------
Signal = Literal["Good", "Caution", "Risk"]
Verdict = Literal["Comfortable", "Tight", "High Risk"]


@dataclass(frozen=True)
class VerdictResult:
    verdict: Verdict
    housing_percent: float
    monthly_buffer: float
    coverage_ratio: Optional[float]
    signals: Dict[str, Signal]
    reasons: Dict[str, str]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def evaluate_verdict(
    *,
    gross_monthly_income: Any,
    housing_cost: Any,
    non_housing_costs: Any,
    savings: Any,
    upfront_costs: Any,
) -> VerdictResult:
    income = _safe_float(gross_monthly_income, 0.0)
    housing = max(_safe_float(housing_cost, 0.0), 0.0)
    non_housing = max(_safe_float(non_housing_costs, 0.0), 0.0)
    sav = max(_safe_float(savings, 0.0), 0.0)
    upfront = max(_safe_float(upfront_costs, 0.0), 0.0)

    # Housing burden
    housing_percent = (housing / income) if income > 0 else 1.0
    if housing_percent <= 0.30:
        housing_signal: Signal = "Good"
        housing_reason = "Housing is within 30% of income (typical affordability guidance)."
    elif housing_percent <= 0.40:
        housing_signal = "Caution"
        housing_reason = "Housing is 30–40% of income, which can feel tight month-to-month."
    else:
        housing_signal = "Risk"
        housing_reason = "Housing exceeds 40% of income, which commonly leads to financial strain."

    # Monthly buffer
    monthly_buffer = income - (housing + non_housing)
    if monthly_buffer >= 1000:
        buffer_signal: Signal = "Good"
        buffer_reason = "Monthly buffer is at least $1,000, providing a healthy cushion."
    elif monthly_buffer >= 300:
        buffer_signal = "Caution"
        buffer_reason = "Monthly buffer is under $1,000; budgeting discipline will matter."
    else:
        buffer_signal = "Risk"
        buffer_reason = "Monthly buffer is under $300, leaving little room for surprises."

    # Upfront coverage
    if upfront <= 0:
        coverage_ratio = None
        upfront_signal: Signal = "Good"
        upfront_reason = "Upfront costs were not provided; treating upfront coverage as OK."
    else:
        coverage_ratio = sav / upfront
        if coverage_ratio >= 1.25:
            upfront_signal = "Good"
            upfront_reason = "Savings cover upfront costs with extra cushion (≥ 1.25×)."
        elif coverage_ratio >= 1.0:
            upfront_signal = "Caution"
            upfront_reason = "Savings cover upfront costs, but with little margin (1.0–1.24×)."
        else:
            upfront_signal = "Risk"
            upfront_reason = "Savings do not fully cover upfront costs (< 1.0×)."

    signals: Dict[str, Signal] = {
        "housing": housing_signal,
        "buffer": buffer_signal,
        "upfront": upfront_signal,
    }
    reasons: Dict[str, str] = {
        "housing": housing_reason,
        "buffer": buffer_reason,
        "upfront": upfront_reason,
    }

    risk_count = sum(1 for s in signals.values() if s == "Risk")
    caution_count = sum(1 for s in signals.values() if s == "Caution")

    if risk_count == 0 and caution_count <= 1:
        verdict: Verdict = "Comfortable"
    elif risk_count <= 1 and caution_count >= 1:
        verdict = "Tight"
    else:
        verdict = "High Risk"

    return VerdictResult(
        verdict=verdict,
        housing_percent=housing_percent,
        monthly_buffer=monthly_buffer,
        coverage_ratio=coverage_ratio,
        signals=signals,
        reasons=reasons,
    )


# -------------------------------------------------
# PDF generation + S3 upload + presign
# -------------------------------------------------
def build_pdf_bytes(submission_id: str, email: str, inputs: Dict[str, Any], results: Dict[str, Any]) -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    _, height = letter

    def line(text: str, y: float, bold: bool = False) -> float:
        c.setFont("Helvetica-Bold" if bold else "Helvetica", 11 if bold else 10)
        c.drawString(72, y, text[:140])
        return y - (18 if bold else 14)

    y = height - 72
    y = line("RBRE Relocation Report", y, bold=True)
    y = line(f"Submission ID: {submission_id}", y)
    y = line(f"Generated: {datetime.now(timezone.utc).isoformat()}", y)
    y = line(f"Email: {email}", y)
    y -= 8

    y = line("Inputs", y, bold=True)
    for k in [
        "household_type",
        "downsizing",
        "net_monthly_income",
        "fixed_monthly_obligations",
        "liquid_savings",
        "timeline",
        "risk_tolerance",
        "current_metro_label",
        "target_metro_labels",
    ]:
        if y < 90:
            c.showPage()
            y = height - 72
        y = line(f"- {k}: {inputs.get(k)}", y)

    y -= 8
    y = line("Results", y, bold=True)
    for k in ["current_cbsa", "target_cbsas", "unmapped_targets"]:
        if y < 90:
            c.showPage()
            y = height - 72
        y = line(f"- {k}: {results.get(k)}", y)

    c.showPage()
    c.save()
    return buf.getvalue()


def upload_pdf(pdf_bytes: bytes, submission_id: str) -> Tuple[str, str]:
    _require_env_core()
    key = f"{S3_PREFIX.rstrip('/')}/{submission_id}.pdf".lstrip("/")
    client = s3_client()
    client.put_object(Bucket=S3_BUCKET, Key=key, Body=pdf_bytes, ContentType="application/pdf")
    return S3_BUCKET, key


def presigned_get_url(bucket: str, key: str, expires_seconds: int = 3600) -> str:
    client = s3_client()
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires_seconds,
    )


# -------------------------------------------------
# DB schema (auto-ensure)
# -------------------------------------------------
CREATE_TABLE_SQL = """
create table if not exists submissions (
  id uuid primary key,
  created_at timestamptz not null default now(),
  event_id text,
  email text,
  inputs jsonb not null,
  results jsonb not null,
  pdf_bucket text not null,
  pdf_key text not null
);
create index if not exists submissions_created_at_idx on submissions(created_at desc);
create index if not exists submissions_email_idx on submissions(email);
"""


def ensure_schema():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not configured")
    with db_conn() as conn:
        with conn.cursor() as cur:
            for stmt in [s.strip() for s in CREATE_TABLE_SQL.split(";") if s.strip()]:
                cur.execute(stmt)
        conn.commit()


# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.get("/")
def health():
    return {
        "status": "alive",
        "version": "2.4",
        "email_enabled": _email_enabled(),
        "metros_loaded": len(COST_DATA),
        "metro_lookup_loaded": len(METRO_LOOKUP),
    }


@app.post("/v1/report.json")
async def generate_report(request: Request):
    # Core env + schema
    try:
        _require_env_core()
    except Exception as e:
        print("ENV missing/invalid:", repr(e), flush=True)
        raise HTTPException(status_code=500, detail={"error": "Server misconfigured", "reason": str(e)})

    try:
        ensure_schema()
    except Exception as e:
        print("DB schema check failed:", repr(e), flush=True)
        raise HTTPException(status_code=500, detail={"error": "DB schema check failed", "reason": str(e)})

    payload = await request.json()
    answers, dbg = _extract_tally_answers(payload)

    print("TALLY PAYLOAD KEYS:", dbg.get("payload_keys"), flush=True)
    print("TALLY SHAPE:", dbg.get("shape"), "FIELDS_COUNT:", dbg.get("fields_count"), flush=True)
    print("TALLY ANSWER KEYS:", sorted(list(answers.keys())), flush=True)

    if not answers:
        raise HTTPException(status_code=400, detail={"error": "Could not extract answers", "debug": dbg})

    # Mapping
    try:
        household_type_raw = pick(answers, ["Household Type?"], required=True)
        downsizing_raw = pick(answers, ["Are You Downsizing?"], required=True)
        net_income_raw = pick(answers, ["Net Monthly Income"], required=True)
        fixed_costs_raw = pick(answers, ["Fixed Monthly Obligations"], required=True)
        savings_raw = pick(answers, ["Liquid Savings Available"], required=True)
        timeline_raw = pick(answers, ["Timeline (How soon do you want to relocate?)"], required=True)
        risk_raw = pick(answers, ["Risk Tolerance"], required=True)
        current_metro_raw = pick(answers, ["Current Metro Area"], required=True)
        targets_raw = pick(answers, ["Metro Areas You Are Considering (Optional)"], required=False)
        email_raw = pick(answers, ["Email Address? (So we can share our insight!)"], required=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail={"error": "Field mapping failed", "reason": str(e)})

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
            "targets": targets_raw,
            "email": email_raw,
        },
        flush=True,
    )

    # Normalize
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
        raise HTTPException(status_code=400, detail={"error": "Normalization failed", "reason": str(e)})

    targets_unwrapped = _unwrap_value(targets_raw)
    if targets_unwrapped is None:
        target_labels: List[str] = []
    elif isinstance(targets_unwrapped, list):
        target_labels = [str(x).strip() for x in targets_unwrapped if x is not None and str(x).strip()]
    else:
        target_labels = [str(targets_unwrapped).strip()] if str(targets_unwrapped).strip() else []

    current_cbsa = label_or_cbsa_to_cbsa(current_metro_raw)
    if not current_cbsa:
        raise HTTPException(status_code=400, detail={"error": "Unknown current metro", "value": _unwrap_value(current_metro_raw)})

    target_cbsas: List[str] = []
    unmapped_targets: List[str] = []
    for t in target_labels:
        cbsa = label_or_cbsa_to_cbsa(t)
        if cbsa:
            target_cbsas.append(cbsa)
        else:
            unmapped_targets.append(t)

    submission_id = str(uuid.uuid4())

    inputs = {
        "email": email,
        "household_type": household_type,
        "downsizing": downsizing,
        "net_monthly_income": net_income,
        "fixed_monthly_obligations": fixed_costs,
        "liquid_savings": savings,
        "timeline": timeline,
        "risk_tolerance": risk_tolerance,
        "current_metro_label": _unwrap_value(current_metro_raw),
        "target_metro_labels": target_labels,
        "eventId": payload.get("eventId"),
        "createdAt": payload.get("createdAt"),
    }

    results = {
        "current_cbsa": current_cbsa,
        "target_cbsas": target_cbsas,
        "unmapped_targets": unmapped_targets,
        "net_monthly_income": net_income,
        "fixed_monthly_obligations": fixed_costs,
        "liquid_savings": savings,
        "timeline": timeline,
        "risk_tolerance": risk_tolerance,
    }

    # PDF
    try:
        pdf_bytes = build_pdf_bytes(submission_id=submission_id, email=email, inputs=inputs, results=results)
    except Exception as e:
        print("PDF generation failed:", repr(e), flush=True)
        raise HTTPException(status_code=500, detail={"error": "PDF generation failed", "reason": str(e)})

    # Upload
    try:
        bucket, key = upload_pdf(pdf_bytes, submission_id=submission_id)
    except Exception as e:
        print("PDF upload failed:", repr(e), flush=True)
        raise HTTPException(status_code=500, detail={"error": "PDF upload failed", "reason": str(e)})

    # DB insert
    try:
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into submissions (id, event_id, email, inputs, results, pdf_bucket, pdf_key)
                    values (%s, %s, %s, %s::jsonb, %s::jsonb, %s, %s)
                    """,
                    (
                        submission_id,
                        payload.get("eventId"),
                        email,
                        json.dumps(inputs),
                        json.dumps(results),
                        bucket,
                        key,
                    ),
                )
            conn.commit()
    except Exception as e:
        print("DB insert failed:", repr(e), flush=True)
        raise HTTPException(status_code=500, detail={"error": "DB insert failed", "reason": str(e)})

    results_url = f"{PUBLIC_BASE_URL.rstrip('/')}/results/{submission_id}"

    # EMAIL (best-effort)
    send_results_email(to_email=email, submission_id=submission_id, results_url=results_url)

    # Final log
    masked_email = email
    if "@" in masked_email:
        local, domain = masked_email.split("@", 1)
        masked_email = (local[:2] + "***@" + domain) if len(local) > 2 else "***@" + domain

    print(
        "FINAL:",
        {
            "submission_id": submission_id,
            "results_url": results_url,
            "email": masked_email,
            "current_cbsa": current_cbsa,
            "targets_count": len(target_cbsas),
            "unmapped_targets": unmapped_targets,
            "email_sent": _email_enabled(),
        },
        flush=True,
    )

    return JSONResponse(content={"status": "processed", "submission_id": submission_id, "results_url": results_url})


@app.get("/results/{submission_id}", response_class=HTMLResponse)
def results_page(request: Request, submission_id: str):
    """
    Renders templates/results.html (premium UI).
    If the template is missing in deploy, falls back to a simple HTML response
    so you never get a “blank page” again.
    """
    # DB fetch
    try:
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("select email, inputs, results from submissions where id = %s", (submission_id,))
                row = cur.fetchone()
    except Exception as e:
        print("DB query failed (results_page):", repr(e), flush=True)
        raise HTTPException(status_code=500, detail={"error": "DB query failed", "reason": str(e)})

    if not row:
        raise HTTPException(status_code=404, detail={"error": "Submission not found"})

    email, inputs, results = row
    if isinstance(inputs, str):
        inputs = json.loads(inputs)
    if isinstance(results, str):
        results = json.loads(results)

    # Pick a "primary" metro for display (first target if present, else current)
    target_cbsas = results.get("target_cbsas") or []
    primary_cbsa = target_cbsas[0] if isinstance(target_cbsas, list) and target_cbsas else results.get("current_cbsa")

    metro_name = _get_metro_name(primary_cbsa, fallback=str(inputs.get("current_metro_label") or "Your metro"))

    # Numbers for verdict
    net_income = float(results.get("net_monthly_income") or 0.0)
    fixed_costs = float(results.get("fixed_monthly_obligations") or 0.0)
    savings = float(results.get("liquid_savings") or 0.0)

    housing_cost = estimate_housing_cost_for_cbsa(primary_cbsa) if primary_cbsa else 0.0

    # Simple upfront estimate: 2 months housing + $1,000 friction
    upfront_costs = (housing_cost * 2.0) + 1000.0 if housing_cost > 0 else 0.0

    verdict_result = evaluate_verdict(
        gross_monthly_income=net_income,
        housing_cost=housing_cost,
        non_housing_costs=fixed_costs,
        savings=savings,
        upfront_costs=upfront_costs,
    )

    pdf_url = f"{PUBLIC_BASE_URL.rstrip('/')}/results/{submission_id}/report.pdf"

    context = {
        "request": request,
        "metro_name": metro_name,
        "pdf_url": pdf_url,
        "restart_url": "/",
        "verdict": verdict_result.verdict,
        "reasons": verdict_result.reasons,
        "signals": verdict_result.signals,
        "housing_cost": _money(housing_cost) if housing_cost else "—",
        "housing_percent": round(verdict_result.housing_percent * 100.0, 1) if net_income > 0 and housing_cost else "—",
        "monthly_buffer": _money(verdict_result.monthly_buffer) if net_income else "—",
        "upfront_costs": _money(upfront_costs) if upfront_costs else "—",
        "coverage_ratio": (round(verdict_result.coverage_ratio, 2) if verdict_result.coverage_ratio is not None else None),
        "submission_id": submission_id,  # handy if you want it in the template
        "email": email,                  # handy if you want it in the template
    }

    # Try to render premium template; fall back safely if missing
    try:
        resp = templates.TemplateResponse("results.html", context)
        # Helpful diagnostic: if you're seeing "blank", confirm we actually output content
        # (Starlette will render later; this still helps confirm route is hit)
        print("RESULTS_PAGE_RENDER: template=results.html verdict=", verdict_result.verdict, flush=True)
        return resp
    except TemplateNotFound:
        print("TEMPLATE MISSING: templates/results.html not found in deploy. Using fallback HTML.", flush=True)
    except Exception as e:
        print("TEMPLATE RENDER ERROR:", repr(e), flush=True)
        print(traceback.format_exc(), flush=True)

    # Fallback HTML (never blank)
    html = f"""
    <html>
      <head>
        <title>RBRE Results</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </head>
      <body style="font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 40px; max-width: 900px;">
        <h1>RBRE Results (Fallback)</h1>
        <p><strong>Metro:</strong> {metro_name}</p>
        <p><strong>Verdict:</strong> {verdict_result.verdict}</p>

        <h2>Why</h2>
        <ul>
          <li><strong>Housing:</strong> {verdict_result.reasons.get("housing")}</li>
          <li><strong>Buffer:</strong> {verdict_result.reasons.get("buffer")}</li>
          <li><strong>Upfront:</strong> {verdict_result.reasons.get("upfront")}</li>
        </ul>

        <h2>Numbers</h2>
        <ul>
          <li><strong>Monthly Housing:</strong> {_money(housing_cost) if housing_cost else "—"}</li>
          <li><strong>Housing %:</strong> {round(verdict_result.housing_percent * 100.0, 1) if net_income > 0 and housing_cost else "—"}</li>
          <li><strong>Monthly Buffer:</strong> {_money(verdict_result.monthly_buffer) if net_income else "—"}</li>
          <li><strong>Upfront Needed:</strong> {_money(upfront_costs) if upfront_costs else "—"}</li>
          <li><strong>Upfront Coverage:</strong> {round(verdict_result.coverage_ratio, 2) if verdict_result.coverage_ratio is not None else "—"}</li>
        </ul>

        <h2>Download</h2>
        <p><a href="{pdf_url}">Download PDF report</a> (link valid ~1 hour)</p>

        <hr />
        <p style="color:#666;font-size:12px;">
          If you're seeing this fallback, it means <code>templates/results.html</code> was not found or failed to render on the server.
          Confirm the file exists and is committed to git.
        </p>
      </body>
    </html>
    """
    print("RESULTS_PAGE_FALLBACK_HTML_LEN:", len(html), flush=True)
    return HTMLResponse(content=html)


@app.get("/results/{submission_id}/report.pdf")
def report_pdf(submission_id: str):
    try:
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("select pdf_bucket, pdf_key from submissions where id = %s", (submission_id,))
                row = cur.fetchone()
    except Exception as e:
        print("DB query failed (report_pdf):", repr(e), flush=True)
        raise HTTPException(status_code=500, detail={"error": "DB query failed", "reason": str(e)})

    if not row:
        raise HTTPException(status_code=404, detail={"error": "Submission not found"})

    bucket, key = row
    try:
        url = presigned_get_url(bucket=bucket, key=key, expires_seconds=3600)
    except Exception as e:
        print("Presign failed:", repr(e), flush=True)
        raise HTTPException(status_code=500, detail={"error": "Presign failed", "reason": str(e)})

    return RedirectResponse(url=url, status_code=302)
