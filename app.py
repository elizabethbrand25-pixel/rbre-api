from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import boto3
import psycopg2
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


# -------------------------------------------------
# ✅ BOOT CONFIRM
# -------------------------------------------------
print("RBRE API BOOT CONFIRM: module imported (app.py)", flush=True)

app = FastAPI(title="RBRE API", version="2.1")


@app.on_event("startup")
async def _startup_log():
    print("RBRE API BOOT CONFIRM: startup event fired", flush=True)


@app.middleware("http")
async def log_every_request(request: Request, call_next):
    print(f"HIT {request.method} {request.url.path}", flush=True)
    resp = await call_next(request)
    print(f"DONE {request.method} {request.url.path} -> {resp.status_code}", flush=True)
    return resp


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


def _require_env():
    missing: List[str] = []
    if not DATABASE_URL:
        missing.append("DATABASE_URL")
    if not S3_ACCESS_KEY_ID:
        missing.append("S3_ACCESS_KEY_ID")
    if not S3_SECRET_ACCESS_KEY:
        missing.append("S3_SECRET_ACCESS_KEY")
    if not S3_BUCKET:
        missing.append("S3_BUCKET")
    # S3_ENDPOINT_URL may be None for AWS, required for R2 typically.
    if S3_ENDPOINT_URL is None:
        # allow AWS, but for R2 this must exist—don't hard fail here.
        pass
    if missing:
        raise RuntimeError(f"Missing required environment variables: {missing}")


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
# PDF generation + S3 upload + presign
# -------------------------------------------------
def build_pdf_bytes(submission_id: str, email: str, inputs: Dict[str, Any], results: Dict[str, Any]) -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter

    def line(text: str, y: float, bold: bool = False) -> float:
        c.setFont("Helvetica-Bold" if bold else "Helvetica", 11 if bold else 10)
        c.drawString(72, y, text[:120])
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
    for k in [
        "current_cbsa",
        "target_cbsas",
        "unmapped_targets",
    ]:
        if y < 90:
            c.showPage()
            y = height - 72
        y = line(f"- {k}: {results.get(k)}", y)

    c.showPage()
    c.save()
    return buf.getvalue()


def upload_pdf(pdf_bytes: bytes, submission_id: str) -> Tuple[str, str]:
    _require_env()
    key = f"{S3_PREFIX.rstrip('/')}/{submission_id}.pdf".lstrip("/")
    client = s3_client()
    client.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=pdf_bytes,
        ContentType="application/pdf",
    )
    return S3_BUCKET, key


def presigned_get_url(bucket: str, key: str, expires_seconds: int = 3600) -> str:
    client = s3_client()
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires_seconds,
    )


# -------------------------------------------------
# DB: ensure table exists (optional safety)
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
        return
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
        "version": "2.1",
        "metros_loaded": len(COST_DATA),
        "metro_lookup_loaded": len(METRO_LOOKUP),
    }


@app.post("/v1/report.json")
async def generate_report(request: Request):
    # Make sure schema exists (safe to run repeatedly)
    try:
        ensure_schema()
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "DB schema check failed", "reason": str(e)})

    payload = await request.json()
    answers, dbg = _extract_tally_answers(payload)

    print("TALLY PAYLOAD KEYS:", dbg.get("payload_keys"), flush=True)
    print("TALLY SHAPE:", dbg.get("shape"), "FIELDS_COUNT:", dbg.get("fields_count"), flush=True)
    print("TALLY ANSWER KEYS:", sorted(list(answers.keys())), flush=True)

    if not answers:
        raise HTTPException(
            status_code=400,
            detail={"error": "Could not extract answers from Tally payload", "debug": dbg},
        )

    # ---- Field mapping ----
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
        raise HTTPException(
            status_code=400,
            detail={"error": "Field mapping failed", "reason": str(e), "received_answer_keys": sorted(list(answers.keys()))},
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
            "targets": targets_raw,
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

    # ---- Build + upload PDF ----
    try:
        pdf_bytes = build_pdf_bytes(submission_id=submission_id, email=email, inputs=inputs, results=results)
        bucket, key = upload_pdf(pdf_bytes, submission_id=submission_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "PDF upload failed", "reason": str(e)})

    # ---- Store in Postgres ----
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
        raise HTTPException(status_code=500, detail={"error": "DB insert failed", "reason": str(e)})

    masked_email = email
    if "@" in masked_email:
        local, domain = masked_email.split("@", 1)
        masked_email = (local[:2] + "***@" + domain) if len(local) > 2 else "***@" + domain

    results_url = f"{PUBLIC_BASE_URL.rstrip('/')}/results/{submission_id}"
    print(
        "FINAL:",
        {
            "submission_id": submission_id,
            "results_url": results_url,
            "email": masked_email,
            "current_cbsa": current_cbsa,
            "targets_count": len(target_cbsas),
            "unmapped_targets": unmapped_targets,
        },
        flush=True,
    )

    # Tally only needs 200; this is for your own debugging / possible redirect/email.
    return JSONResponse(
        content={
            "status": "processed",
            "submission_id": submission_id,
            "results_url": results_url,
        }
    )


@app.get("/results/{submission_id}", response_class=HTMLResponse)
def results_page(submission_id: str):
    if not DATABASE_URL:
        raise HTTPException(status_code=500, detail={"error": "DATABASE_URL not configured"})

    try:
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("select email, inputs, results from submissions where id = %s", (submission_id,))
                row = cur.fetchone()
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "DB query failed", "reason": str(e)})

    if not row:
        raise HTTPException(status_code=404, detail={"error": "Submission not found"})

    email, inputs, results = row
    if isinstance(inputs, str):
        inputs = json.loads(inputs)
    if isinstance(results, str):
        results = json.loads(results)

    pdf_link = f"{PUBLIC_BASE_URL.rstrip('/')}/results/{submission_id}/report.pdf"

    html = f"""
    <html>
      <head>
        <title>RBRE Results</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </head>
      <body style="font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 40px; max-width: 900px;">
        <h1>RBRE Results</h1>
        <p><strong>Submission ID:</strong> {submission_id}</p>
        <p><strong>Email:</strong> {email}</p>

        <h2>Summary</h2>
        <ul>
          <li><strong>Current CBSA:</strong> {results.get("current_cbsa")}</li>
          <li><strong>Targets:</strong> {", ".join(results.get("target_cbsas", [])) or "(none)"}</li>
          <li><strong>Unmapped targets:</strong> {", ".join(results.get("unmapped_targets", [])) or "(none)"}</li>
        </ul>

        <h2>Download</h2>
        <p><a href="{pdf_link}">Download PDF report</a> (link valid ~1 hour)</p>

        <details style="margin-top: 18px;">
          <summary style="cursor: pointer;">Show raw data</summary>
          <pre style="background:#f6f6f6; padding:12px; border-radius:8px; overflow:auto;">{json.dumps({"inputs": inputs, "results": results}, indent=2)}</pre>
        </details>
      </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/results/{submission_id}/report.pdf")
def report_pdf(submission_id: str):
    if not DATABASE_URL:
        raise HTTPException(status_code=500, detail={"error": "DATABASE_URL not configured"})

    try:
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("select pdf_bucket, pdf_key from submissions where id = %s", (submission_id,))
                row = cur.fetchone()
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "DB query failed", "reason": str(e)})

    if not row:
        raise HTTPException(status_code=404, detail={"error": "Submission not found"})

    bucket, key = row
    try:
        url = presigned_get_url(bucket=bucket, key=key, expires_seconds=3600)
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "Presign failed", "reason": str(e)})

    return RedirectResponse(url=url, status_code=302)
