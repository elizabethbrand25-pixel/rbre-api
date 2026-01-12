from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple

import requests
import pandas as pd

HUD_BASE = "https://www.huduser.gov/hudapi/public/fmr"
EIA_TABLE_5A_XLSX = "https://www.eia.gov/electricity/sales_revenue_price/xls/table_5A.xlsx"


@dataclass
class MetroCostProfile:
    cbsa: str
    metro_name: str
    state: str
    rent_1br: int
    rent_2br: int
    utilities_proxy_monthly: Optional[float] = None
    transport_proxy_monthly: float = 230.0
    entry_friction: float = 0.50


def parse_cbsa5(hud_entityid: str) -> str:
    m = re.search(r"(\d{5})", str(hud_entityid))
    if not m:
        raise ValueError(f"Could not parse 5-digit CBSA from HUD entity id: {hud_entityid}")
    return m.group(1)


def extract_primary_state(area_name: str) -> str:
    # "Portland-Vancouver-Hillsboro, OR-WA MSA" -> "OR"
    m = re.search(r",\s*([A-Z]{2})", str(area_name))
    return m.group(1) if m else "NA"


def hud_get_json(url: str, token: str) -> Tuple[int, Any]:
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(url, headers=headers, timeout=60)
    try:
        payload = r.json()
    except Exception:
        raise RuntimeError(
            f"HUD endpoint did not return JSON. HTTP {r.status_code}. First 200 chars: {r.text[:200]!r}"
        )
    return r.status_code, payload


def hud_list_metros(token: str) -> list[dict]:
    status, payload = hud_get_json(f"{HUD_BASE}/listMetroAreas", token)
    if status != 200:
        raise RuntimeError(f"HUD listMetroAreas error HTTP {status}: {payload}")

    # HUD returns either list or dict; you observed list
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return payload["data"]

    raise RuntimeError(f"Unexpected HUD listMetroAreas payload shape: {type(payload)} -> {payload!r}")


def hud_get_fmr(token: str, entityid: str, year: Optional[int] = None) -> Optional[dict]:
    url = f"{HUD_BASE}/data/{entityid}"
    if year:
        url += f"?year={year}"

    status, payload = hud_get_json(url, token)

    # Some listed entity IDs legitimately have no data; skip them
    if status == 404:
        return None

    if status != 200:
        raise RuntimeError(f"HUD FMR data error HTTP {status} for {entityid}: {payload}")

    return payload




def extract_1br_2br(fmr_payload: dict) -> Tuple[Optional[int], Optional[int]]:
    """
    HUD FMR API puts bedroom rents inside:
      payload["data"]["basicdata"]

    basicdata is either:
      - dict with keys "One-Bedroom", "Two-Bedroom", ...
      - list of dicts (Small Area FMR metros), where one row has zip_code == "MSA level"
    """
    if not isinstance(fmr_payload, dict):
        return None, None

    data = fmr_payload.get("data")
    if not isinstance(data, dict):
        return None, None

    basic = data.get("basicdata")
    if basic is None:
        return None, None

    # Case 1: basicdata is a dict
    if isinstance(basic, dict):
        one = basic.get("One-Bedroom") or basic.get("One Bedroom")
        two = basic.get("Two-Bedroom") or basic.get("Two Bedroom")
        try:
            return int(float(one)), int(float(two))
        except Exception:
            return None, None

    # Case 2: basicdata is a list (Small Area FMR metros)
    if isinstance(basic, list) and basic:
        msa_row = None
        for row in basic:
            if isinstance(row, dict) and str(row.get("zip_code", "")).strip().lower() == "msa level":
                msa_row = row
                break
        if msa_row is None:
            msa_row = basic[0] if isinstance(basic[0], dict) else None
        if not isinstance(msa_row, dict):
            return None, None

        one = msa_row.get("One-Bedroom") or msa_row.get("One Bedroom")
        two = msa_row.get("Two-Bedroom") or msa_row.get("Two Bedroom")
        try:
            return int(float(one)), int(float(two))
        except Exception:
            return None, None

    return None, None

STATE_NAME_TO_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "District of Columbia": "DC",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL",
    "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
    "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR",
    "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
    "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA",
    "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}

def load_eia_state_bills() -> Dict[str, float]:
    raw = pd.read_excel(EIA_TABLE_5A_XLSX, header=None)

    header_row = None
    for i in range(min(30, len(raw))):
        row = raw.iloc[i].astype(str).str.strip().str.lower().tolist()
        if "state" in row:
            header_row = i
            break

    if header_row is None:
        raise RuntimeError("Could not find header row containing 'State' in EIA table.")

    df = pd.read_excel(EIA_TABLE_5A_XLSX, header=header_row)
    df.columns = [str(c).strip() for c in df.columns]

    state_col = None
    bill_col = None

    for c in df.columns:
        if c.lower() == "state":
            state_col = c
        if "average" in c.lower() and "bill" in c.lower():
            bill_col = c

    if not state_col or not bill_col:
        raise RuntimeError(f"Could not locate required columns. Found: {df.columns.tolist()}")

    out: Dict[str, float] = {}

    for _, row in df.iterrows():
        state_raw = row.get(state_col)
        bill_raw = row.get(bill_col)

        if not state_raw:
            continue

        state = str(state_raw).strip()

        if len(state) == 2 and state.isalpha():
            state_abbr = state.upper()
        else:
            state_abbr = STATE_NAME_TO_ABBR.get(state.title())

        if not state_abbr:
            continue

        bill = pd.to_numeric(bill_raw, errors="coerce")
        if pd.isna(bill):
            continue

        out[state_abbr] = float(bill)

    if not out:
        raise RuntimeError("Parsed 0 state bills from EIA table.")

    return out



def entry_friction_from_rent(rent_2br: int) -> float:
    if rent_2br >= 3000:
        return 0.80
    if rent_2br >= 2500:
        return 0.70
    if rent_2br >= 2000:
        return 0.60
    if rent_2br >= 1600:
        return 0.50
    if rent_2br >= 1300:
        return 0.42
    return 0.35

def main() -> None:
    token = os.environ.get("HUD_TOKEN", "").strip()
    if not token:
        raise SystemExit(
            'Missing HUD_TOKEN. Set it with: setx HUD_TOKEN "YOUR_TOKEN" then reopen PowerShell.'
        )

    year_env = os.environ.get("HUD_YEAR", "").strip()
    year = int(year_env) if year_env.isdigit() else None

    print("Loading EIA utility proxy (Table 5A)...")
    state_bills = load_eia_state_bills()

    print("Fetching metro list from HUD...")
    metros = hud_list_metros(token)
    print(f"HUD metros returned: {len(metros)}")

    out: Dict[str, Dict[str, Any]] = {}
    skipped = 0

    for m in metros:
        entityid = m.get("cbsa_code")
        name = m.get("area_name")
        category = m.get("category")

        if not entityid or not name or category != "MetroArea":
            skipped += 1
            continue

        cbsa = parse_cbsa5(entityid)
        state = extract_primary_state(name)

        fmr = hud_get_fmr(token, entityid, year=year)
        if fmr is None:
            skipped += 1
            continue

        rent_1br, rent_2br = extract_1br_2br(fmr)
        if rent_1br is None or rent_2br is None:
            skipped += 1
            continue

        prof = MetroCostProfile(
            cbsa=cbsa,
            metro_name=name,
            state=state,
            rent_1br=rent_1br,
            rent_2br=rent_2br,
            utilities_proxy_monthly=state_bills.get(state),
            transport_proxy_monthly=230.0,
            entry_friction=entry_friction_from_rent(rent_2br),
        )

        out[cbsa] = asdict(prof)

    with open("cost_data_metro.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"✅ Wrote {len(out)} metros to cost_data_metro.json")
    print(f"Skipped {skipped} rows.")




if __name__ == "__main__":
    main()
