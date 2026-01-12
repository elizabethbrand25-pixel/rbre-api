# build_metro_cost_data.py
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple

import requests
import pandas as pd


HUD_BASE = "https://www.huduser.gov/hudapi/public/fmr"  # :contentReference[oaicite:12]{index=12}


@dataclass
class MetroCostProfile:
    cbsa: str                 # numeric CBSA code, e.g. "42660"
    metro_name: str           # e.g. "Seattle-Tacoma-Bellevue, WA MSA"
    state: str                # primary state abbrev extracted from name
    rent_1br: int             # HUD FMR
    rent_2br: int             # HUD FMR
    rpp_all_items: Optional[float] = None  # BEA metro RPP index (US=100)
    utilities_proxy_monthly: Optional[float] = None  # EIA avg monthly bill, by state
    transport_proxy_monthly: float = 230.0  # simple default
    entry_friction: float = 0.50            # heuristic 0..1, refine later


def parse_cbsa_numeric(hud_cbsa_code: str) -> str:
    """
    HUD listMetroAreas returns cbsa_code strings like:
      "METRO10180M10180" or "METRO29180N22001" (HUD metro FMR areas can differ from OMB CBSAs)
    We extract the first 5-digit code as the CBSA-like identifier.
    """
    m = re.search(r"(\d{5})", hud_cbsa_code)
    if not m:
        raise ValueError(f"Could not parse CBSA from HUD cbsa_code: {hud_cbsa_code}")
    return m.group(1)


def extract_primary_state(area_name: str) -> str:
    """
    Extract trailing state abbreviation if present, e.g. '..., WA MSA' -> 'WA'.
    If multi-state, you may want to store 'MULTI' or first state; MVP uses first.
    """
    m = re.search(r",\s*([A-Z]{2})(\s|$)", area_name)
    return m.group(1) if m else "NA"


def hud_get_metro_list(hud_token: str, updated_year: Optional[int] = None) -> List[Dict[str, Any]]:
    headers = {"Authorization": f"Bearer {hud_token}"}
    url = f"{HUD_BASE}/listMetroAreas"
    if updated_year:
        url += f"?updated={updated_year}"
    r = requests.get(url, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json()["data"]


def hud_get_fmr_for_entity(hud_token: str, entityid: str, year: Optional[int] = None) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {hud_token}"}
    url = f"{HUD_BASE}/data/{entityid}"
    if year:
        url += f"?year={year}"
    r = requests.get(url, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json()


def build_from_hud(hud_token: str, year: Optional[int] = None) -> Dict[str, MetroCostProfile]:
    metros = hud_get_metro_list(hud_token)
    out: Dict[str, MetroCostProfile] = {}

    for m in metros:
        entityid = m["cbsa_code"]         # HUD entity id for /fmr/data/{entityid} :contentReference[oaicite:13]{index=13}
        area_name = m["area_name"]
        cbsa = parse_cbsa_numeric(entityid)
        state = extract_primary_state(area_name)

        fmr = hud_get_fmr_for_entity(hud_token, entityid, year=year)

        # HUD response structure varies slightly; these keys are commonly present:
        # We try several known patterns.
        one = None
        two = None

        # Pattern 1: fields like "One-Bedroom", "Two-Bedroom"
        if "data" in fmr and isinstance(fmr["data"], dict):
            d = fmr["data"]
            one = d.get("One-Bedroom") or d.get("One Bedroom") or d.get("one_bedroom")
            two = d.get("Two-Bedroom") or d.get("Two Bedroom") or d.get("two_bedroom")

        # Pattern 2: list of items (statewide responses etc.)
        if (one is None or two is None) and "data" in fmr and isinstance(fmr["data"], list):
            # try to find matching entity
            for row in fmr["data"]:
                if str(row.get("code", "")).find(cbsa) >= 0 or row.get("name") == area_name:
                    one = row.get("One-Bedroom") or row.get("One Bedroom")
                    two = row.get("Two-Bedroom") or row.get("Two Bedroom")
                    break

        if one is None or two is None:
            # Skip if we can't parse (you can log these)
            continue

        prof = MetroCostProfile(
            cbsa=cbsa,
            metro_name=area_name,
            state=state,
            rent_1br=int(float(one)),
            rent_2br=int(float(two)),
        )
        out[cbsa] = prof

    return out


def load_bea_rpp_metro_csv(path: str, year: Optional[int] = None) -> Dict[str, float]:
    """
    Load a BEA RPP metro file you downloaded from the BEA RPP page.
    You'll need to identify column names once (they vary by BEA download format).

    Expected: a CBSA/GeoFIPS-like code + an 'All items' RPP value for a year.
    BEA RPP metro page: :contentReference[oaicite:14]{index=14}
    """
    df = pd.read_csv(path)

    # Try common column name candidates
    code_cols = ["CBSA", "cbsa", "GeoFIPS", "geofips", "LineCode", "Linecode", "Code"]
    value_cols = ["RPP", "rpp", "All items", "All_Items", "AllItems", "RPP All items", "Value", "value"]

    code_col = next((c for c in code_cols if c in df.columns), None)
    val_col = next((c for c in value_cols if c in df.columns), None)

    if code_col is None or val_col is None:
        raise ValueError(f"Could not find CBSA/value columns in BEA file. Columns: {list(df.columns)}")

    # Optional year filtering if the file contains multiple years
    if year is not None and "Year" in df.columns:
        df = df[df["Year"] == year]

    out: Dict[str, float] = {}
    for _, row in df.iterrows():
        cbsa = str(row[code_col]).zfill(5)
        try:
            out[cbsa] = float(row[val_col])
        except Exception:
            continue
    return out


def load_eia_table5a_xlsx(path: str) -> Dict[str, float]:
    """
    Load EIA Table 5A XLSX (Residential Average Monthly Bill by State).
    EIA provides XLSX for Table 5A: :contentReference[oaicite:15]{index=15}
    """
    df = pd.read_excel(path)

    # The file often has header rows; find a row where "State" appears as a column or value.
    # We'll attempt common patterns.
    # Try to locate columns by name
    cols = [c.strip() if isinstance(c, str) else c for c in df.columns]
    df.columns = cols

    # Common columns in Table 5A exports:
    state_col = next((c for c in df.columns if str(c).lower() == "state"), None)
    bill_col = None
    for c in df.columns:
        if "average monthly bill" in str(c).lower():
            bill_col = c
            break

    if state_col is None or bill_col is None:
        # fallback: scan for likely columns
        raise ValueError(f"Could not find expected columns in EIA table. Columns: {list(df.columns)}")

    out: Dict[str, float] = {}
    for _, row in df.iterrows():
        st = row.get(state_col)
        bill = row.get(bill_col)
        if isinstance(st, str) and len(st.strip()) == 2:
            try:
                out[st.strip().upper()] = float(bill)
            except Exception:
                pass
    return out


def apply_simple_entry_friction(profile: MetroCostProfile) -> float:
    """
    MVP heuristic:
    - Higher rents tend to correlate with tighter markets (not always, but acceptable proxy v1)
    Scale rent_2br roughly into 0.25..0.85
    """
    r = profile.rent_2br
    # tune these thresholds as you learn from user outcomes
    if r >= 3000:
        return 0.80
    if r >= 2500:
        return 0.70
    if r >= 2000:
        return 0.60
    if r >= 1600:
        return 0.50
    if r >= 1300:
        return 0.42
    return 0.35


def main():
    hud_token = os.environ.get("HUD_TOKEN")
    if not hud_token:
        raise SystemExit("Set HUD_TOKEN env var with your HUD User API token.")

    # 1) Pull metro rents from HUD
    metro_profiles = build_from_hud(hud_token=hud_token, year=None)

    # 2) Load BEA RPP metro file (downloaded manually)
    #    Replace with your path
    bea_path = "bea_rpp_metro.csv"
    bea_rpp = load_bea_rpp_metro_csv(bea_path, year=None) if os.path.exists(bea_path) else {}

    # 3) Load EIA Table 5A XLSX (downloaded manually)
    eia_path = "table_5A.xlsx"
    eia_bills = load_eia_table5a_xlsx(eia_path) if os.path.exists(eia_path) else {}

    # 4) Merge
    for cbsa, prof in metro_profiles.items():
        prof.rpp_all_items = bea_rpp.get(cbsa)
        prof.utilities_proxy_monthly = eia_bills.get(prof.state)
        prof.entry_friction = apply_simple_entry_friction(prof)

    # 5) Output JSON
    out = {cbsa: asdict(prof) for cbsa, prof in metro_profiles.items()}
    with open("cost_data_metro.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {len(out)} metros to cost_data_metro.json")


if __name__ == "__main__":
    main()
