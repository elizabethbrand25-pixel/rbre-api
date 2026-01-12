"""
Relocation Budget Reality Engine (RBRE) — Domestic US
Engine-only code (math + rules + affordability + paths). No UI, no payments, no PDF.

Design goals:
- Deterministic, auditable outcomes (no AI needed for decisions)
- Accepts a single JSON-like payload
- Returns a structured result JSON (ready for templating into PDF/email)
- Uses a lightweight "cost index" dataset you can extend/refresh quarterly

You can run:
    python rbre_engine.py

Then integrate into:
- a serverless function (AWS Lambda, Cloud Run, etc.)
- a backend API (FastAPI/Flask)
- Make/Zapier code module
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Literal
import math
import re
import uuid
from datetime import datetime


HouseholdType = Literal["individual", "couple"]
TimelineFlex = Literal["tight", "moderate", "flexible"]   # <3 months, 3-6, 6-12
RiskTol = Literal["low", "medium", "high"]
ProductTier = Literal["snapshot", "full"]


# -----------------------------
# Lightweight datasets (extendable)
# -----------------------------

@dataclass(frozen=True)
class LocationCostProfile:
    """
    Minimal, explainable cost model:
    - rent_1br / rent_2br approximate medians
    - utilities_proxy monthly
    - transport_proxy monthly
    - entry_friction: 0..1 (higher = harder to secure housing quickly; market tightness)
    - notes: short human-readable qualifiers
    """
    city: str
    state: str
    rent_1br: int
    rent_2br: int
    utilities_proxy: int
    transport_proxy: int
    entry_friction: float
    notes: str = ""


# Sample seed dataset. Replace/extend with your own curated list.
# Tip: Start with ~50–200 cities; refresh quarterly.
COST_DATA: Dict[str, LocationCostProfile] = {
    "PORTLAND, OR": LocationCostProfile("Portland", "OR", rent_1br=1700, rent_2br=2200, utilities_proxy=220, transport_proxy=220, entry_friction=0.65, notes="High demand, limited vacancy in some neighborhoods."),
    "SEATTLE, WA": LocationCostProfile("Seattle", "WA", rent_1br=2100, rent_2br=2800, utilities_proxy=240, transport_proxy=240, entry_friction=0.75),
    "BOISE, ID": LocationCostProfile("Boise", "ID", rent_1br=1400, rent_2br=1800, utilities_proxy=210, transport_proxy=230, entry_friction=0.55),
    "SPOKANE, WA": LocationCostProfile("Spokane", "WA", rent_1br=1200, rent_2br=1500, utilities_proxy=200, transport_proxy=220, entry_friction=0.45),
    "TULSA, OK": LocationCostProfile("Tulsa", "OK", rent_1br=1100, rent_2br=1350, utilities_proxy=190, transport_proxy=210, entry_friction=0.35),
    "KANSAS CITY, MO": LocationCostProfile("Kansas City", "MO", rent_1br=1250, rent_2br=1600, utilities_proxy=200, transport_proxy=220, entry_friction=0.40),
    "PITTSBURGH, PA": LocationCostProfile("Pittsburgh", "PA", rent_1br=1350, rent_2br=1700, utilities_proxy=210, transport_proxy=200, entry_friction=0.40),
    "CLEVELAND, OH": LocationCostProfile("Cleveland", "OH", rent_1br=1150, rent_2br=1450, utilities_proxy=200, transport_proxy=200, entry_friction=0.35),
    "ALBUQUERQUE, NM": LocationCostProfile("Albuquerque", "NM", rent_1br=1200, rent_2br=1500, utilities_proxy=210, transport_proxy=230, entry_friction=0.40),
    "RALEIGH, NC": LocationCostProfile("Raleigh", "NC", rent_1br=1500, rent_2br=1900, utilities_proxy=210, transport_proxy=230, entry_friction=0.55),
    "CHARLOTTE, NC": LocationCostProfile("Charlotte", "NC", rent_1br=1500, rent_2br=1900, utilities_proxy=210, transport_proxy=240, entry_friction=0.55),
    "ATLANTA, GA": LocationCostProfile("Atlanta", "GA", rent_1br=1550, rent_2br=2000, utilities_proxy=220, transport_proxy=250, entry_friction=0.55),
    "PHOENIX, AZ": LocationCostProfile("Phoenix", "AZ", rent_1br=1500, rent_2br=1900, utilities_proxy=240, transport_proxy=250, entry_friction=0.60, notes="Summer utilities can spike."),
}


# Candidate pool keys. In a production system, you can include more and refresh regularly.
DEFAULT_CANDIDATE_POOL: List[str] = [
    "TULSA, OK", "KANSAS CITY, MO", "CLEVELAND, OH", "PITTSBURGH, PA",
    "ALBUQUERQUE, NM", "SPOKANE, WA", "BOISE, ID", "RALEIGH, NC",
    "CHARLOTTE, NC", "ATLANTA, GA", "PHOENIX, AZ"
]


# -----------------------------
# Input & output models
# -----------------------------

@dataclass
class RBREInput:
    product_tier: ProductTier
    household_type: HouseholdType
    downsizing: bool

    # Financials (USD)
    net_monthly_income: float
    fixed_monthly_obligations: float
    liquid_savings: float

    timeline: TimelineFlex
    risk_tolerance: RiskTol

    current_location: str  # "City, ST"
    targets: Optional[List[str]] = None  # up to 3 like ["Boise, ID", "Spokane, WA"]


@dataclass
class BudgetEnvelope:
    safety_buffer: float
    deployable_relocation_capital: float
    safe_ceiling: float
    stretch_ceiling: float
    failure_threshold: float
    free_cash_flow: float
    monthly_living_cost_proxy: float
    buffer_months: int


@dataclass
class Flags:
    infeasible: bool
    high_risk_cashflow: bool
    negative_cashflow: bool
    tight_timeline: bool
    high_entry_friction: bool
    downsizing_risk: bool


@dataclass
class LocationAssessment:
    location_key: str
    city: str
    state: str

    est_monthly_cost: float
    affordability_delta_pct: float  # relative to current location

    feasibility_score: int  # 0-100
    dominant_risk_driver: str
    sensitivity_note: str

    recommended: bool
    tags: List[str]


@dataclass
class ExecutionPath:
    name: str  # Conservative/Balanced/Aggressive
    summary: str
    timeline_weeks: Tuple[int, int]
    budget_allocation: Dict[str, float]
    milestones: List[str]
    abort_conditions: List[str]
    risk_notes: List[str]


@dataclass
class RBREResult:
    report_id: str
    generated_at_utc: str
    input_echo: Dict[str, Any]
    envelope: Dict[str, Any]
    flags: Dict[str, Any]
    baseline: Dict[str, Any]
    target_assessments: List[Dict[str, Any]]
    affordable_recommendations: List[Dict[str, Any]]
    paths: Dict[str, List[Dict[str, Any]]]  # keyed by location_key


# -----------------------------
# Helpers: normalization & validation
# -----------------------------

def normalize_location(loc: str) -> str:
    """
    Normalize "City, ST" -> "CITY, ST" and remove extra spaces.
    """
    if not isinstance(loc, str) or not loc.strip():
        raise ValueError("location must be a non-empty string like 'City, ST'")
    loc = loc.strip()
    # Allow "City ST" or "City,ST"
    loc = re.sub(r"\s*,\s*", ", ", loc)
    if "," not in loc:
        # try last token as state
        parts = loc.split()
        if len(parts) < 2:
            raise ValueError(f"Invalid location format: {loc!r}. Expected 'City, ST'.")
        loc = " ".join(parts[:-1]) + ", " + parts[-1]
    city, state = [p.strip() for p in loc.split(",", 1)]
    if len(state) != 2:
        # allow longer but you should standardize in UI
        state = state[:2]
    return f"{city.upper()}, {state.upper()}"


def clamp(n: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, n))


def round_money(x: float) -> float:
    return float(round(x, 2))


# -----------------------------
# Core engine
# -----------------------------

class RBREEngine:
    def __init__(
        self,
        cost_data: Dict[str, LocationCostProfile],
        candidate_pool: Optional[List[str]] = None
    ) -> None:
        self.cost_data = cost_data
        self.candidate_pool = candidate_pool or DEFAULT_CANDIDATE_POOL

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        inp = self._parse_input(payload)

        report_id = str(uuid.uuid4())
        now_utc = datetime.utcnow().isoformat(timespec="seconds") + "Z"

        current_key = normalize_location(inp.current_location)
        baseline_profile = self._require_profile(current_key)

        envelope = self._compute_budget_envelope(inp, baseline_profile)
        flags = self._compute_flags(inp, envelope, baseline_profile)

        # Determine which locations to assess:
        target_keys = self._build_target_set(inp, current_key)
        assessments = []
        for k in target_keys:
            if k == current_key:
                continue
            prof = self._require_profile(k)
            assessments.append(
                self._assess_location(inp, envelope, baseline_profile, prof)
            )

        # Separate "affordable recommendations" (relative savings + fit-qualified)
        affordable = self._select_affordable_recommendations(assessments)

        # Build execution paths
        paths = self._build_paths(inp, envelope, flags, baseline_profile, assessments, affordable)

        result = RBREResult(
            report_id=report_id,
            generated_at_utc=now_utc,
            input_echo=self._safe_input_echo(inp),
            envelope=asdict(envelope),
            flags=asdict(flags),
            baseline={
                "current_location_key": current_key,
                "baseline_monthly_cost_est": round_money(self._estimate_monthly_cost(inp, baseline_profile)),
                "baseline_rent_used": self._rent_for_household(inp, baseline_profile),
                "notes": baseline_profile.notes,
            },
            target_assessments=[asdict(a) for a in assessments],
            affordable_recommendations=[asdict(a) for a in affordable],
            paths={k: [asdict(p) for p in v] for k, v in paths.items()},
        )
        return asdict(result)

    # -------------------------
    # Parsing
    # -------------------------

    def _parse_input(self, payload: Dict[str, Any]) -> RBREInput:
        def req(key: str) -> Any:
            if key not in payload:
                raise ValueError(f"Missing required field: {key}")
            return payload[key]

        product_tier = req("product_tier")
        if product_tier not in ("snapshot", "full"):
            raise ValueError("product_tier must be 'snapshot' or 'full'")

        household_type = req("household_type")
        if household_type not in ("individual", "couple"):
            raise ValueError("household_type must be 'individual' or 'couple'")

        downsizing = bool(req("downsizing"))

        nmi = float(req("net_monthly_income"))
        fmo = float(req("fixed_monthly_obligations"))
        sav = float(req("liquid_savings"))
        if nmi < 0 or fmo < 0 or sav < 0:
            raise ValueError("Income, obligations, and savings must be non-negative numbers.")

        timeline = req("timeline")
        if timeline not in ("tight", "moderate", "flexible"):
            raise ValueError("timeline must be 'tight', 'moderate', or 'flexible'")

        risk = req("risk_tolerance")
        if risk not in ("low", "medium", "high"):
            raise ValueError("risk_tolerance must be 'low', 'medium', or 'high'")

        current_location = req("current_location")
        targets = payload.get("targets") or []
        if targets and not isinstance(targets, list):
            raise ValueError("targets must be a list like ['Boise, ID', 'Spokane, WA']")
        targets = targets[:3]  # enforce cap

        return RBREInput(
            product_tier=product_tier,
            household_type=household_type,
            downsizing=downsizing,
            net_monthly_income=nmi,
            fixed_monthly_obligations=fmo,
            liquid_savings=sav,
            timeline=timeline,
            risk_tolerance=risk,
            current_location=current_location,
            targets=targets,
        )

    def _safe_input_echo(self, inp: RBREInput) -> Dict[str, Any]:
        # Avoid echoing anything sensitive beyond what's needed; you can tailor this.
        return {
            "product_tier": inp.product_tier,
            "household_type": inp.household_type,
            "downsizing": inp.downsizing,
            "net_monthly_income": round_money(inp.net_monthly_income),
            "fixed_monthly_obligations": round_money(inp.fixed_monthly_obligations),
            "liquid_savings": round_money(inp.liquid_savings),
            "timeline": inp.timeline,
            "risk_tolerance": inp.risk_tolerance,
            "current_location": normalize_location(inp.current_location),
            "targets": [normalize_location(t) for t in (inp.targets or [])],
        }

    # -------------------------
    # Data lookup
    # -------------------------

    def _require_profile(self, location_key: str) -> LocationCostProfile:
        if location_key not in self.cost_data:
            # In production, you can:
            # - fall back to state-level averages
            # - or request "closest match"
            raise KeyError(f"Location not found in COST_DATA: {location_key}")
        return self.cost_data[location_key]

    # -------------------------
    # Budget math & buffers
    # -------------------------

    def _buffer_months(self, inp: RBREInput) -> int:
        # Conservative defaults for trust.
        base = 4 if inp.household_type == "individual" else 5
        if inp.downsizing:
            base += 1

        # Risk tolerance adjusts buffer months slightly
        if inp.risk_tolerance == "low":
            base += 1
        elif inp.risk_tolerance == "high":
            base -= 1

        # Timeline tight increases fragility; require more buffer.
        if inp.timeline == "tight":
            base += 1

        return int(clamp(base, 3, 8))

    def _estimate_variable_expenses(self, inp: RBREInput, baseline: LocationCostProfile) -> float:
        """
        Variable expense proxy; keep it simple and transparent.
        We assume variable costs scale somewhat with baseline utilities+transport.
        """
        # A small conservative floor so low-obligation users don't under-budget:
        floor = 600 if inp.household_type == "individual" else 900

        # Use baseline utilities + transport plus a cushion:
        prox = baseline.utilities_proxy + baseline.transport_proxy + (350 if inp.household_type == "individual" else 550)

        if inp.downsizing:
            prox += 75  # downsizing transitions often create misc spending

        return float(max(floor, prox))

    def _compute_budget_envelope(self, inp: RBREInput, baseline: LocationCostProfile) -> BudgetEnvelope:
        fcf = inp.net_monthly_income - inp.fixed_monthly_obligations

        var = self._estimate_variable_expenses(inp, baseline)
        mlc = inp.fixed_monthly_obligations + var

        months = self._buffer_months(inp)
        sb = mlc * months
        drc = inp.liquid_savings - sb

        safe = max(0.0, drc * 0.70)
        stretch = max(0.0, drc * 0.90)
        fail = max(0.0, drc)

        return BudgetEnvelope(
            safety_buffer=round_money(sb),
            deployable_relocation_capital=round_money(drc),
            safe_ceiling=round_money(safe),
            stretch_ceiling=round_money(stretch),
            failure_threshold=round_money(fail),
            free_cash_flow=round_money(fcf),
            monthly_living_cost_proxy=round_money(mlc),
            buffer_months=months,
        )

    # -------------------------
    # Flags & risk signals
    # -------------------------

    def _compute_flags(self, inp: RBREInput, env: BudgetEnvelope, baseline: LocationCostProfile) -> Flags:
        negative_cf = env.free_cash_flow < 0
        high_risk_cf = env.free_cash_flow <= 0 or (env.free_cash_flow / max(1.0, inp.net_monthly_income)) < 0.05
        infeasible = env.deployable_relocation_capital <= 0
        tight = inp.timeline == "tight"
        friction = baseline.entry_friction >= 0.70
        downsizing_risk = inp.downsizing

        return Flags(
            infeasible=bool(infeasible),
            high_risk_cashflow=bool(high_risk_cf),
            negative_cashflow=bool(negative_cf),
            tight_timeline=bool(tight),
            high_entry_friction=bool(friction),
            downsizing_risk=bool(downsizing_risk),
        )

    # -------------------------
    # Location selection & affordability
    # -------------------------

    def _build_target_set(self, inp: RBREInput, current_key: str) -> List[str]:
        keys: List[str] = []
        # User targets first
        for t in (inp.targets or []):
            k = normalize_location(t)
            if k not in keys:
                keys.append(k)

        # Add candidate pool as fallback / supplement
        for k in self.candidate_pool:
            if k not in keys and k in self.cost_data:
                keys.append(k)

        # Ensure baseline included for comparisons
        if current_key not in keys:
            keys.insert(0, current_key)

        # Cap assessments to keep runtime predictable in serverless settings
        return keys[:40]

    def _rent_for_household(self, inp: RBREInput, prof: LocationCostProfile) -> float:
        return float(prof.rent_1br if inp.household_type == "individual" else prof.rent_2br)

    def _estimate_monthly_cost(self, inp: RBREInput, prof: LocationCostProfile) -> float:
        """
        Simple, explainable monthly cost estimate:
        rent + utilities + transport + a generic 'other variable' pad.
        This is not intended to be perfect; it should be consistent and honest.
        """
        rent = self._rent_for_household(inp, prof)

        # Other variable pad: groceries/household/etc. Keep stable and modest.
        other = 700 if inp.household_type == "individual" else 1100

        if inp.downsizing:
            other += 50

        return float(rent + prof.utilities_proxy + prof.transport_proxy + other)

    def _assess_location(
        self,
        inp: RBREInput,
        env: BudgetEnvelope,
        baseline_prof: LocationCostProfile,
        target_prof: LocationCostProfile,
    ) -> LocationAssessment:
        baseline_cost = self._estimate_monthly_cost(inp, baseline_prof)
        target_cost = self._estimate_monthly_cost(inp, target_prof)

        delta_pct = (baseline_cost - target_cost) / max(1.0, baseline_cost) * 100.0

        # Feasibility score components (0..100), weighted
        # 1) affordability pressure vs income (rent-to-income proxy)
        rent = self._rent_for_household(inp, target_prof)
        rent_ratio = rent / max(1.0, inp.net_monthly_income)  # monthly rent / monthly net income
        rent_score = 100 - clamp((rent_ratio - 0.25) * 200, 0, 70)  # 25% is comfy; above that decreases

        # 2) entry friction penalty
        friction_pen = target_prof.entry_friction * 20  # up to 20 points

        # 3) timeline compression penalty
        timeline_pen = 0
        if inp.timeline == "tight":
            timeline_pen = 12
        elif inp.timeline == "moderate":
            timeline_pen = 6

        # 4) buffer erosion speed (how quickly monthly cost burns DRC if income gap occurs)
        # Assume a 1-month gap scenario: target_cost must be coverable by safe ceiling?
        # This is simplistic but effective.
        gap_pen = 0
        if env.safe_ceiling > 0:
            # If safe ceiling can't cover ~1 month of target_cost + typical move fixed costs, penalize.
            if env.safe_ceiling < (target_cost * 1.2):
                gap_pen = 10

        # 5) downsizing misc risk
        down_pen = 4 if inp.downsizing else 0

        raw = rent_score - friction_pen - timeline_pen - gap_pen - down_pen
        score = int(clamp(raw, 0, 100))

        # dominant risk driver
        drivers = []
        drivers.append(("Housing affordability pressure", 100 - rent_score))
        drivers.append(("Market entry friction", friction_pen))
        drivers.append(("Timeline compression", timeline_pen))
        drivers.append(("Buffer fragility", gap_pen))
        drivers.append(("Downsizing transition risk", down_pen))
        dominant = max(drivers, key=lambda x: x[1])[0]

        # sensitivity: rent +$200 impact (score change)
        rent_ratio_200 = (rent + 200) / max(1.0, inp.net_monthly_income)
        rent_score_200 = 100 - clamp((rent_ratio_200 - 0.25) * 200, 0, 70)
        raw_200 = rent_score_200 - friction_pen - timeline_pen - gap_pen - down_pen
        score_200 = int(clamp(raw_200, 0, 100))
        sensitivity = f"If rent is $200 higher than expected, feasibility score drops from {score} to {score_200}."

        # recommendation logic for "affordable alternatives"
        recommended = (delta_pct >= 15.0) and (score >= 60) and (target_prof.entry_friction <= 0.70)

        tags = []
        if delta_pct >= 15:
            tags.append("meaningfully_cheaper")
        elif delta_pct >= 5:
            tags.append("moderately_cheaper")
        elif delta_pct <= -5:
            tags.append("more_expensive")
        else:
            tags.append("comparable")

        if target_prof.entry_friction >= 0.70:
            tags.append("tight_market")
        if rent_ratio >= 0.35:
            tags.append("high_rent_pressure")
        if inp.timeline == "tight":
            tags.append("tight_timeline")

        return LocationAssessment(
            location_key=f"{target_prof.city.upper()}, {target_prof.state.upper()}",
            city=target_prof.city,
            state=target_prof.state,
            est_monthly_cost=round_money(target_cost),
            affordability_delta_pct=round_money(delta_pct),
            feasibility_score=score,
            dominant_risk_driver=dominant,
            sensitivity_note=sensitivity,
            recommended=recommended,
            tags=tags,
        )

    def _select_affordable_recommendations(self, assessments: List[LocationAssessment]) -> List[LocationAssessment]:
        # Only show 3–6 max; prioritize meaningful savings + feasibility.
        candidates = [a for a in assessments if a.affordability_delta_pct >= 5 and a.feasibility_score >= 55]
        # Primary filter: meaningful cheaper
        meaningful = [a for a in candidates if a.affordability_delta_pct >= 15 and a.feasibility_score >= 60]
        moderate = [a for a in candidates if 5 <= a.affordability_delta_pct < 15 and a.feasibility_score >= 65]

        # Sort by (delta, score)
        meaningful.sort(key=lambda x: (x.affordability_delta_pct, x.feasibility_score), reverse=True)
        moderate.sort(key=lambda x: (x.affordability_delta_pct, x.feasibility_score), reverse=True)

        recs = meaningful[:6]
        if len(recs) < 3:
            recs += moderate[: (6 - len(recs))]

        return recs[:6]

    # -------------------------
    # Execution paths
    # -------------------------

    def _build_paths(
        self,
        inp: RBREInput,
        env: BudgetEnvelope,
        flags: Flags,
        baseline_prof: LocationCostProfile,
        assessments: List[LocationAssessment],
        affordable: List[LocationAssessment],
    ) -> Dict[str, List[ExecutionPath]]:
        """
        Returns execution paths keyed by location_key.
        - Snapshot tier: only Balanced for recommended affordable locations + any user targets
        - Full tier: Conservative + Balanced + Aggressive for recommended (and for user targets, if present)
        """
        # Which locations get paths?
        user_targets = {normalize_location(t) for t in (inp.targets or [])}
        # Build a set of prioritized location keys:
        prioritized = []
        for a in assessments:
            is_user_target = a.location_key in user_targets
            if is_user_target or a.recommended:
                prioritized.append(a.location_key)
        prioritized = prioritized[:6]  # cap for report cleanliness

        out: Dict[str, List[ExecutionPath]] = {}
        for loc_key in prioritized:
            a = next(x for x in assessments if x.location_key == loc_key)
            prof = self._require_profile(loc_key)

            if flags.infeasible:
                # If infeasible overall, provide a single "stabilize first" path.
                out[loc_key] = [self._path_stabilize_first(inp, env, a)]
                continue

            if inp.product_tier == "snapshot":
                out[loc_key] = [self._path_balanced(inp, env, a, prof)]
            else:
                out[loc_key] = [
                    self._path_conservative(inp, env, a, prof),
                    self._path_balanced(inp, env, a, prof),
                    self._path_aggressive(inp, env, a, prof),
                ]
        return out

    def _common_budget_buckets(self, env: BudgetEnvelope, aggressiveness: float) -> Dict[str, float]:
        """
        aggressiveness: 0.0 conservative, 0.5 balanced, 1.0 aggressive
        Allocate within the *safe or stretch* ceiling (not failure threshold).
        """
        base_cap = env.safe_ceiling if aggressiveness <= 0.5 else env.stretch_ceiling
        cap = max(0.0, base_cap)

        # Allocation heuristics:
        # - deposits/upfront rent dominate; overlap is big risk; buffer reserved matters
        deposits = cap * (0.30 + 0.10 * aggressiveness)
        move_exec = cap * (0.18 + 0.06 * aggressiveness)
        overlap = cap * (0.20 - 0.05 * aggressiveness)
        setup = cap * 0.10
        variance = cap * (0.12 - 0.04 * aggressiveness)
        reserve = cap - (deposits + move_exec + overlap + setup + variance)

        # Guardrails
        reserve = max(0.0, reserve)
        return {
            "Deposits & upfront rent": round_money(deposits),
            "Move execution": round_money(move_exec),
            "Overlap housing & utilities": round_money(overlap),
            "Post-move setup": round_money(setup),
            "Variance buffer": round_money(variance),
            "Remaining reserve within ceiling": round_money(reserve),
        }

    def _abort_conditions_base(self, env: BudgetEnvelope) -> List[str]:
        return [
            f"If liquid savings fall below your safety buffer (${env.safety_buffer:,.0f}), pause relocation planning and stabilize.",
            f"If you exceed your safe ceiling (${env.safe_ceiling:,.0f}) before securing housing, stop and re-scope the plan.",
        ]

    def _path_stabilize_first(self, inp: RBREInput, env: BudgetEnvelope, a: LocationAssessment) -> ExecutionPath:
        milestones = [
            "Reduce fixed obligations or increase income to restore positive free cash flow.",
            "Build liquid savings above the safety buffer threshold.",
            "Re-run feasibility with updated numbers before signing any lease or making non-refundable payments.",
        ]
        aborts = [
            "Do not commit to a lease, deposit, or moving contract while deployable relocation capital is <= $0.",
            "Avoid financing relocation costs (high default risk during transition).",
        ]
        summary = (
            "Your current financial constraints make relocation high-risk right now. "
            "This plan focuses on stabilizing cash flow and rebuilding a safety buffer before attempting a move."
        )
        return ExecutionPath(
            name="Stabilize First",
            summary=summary,
            timeline_weeks=(8, 24),
            budget_allocation={
                "Primary focus": 0.0
            },
            milestones=milestones,
            abort_conditions=aborts,
            risk_notes=[
                "Attempting relocation without deployable capital often leads to late fees, debt, or forced return moves.",
                "Stability first reduces the probability of cascading failure in the first 3–6 months after moving.",
            ],
        )

    def _path_conservative(self, inp: RBREInput, env: BudgetEnvelope, a: LocationAssessment, prof: LocationCostProfile) -> ExecutionPath:
        allocation = self._common_budget_buckets(env, aggressiveness=0.0)
        timeline = (10, 20) if inp.timeline != "tight" else (8, 14)
        milestones = [
            "Confirm housing ranges and entry requirements (income multiple, deposits, fees).",
            "Run a ‘two-month overlap’ stress test and ensure it stays within safe ceiling.",
            "Book move logistics with cancellation flexibility.",
            "Sign housing only after costs match the plan and buffer remains intact.",
            "Post-move: keep spending tight for 60 days to rebuild deployable capital.",
        ]
        aborts = self._abort_conditions_base(env) + [
            "If required deposit + fees exceed 1.5× expected upfront costs, switch units/area or delay move.",
        ]
        notes = [
            "Higher success probability; slower timeline; fewer surprises.",
            "Best for low risk tolerance or variable income.",
        ]
        return ExecutionPath(
            name="Conservative",
            summary="A low-risk execution plan prioritizing buffer protection and controlled commitments.",
            timeline_weeks=timeline,
            budget_allocation=allocation,
            milestones=milestones,
            abort_conditions=aborts,
            risk_notes=notes,
        )

    def _path_balanced(self, inp: RBREInput, env: BudgetEnvelope, a: LocationAssessment, prof: LocationCostProfile) -> ExecutionPath:
        allocation = self._common_budget_buckets(env, aggressiveness=0.5)
        timeline = (6, 12) if inp.timeline != "flexible" else (8, 14)
        milestones = [
            "Validate the top 2 housing options and total move-in costs.",
            "Choose a move window that minimizes overlap (target <= 4 weeks).",
            "Lock moving method (DIY vs hired) based on cost/effort tradeoff.",
            "Use a single ‘go/no-go’ checkpoint: confirm you’re under safe ceiling before paying deposits.",
            "Post-move: set a 30-day stabilization budget and rebuild reserves.",
        ]
        aborts = self._abort_conditions_base(env)
        notes = [
            "Most realistic for most users: clear tradeoffs, manageable stress.",
            "If timeline becomes tight, increase buffer and reduce overlap risk immediately.",
        ]
        return ExecutionPath(
            name="Balanced",
            summary="A practical plan balancing speed and safety while keeping costs within a realistic ceiling.",
            timeline_weeks=timeline,
            budget_allocation=allocation,
            milestones=milestones,
            abort_conditions=aborts,
            risk_notes=notes,
        )

    def _path_aggressive(self, inp: RBREInput, env: BudgetEnvelope, a: LocationAssessment, prof: LocationCostProfile) -> ExecutionPath:
        allocation = self._common_budget_buckets(env, aggressiveness=1.0)
        timeline = (3, 7) if inp.timeline == "tight" else (4, 8)
        milestones = [
            "Pre-approve housing requirements (proof of income, references, funds ready).",
            "Keep overlap <= 2 weeks; avoid non-refundable commitments until housing is secured.",
            "Choose the simplest move method to reduce coordination risk.",
            "Set strict spending rules for the first 60 days after moving.",
        ]
        aborts = [
            f"If expected total relocation costs exceed your stretch ceiling (${env.stretch_ceiling:,.0f}), do not proceed.",
            f"If housing cannot be secured within 14 days, pause and switch to the Balanced plan.",
        ]
        notes = [
            "Fastest path, but higher failure risk if anything slips (housing delays, surprise fees, income disruption).",
            "Only appropriate if income is stable and you can respond quickly to setbacks.",
        ]
        return ExecutionPath(
            name="Aggressive",
            summary="A fast execution plan with thin margins; explicitly higher risk and strict abort triggers.",
            timeline_weeks=timeline,
            budget_allocation=allocation,
            milestones=milestones,
            abort_conditions=aborts,
            risk_notes=notes,
        )


# -----------------------------
# Demo / local test
# -----------------------------

if __name__ == "__main__":
    engine = RBREEngine(cost_data=COST_DATA)

    sample_payload = {
        "product_tier": "full",
        "household_type": "couple",
        "downsizing": True,
        "net_monthly_income": 6500,
        "fixed_monthly_obligations": 3200,
        "liquid_savings": 22000,
        "timeline": "moderate",
        "risk_tolerance": "medium",
        "current_location": "Portland, OR",
        "targets": ["Boise, ID", "Spokane, WA"]
    }

    try:
        out = engine.run(sample_payload)
        # Pretty-print without external deps
        import json
        print(json.dumps(out, indent=2))
    except Exception as e:
        print("ERROR:", str(e))
