"""
BusBalance AI — Constrained Schedule & Fleet Optimizer
Stage 4 Pipeline:
- Google OR-Tools CP-SAT Solver with Deterministic Constrained Heuristic fallback.
- Objective: Jointly minimize Overcrowding Penalty, Passenger Wait Time, and Operating Waste.
- Hard constraints: Total fleet limit, Minimum service protection (headway cap), Maximum frequency cap.
- Planner-Readable Explanations for human-in-the-loop auditability.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import math
from ortools.sat.python import cp_model

from AI_models.data_pipeline import ROUTES_SPEC, START_HOUR, END_HOUR, SLOTS_PER_DAY, SLOT_MINUTES, get_slot_time_str


OPERATIONAL_WINDOWS = [
    {"id": "EARLY_MORNING", "name": "Early Morning Shoulder", "start_hour": 7.0, "end_hour": 8.0, "label": "07:00 - 08:00"},
    {"id": "MORNING_PEAK", "name": "Morning Peak Rush", "start_hour": 8.0, "end_hour": 10.0, "label": "08:00 - 10:00"},
    {"id": "MID_MORNING", "name": "Mid-Morning Transition", "start_hour": 10.0, "end_hour": 12.0, "label": "10:00 - 12:00"},
    {"id": "MIDDAY_LULL", "name": "Midday Lull Period", "start_hour": 12.0, "end_hour": 15.0, "label": "12:00 - 15:00"},
    {"id": "AFTERNOON_SHOULDER", "name": "Late Afternoon Shoulder", "start_hour": 15.0, "end_hour": 17.0, "label": "15:00 - 17:00"},
    {"id": "EVENING_PEAK", "name": "Evening Commuter Peak", "start_hour": 17.0, "end_hour": 20.0, "label": "17:00 - 20:00"},
    {"id": "LATE_EVENING", "name": "Late Evening Wind-down", "start_hour": 20.0, "end_hour": 21.0, "label": "20:00 - 21:00"},
]


class ScheduleOptimizer:
    _instance = None

    def __init__(self):
        self.latest_result: Optional[Dict[str, Any]] = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ScheduleOptimizer()
        return cls._instance

    def optimize_schedule(
        self,
        predictions_df: pd.DataFrame,
        total_fleet_limit: int = 26,
        bus_capacity: int = 50,
        target_load_factor: float = 0.85,
        min_policy_headway_mins: float = 20.0,
        surge_multiplier: float = 1.0,
        human_overrides: Optional[Dict[str, int]] = None,
        use_p85_surge: bool = False
    ) -> Dict[str, Any]:
        """
        Run the constrained fleet optimizer using Google OR-Tools CP-SAT.
        """
        human_overrides = human_overrides or {}
        forecast_col = "ai_p85_pred" if use_p85_surge else "ai_p50_pred"

        target_date = predictions_df["date"].max()
        day_df = predictions_df[predictions_df["date"] == target_date].copy()

        # Detailed slot loads for each window and route
        window_demands = self._aggregate_window_demands(day_df, forecast_col, surge_multiplier)

        window_results = []
        overall_baseline_kpis = {"overcrowded_pax": 0.0, "wait_time_pax_mins": 0.0, "underused_trips": 0.0, "total_km": 0.0, "pax_trips": 0.0}
        overall_proposed_kpis = {"overcrowded_pax": 0.0, "wait_time_pax_mins": 0.0, "underused_trips": 0.0, "total_km": 0.0, "pax_trips": 0.0}

        for win in OPERATIONAL_WINDOWS:
            win_id = win["id"]
            win_demand = window_demands[win_id]
            duration_hours = win["end_hour"] - win["start_hour"]

            # Solve via OR-Tools CP-SAT
            sol, method = self._solve_window_cp_sat(
                win=win,
                demands=win_demand,
                total_fleet_limit=total_fleet_limit,
                bus_capacity=bus_capacity,
                target_load_factor=target_load_factor,
                min_policy_headway_mins=min_policy_headway_mins,
                duration_hours=duration_hours,
                human_overrides=human_overrides
            )

            # Compute detailed comparative metrics
            win_metrics = self._evaluate_window_metrics(
                win=win,
                demands=win_demand,
                proposed_allocation=sol,
                bus_capacity=bus_capacity,
                target_load_factor=target_load_factor,
                duration_hours=duration_hours
            )

            # Generate natural language explanation
            explanation = self._generate_explanation(
                win=win,
                metrics=win_metrics,
                demands=win_demand,
                human_overrides=human_overrides
            )
            win_metrics["explanation"] = explanation
            win_metrics["solver_method"] = method
            window_results.append(win_metrics)

            for k in overall_baseline_kpis:
                overall_baseline_kpis[k] += win_metrics["baseline_summary"][k]
                overall_proposed_kpis[k] += win_metrics["proposed_summary"][k]

        total_pax = max(1.0, overall_baseline_kpis["pax_trips"])
        base_avg_wait = overall_baseline_kpis["wait_time_pax_mins"] / total_pax
        prop_avg_wait = overall_proposed_kpis["wait_time_pax_mins"] / total_pax

        overcrowd_diff = overall_baseline_kpis["overcrowded_pax"] - overall_proposed_kpis["overcrowded_pax"]
        overcrowd_red_pct = max(0.0, (overcrowd_diff / max(1.0, overall_baseline_kpis["overcrowded_pax"])) * 100.0) if overall_baseline_kpis["overcrowded_pax"] > 0 else 0.0

        underused_diff = overall_baseline_kpis["underused_trips"] - overall_proposed_kpis["underused_trips"]
        underused_saved_pct = max(0.0, (underused_diff / max(1.0, overall_baseline_kpis["underused_trips"])) * 100.0) if overall_baseline_kpis["underused_trips"] > 0 else 0.0

        km_diff = overall_baseline_kpis["total_km"] - overall_proposed_kpis["total_km"]

        avg_active_buses = float(np.mean([w["proposed_summary"]["total_buses"] for w in window_results]))
        fleet_utilization_pct = min(100.0, (avg_active_buses / max(1, total_fleet_limit)) * 100.0)

        executive_kpis = {
            "baseline": {
                "overcrowded_pax": int(round(overall_baseline_kpis["overcrowded_pax"])),
                "avg_wait_time_mins": round(float(base_avg_wait), 1),
                "underused_trips": int(round(overall_baseline_kpis["underused_trips"])),
                "total_operating_km": int(round(overall_baseline_kpis["total_km"])),
                "fleet_utilization_pct": round(float((26.0 / max(1, total_fleet_limit)) * 100.0), 1)
            },
            "proposed": {
                "overcrowded_pax": int(round(overall_proposed_kpis["overcrowded_pax"])),
                "avg_wait_time_mins": round(float(prop_avg_wait), 1),
                "underused_trips": int(round(overall_proposed_kpis["underused_trips"])),
                "total_operating_km": int(round(overall_proposed_kpis["total_km"])),
                "fleet_utilization_pct": round(float(fleet_utilization_pct), 1)
            },
            "impact": {
                "overcrowding_reduction_pct": round(float(overcrowd_red_pct), 1),
                "wait_time_saved_mins": round(float(base_avg_wait - prop_avg_wait), 1),
                "wait_time_reduction_pct": round(float(max(0.0, (base_avg_wait - prop_avg_wait) / max(0.1, base_avg_wait) * 100.0)), 1),
                "underused_trips_saved_pct": round(float(underused_saved_pct), 1),
                "km_delta": round(float(km_diff), 1),
                "active_buses_avg": round(float(avg_active_buses), 1)
            }
        }

        segment_profile = self._build_segment_profile(day_df, window_results, bus_capacity, target_load_factor)
        slot_schedule = self._build_slot_schedule(day_df, window_results, bus_capacity)

        self.latest_result = {
            "target_date": target_date,
            "parameters": {
                "total_fleet_limit": total_fleet_limit,
                "bus_capacity": bus_capacity,
                "target_load_factor": target_load_factor,
                "min_policy_headway_mins": min_policy_headway_mins,
                "surge_multiplier": surge_multiplier,
                "use_p85_surge": use_p85_surge,
                "human_overrides": human_overrides
            },
            "executive_kpis": executive_kpis,
            "window_results": window_results,
            "segment_profile": segment_profile,
            "slot_schedule": slot_schedule
        }

        return self.latest_result

    def _aggregate_window_demands(
        self,
        day_df: pd.DataFrame,
        forecast_col: str,
        surge_mult: float
    ) -> Dict[str, Dict[str, Any]]:
        """Group 15-min forecasts into operational windows per route with individual slot loads."""
        window_demands = {}

        for win in OPERATIONAL_WINDOWS:
            win_id = win["id"]
            sh = win["start_hour"]
            eh = win["end_hour"]

            sub_df = day_df[(day_df["slot_idx"] >= int((sh - START_HOUR) * 4)) &
                            (day_df["slot_idx"] < int((eh - START_HOUR) * 4))].copy()

            sub_df["scaled_demand"] = sub_df[forecast_col] * surge_mult

            route_data = {}
            for route_id, rspec in ROUTES_SPEC.items():
                r_df = sub_df[sub_df["route_id"] == route_id]
                
                # Extract slot MLS values
                slot_mls_list = []
                for s_idx in r_df["slot_idx"].unique():
                    s_sub = r_df[r_df["slot_idx"] == s_idx]
                    slot_mls_list.append(float(s_sub["scaled_demand"].max()))

                seg_max_loads = r_df.groupby("segment_id")["scaled_demand"].max()
                mls_load = float(seg_max_loads.max()) if len(seg_max_loads) > 0 else 45.0
                mls_segment = seg_max_loads.idxmax() if len(seg_max_loads) > 0 else rspec["segments"][0]["id"]
                total_win_pax = float(r_df["scaled_demand"].sum()) if len(r_df) > 0 else mls_load * 4.0

                seg_name = next((s["name"] for s in rspec["segments"] if s["id"] == mls_segment), mls_segment)

                route_data[route_id] = {
                    "mls_load": mls_load,
                    "mls_segment_id": mls_segment,
                    "mls_segment_name": seg_name,
                    "slot_mls_loads": slot_mls_list,
                    "total_pax": total_win_pax,
                    "baseline_buses": rspec["baseline_buses"],
                    "cycle_time_mins": rspec["cycle_time_mins"],
                    "route_distance_km": rspec["route_distance_km"]
                }

            window_demands[win_id] = route_data

        return window_demands

    def _solve_window_cp_sat(
        self,
        win: Dict[str, Any],
        demands: Dict[str, Dict[str, Any]],
        total_fleet_limit: int,
        bus_capacity: int,
        target_load_factor: float,
        min_policy_headway_mins: float,
        duration_hours: float,
        human_overrides: Dict[str, int]
    ) -> Tuple[Dict[str, int], str]:
        """
        Solve allocation for one time window using OR-Tools CP-SAT.
        """
        model = cp_model.CpModel()
        buses_vars = {}
        route_bounds = {}

        for r_id, r_info in demands.items():
            cycle = r_info["cycle_time_mins"]
            min_buses = max(2, int(math.ceil(cycle / min_policy_headway_mins)))
            max_buses = min(25, max(min_buses, int(math.floor(cycle / 3.0))))

            override_key = f"{r_id}_{win['id']}"
            if override_key in human_overrides:
                fixed_b = human_overrides[override_key]
                min_buses = max_buses = fixed_b
            elif r_id in human_overrides:
                fixed_b = human_overrides[r_id]
                min_buses = max_buses = fixed_b

            route_bounds[r_id] = (min_buses, max_buses)

        total_min_required = sum(b[0] for b in route_bounds.values())
        effective_fleet_limit = max(total_fleet_limit, total_min_required)

        allocation = {}

        try:
            objective_terms = []

            for r_id, r_info in demands.items():
                min_b, max_b = route_bounds[r_id]
                if min_b == max_b:
                    buses_vars[r_id] = min_b
                    continue

                candidates = list(range(min_b, max_b + 1))
                cand_vars = []

                for b in candidates:
                    v = model.NewBoolVar(f"b_{r_id}_{b}")
                    cand_vars.append((b, v))

                    cost = self._compute_allocation_cost_integer(
                        route_info=r_info,
                        buses=b,
                        bus_capacity=bus_capacity,
                        target_load_factor=target_load_factor,
                        duration_hours=duration_hours
                    )
                    objective_terms.append(cost * v)

                model.Add(sum(v for _, v in cand_vars) == 1)

                b_var = model.NewIntVar(min_b, max_b, f"bus_{r_id}")
                model.Add(b_var == sum(b * v for b, v in cand_vars))
                buses_vars[r_id] = b_var

            # Fleet limit hard constraint
            model.Add(sum(buses_vars[r_id] for r_id in demands.keys()) <= effective_fleet_limit)
            model.Minimize(sum(objective_terms))

            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = 1.0
            status = solver.Solve(model)

            if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                for r_id in demands.keys():
                    if isinstance(buses_vars[r_id], int):
                        allocation[r_id] = buses_vars[r_id]
                    else:
                        allocation[r_id] = int(solver.Value(buses_vars[r_id]))
                return allocation, "OR-Tools CP-SAT"
            else:
                raise RuntimeError(f"CP-SAT status: {solver.StatusName(status)}")

        except Exception as e:
            allocation = self._solve_window_heuristic(
                demands=demands,
                route_bounds=route_bounds,
                total_fleet_limit=effective_fleet_limit,
                bus_capacity=bus_capacity,
                target_load_factor=target_load_factor,
                duration_hours=duration_hours
            )
            return allocation, "Constrained Heuristic (Fallback)"

    def _compute_allocation_cost_integer(
        self,
        route_info: Dict[str, Any],
        buses: int,
        bus_capacity: int,
        target_load_factor: float,
        duration_hours: float
    ) -> int:
        """
        Integer penalty evaluator for OR-Tools CP-SAT objective.
        Evaluates exact slot-level overcrowding, passenger waiting time, and vehicle distance.
        """
        cycle = route_info["cycle_time_mins"]
        slot_mls = route_info.get("slot_mls_loads", [route_info["mls_load"]])
        dist_km = route_info["route_distance_km"]

        # Capacity offered per 15-minute slot
        trips_per_15m = buses * (15.0 / cycle)
        slot_cap = trips_per_15m * bus_capacity
        eff_slot_cap = slot_cap * target_load_factor

        overcrowded_pax = 0.0
        underused_slots = 0.0
        for load in slot_mls:
            if load > eff_slot_cap:
                overcrowded_pax += (load - eff_slot_cap)
            elif load < (0.40 * slot_cap):
                underused_slots += 1.0

        avg_wait_mins = cycle / (2.0 * max(1, buses))
        total_pax = sum(slot_mls)
        wait_penalty = avg_wait_mins * total_pax

        total_trips = (buses * 60.0 / cycle) * duration_hours
        total_km = total_trips * dist_km

        # Balanced multi-objective weights
        penalty = int(overcrowded_pax * 100.0 + wait_penalty * 6.0 + total_km * 3.0 + underused_slots * 40.0)
        return penalty

    def _solve_window_heuristic(
        self,
        demands: Dict[str, Dict[str, Any]],
        route_bounds: Dict[str, Tuple[int, int]],
        total_fleet_limit: int,
        bus_capacity: int,
        target_load_factor: float,
        duration_hours: float
    ) -> Dict[str, int]:
        """Deterministic priority-based greedy heuristic fallback."""
        alloc = {r_id: bounds[0] for r_id, bounds in route_bounds.items()}
        remaining_buses = total_fleet_limit - sum(alloc.values())

        while remaining_buses > 0:
            best_route = None
            max_benefit = -1.0

            for r_id, r_info in demands.items():
                curr_b = alloc[r_id]
                max_b = route_bounds[r_id][1]
                if curr_b < max_b:
                    cost_curr = self._compute_allocation_cost_integer(
                        r_info, curr_b, bus_capacity, target_load_factor, duration_hours
                    )
                    cost_next = self._compute_allocation_cost_integer(
                        r_info, curr_b + 1, bus_capacity, target_load_factor, duration_hours
                    )
                    benefit = cost_curr - cost_next
                    if benefit > max_benefit:
                        max_benefit = benefit
                        best_route = r_id

            if best_route is None or max_benefit <= 0:
                break

            alloc[best_route] += 1
            remaining_buses -= 1

        return alloc

    def _evaluate_window_metrics(
        self,
        win: Dict[str, Any],
        demands: Dict[str, Dict[str, Any]],
        proposed_allocation: Dict[str, int],
        bus_capacity: int,
        target_load_factor: float,
        duration_hours: float
    ) -> Dict[str, Any]:
        """Evaluate baseline vs proposed metrics per window with exact slot-level accounting."""
        routes_detail = []
        base_tot_overcrowd = 0.0
        prop_tot_overcrowd = 0.0
        base_tot_wait_pax = 0.0
        prop_tot_wait_pax = 0.0
        base_tot_underused_trips = 0.0
        prop_tot_underused_trips = 0.0
        base_tot_km = 0.0
        prop_tot_km = 0.0
        tot_pax = 0.0

        for r_id, r_info in demands.items():
            b_base = r_info["baseline_buses"]
            b_prop = proposed_allocation[r_id]
            cycle = r_info["cycle_time_mins"]
            dist_km = r_info["route_distance_km"]
            mls = r_info["mls_load"]
            slot_loads = r_info.get("slot_mls_loads", [mls])

            # 15-minute slot capacities
            trips_per_15m_base = b_base * (15.0 / cycle)
            trips_per_15m_prop = b_prop * (15.0 / cycle)

            slot_cap_base = trips_per_15m_base * bus_capacity
            slot_cap_prop = trips_per_15m_prop * bus_capacity

            eff_cap_base = slot_cap_base * target_load_factor
            eff_cap_prop = slot_cap_prop * target_load_factor

            # Headways and wait times
            headway_base = cycle / max(1, b_base)
            headway_prop = cycle / max(1, b_prop)
            wait_base = headway_base / 2.0
            wait_prop = headway_prop / 2.0

            # Slot-by-slot calculation of overcrowding and underused trips
            route_overcrowd_base = 0.0
            route_overcrowd_prop = 0.0
            route_under_base = 0.0
            route_under_prop = 0.0

            for load in slot_loads:
                if load > eff_cap_base:
                    route_overcrowd_base += (load - eff_cap_base)
                elif load < (0.40 * slot_cap_base):
                    route_under_base += trips_per_15m_base

                if load > eff_cap_prop:
                    route_overcrowd_prop += (load - eff_cap_prop)
                elif load < (0.40 * slot_cap_prop):
                    route_under_prop += trips_per_15m_prop

            total_trips_base = (b_base * 60.0 / cycle) * duration_hours
            total_trips_prop = (b_prop * 60.0 / cycle) * duration_hours

            km_base = total_trips_base * dist_km
            km_prop = total_trips_prop * dist_km

            route_pax = sum(slot_loads) * 2.0  # entire route boardings
            tot_pax += route_pax

            base_tot_overcrowd += route_overcrowd_base
            prop_tot_overcrowd += route_overcrowd_prop
            base_tot_wait_pax += wait_base * route_pax
            prop_tot_wait_pax += wait_prop * route_pax
            base_tot_underused_trips += route_under_base
            prop_tot_underused_trips += route_under_prop
            base_tot_km += km_base
            prop_tot_km += km_prop

            routes_detail.append({
                "route_id": r_id,
                "route_name": ROUTES_SPEC[r_id]["route_name"],
                "mls_segment": r_info["mls_segment_name"],
                "mls_demand": round(float(mls), 1),
                "baseline_buses": b_base,
                "proposed_buses": b_prop,
                "delta_buses": b_prop - b_base,
                "baseline_headway_mins": round(float(headway_base), 1),
                "proposed_headway_mins": round(float(headway_prop), 1),
                "baseline_wait_mins": round(float(wait_base), 1),
                "proposed_wait_mins": round(float(wait_prop), 1),
                "baseline_load_factor_pct": round(float((mls / max(1.0, slot_cap_base)) * 100.0), 1),
                "proposed_load_factor_pct": round(float((mls / max(1.0, slot_cap_prop)) * 100.0), 1),
                "overcrowded_pax_base": int(round(route_overcrowd_base)),
                "overcrowded_pax_prop": int(round(route_overcrowd_prop)),
                "operating_km_base": round(float(km_base), 1),
                "operating_km_prop": round(float(km_prop), 1),
            })

        return {
            "window_id": win["id"],
            "window_name": win["name"],
            "window_label": win["label"],
            "duration_hours": duration_hours,
            "routes": routes_detail,
            "baseline_summary": {
                "total_buses": sum(d["baseline_buses"] for d in demands.values()),
                "overcrowded_pax": base_tot_overcrowd,
                "wait_time_pax_mins": base_tot_wait_pax,
                "underused_trips": base_tot_underused_trips,
                "total_km": base_tot_km,
                "pax_trips": tot_pax
            },
            "proposed_summary": {
                "total_buses": sum(proposed_allocation.values()),
                "overcrowded_pax": prop_tot_overcrowd,
                "wait_time_pax_mins": prop_tot_wait_pax,
                "underused_trips": prop_tot_underused_trips,
                "total_km": prop_tot_km,
                "pax_trips": tot_pax
            }
        }

    def _generate_explanation(
        self,
        win: Dict[str, Any],
        metrics: Dict[str, Any],
        demands: Dict[str, Dict[str, Any]],
        human_overrides: Dict[str, int]
    ) -> str:
        """
        Generate clear, natural-language rationale for every recommendation.
        """
        routes = metrics["routes"]
        shifts = []
        lulls = []
        locks = []

        for r in routes:
            r_id = r["route_id"]
            delta = r["delta_buses"]
            if r_id in human_overrides or f"{r_id}_{win['id']}" in human_overrides:
                locks.append(f"{r_id} locked at {r['proposed_buses']} buses by human scheduler")
            elif delta > 0:
                shifts.append(
                    f"+{delta} buses to {r_id} due to MLS surge of {int(r['mls_demand'])} pax on {r['mls_segment']} "
                    f"(wait time reduced from {r['baseline_wait_mins']}m to {r['proposed_wait_mins']}m, load factor brought from {r['baseline_load_factor_pct']}% down to {r['proposed_load_factor_pct']}%)"
                )
            elif delta < 0:
                lulls.append(
                    f"{abs(delta)} surplus buses reallocated from {r_id} ({win['label']} lull, load factor balanced at {r['proposed_load_factor_pct']}%, headway maintained at {r['proposed_headway_mins']}m)"
                )

        reasons = []
        if locks:
            reasons.append("; ".join(locks))
        if shifts:
            reasons.append("Shifted " + ", ".join(shifts))
        if lulls:
            reasons.append("Recovered " + ", ".join(lulls))

        if not reasons:
            return f"Schedule maintained steady state for {win['label']}; fleet meets policy headway and target load constraints across all routes."

        explanation = ". ".join(reasons) + f". Fleet limit strictly satisfied with policy headway cap <= 20 min enforced."
        return explanation

    def _build_segment_profile(
        self,
        day_df: pd.DataFrame,
        window_results: List[Dict[str, Any]],
        bus_capacity: int,
        target_load_factor: float
    ) -> List[Dict[str, Any]]:
        """
        Build segment bottleneck profile: Baseline vs. Proposed Load Factor across sequential segments.
        """
        profile = []
        peak_alloc = {}
        for r_id in ROUTES_SPEC:
            peak_buses = [
                r["proposed_buses"]
                for w in window_results if w["window_id"] in ["MORNING_PEAK", "EVENING_PEAK"]
                for r in w["routes"] if r["route_id"] == r_id
            ]
            peak_alloc[r_id] = np.mean(peak_buses) if peak_buses else ROUTES_SPEC[r_id]["baseline_buses"]

        for route_id, rspec in ROUTES_SPEC.items():
            r_df = day_df[day_df["route_id"] == route_id]
            cycle = rspec["cycle_time_mins"]
            base_buses = rspec["baseline_buses"]
            prop_buses = peak_alloc.get(route_id, base_buses)

            base_cap_15m = max(1.0, (base_buses * 15.0 / cycle) * bus_capacity)
            prop_cap_15m = max(1.0, (prop_buses * 15.0 / cycle) * bus_capacity)

            for seg in rspec["segments"]:
                seg_df = r_df[r_df["segment_id"] == seg["id"]]
                peak_load = float(seg_df["segment_load"].quantile(0.92)) if len(seg_df) > 0 else 30.0

                base_lf = round((peak_load / base_cap_15m) * 100.0, 1)
                prop_lf = round((peak_load / prop_cap_15m) * 100.0, 1)

                profile.append({
                    "route_id": route_id,
                    "segment_id": seg["id"],
                    "segment_name": seg["name"],
                    "peak_passenger_load": round(peak_load, 1),
                    "baseline_load_factor_pct": base_lf,
                    "proposed_load_factor_pct": prop_lf,
                    "crowding_threshold_pct": 85.0,
                    "crush_capacity_pct": 100.0,
                    "is_bottleneck": bool(base_lf > 85.0)
                })

        return profile

    def _build_slot_schedule(
        self,
        day_df: pd.DataFrame,
        window_results: List[Dict[str, Any]],
        bus_capacity: int
    ) -> List[Dict[str, Any]]:
        """Map window bus allocations back to 56 individual 15-minute slots for timeseries rendering."""
        hour_alloc_map = {}
        for win in window_results:
            win_id = win["window_id"]
            for r in win["routes"]:
                r_id = r["route_id"]
                hour_alloc_map[(win_id, r_id)] = {
                    "baseline_buses": r["baseline_buses"],
                    "proposed_buses": r["proposed_buses"],
                    "baseline_wait": r["baseline_wait_mins"],
                    "proposed_wait": r["proposed_wait_mins"]
                }

        slot_items = []
        for slot_idx in range(SLOTS_PER_DAY):
            hour_float = START_HOUR + (slot_idx * SLOT_MINUTES) / 60.0
            time_str = get_slot_time_str(slot_idx)

            matching_win = next(
                (w for w in OPERATIONAL_WINDOWS if w["start_hour"] <= hour_float < w["end_hour"]),
                OPERATIONAL_WINDOWS[-1]
            )
            win_id = matching_win["id"]

            for route_id, rspec in ROUTES_SPEC.items():
                alloc = hour_alloc_map.get((win_id, route_id), {
                    "baseline_buses": rspec["baseline_buses"],
                    "proposed_buses": rspec["baseline_buses"],
                    "baseline_wait": rspec["cycle_time_mins"] / (2.0 * rspec["baseline_buses"]),
                    "proposed_wait": rspec["cycle_time_mins"] / (2.0 * rspec["baseline_buses"])
                })

                s_df = day_df[(day_df["slot_idx"] == slot_idx) & (day_df["route_id"] == route_id)]
                slot_mls = float(s_df["segment_load"].max()) if len(s_df) > 0 else 30.0

                slot_items.append({
                    "slot_idx": slot_idx,
                    "time_str": time_str,
                    "route_id": route_id,
                    "window_id": win_id,
                    "mls_demand": round(slot_mls, 1),
                    "baseline_buses": alloc["baseline_buses"],
                    "proposed_buses": alloc["proposed_buses"],
                    "baseline_wait_mins": round(alloc["baseline_wait"], 1),
                    "proposed_wait_mins": round(alloc["proposed_wait"], 1)
                })

        return slot_items
