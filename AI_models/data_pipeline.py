"""
BusBalance AI — Data Pipeline & Synthetic Seed Generator
Realistic transit dataset generator for Ahmedabad–Gandhinagar Transport Corridor (BRTS Janmarg & GSRTC).
Enforces Conservation of Flow and Spatiotemporal Feature Engineering.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import os

# Ahmedabad–Gandhinagar Route Definitions
ROUTES_SPEC = {
    "R101": {
        "route_name": "RTO Circle -> ISKCON (BRTS Trunk Corridor)",
        "cycle_time_mins": 75,
        "route_distance_km": 18.0,
        "baseline_buses": 10,
        "segments": [
            {"id": "R101_S1", "name": "RTO Circle -> Visat", "distance_km": 3.8, "base_demand_weight": 0.85},
            {"id": "R101_S2", "name": "Visat -> Memnagar", "distance_km": 5.2, "base_demand_weight": 1.45},
            {"id": "R101_S3", "name": "Memnagar -> Shivranjani", "distance_km": 4.6, "base_demand_weight": 1.65},
            {"id": "R101_S4", "name": "Shivranjani -> ISKCON", "distance_km": 4.4, "base_demand_weight": 1.10},
        ],
        "morning_peak_boost": 1.45,   # +45% morning peak spike (08:00 - 10:00)
        "midday_lull_factor": 0.65,    # -35% midday lull (12:00 - 15:00)
        "evening_peak_boost": 1.30,   # +30% evening rush (17:30 - 20:00)
    },
    "R201": {
        "route_name": "RTO Circle -> Narol via Kalupur Railway Stn (Transit Hub)",
        "cycle_time_mins": 60,
        "route_distance_km": 14.0,
        "baseline_buses": 8,
        "segments": [
            {"id": "R201_S1", "name": "RTO Circle -> Kalupur Station", "distance_km": 4.5, "base_demand_weight": 1.25},
            {"id": "R201_S2", "name": "Kalupur Station -> Geeta Mandir", "distance_km": 3.2, "base_demand_weight": 1.55},
            {"id": "R201_S3", "name": "Geeta Mandir -> Narol", "distance_km": 6.3, "base_demand_weight": 1.05},
        ],
        "morning_peak_boost": 1.30,   # Railway connection peaks
        "midday_lull_factor": 0.80,    # Steady transit-hub baseline
        "evening_peak_boost": 1.35,   # Intercity rail/bus transfers
    },
    "G301": {
        "route_name": "Geeta Mandir -> Gandhinagar Sachivalaya (Twin-City Commuter)",
        "cycle_time_mins": 90,
        "route_distance_km": 25.0,
        "baseline_buses": 8,
        "segments": [
            {"id": "G301_S1", "name": "Geeta Mandir -> Income Tax", "distance_km": 3.5, "base_demand_weight": 0.90},
            {"id": "G301_S2", "name": "Income Tax -> Koteshwar/Motera", "distance_km": 7.5, "base_demand_weight": 1.35},
            {"id": "G301_S3", "name": "Koteshwar/Motera -> Adalaj", "distance_km": 8.0, "base_demand_weight": 1.50},
            {"id": "G301_S4", "name": "Adalaj -> Gandhinagar Sachivalaya", "distance_km": 6.0, "base_demand_weight": 1.40},
        ],
        "morning_peak_boost": 1.50,   # Heavy government secretariat influx
        "midday_lull_factor": 0.55,    # Pronounced non-commuter lull
        "evening_peak_boost": 1.45,   # Evening return exodus
    }
}

SLOT_MINUTES = 15
START_HOUR = 7    # 07:00
END_HOUR = 21     # 21:00
SLOTS_PER_DAY = (END_HOUR - START_HOUR) * (60 // SLOT_MINUTES)  # 56 slots


def get_slot_time_str(slot_idx: int) -> str:
    """Convert slot index (0..55) to 'HH:MM' string starting from 07:00."""
    total_minutes = START_HOUR * 60 + slot_idx * SLOT_MINUTES
    hour = total_minutes // 60
    minute = total_minutes % 60
    return f"{hour:02d}:{minute:02d}"


def generate_synthetic_transit_data(num_days: int = 28, seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic transit corridor data covering 3 routes with 15-minute resolution.
    Strictly guarantees Conservation of Flow:
    segment_load[i] = segment_load[i-1] + boardings[i] - alightings[i].
    """
    np.random.seed(seed)
    base_date = datetime.now().date() - timedelta(days=num_days)
    records = []

    for d in range(num_days):
        current_date = base_date + timedelta(days=d)
        day_of_week = current_date.weekday()  # 0=Monday, 6=Sunday
        is_weekend = 1 if day_of_week >= 5 else 0

        # Simulate sporadic weather (monsoon showers) and special events (e.g. match at Motera)
        rain_mm = 0.0
        if np.random.rand() < 0.15:
            rain_mm = round(float(np.random.uniform(5.0, 32.0)), 1)

        special_event_flag = 0
        if (day_of_week == 5 or day_of_week == 6) and np.random.rand() < 0.25:
            special_event_flag = 1  # Weekend event at Motera / SG Highway

        for route_id, rspec in ROUTES_SPEC.items():
            num_segments = len(rspec["segments"])

            for slot_idx in range(SLOTS_PER_DAY):
                time_str = get_slot_time_str(slot_idx)
                hour_float = START_HOUR + (slot_idx * SLOT_MINUTES) / 60.0

                # Calculate temporal demand multiplier based on real Ahmedabad corridor patterns
                # Morning peak: 08:00 - 10:00 (peak around 09:00)
                if 8.0 <= hour_float < 10.0:
                    peak_intensity = np.sin((hour_float - 8.0) / 2.0 * np.pi)
                    time_mult = 1.0 + (rspec["morning_peak_boost"] - 1.0) * peak_intensity
                # Midday lull: 12:00 - 15:00 (lowest around 13:30)
                elif 12.0 <= hour_float < 15.0:
                    lull_depth = np.sin((hour_float - 12.0) / 3.0 * np.pi)
                    time_mult = 1.0 - (1.0 - rspec["midday_lull_factor"]) * lull_depth
                # Evening peak: 17:00 - 20:00 (peak around 18:30)
                elif 17.0 <= hour_float < 20.0:
                    peak_intensity = np.sin((hour_float - 17.0) / 3.0 * np.pi)
                    time_mult = 1.0 + (rspec["evening_peak_boost"] - 1.0) * peak_intensity
                else:
                    time_mult = 0.85

                # Weekend adjustments
                if is_weekend:
                    if route_id == "G301":
                        time_mult *= 0.45  # Government offices closed on weekends
                    elif route_id == "R101":
                        time_mult *= 0.90  # Leisure/shopping on SG Highway
                    else:
                        time_mult *= 0.80

                # Weather surge adjustment (rain slows mobility, causes boarding bunches)
                if rain_mm > 0:
                    time_mult *= (1.0 + min(0.35, rain_mm * 0.015))

                # Special event surge (specifically impacts Motera on G301 and ISKCON on R101)
                if special_event_flag and hour_float >= 16.0:
                    if route_id in ["G301", "R101"]:
                        time_mult *= 1.45

                timestamp_dt = datetime.combine(
                    current_date,
                    datetime.min.time()
                ) + timedelta(minutes=int(slot_idx * SLOT_MINUTES + START_HOUR * 60))

                # Sequential segment passenger flow (Conservation of Flow)
                running_bus_load = 0

                for seg_idx, seg in enumerate(rspec["segments"]):
                    base_rate = 28.0 * seg["base_demand_weight"] * time_mult
                    # Add realistic noise
                    noise_factor = np.random.uniform(0.85, 1.18)
                    expected_boardings = base_rate * noise_factor

                    boardings = int(np.maximum(2, np.random.poisson(expected_boardings)))

                    # Alightings: Origin segment has 0 or minimal alightings; terminal has all alightings
                    if seg_idx == 0:
                        alightings = int(np.random.binomial(boardings, 0.05)) if boardings > 0 else 0
                    elif seg_idx == num_segments - 1:
                        # At terminal segment, all passengers aboard plus earlier alight
                        potential_alight = running_bus_load + boardings
                        # Must alight most of the load as trip terminates
                        alightings = int(potential_alight * np.random.uniform(0.88, 1.0))
                        alightings = min(alightings, running_bus_load + boardings)
                    else:
                        # Intermediate segments: alight a fraction of onboard passengers
                        alighting_fraction = np.random.uniform(0.30, 0.55)
                        alightings = int(np.random.binomial(max(1, running_bus_load), min(0.95, alighting_fraction)))

                    # Enforce Conservation of Flow strictly
                    new_load = max(0, running_bus_load + boardings - alightings)
                    running_bus_load = new_load

                    records.append({
                        "timestamp": timestamp_dt.strftime("%Y-%m-%d %H:%M:%S"),
                        "date": current_date.strftime("%Y-%m-%d"),
                        "time_str": time_str,
                        "slot_idx": slot_idx,
                        "route_id": route_id,
                        "route_name": rspec["route_name"],
                        "segment_id": seg["id"],
                        "segment_name": seg["name"],
                        "segment_order": seg_idx,
                        "day_of_week": day_of_week,
                        "is_weekend": is_weekend,
                        "rain_mm": rain_mm,
                        "special_event_flag": special_event_flag,
                        "boardings": boardings,
                        "alightings": alightings,
                        "segment_load": new_load,
                        "cycle_time_mins": rspec["cycle_time_mins"],
                        "route_distance_km": rspec["route_distance_km"],
                        "segment_distance_km": seg["distance_km"],
                        "baseline_buses": rspec["baseline_buses"]
                    })

    df = pd.DataFrame(records)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer spatiotemporal features:
    - Cyclical time encodings (slot_sin, slot_cos)
    - Temporal lags (load_lag_1, load_lag_2, load_lag_yesterday)
    - Spatial lag (upstream_load_lag_1 from segment i-1)
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by=["route_id", "segment_id", "timestamp"]).reset_index(drop=True)

    # Cyclical slot encodings
    df["slot_sin"] = np.sin(2 * np.pi * df["slot_idx"] / float(SLOTS_PER_DAY))
    df["slot_cos"] = np.cos(2 * np.pi * df["slot_idx"] / float(SLOTS_PER_DAY))

    # Temporal lags per route & segment
    grouped = df.groupby(["route_id", "segment_id"])
    df["load_lag_1"] = grouped["segment_load"].shift(1)
    df["load_lag_2"] = grouped["segment_load"].shift(2)
    # 56 slots per day for daily lag
    df["load_lag_yesterday"] = grouped["segment_load"].shift(SLOTS_PER_DAY)

    # Fill initial missing lag values with segment median / current load
    df["load_lag_1"] = df["load_lag_1"].fillna(df["segment_load"])
    df["load_lag_2"] = df["load_lag_2"].fillna(df["load_lag_1"])
    df["load_lag_yesterday"] = df["load_lag_yesterday"].fillna(df["segment_load"])

    # Spatial lag: load from upstream segment (segment_order - 1) at the same timestamp
    # Create lookup map of (timestamp, route_id, segment_order) -> segment_load
    upstream_map = df.set_index(["timestamp", "route_id", "segment_order"])["segment_load"].to_dict()

    def get_upstream(row):
        if row["segment_order"] == 0:
            return 0  # Origin segment has no upstream segment
        key = (row["timestamp"], row["route_id"], row["segment_order"] - 1)
        return upstream_map.get(key, 0)

    df["upstream_load_lag_1"] = df.apply(get_upstream, axis=1)

    return df


class DataStore:
    """In-memory data store for BusBalance AI."""
    _instance = None

    def __init__(self):
        self.raw_df: Optional[pd.DataFrame] = None
        self.featured_df: Optional[pd.DataFrame] = None
        self.is_initialized: bool = False

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = DataStore()
        return cls._instance

    def initialize_data(self, force_refresh: bool = False, csv_path: Optional[str] = None) -> pd.DataFrame:
        if self.is_initialized and not force_refresh and self.featured_df is not None:
            return self.featured_df

        if csv_path and os.path.exists(csv_path):
            print(f"[DataStore] Loading corridor data from {csv_path}...")
            self.raw_df = pd.read_csv(csv_path)
        else:
            print("[DataStore] Generating synthetic seed data for Ahmedabad-Gandhinagar corridor...")
            self.raw_df = generate_synthetic_transit_data(num_days=28, seed=42)
            if csv_path:
                os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                self.raw_df.to_csv(csv_path, index=False)
                print(f"[DataStore] Saved seed dataset to {csv_path}")

        print("[DataStore] Engineering spatiotemporal features...")
        self.featured_df = engineer_features(self.raw_df)
        self.is_initialized = True
        print(f"[DataStore] Initialized successfully. Total rows: {len(self.featured_df)}")
        return self.featured_df

    def load_custom_csv(self, df_upload: pd.DataFrame) -> pd.DataFrame:
        """Load uploaded custom CSV dataset and engineer required features."""
        required_cols = {"timestamp", "route_id", "segment_id", "slot_idx", "segment_load", "baseline_buses"}
        missing = required_cols - set(df_upload.columns)
        if missing:
            raise ValueError(f"Uploaded CSV is missing mandatory columns: {missing}")

        self.raw_df = df_upload
        self.featured_df = engineer_features(df_upload)
        self.is_initialized = True
        return self.featured_df
