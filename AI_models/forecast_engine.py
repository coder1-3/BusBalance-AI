"""
BusBalance AI — Demand Forecasting Engine
Stage 3 Pipeline:
- Time-based Train/Test Split (no future leakage).
- Dual model evaluation: Historical Baseline vs. HistGradientBoostingRegressor (P50 & P85 surge).
- Accuracy metrics: MAE, RMSE, MAPE (%).
- Maximum Load Section (MLS) detection & overcrowding risk tagging.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import joblib
import os

FEATURE_COLS = [
    "slot_idx",
    "slot_sin",
    "slot_cos",
    "day_of_week",
    "is_weekend",
    "rain_mm",
    "special_event_flag",
    "segment_order",
    "load_lag_1",
    "load_lag_2",
    "load_lag_yesterday",
    "upstream_load_lag_1",
    "route_code",
    "segment_code"
]

def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1.0) -> float:
    """Mean Absolute Percentage Error with epsilon to prevent division by zero."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, epsilon))) * 100.0)


class ForecastEngine:
    _instance = None

    def __init__(self):
        self.model_p50: Optional[HistGradientBoostingRegressor] = None
        self.model_p85: Optional[HistGradientBoostingRegressor] = None
        self.baseline_lookup: Dict[Tuple[str, str, int], float] = {}
        self.metrics: Dict[str, Any] = {}
        self.route_map: Dict[str, int] = {}
        self.segment_map: Dict[str, int] = {}
        self.is_trained: bool = False
        self.latest_predictions_df: Optional[pd.DataFrame] = None
        self.train_test_split_date: Optional[str] = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ForecastEngine()
        return cls._instance

    def train_and_evaluate(
        self,
        df: pd.DataFrame,
        test_days: int = 7,
        bus_capacity: int = 50,
        target_load_factor: float = 0.85
    ) -> Dict[str, Any]:
        """
        Train both Historical Baseline and ML Forecasters using a strict time-based split.
        Computes MAE, RMSE, MAPE on the held-out test window.
        """
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(by="timestamp").reset_index(drop=True)

        # Categorical encoding for route and segment
        unique_routes = sorted(df["route_id"].unique())
        unique_segments = sorted(df["segment_id"].unique())
        self.route_map = {r: i for i, r in enumerate(unique_routes)}
        self.segment_map = {s: i for i, s in enumerate(unique_segments)}

        df["route_code"] = df["route_id"].map(self.route_map)
        df["segment_code"] = df["segment_id"].map(self.segment_map)

        # Strict Time-Based Train/Test Split
        unique_dates = sorted(df["date"].unique())
        if len(unique_dates) <= test_days:
            split_idx = max(1, int(len(unique_dates) * 0.75))
            train_dates = unique_dates[:split_idx]
            test_dates = unique_dates[split_idx:]
        else:
            train_dates = unique_dates[:-test_days]
            test_dates = unique_dates[-test_days:]

        self.train_test_split_date = test_dates[0]
        train_mask = df["date"].isin(train_dates)
        test_mask = df["date"].isin(test_dates)

        train_df = df[train_mask]
        test_df = df[test_mask].copy()

        # 1. Historical Baseline Forecaster (Mean passenger load per route, segment, slot)
        baseline_group = train_df.groupby(["route_id", "segment_id", "slot_idx"])["segment_load"].mean().to_dict()
        self.baseline_lookup = baseline_group

        # Overall segment fallback
        segment_fallback = train_df.groupby(["segment_id"])["segment_load"].mean().to_dict()

        def get_baseline_pred(row):
            key = (row["route_id"], row["segment_id"], row["slot_idx"])
            return baseline_group.get(key, segment_fallback.get(row["segment_id"], 25.0))

        test_baseline_preds = test_df.apply(get_baseline_pred, axis=1).values

        # 2. AI/ML Forecaster (HistGradientBoostingRegressor with quantile loss)
        X_train = train_df[FEATURE_COLS]
        y_train = train_df["segment_load"].values

        X_test = test_df[FEATURE_COLS]
        y_test = test_df["segment_load"].values

        # Median forecast (P50)
        self.model_p50 = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=0.50,
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=31,
            random_state=42
        )
        self.model_p50.fit(X_train, y_train)
        test_ai_p50 = np.maximum(0, self.model_p50.predict(X_test))

        # Peak surge forecast (P85) for safety buffer
        self.model_p85 = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=0.85,
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=31,
            random_state=42
        )
        self.model_p85.fit(X_train, y_train)
        test_ai_p85 = np.maximum(test_ai_p50, self.model_p85.predict(X_test))

        # Accuracy Metrics on Test Set
        base_mae = float(mean_absolute_error(y_test, test_baseline_preds))
        base_rmse = float(root_mean_squared_error(y_test, test_baseline_preds))
        base_mape = float(calculate_mape(y_test, test_baseline_preds))

        ai_mae = float(mean_absolute_error(y_test, test_ai_p50))
        ai_rmse = float(root_mean_squared_error(y_test, test_ai_p50))
        ai_mape = float(calculate_mape(y_test, test_ai_p50))

        mae_improvement_pct = max(0.0, ((base_mae - ai_mae) / max(base_mae, 1e-4)) * 100.0)

        self.metrics = {
            "baseline": {
                "mae": round(base_mae, 2),
                "rmse": round(base_rmse, 2),
                "mape_pct": round(base_mape, 2)
            },
            "ai_model": {
                "mae": round(ai_mae, 2),
                "rmse": round(ai_rmse, 2),
                "mape_pct": round(ai_mape, 2),
                "mae_improvement_pct": round(mae_improvement_pct, 1)
            },
            "train_samples": int(len(train_df)),
            "test_samples": int(len(test_df)),
            "split_date": self.train_test_split_date,
            "test_dates": test_dates
        }

        # Attach predictions to the test set for evaluation and dashboard consumption
        test_df["baseline_pred"] = np.round(test_baseline_preds, 1)
        test_df["ai_p50_pred"] = np.round(test_ai_p50, 1)
        test_df["ai_p85_pred"] = np.round(test_ai_p85, 1)

        # Compute Maximum Load Section (MLS) and classify crowding status
        self.latest_predictions_df = self._compute_mls_and_crowding(test_df, bus_capacity, target_load_factor)
        self.is_trained = True

        print(f"[ForecastEngine] Training complete. Baseline MAE={base_mae:.2f}, AI MAE={ai_mae:.2f} ({mae_improvement_pct:.1f}% improvement)")
        return self.metrics

    def _compute_mls_and_crowding(
        self,
        df: pd.DataFrame,
        bus_capacity: int,
        target_load_factor: float
    ) -> pd.DataFrame:
        """
        Flag each (route_id, time_slot) based on the Maximum Load Section (MLS):
        - LIKELY OVERCROWDED (Peak Load Factor > target_load_factor, e.g. 85%)
        - BALANCED (Load Factor 40% – 85%)
        - POTENTIALLY UNDERUSED (Load Factor < 40%)
        """
        df = df.copy()
        
        # Calculate offered baseline passenger capacity in 15-min slot:
        # trips_per_hour = baseline_buses * (60 / cycle_time_mins)
        # trips_in_15m = trips_per_hour * (15 / 60) = baseline_buses * 15 / cycle_time_mins
        # offered_capacity = trips_in_15m * bus_capacity
        df["baseline_trips_15m"] = df["baseline_buses"] * (15.0 / df["cycle_time_mins"])
        df["baseline_offered_capacity"] = np.maximum(1.0, df["baseline_trips_15m"] * bus_capacity)

        # Baseline and AI load factors per segment
        df["baseline_load_factor"] = np.round(df["segment_load"] / df["baseline_offered_capacity"] * 100.0, 1)
        df["ai_load_factor"] = np.round(df["ai_p50_pred"] / df["baseline_offered_capacity"] * 100.0, 1)

        # Group by (date, route_id, slot_idx) to find MLS (Maximum Load Section)
        mls_group = df.groupby(["date", "route_id", "slot_idx"])["ai_p50_pred"].transform("max")
        df["mls_load"] = mls_group
        df["mls_load_factor"] = np.round(df["mls_load"] / df["baseline_offered_capacity"] * 100.0, 1)

        def assign_status(load_factor_pct: float) -> str:
            if load_factor_pct > (target_load_factor * 100.0):
                return "LIKELY OVERCROWDED"
            elif load_factor_pct < 40.0:
                return "POTENTIALLY UNDERUSED"
            else:
                return "BALANCED"

        df["crowding_status"] = df["mls_load_factor"].apply(assign_status)
        return df

    def get_aggregated_slot_forecast(
        self,
        route_id: Optional[str] = None,
        date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get aggregated slot-by-slot forecast for UI charts and optimization.
        Provides actual load, baseline historical average, AI P50, and AI P85 upper bound.
        """
        if not self.is_trained or self.latest_predictions_df is None:
            return []

        df = self.latest_predictions_df
        if date:
            df = df[df["date"] == date]
        else:
            # Pick the most recent test date
            most_recent_date = df["date"].max()
            df = df[df["date"] == most_recent_date]

        if route_id and route_id != "ALL":
            df = df[df["route_id"] == route_id]

        # Group by slot_idx
        grouped = df.groupby(["slot_idx", "time_str"]).agg({
            "segment_load": "sum",
            "baseline_pred": "sum",
            "ai_p50_pred": "sum",
            "ai_p85_pred": "sum",
            "baseline_offered_capacity": "sum",
            "mls_load": "max",
            "mls_load_factor": "max"
        }).reset_index()

        results = []
        for _, row in grouped.iterrows():
            lf = row["mls_load_factor"]
            status = "LIKELY OVERCROWDED" if lf > 85.0 else ("POTENTIALLY UNDERUSED" if lf < 40.0 else "BALANCED")
            results.append({
                "slot_idx": int(row["slot_idx"]),
                "time_str": row["time_str"],
                "actual_load": round(float(row["segment_load"]), 1),
                "baseline_pred": round(float(row["baseline_pred"]), 1),
                "ai_p50_pred": round(float(row["ai_p50_pred"]), 1),
                "ai_p85_pred": round(float(row["ai_p85_pred"]), 1),
                "offered_capacity": round(float(row["baseline_offered_capacity"]), 1),
                "mls_load": round(float(row["mls_load"]), 1),
                "mls_load_factor": round(float(lf), 1),
                "status": status
            })

        return results
