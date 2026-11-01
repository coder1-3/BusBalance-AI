"""
BusBalance AI — Pydantic Data Models & Request/Response Schemas
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any


class OptimizeRequest(BaseModel):
    total_fleet_limit: int = Field(default=26, ge=12, le=50, description="Total available buses across corridor")
    bus_capacity: int = Field(default=50, ge=20, le=120, description="Nominal seating and comfortable standing capacity")
    target_load_factor: float = Field(default=0.85, ge=0.50, le=1.0, description="Target policy load factor threshold (0.85 = 85%)")
    min_policy_headway_mins: float = Field(default=20.0, ge=5.0, le=45.0, description="Policy headway cap to protect citizen access")
    surge_multiplier: float = Field(default=1.0, ge=0.5, le=2.5, description="What-if simulator multiplier for weather/event demand")
    use_p85_surge: bool = Field(default=False, description="Use AI P85 quantile forecast instead of P50 for peak surge hedging")
    human_overrides: Optional[Dict[str, int]] = Field(default_factory=dict, description="Manual scheduler bus locks per route or route_window")


class PlannerActionRequest(BaseModel):
    action: str = Field(..., description="'APPROVE', 'OVERRIDE', or 'RESET'")
    window_id: Optional[str] = None
    route_id: Optional[str] = None
    override_buses: Optional[int] = None
    notes: Optional[str] = None


class ForecastRunRequest(BaseModel):
    test_days: int = Field(default=7, ge=3, le=14)
    bus_capacity: int = Field(default=50, ge=20, le=120)
    target_load_factor: float = Field(default=0.85, ge=0.50, le=1.0)


class SystemStatus(BaseModel):
    status: str
    is_data_loaded: bool
    total_corridor_rows: int
    is_model_trained: bool
    routes_count: int
    latest_metrics: Optional[Dict[str, Any]] = None
