"""Pydantic schemas for GreenGen REST API requests and responses."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PreprocessRequest(BaseModel):
    missing_numeric_strategy: str = Field(default="median", description="median, mean, zero, or drop")
    missing_categorical_strategy: str = Field(default="mode", description="mode, unknown, or drop")
    drop_duplicates: bool = Field(default=True)
    outlier_method: str = Field(default="none", description="none, iqr_clip, or zscore_clip")
    type_overrides: Optional[Dict[str, str]] = Field(default=None, description="Column -> numeric|categorical|datetime")
    concept_overrides: Optional[Dict[str, str]] = Field(default=None, description="Concept -> Column name")
    emission_factors: Optional[Dict[str, float]] = Field(
        default=None,
        description="Optional user-supplied emission factors (e.g., electricity_kg_per_kwh, fuel_kg_per_liter) to compute CO2 when target is missing",
    )


class TrainRequest(BaseModel):
    target_column: str
    target_unit: str = "tCO2e"
    feature_columns: List[str]
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    split_strategy: str = Field(default="random", description="random or chronological")
    time_column: Optional[str] = None
    gpr_max_samples: int = Field(default=500, ge=50, le=2500)


class PredictRequest(BaseModel):
    model_name: Optional[str] = None
    target_column: Optional[str] = None
    input_values: Dict[str, Any]


class ForecastRequest(BaseModel):
    model_name: Optional[str] = None
    target_column: Optional[str] = None
    horizon_periods: int = Field(default=5, ge=1, le=24)
    time_column: Optional[str] = None
    feature_growth_rates: Optional[Dict[str, float]] = None


class VariableBound(BaseModel):
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    max_change_pct: Optional[float] = Field(default=35.0, ge=0.0, le=100.0)
    unit_cost: Optional[float] = None


class OptimizeRequest(BaseModel):
    model_name: Optional[str] = None
    target_column: Optional[str] = None
    target_reduction_pct: float = Field(default=15.0, ge=0.1, le=95.0)
    max_budget: Optional[float] = None
    baseline_row_index: Optional[int] = None
    baseline_inputs: Optional[Dict[str, Any]] = None
    variable_bounds: Optional[Dict[str, VariableBound]] = None
    preserve_production_column: Optional[str] = None
    min_production_ratio: float = Field(default=1.0, ge=0.5, le=1.5)


class ScenarioItem(BaseModel):
    name: str
    description: Optional[str] = ""
    adjustments_pct: Dict[str, float] = Field(description="Column name -> percentage change (-20 means -20%)")


class ScenariosRequest(BaseModel):
    model_name: Optional[str] = None
    target_column: Optional[str] = None
    baseline_inputs: Optional[Dict[str, Any]] = None
    scenarios: List[ScenarioItem]


class AssistantChatRequest(BaseModel):
    question: str
    sector_filter: Optional[str] = None
    top_k: int = Field(default=4, ge=1, le=8)


class ActionPlanItem(BaseModel):
    id: str
    horizon: str = Field(description="Immediate (0-3m) | Short-Term (3-6m) | Medium-Term (6-12m) | Long-Term (1-3y)")
    description: str
    priority: str = Field(default="High")
    responsible_role: str
    target_completion_date: str
    required_resources: str
    estimated_cost: Optional[str] = "Requires site-specific capex/opex quotes"
    expected_outcome: str
    kpi: str
    progress_status: str = Field(default="Not Started")


class ActionPlanRequest(BaseModel):
    items: List[ActionPlanItem]


class ReportRequest(BaseModel):
    format: str = Field(default="pdf", description="pdf, csv, or xlsx")
    organization_name: Optional[str] = "Uploaded Dataset Organization"
    notes: Optional[str] = None
