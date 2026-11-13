"""Automatic semantic concept detection for environmental and operational dataset columns."""
import re
from typing import Any, Dict, List
import pandas as pd

ENVIRONMENTAL_CONCEPTS = {
    "co2_emissions": {
        "label": "CO2 / GHG Emissions",
        "unit_default": "tCO2e",
        "patterns": [r"co2", r"carbon", r"ghg", r"emission", r"scope_?[123]", r"tco2"],
        "is_target_candidate": True,
        "is_controllable": False,
    },
    "electricity_consumption": {
        "label": "Electricity Consumption",
        "unit_default": "kWh",
        "patterns": [r"elec", r"kwh", r"mwh", r"grid_power", r"power_usage"],
        "is_target_candidate": True,
        "is_controllable": True,
    },
    "energy_consumption": {
        "label": "Total Energy Consumption",
        "unit_default": "GJ",
        "patterns": [r"energy", r"joule", r"btu", r"thermal"],
        "is_target_candidate": True,
        "is_controllable": True,
    },
    "renewable_energy": {
        "label": "Renewable Energy Share / Usage",
        "unit_default": "%",
        "patterns": [r"renew", r"solar", r"wind", r"green_power", r"clean_energy", r"re_share"],
        "is_target_candidate": False,
        "is_controllable": True,
    },
    "fossil_fuel": {
        "label": "Fossil Fuel Consumption",
        "unit_default": "liters",
        "patterns": [r"fossil", r"diesel", r"gasoline", r"petrol", r"coal", r"natural_gas", r"fuel_liters", r"fuel_usage", r"fuel_cons"],
        "is_target_candidate": True,
        "is_controllable": True,
    },
    "transportation": {
        "label": "Transportation & Logistics Activity",
        "unit_default": "km",
        "patterns": [r"transport", r"logistics", r"fleet", r"distance", r"vehicle", r"km", r"miles", r"freight", r"tkm"],
        "is_target_candidate": False,
        "is_controllable": True,
    },
    "waste_generation": {
        "label": "Waste Generation",
        "unit_default": "tonnes",
        "patterns": [r"waste", r"landfill", r"scrap", r"refuse", r"garbage"],
        "is_target_candidate": True,
        "is_controllable": True,
    },
    "water_consumption": {
        "label": "Water Consumption",
        "unit_default": "m3",
        "patterns": [r"water", r"h2o", r"m3", r"effluent", r"wastewater"],
        "is_target_candidate": True,
        "is_controllable": True,
    },
    "industrial_production": {
        "label": "Industrial Production / Output",
        "unit_default": "units",
        "patterns": [r"prod", r"output", r"manufactur", r"volume", r"units", r"throughput", r"revenue"],
        "is_target_candidate": False,
        "is_controllable": False,
    },
    "cost_budget": {
        "label": "Cost / Financial Expenditure",
        "unit_default": "USD",
        "patterns": [r"cost", r"budget", r"capex", r"opex", r"usd", r"price", r"expenditure", r"spend"],
        "is_target_candidate": False,
        "is_controllable": False,
    },
    "country_region": {
        "label": "Country or Region",
        "unit_default": "",
        "patterns": [r"country", r"region", r"nation", r"location", r"state", r"city", r"site"],
        "is_target_candidate": False,
        "is_controllable": False,
    },
    "industry_sector": {
        "label": "Industry or Sector",
        "unit_default": "",
        "patterns": [r"sector", r"industry", r"department", r"category", r"division", r"segment"],
        "is_target_candidate": False,
        "is_controllable": False,
    },
    "organization": {
        "label": "Organization / Facility",
        "unit_default": "",
        "patterns": [r"org", r"company", r"facility", r"plant", r"entity", r"institution", r"firm"],
        "is_target_candidate": False,
        "is_controllable": False,
    },
    "temporal": {
        "label": "Year, Month, or Date",
        "unit_default": "",
        "patterns": [r"^year$", r"^month$", r"date", r"period", r"timestamp", r"quarter", r"time"],
        "is_target_candidate": False,
        "is_controllable": False,
    },
}


def detect_column_concepts(df: pd.DataFrame) -> Dict[str, Any]:
    """Maps dataset columns to environmental concepts using column names, data types, and value ranges."""
    concept_to_columns: Dict[str, List[str]] = {k: [] for k in ENVIRONMENTAL_CONCEPTS}
    column_to_concept: Dict[str, str] = {}
    column_units: Dict[str, str] = {}

    for col in df.columns:
        col_lower = col.lower().strip()
        is_num = pd.api.types.is_numeric_dtype(df[col])
        matched_concept = None

        # Priority matching
        for concept_key, meta in ENVIRONMENTAL_CONCEPTS.items():
            # Numeric-only concepts should not match pure text columns (except temporal/country/sector/org)
            if concept_key not in ("country_region", "industry_sector", "organization", "temporal") and not is_num:
                continue
            for pat in meta["patterns"]:
                if re.search(pat, col_lower):
                    matched_concept = concept_key
                    break
            if matched_concept:
                break

        # Value-pattern heuristic for temporal year column
        if not matched_concept and is_num:
            non_null = df[col].dropna()
            if len(non_null) > 0:
                min_v, max_v = non_null.min(), non_null.max()
                if 1990 <= min_v <= 2050 and 1990 <= max_v <= 2050 and (non_null % 1 == 0).all():
                    matched_concept = "temporal"

        if matched_concept:
            concept_to_columns[matched_concept].append(col)
            column_to_concept[col] = matched_concept
            column_units[col] = ENVIRONMENTAL_CONCEPTS[matched_concept]["unit_default"]
        else:
            column_to_concept[col] = "other_numeric" if is_num else "other_categorical"
            column_units[col] = "units" if is_num else ""

    # Determine primary target and controllable features
    co2_cols = concept_to_columns["co2_emissions"]
    primary_target = co2_cols[0] if co2_cols else None

    target_candidates = []
    for c_key, meta in ENVIRONMENTAL_CONCEPTS.items():
        if meta["is_target_candidate"]:
            for c_name in concept_to_columns[c_key]:
                if c_name not in target_candidates:
                    target_candidates.append(c_name)

    # Also allow any numeric column that is not a temporal ID to be selected as a target by user
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and column_to_concept.get(col) != "temporal":
            if col not in target_candidates:
                target_candidates.append(col)

    controllable_columns = []
    for c_key, meta in ENVIRONMENTAL_CONCEPTS.items():
        if meta["is_controllable"]:
            for c_name in concept_to_columns[c_key]:
                if c_name != primary_target and c_name not in controllable_columns:
                    controllable_columns.append(c_name)

    has_co2_target = len(co2_cols) > 0
    has_activity_data = any(
        len(concept_to_columns[k]) > 0
        for k in ["electricity_consumption", "energy_consumption", "fossil_fuel", "transportation", "waste_generation"]
    )
    has_temporal = len(concept_to_columns["temporal"]) > 0
    has_cost = len(concept_to_columns["cost_budget"]) > 0

    capabilities = {
        "direct_co2_modeling": has_co2_target,
        "activity_based_co2_calculation_possible": (not has_co2_target) and has_activity_data,
        "time_series_forecasting": has_temporal and len(df) >= 8,
        "cost_constrained_optimization": has_cost,
        "controllable_optimization_variables": controllable_columns,
    }

    if has_co2_target:
        guidance_message = (
            f"Detected CO2 target column '{primary_target}' along with {len(controllable_columns)} controllable "
            f"operational variables. Full ML prediction, model comparison, and constrained optimization are supported."
        )
    elif has_activity_data:
        guidance_message = (
            "No direct CO2 emissions column was detected, but operational activity columns (electricity, fuel, transport, or waste) "
            "are present. You can either supply emission factors in Dataset Preprocessing to calculate a derived CO2 column, "
            "or select an existing environmental metric (e.g., electricity or waste) as your target variable."
        )
    else:
        guidance_message = (
            "No standard CO2 or environmental activity columns were automatically recognized by name. "
            "Please map your columns manually or select a numeric target variable for analysis."
        )

    return {
        "concept_to_columns": concept_to_columns,
        "column_to_concept": column_to_concept,
        "column_units": column_units,
        "primary_target": primary_target or (target_candidates[0] if target_candidates else None),
        "target_candidates": target_candidates,
        "controllable_columns": controllable_columns,
        "capabilities": capabilities,
        "guidance_message": guidance_message,
    }
