"""Comprehensive data profiling, summary statistics, and KPI extraction from uploaded datasets."""
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def profile_dataset(df: pd.DataFrame, mapping: Dict[str, Any]) -> Dict[str, Any]:
    total_rows = int(len(df))
    total_cols = int(len(df.columns))
    total_cells = max(1, total_rows * total_cols)
    total_missing = int(df.isna().sum().sum())
    missing_pct = round((total_missing / total_cells) * 100.0, 2)
    duplicate_rows = int(df.duplicated().sum())

    columns_info: List[Dict[str, Any]] = []
    numerical_columns: List[str] = []
    categorical_columns: List[str] = []
    datetime_columns: List[str] = []

    col_to_concept = mapping.get("column_to_concept", {})
    col_units = mapping.get("column_units", {})

    for col in df.columns:
        s = df[col]
        missing_count = int(s.isna().sum())
        col_missing_pct = round((missing_count / max(1, total_rows)) * 100.0, 2)
        unique_count = int(s.nunique(dropna=True))

        is_num = pd.api.types.is_numeric_dtype(s)
        is_dt = pd.api.types.is_datetime64_any_dtype(s)

        if not is_num and not is_dt and col_to_concept.get(col) == "temporal":
            # Check if parseable as date
            try:
                parsed = pd.to_datetime(s.dropna().head(20), errors="raise")
                if len(parsed) > 0:
                    is_dt = True
            except Exception:
                pass

        if is_num:
            dtype_label = "numeric"
            numerical_columns.append(col)
            clean_s = s.dropna()
            if len(clean_s) > 0:
                q1 = float(clean_s.quantile(0.25))
                q3 = float(clean_s.quantile(0.75))
                iqr = q3 - q1
                outliers = int(((clean_s < (q1 - 1.5 * iqr)) | (clean_s > (q3 + 1.5 * iqr))).sum()) if iqr > 0 else 0
                stats = {
                    "mean": round(float(clean_s.mean()), 4),
                    "std": round(float(clean_s.std()), 4) if len(clean_s) > 1 else 0.0,
                    "min": round(float(clean_s.min()), 4),
                    "median": round(float(clean_s.median()), 4),
                    "max": round(float(clean_s.max()), 4),
                    "q25": round(q1, 4),
                    "q75": round(q3, 4),
                    "iqr_outliers": outliers,
                }
            else:
                stats = {"mean": None, "std": None, "min": None, "median": None, "max": None, "iqr_outliers": 0}
        elif is_dt:
            dtype_label = "datetime"
            datetime_columns.append(col)
            clean_s = s.dropna().astype(str)
            stats = {
                "min": str(clean_s.min()) if len(clean_s) else None,
                "max": str(clean_s.max()) if len(clean_s) else None,
            }
        else:
            dtype_label = "categorical"
            categorical_columns.append(col)
            top_vals = s.dropna().astype(str).value_counts().head(5).to_dict()
            stats = {"top_values": {str(k): int(v) for k, v in top_vals.items()}}

        columns_info.append({
            "name": col,
            "dtype": dtype_label,
            "pandas_dtype": str(s.dtype),
            "concept": col_to_concept.get(col, "unmapped"),
            "unit": col_units.get(col, ""),
            "missing_count": missing_count,
            "missing_pct": col_missing_pct,
            "unique_count": unique_count,
            "stats": stats,
        })

    # Quality score (0-100) based on completeness, uniqueness, and sample size
    completeness_score = max(0.0, 100.0 - missing_pct * 2.0)
    dup_penalty = min(25.0, (duplicate_rows / max(1, total_rows)) * 100.0)
    size_penalty = 15.0 if total_rows < 20 else (5.0 if total_rows < 40 else 0.0)
    quality_score = round(max(10.0, completeness_score - dup_penalty - size_penalty), 1)

    return {
        "row_count": total_rows,
        "column_count": total_cols,
        "missing_cells": total_missing,
        "missing_pct": missing_pct,
        "duplicate_rows": duplicate_rows,
        "quality_score": quality_score,
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
        "columns": columns_info,
    }


def compute_dashboard_kpis(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Computes strictly grounded dashboard KPIs and aggregations from the uploaded dataset."""
    filtered_df = df.copy()
    applied_filters: Dict[str, Any] = {}

    if filters:
        for col, val in filters.items():
            if val is not None and val != "" and val != "All" and col in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[col].astype(str) == str(val)]
                applied_filters[col] = val

    c2c = mapping.get("concept_to_columns", {})
    co2_col = (c2c.get("co2_emissions") or [None])[0] or mapping.get("primary_target")
    elec_col = (c2c.get("electricity_consumption") or c2c.get("energy_consumption") or [None])[0]
    ren_col = (c2c.get("renewable_energy") or [None])[0]
    year_col = None
    for cand in c2c.get("temporal", []):
        if "year" in cand.lower() or pd.api.types.is_numeric_dtype(filtered_df[cand]):
            year_col = cand
            break
    if not year_col and c2c.get("temporal"):
        year_col = c2c["temporal"][0]

    sector_col = (c2c.get("industry_sector") or [None])[0]
    country_col = (c2c.get("country_region") or [None])[0]
    org_col = (c2c.get("organization") or [None])[0]

    kpis: Dict[str, Any] = {
        "row_count": int(len(filtered_df)),
        "applied_filters": applied_filters,
        "co2_column": co2_col,
        "co2_unit": mapping.get("column_units", {}).get(co2_col, "tCO2e") if co2_col else None,
        "total_emissions": None,
        "average_emissions": None,
        "latest_emissions": None,
        "Reporting_period": "Not specified in dataset",
        "total_energy": None,
        "energy_column": elec_col,
        "energy_unit": mapping.get("column_units", {}).get(elec_col, "kWh") if elec_col else None,
        "avg_renewable_share": None,
        "renewable_column": ren_col,
        "emissions_by_year": [],
        "emissions_by_sector": [],
        "emissions_by_country": [],
        "highest_emission_category": None,
        "filter_options": {},
    }

    # Populate filter dropdown options from full dataset
    for fcol in [year_col, country_col, sector_col, org_col]:
        if fcol and fcol in df.columns:
            vals = sorted([str(x) for x in df[fcol].dropna().unique()])[:50]
            kpis["filter_options"][fcol] = vals

    if len(filtered_df) == 0:
        return kpis

    if year_col and year_col in filtered_df.columns:
        non_null_yr = filtered_df[year_col].dropna()
        if len(non_null_yr) > 0:
            kpis["reporting_period"] = f"{non_null_yr.min()} – {non_null_yr.max()}"

    if co2_col and co2_col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[co2_col]):
        s_co2 = filtered_df[co2_col].dropna()
        if len(s_co2) > 0:
            kpis["total_emissions"] = round(float(s_co2.sum()), 2)
            kpis["average_emissions"] = round(float(s_co2.mean()), 2)
            kpis["latest_emissions"] = round(float(s_co2.iloc[-1]), 2)

            if year_col and year_col in filtered_df.columns:
                grp = (
                    filtered_df.groupby(year_col, dropna=True)[co2_col]
                    .agg(["sum", "mean", "count"])
                    .reset_index()
                    .sort_values(by=year_col)
                )
                kpis["emissions_by_year"] = [
                    {
                        "period": str(row[year_col]),
                        "total_emissions": round(float(row["sum"]), 2),
                        "avg_emissions": round(float(row["mean"]), 2),
                        "records": int(row["count"]),
                    }
                    for _, row in grp.iterrows()
                ]

            if sector_col and sector_col in filtered_df.columns:
                grp_s = (
                    filtered_df.groupby(sector_col, dropna=True)[co2_col]
                    .sum()
                    .reset_index()
                    .sort_values(by=co2_col, ascending=False)
                )
                kpis["emissions_by_sector"] = [
                    {"category": str(row[sector_col]), "emissions": round(float(row[co2_col]), 2)}
                    for _, row in grp_s.iterrows()
                ]
                if len(kpis["emissions_by_sector"]) > 0:
                    top = kpis["emissions_by_sector"][0]
                    kpis["highest_emission_category"] = {
                        "dimension": sector_col,
                        "name": top["category"],
                        "emissions": top["emissions"],
                    }

            if country_col and country_col in filtered_df.columns:
                grp_c = (
                    filtered_df.groupby(country_col, dropna=True)[co2_col]
                    .sum()
                    .reset_index()
                    .sort_values(by=co2_col, ascending=False)
                )
                kpis["emissions_by_country"] = [
                    {"country": str(row[country_col]), "emissions": round(float(row[co2_col]), 2)}
                    for _, row in grp_c.iterrows()
                ]
                if not kpis["highest_emission_category"] and len(kpis["emissions_by_country"]) > 0:
                    top_c = kpis["emissions_by_country"][0]
                    kpis["highest_emission_category"] = {
                        "dimension": country_col,
                        "name": top_c["country"],
                        "emissions": top_c["emissions"],
                    }

    if elec_col and elec_col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[elec_col]):
        s_el = filtered_df[elec_col].dropna()
        if len(s_el) > 0:
            kpis["total_energy"] = round(float(s_el.sum()), 2)

    if ren_col and ren_col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[ren_col]):
        s_ren = filtered_df[ren_col].dropna()
        if len(s_ren) > 0:
            kpis["avg_renewable_share"] = round(float(s_ren.mean()), 2)

    return kpis
