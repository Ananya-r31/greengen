"""Evidence-based Sustainability Recommendations and Editable Decarbonization Action Plan Generator."""
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def generate_recommendations(
    df: pd.DataFrame,
    profile: Dict[str, Any],
    mapping: Dict[str, Any],
    kpis: Dict[str, Any],
    latest_model_run: Optional[Dict[str, Any]] = None,
    latest_opt_run: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    c2c = mapping.get("concept_to_columns", {})
    co2_col = kpis.get("co2_column") or mapping.get("primary_target")
    co2_unit = kpis.get("co2_unit") or "tCO2e"
    cost_cols = c2c.get("cost_budget", [])
    has_cost = len(cost_cols) > 0

    recommendations: List[Dict[str, Any]] = []

    # Map optimization variable adjustments if available
    opt_changes_by_var: Dict[str, Dict[str, Any]] = {}
    if latest_opt_run and "result" in latest_opt_run:
        for vc in latest_opt_run["result"].get("variable_changes", []):
            opt_changes_by_var[vc["variable"]] = vc

    # 1. Energy Efficiency Recommendation
    elec_cols = c2c.get("electricity_consumption", []) + c2c.get("energy_consumption", [])
    if elec_cols and elec_cols[0] in df.columns:
        ecol = elec_cols[0]
        s_el = df[ecol].dropna()
        mean_el = float(s_el.mean()) if len(s_el) else 0.0
        max_el = float(s_el.max()) if len(s_el) else 0.0
        opt_vc = opt_changes_by_var.get(ecol)
        if opt_vc and opt_vc["pct_change"] < 0:
            impact_str = (
                f"Data-supported optimization indicates reducing '{ecol}' by {abs(opt_vc['pct_change']):.1f}% "
                f"({opt_vc['baseline_value']:,.1f} → {opt_vc['optimized_value']:,.1f}) contributes to a "
                f"{latest_opt_run['percentage_reduction']:.1f}% ({latest_opt_run['absolute_reduction']:.2f} {co2_unit}) total emission reduction."
            )
            evidence_type = "Quantitative (Supported by Dataset & SLSQP Optimization)"
        else:
            impact_str = (
                f"Qualitative/Directional: Reducing '{ecol}' peak consumption ({max_el:,.1f}) toward the dataset median "
                f"({float(s_el.median()):,.1f}) lowers Scope 2 grid emissions. Run Carbon Optimization to quantify exact tCO2e impact."
            )
            evidence_type = "Dataset Statistical Evidence"

        recommendations.append({
            "id": "REC-ENE-01",
            "category": "Energy efficiency",
            "problem_or_opportunity": f"High operational electricity/energy demand in '{ecol}' (Mean: {mean_el:,.1f}, Peak: {max_el:,.1f}).",
            "dataset_evidence": f"Column '{ecol}' ranges from {float(s_el.min()):,.1f} to {max_el:,.1f} across {len(s_el)} records.",
            "proposed_action": "Deploy Variable Frequency Drives (VFDs) on motors/pumps, upgrade to IE4 high-efficiency equipment, and implement ISO 50001 automated sub-metering.",
            "expected_environmental_impact": impact_str,
            "estimated_cost": (
                f"Dataset mean expenditure ('{cost_cols[0]}') is {float(df[cost_cols[0]].dropna().mean()):,.2f}; specific VFD capex requires engineering quotes."
                if has_cost
                else "Unavailable in dataset (requires equipment capex and local electricity tariff $/kWh)."
            ),
            "implementation_difficulty": "Medium",
            "priority": "High",
            "suggested_timeline": "3–6 months",
            "required_resources": "Facility Energy Engineer, PLC/BMS sub-meters, HVAC/motor retrofit contractors",
            "relevant_constraints": "Must maintain minimum industrial production throughput and process thermal stability.",
            "measurement_method": f"Continuous smart-meter tracking of '{ecol}' normalized by production volume (kWh/unit).",
            "confidence_level": evidence_type,
        })

    # 2. Renewable-Energy Adoption
    ren_cols = c2c.get("renewable_energy", [])
    if ren_cols and ren_cols[0] in df.columns:
        rcol = ren_cols[0]
        s_ren = df[rcol].dropna()
        mean_ren = float(s_ren.mean()) if len(s_ren) else 0.0
        opt_vc = opt_changes_by_var.get(rcol)
        if opt_vc and opt_vc["pct_change"] > 0:
            impact_str = (
                f"Optimization model recommends increasing '{rcol}' from {opt_vc['baseline_value']:.1f}% to "
                f"{opt_vc['optimized_value']:.1f}% ({opt_vc['pct_change']:+.1f}%), directly lowering Scope 2 emissions."
            )
            conf = "High (Data & Optimization Supported)"
        else:
            impact_str = f"Increasing '{rcol}' from current average ({mean_ren:.1f}%) toward 80–100% via PPAs or on-site solar PV cuts market-based Scope 2 emissions."
            conf = "Medium-High (Empirical Dataset Correlation)"

        recommendations.append({
            "id": "REC-REN-02",
            "category": "Renewable-energy adoption",
            "problem_or_opportunity": f"Current renewable energy share ('{rcol}') averages {mean_ren:.1f}%, leaving significant fossil grid exposure.",
            "dataset_evidence": f"Observed '{rcol}' spans {float(s_ren.min()):.1f}% to {float(s_ren.max()):.1f}% in uploaded records.",
            "proposed_action": "Execute utility-scale Renewable Power Purchase Agreements (PPAs) and expand behind-the-meter rooftop solar PV.",
            "expected_environmental_impact": impact_str,
            "estimated_cost": "Unavailable in dataset (requires PPA strike price $/MWh vs local grid tariff comparison).",
            "implementation_difficulty": "Medium",
            "priority": "High",
            "suggested_timeline": "6–12 months",
            "required_resources": "Energy procurement team, roof structural capacity audit, utility interconnection approval",
            "relevant_constraints": "Grid interconnection limits, rooftop area availability, and diurnal solar profile matching.",
            "measurement_method": f"Monthly tracking of '{rcol}' (%) and retired Energy Attribute Certificates (RECs/GOs).",
            "confidence_level": conf,
        })

    # 3. Fuel & Transportation Efficiency
    fuel_trans_cols = c2c.get("fossil_fuel", []) + c2c.get("transportation", [])
    if fuel_trans_cols and fuel_trans_cols[0] in df.columns:
        ftcol = fuel_trans_cols[0]
        s_ft = df[ftcol].dropna()
        mean_ft = float(s_ft.mean()) if len(s_ft) else 0.0
        opt_vc = opt_changes_by_var.get(ftcol)
        impact_str = (
            f"Constrained optimizer targets a {abs(opt_vc['pct_change']):.1f}% reduction in '{ftcol}' "
            f"({opt_vc['baseline_value']:,.1f} → {opt_vc['optimized_value']:,.1f})."
            if (opt_vc and opt_vc["pct_change"] < 0)
            else f"Reducing stationary/mobile fuel and logistics activity ('{ftcol}', mean {mean_ft:,.1f}) directly cuts Scope 1 & 3 combustion emissions."
        )
        recommendations.append({
            "id": "REC-TRN-03",
            "category": "Fuel and transportation efficiency",
            "problem_or_opportunity": f"Direct fuel combustion and transport activity ('{ftcol}') averages {mean_ft:,.1f} per reporting period.",
            "dataset_evidence": f"'{ftcol}' total volume across dataset is {float(s_ft.sum()):,.1f} (Max: {float(s_ft.max()):,.1f}).",
            "proposed_action": "Optimize logistics routing, shift long-haul road freight to rail, and phase in electric fleet vehicles (BEVs) or HVO renewable diesel.",
            "expected_environmental_impact": impact_str,
            "estimated_cost": "Unavailable in dataset (requires fleet TCO and charging infrastructure capex data).",
            "implementation_difficulty": "Medium-High",
            "priority": "High",
            "suggested_timeline": "6–12 months",
            "required_resources": "Fleet & Logistics Operations Manager, telematics routing platform, EV charging stations",
            "relevant_constraints": "Delivery SLA timelines, regional EV charger density, and heavy payload requirements.",
            "measurement_method": f"Telematics fuel logs and '{ftcol}' intensity per ton-kilometer (ISO 14083).",
            "confidence_level": "High (Supported by Dataset Activity Records)",
        })

    # 4. Waste Reduction & Circularity
    waste_cols = c2c.get("waste_generation", [])
    if waste_cols and waste_cols[0] in df.columns:
        wcol = waste_cols[0]
        s_w = df[wcol].dropna()
        recommendations.append({
            "id": "REC-WST-04",
            "category": "Waste reduction",
            "problem_or_opportunity": f"Operational waste generation ('{wcol}') totals {float(s_w.sum()):,.2f} (Mean: {float(s_w.mean()):,.2f}).",
            "dataset_evidence": f"Observed '{wcol}' ranges between {float(s_w.min()):,.2f} and {float(s_w.max()):,.2f}.",
            "proposed_action": "Implement closed-loop scrap recovery, segregate organic streams for anaerobic digestion, and divert landfill waste to material recycling.",
            "expected_environmental_impact": "Reduces Scope 3 Category 5 landfill methane emissions and embodied raw-material extraction footprints.",
            "estimated_cost": "Unavailable in dataset (requires landfill tipping fee vs recycling vendor contract rates).",
            "implementation_difficulty": "Low-Medium",
            "priority": "Medium",
            "suggested_timeline": "0–3 months",
            "required_resources": "EHS & Waste Coordinator, on-site material segregation bins, certified recycling partners",
            "relevant_constraints": "Local municipal recycling infrastructure and material contamination tolerances.",
            "measurement_method": f"Weighbridge transfer notes and monthly tracking of '{wcol}' diversion rate (%).",
            "confidence_level": "Medium-High (Dataset Activity Supported)",
        })

    # 5. Water Conservation (Only when supported by dataset!)
    water_cols = c2c.get("water_consumption", [])
    if water_cols and water_cols[0] in df.columns:
        wtcol = water_cols[0]
        s_wt = df[wtcol].dropna()
        recommendations.append({
            "id": "REC-WAT-05",
            "category": "Water conservation",
            "problem_or_opportunity": f"Water withdrawal ('{wtcol}') averages {float(s_wt.mean()):,.1f} per record (Total: {float(s_wt.sum()):,.1f}).",
            "dataset_evidence": f"Dataset tracks '{wtcol}' across {len(s_wt)} observations (Max: {float(s_wt.max()):,.1f}).",
            "proposed_action": "Install ultrasonic flow leak sensors, closed-loop cooling tower water recirculation, and cascade process water reuse.",
            "expected_environmental_impact": "Lowers embodied Scope 3 water supply/treatment emissions and thermal pumping energy demand.",
            "estimated_cost": "Unavailable in dataset (requires municipal water/effluent tariff data).",
            "implementation_difficulty": "Low-Medium",
            "priority": "Medium",
            "suggested_timeline": "3–6 months",
            "required_resources": "Facilities Utilities Team, flow sub-meters, water treatment filtration",
            "relevant_constraints": "Process water purity and hygiene standards.",
            "measurement_method": f"Sub-metered withdrawal volume in '{wtcol}' per unit of output.",
            "confidence_level": "Medium (Dataset Supported)",
        })

    # 6. Emission Monitoring, Data Quality & ESG Carbon Accounting (Always grounded in profile findings)
    missing_pct = profile.get("missing_pct", 0.0)
    dup_rows = profile.get("duplicate_rows", 0)
    recommendations.append({
        "id": "REC-DQ-06",
        "category": "Emission monitoring and data quality",
        "problem_or_opportunity": (
            f"Dataset contains {missing_pct}% missing cells, {dup_rows} duplicate rows, and {profile.get('row_count', 0)} total observations."
        ),
        "dataset_evidence": f"Data quality score is {profile.get('quality_score', 100)}/100 based on completeness and sample size.",
        "proposed_action": "Automate ERP/utility EDI data ingestion, record facility-specific unit costs ($/kWh, $/L), and log supplier-specific Scope 3 emission factors.",
        "expected_environmental_impact": "Qualitative: Reduces model prediction uncertainty and enables cost-optimal Marginal Abatement Cost Curve (MACC) ranking.",
        "estimated_cost": "Internal data engineering staff time.",
        "implementation_difficulty": "Low",
        "priority": "High" if missing_pct > 2.0 or len(df) < 30 else "Medium",
        "suggested_timeline": "0–3 months",
        "required_resources": "Sustainability Data Analyst, ERP/IoT API connectors",
        "relevant_constraints": "Legacy meter reading schedules and supplier data availability.",
        "measurement_method": "Data completeness ≥99.5% and reduction in held-out model RMSE.",
        "confidence_level": "High (Direct Data Profile Audit)",
    })

    recommendations.append({
        "id": "REC-ESG-07",
        "category": "ESG reporting and carbon accounting",
        "problem_or_opportunity": "Align operational emissions tracking with auditable GHG Protocol Scopes 1–3, GRI 305, and SBTi 1.5°C pathways.",
        "dataset_evidence": f"Primary tracked emissions metric is '{co2_col}' ({co2_unit}) over {kpis.get('reporting_period', 'recorded periods')}.",
        "proposed_action": "Establish dual Scope 2 reporting (location-based and market-based) and formalize a 4.2% annual linear reduction target aligned with SBTi.",
        "expected_environmental_impact": "Qualitative: Ensures audit readiness for CSRD/ESRS E1 and IFRS S2 climate disclosures.",
        "estimated_cost": "Unavailable in dataset (depends on third-party assurance scope).",
        "implementation_difficulty": "Low-Medium",
        "priority": "Medium",
        "suggested_timeline": "1–3 years",
        "required_resources": "ESG Reporting Lead, Internal Audit, GHG Inventory Software",
        "relevant_constraints": "Requires verified regional grid emission factors and contractual instrument tracking.",
        "measurement_method": "Annual GHG Inventory completeness and % target achievement.",
        "confidence_level": "High (Standards Grounded)",
    })

    return {
        "dataset_row_count": len(df),
        "recommendations_count": len(recommendations),
        "recommendations": recommendations,
    }


def generate_default_action_plan(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
    kpis: Dict[str, Any],
    latest_opt_run: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    co2_col = kpis.get("co2_column") or mapping.get("primary_target") or "co2_emissions"
    co2_unit = kpis.get("co2_unit") or "tCO2e"
    c2c = mapping.get("concept_to_columns", {})
    elec_col = (c2c.get("electricity_consumption") or c2c.get("energy_consumption") or ["electricity_kwh"])[0]
    ren_col = (c2c.get("renewable_energy") or ["renewable_share_pct"])[0]
    fuel_col = (c2c.get("fossil_fuel") or c2c.get("transportation") or ["fossil_fuel_liters"])[0]

    opt_pct = latest_opt_run["percentage_reduction"] if latest_opt_run else None
    opt_outcome = (
        f"Achieve modeled {opt_pct:.1f}% ({latest_opt_run['absolute_reduction']:.2f} {co2_unit}) emissions reduction."
        if opt_pct is not None
        else f"Establish baseline abatement trajectory for '{co2_col}'."
    )

    return [
        {
            "id": "ACT-01",
            "horizon": "Immediate (0-3m)",
            "description": f"Audit sub-metering for '{elec_col}' and '{fuel_col}', eliminate compressed-air/HVAC idle loads, and standardize GHG data capture.",
            "priority": "High",
            "responsible_role": "Facility Energy & Sustainability Lead",
            "target_completion_date": "2027-01-31",
            "required_resources": "Smart sub-meters, BMS scheduling access, data validation pipeline",
            "estimated_cost": "Requires site-specific opex quote (not in dataset)",
            "expected_outcome": "Zero missing records and 3–5% quick-win reduction in baseline idle energy waste.",
            "kpi": f"Completeness (%) & '{elec_col}' base-load kWh",
            "progress_status": "In Progress",
        },
        {
            "id": "ACT-02",
            "horizon": "Short-Term (3-6m)",
            "description": f"Execute operational efficiency retrofits (VFDs, process heat recovery, and waste segregation) targeting '{elec_col}'.",
            "priority": "High",
            "responsible_role": "Operations & Plant Engineering Team",
            "target_completion_date": "2027-04-30",
            "required_resources": "VFD hardware, mechanical contractors, ISO 50001 energy audit",
            "estimated_cost": "Requires equipment capex quote (not in dataset)",
            "expected_outcome": opt_outcome,
            "kpi": f"{elec_col} per production unit",
            "progress_status": "Planned",
        },
        {
            "id": "ACT-03",
            "horizon": "Medium-Term (6-12m)",
            "description": f"Expand renewable electricity share ('{ren_col}') via solar PV / utility PPA procurement and optimize '{fuel_col}' logistics routes.",
            "priority": "High",
            "responsible_role": "Procurement & Logistics Directors",
            "target_completion_date": "2027-10-31",
            "required_resources": "PPA legal/financial advisory, fleet route optimization software",
            "estimated_cost": "Requires PPA tariff & fleet transition model",
            "expected_outcome": f"Increase '{ren_col}' and reduce Scope 1/2 emissions intensity by 10–18%.",
            "kpi": f"'{ren_col}' (%) & '{fuel_col}' consumption",
            "progress_status": "Not Started",
        },
        {
            "id": "ACT-04",
            "horizon": "Long-Term (1-3y)",
            "description": "Electrify thermal processes and heavy fleet vehicles, integrate supplier-specific Scope 3 PCFs, and align with SBTi Net-Zero.",
            "priority": "Medium",
            "responsible_role": "Chief Sustainability Officer & Executive Committee",
            "target_completion_date": "2029-10-31",
            "required_resources": "Multi-year decarbonization capex budget, supplier engagement portal",
            "estimated_cost": "Requires multi-year capital allocation plan",
            "expected_outcome": f"Sustained ≥25–40% structural reduction in total '{co2_col}' aligned with 1.5°C pathway.",
            "kpi": f"Total '{co2_col}' ({co2_unit}) vs baseline year",
            "progress_status": "Not Started",
        },
    ]
