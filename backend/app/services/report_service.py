"""Sustainability Report Generator supporting downloadable PDF, CSV, and Excel (.xlsx) formats.
Covers all 14 required sections and strictly marks unavailable metrics as unavailable instead of fabricating values."""
import io
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill
from fpdf import FPDF

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def _ascii_safe(text: Any) -> str:
    """Normalizes Unicode symbols for FPDF Latin-1 compatibility."""
    if text is None:
        return "Unavailable"
    s = str(text)
    replacements = {
        "₂": "2",
        "²": "2",
        "³": "3",
        "—": "-",
        "–": "-",
        "→": "->",
        "≥": ">=",
        "≤": "<=",
        "×": "x",
        "°": " deg ",
        "√": "sqrt",
        "•": "*",
        "“": '"',
        "”": '"',
        "’": "'",
        "‘": "'",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    return s.encode("latin-1", errors="replace").decode("latin-1")


def generate_sustainability_report(
    dataset_record: Dict[str, Any],
    df: pd.DataFrame,
    profile: Dict[str, Any],
    mapping: Dict[str, Any],
    kpis: Dict[str, Any],
    latest_model_run: Optional[Dict[str, Any]],
    latest_opt_run: Optional[Dict[str, Any]],
    recommendations: List[Dict[str, Any]],
    action_plan: List[Dict[str, Any]],
    forecast_data: Optional[Dict[str, Any]] = None,
    scenarios_data: Optional[Dict[str, Any]] = None,
    report_format: str = "pdf",
    organization_name: str = "Uploaded Dataset Organization",
) -> Tuple[str, str]:
    dataset_id = dataset_record["dataset_id"]
    co2_col = kpis.get("co2_column") or mapping.get("primary_target") or "Unavailable"
    co2_unit = kpis.get("co2_unit") or "tCO2e"
    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    fmt = report_format.lower().strip()

    if fmt == "csv":
        report_path = os.path.join(REPORTS_DIR, f"greengen_report_{dataset_id}.csv")
        rows: List[Dict[str, Any]] = [
            {"Section": "1. Executive Summary", "Metric_or_Item": "Organization / Dataset", "Value": f"{organization_name} ({dataset_record['filename']})", "Unit_or_Notes": "Not independently verified / Non-regulatory analytics report"},
            {"Section": "2. Dataset Description", "Metric_or_Item": "Reporting Period & Dimensions", "Value": f"{kpis.get('reporting_period', 'Unavailable')} ({len(df)} rows, {len(df.columns)} cols)", "Unit_or_Notes": "Synthetic Dev Data" if dataset_record.get("is_synthetic") else "User Uploaded Data"},
            {"Section": "3. Data-Quality Findings", "Metric_or_Item": "Missing Cells / Duplicate Rows / Quality Score", "Value": f"{profile.get('missing_pct', 0)}% missing, {profile.get('duplicate_rows', 0)} duplicates", "Unit_or_Notes": f"Score: {profile.get('quality_score', 0)}/100"},
            {"Section": "4. Environmental Performance Indicators", "Metric_or_Item": f"Total {co2_col}", "Value": kpis.get("total_emissions", "Unavailable"), "Unit_or_Notes": co2_unit},
            {"Section": "4. Environmental Performance Indicators", "Metric_or_Item": f"Average {co2_col}", "Value": kpis.get("average_emissions", "Unavailable"), "Unit_or_Notes": co2_unit},
            {"Section": "4. Environmental Performance Indicators", "Metric_or_Item": "Average Renewable Share", "Value": kpis.get("avg_renewable_share", "Unavailable"), "Unit_or_Notes": "%"},
        ]
        for yr in kpis.get("emissions_by_year", []):
            rows.append({"Section": "5. Historical Emissions Analysis", "Metric_or_Item": f"Period {yr['period']}", "Value": yr["total_emissions"], "Unit_or_Notes": f"Avg: {yr['avg_emissions']} {co2_unit} ({yr['records']} records)"})

        if latest_model_run:
            rows.append({"Section": "6. Machine Learning Methodology", "Metric_or_Item": "Split & Preprocessing", "Value": latest_model_run.get("split_strategy", "random"), "Unit_or_Notes": "Zero-leakage train-only scaling & imputation"})
            for m in latest_model_run.get("models_summary", {}).values():
                rows.append({
                    "Section": "7. Model Comparison & Evaluation",
                    "Metric_or_Item": m["model_name"],
                    "Value": f"RMSE={m['rmse']}, MAE={m['mae']}, R2={m['r2']}",
                    "Unit_or_Notes": "BEST MODEL" if m.get("is_best") else "Evaluated on held-out test split",
                })
        else:
            rows.append({"Section": "7. Model Comparison & Evaluation", "Metric_or_Item": "Model Training Status", "Value": "Unavailable (No models trained yet)", "Unit_or_Notes": ""})

        if forecast_data and forecast_data.get("forecast_series"):
            for fs in forecast_data["forecast_series"]:
                rows.append({"Section": "8. Emissions Forecasts", "Metric_or_Item": f"Period {fs['period']}", "Value": fs["predicted_emissions"], "Unit_or_Notes": f"95% CI: [{fs['lower_bound_95']}, {fs['upper_bound_95']}] {co2_unit}"})
        else:
            rows.append({"Section": "8. Emissions Forecasts", "Metric_or_Item": "Forecast Status", "Value": "Unavailable (Run forecast or provide temporal data)", "Unit_or_Notes": ""})

        if latest_opt_run:
            rows.append({
                "Section": "9. Optimization Methodology & Results",
                "Metric_or_Item": f"SLSQP Optimization ({latest_opt_run['model_name']})",
                "Value": f"Baseline {latest_opt_run['baseline_emissions']} -> Optimized {latest_opt_run['optimized_emissions']}",
                "Unit_or_Notes": f"Reduction: {latest_opt_run['absolute_reduction']} {co2_unit} ({latest_opt_run['percentage_reduction']}%)",
            })
        else:
            rows.append({"Section": "9. Optimization Methodology & Results", "Metric_or_Item": "Optimization Status", "Value": "Unavailable (No optimization run recorded)", "Unit_or_Notes": ""})

        if scenarios_data and scenarios_data.get("scenarios"):
            for sc in scenarios_data["scenarios"]:
                rows.append({"Section": "10. What-If Scenario Comparisons", "Metric_or_Item": sc["scenario_name"], "Value": sc["scenario_emissions"], "Unit_or_Notes": f"Change: {-sc['percentage_reduction']:+.2f}% vs baseline"})
        else:
            rows.append({"Section": "10. What-If Scenario Comparisons", "Metric_or_Item": "Scenario Status", "Value": "Unavailable", "Unit_or_Notes": ""})

        for rec in recommendations:
            rows.append({"Section": "11. Recommendations", "Metric_or_Item": f"{rec['category']} ({rec['priority']})", "Value": rec["proposed_action"], "Unit_or_Notes": rec["expected_environmental_impact"]})

        for act in action_plan:
            rows.append({"Section": "12. Decarbonization Action Plan", "Metric_or_Item": f"{act['horizon']} - {act['responsible_role']}", "Value": act["description"], "Unit_or_Notes": f"KPI: {act['kpi']} | Status: {act['progress_status']}"})

        rows.append({"Section": "13. Assumptions & Limitations", "Metric_or_Item": "Analytical Disclaimer", "Value": "SLSQP finds local optima; tree models do not extrapolate beyond training bounds. Not an audited ESG certification.", "Unit_or_Notes": "Strictly grounded in uploaded dataset"})
        rows.append({"Section": "14. Data Sources & References", "Metric_or_Item": "Standards & Dataset", "Value": f"Uploaded file '{dataset_record['filename']}', GHG Protocol, IPCC EF Hub, ISO 50001, SBTi v1.2", "Unit_or_Notes": f"Generated {now_str}"})

        pd.DataFrame(rows).to_csv(report_path, index=False)
        return report_path, "text/csv"

    if fmt == "xlsx":
        report_path = os.path.join(REPORTS_DIR, f"greengen_report_{dataset_id}.xlsx")
        wb = openpyxl.Workbook()
        ws_sum = wb.active
        ws_sum.title = "Executive Summary & KPIs"

        header_fill = PatternFill(start_color="065F46", end_color="065F46", fill_type="solid")
        header_font = Font(color="FFFFFF", bold=True, size=11)

        summary_rows = [
            ["GreenGen Sustainability & Carbon Optimization Report", "", ""],
            ["Generated Timestamp", now_str, "Non-certified decision-support report"],
            ["Organization", organization_name, ""],
            ["Dataset Filename", dataset_record["filename"], "Synthetic Dev Data" if dataset_record.get("is_synthetic") else "Uploaded Dataset"],
            ["Reporting Period", kpis.get("reporting_period", "Unavailable"), ""],
            ["Total Rows / Columns", f"{len(df)} rows / {len(df.columns)} columns", f"Quality Score: {profile.get('quality_score', 0)}/100"],
            ["Missing Data (%)", f"{profile.get('missing_pct', 0)}%", f"Duplicate Rows: {profile.get('duplicate_rows', 0)}"],
            ["Primary Emissions Metric", co2_col, co2_unit],
            ["Total Recorded Emissions", kpis.get("total_emissions", "Unavailable"), co2_unit],
            ["Average Emissions per Period", kpis.get("average_emissions", "Unavailable"), co2_unit],
            ["Latest Recorded Emissions", kpis.get("latest_emissions", "Unavailable"), co2_unit],
            ["Total Energy Consumption", kpis.get("total_energy", "Unavailable"), kpis.get("energy_unit") or ""],
            ["Average Renewable Share", kpis.get("avg_renewable_share", "Unavailable"), "%"],
        ]
        for r_idx, r_vals in enumerate(summary_rows, start=1):
            ws_sum.append(r_vals)
            if r_idx == 1:
                for cell in ws_sum[r_idx]:
                    cell.fill = header_fill
                    cell.font = header_font

        # Sheet 2: Model Comparison & Optimization
        ws_ml = wb.create_sheet("ML Models & Optimization")
        ws_ml.append(["Model Name", "Status", "MAE", "RMSE", "R2", "MAPE (%)", "Selection Status"])
        for cell in ws_ml[1]:
            cell.fill = header_fill
            cell.font = header_font
        if latest_model_run:
            for m in latest_model_run.get("models_summary", {}).values():
                ws_ml.append([
                    m["model_name"],
                    m["status"],
                    m["mae"],
                    m["rmse"],
                    m["r2"],
                    m["mape"] if m["mape"] is not None else "Omitted (Near-zero target)",
                    "Best Model" if m.get("is_best") else "Evaluated",
                ])
        else:
            ws_ml.append(["Unavailable", "No models trained yet", "-", "-", "-", "-", "-"])

        ws_ml.append([])
        ws_ml.append(["Optimization Parameter", "Baseline Value", "Optimized Value", "Change (%)", "Bounds", "Within Training Dist."])
        if latest_opt_run and "result" in latest_opt_run:
            for vc in latest_opt_run["result"].get("variable_changes", []):
                ws_ml.append([
                    vc["variable"],
                    vc["baseline_value"],
                    vc["optimized_value"],
                    f"{vc['pct_change']:+.2f}%",
                    f"[{vc['bound_min']}, {vc['bound_max']}]",
                    "Yes" if vc["within_training_distribution"] else "Warning",
                ])
        else:
            ws_ml.append(["Unavailable (No optimization run executed)", "-", "-", "-", "-", "-"])

        # Sheet 3: Recommendations & Action Plan
        ws_act = wb.create_sheet("Recommendations & Action Plan")
        ws_act.append(["Horizon / Category", "Priority", "Owner / Role", "Proposed Action", "Expected Impact / Outcome", "Estimated Cost", "Status"])
        for cell in ws_act[1]:
            cell.fill = header_fill
            cell.font = header_font
        for rec in recommendations:
            ws_act.append([rec["category"], rec["priority"], rec["required_resources"], rec["proposed_action"], rec["expected_environmental_impact"], rec["estimated_cost"], rec["confidence_level"]])
        for act in action_plan:
            ws_act.append([act["horizon"], act["priority"], act["responsible_role"], act["description"], act["expected_outcome"], act["estimated_cost"], act["progress_status"]])

        for ws in [ws_sum, ws_ml, ws_act]:
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = 28

        wb.save(report_path)
        return report_path, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    # Default: PDF Report using FPDF with all 14 structured sections
    report_path = os.path.join(REPORTS_DIR, f"greengen_report_{dataset_id}.pdf")
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    def add_section_header(title: str) -> None:
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_fill_color(6, 95, 70)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 8, _ascii_safe(f"  {title}"), ln=True, fill=True)
        pdf.set_text_color(20, 20, 20)
        pdf.ln(2)

    def add_body_line(text: str, bold_prefix: str = "") -> None:
        pdf.set_font("Helvetica", "", 9.5)
        if bold_prefix:
            pdf.set_font("Helvetica", "B", 9.5)
            pdf.write(5, _ascii_safe(bold_prefix + " "))
            pdf.set_font("Helvetica", "", 9.5)
        pdf.multi_cell(0, 5, _ascii_safe(text))
        pdf.ln(1)

    # Banner
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(6, 95, 70)
    pdf.cell(0, 9, "GREENGEN - SUSTAINABILITY & CARBON OPTIMIZATION REPORT", ln=True)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(90, 90, 90)
    pdf.cell(
        0,
        5,
        _ascii_safe(
            f"Generated: {now_str} | Dataset: {dataset_record['filename']} | Notice: Analytical Decision-Support Report (Not a Certified Regulatory Audit)"
        ),
        ln=True,
    )
    pdf.ln(3)

    # 1. Executive Summary
    add_section_header("1. Executive Summary")
    add_body_line(
        f"This report synthesizes empirical environmental performance, machine learning regression benchmarks, "
        f"SciPy SLSQP constrained carbon optimization, and evidence-based decarbonization recommendations derived strictly from "
        f"dataset '{dataset_record['filename']}' ({len(df)} records, {len(df.columns)} columns)."
    )

    # 2. Dataset Description & Reporting Period
    add_section_header("2. Dataset Description and Reporting Period")
    add_body_line(
        f"Reporting Period: {kpis.get('reporting_period', 'Unavailable')} | Source Classification: "
        f"{'Synthetic Development Dataset' if dataset_record.get('is_synthetic') else 'User-Uploaded Operational Dataset'}."
    )
    add_body_line(f"Primary Target Variable: {co2_col} ({co2_unit}) | Controllable Variables: {', '.join(mapping.get('controllable_columns', [])) or 'None detected'}.")

    # 3. Data-Quality Findings
    add_section_header("3. Data-Quality Findings")
    add_body_line(
        f"Overall Quality Score: {profile.get('quality_score', 0)}/100 | Missing Cells: {profile.get('missing_cells', 0)} "
        f"({profile.get('missing_pct', 0)}%) | Exact Duplicate Rows: {profile.get('duplicate_rows', 0)}."
    )

    # 4. Environmental Performance Indicators
    add_section_header("4. Environmental Performance Indicators (KPIs)")
    add_body_line(
        f"Total {co2_col}: {kpis.get('total_emissions', 'Unavailable')} {co2_unit} | "
        f"Average per Observation: {kpis.get('average_emissions', 'Unavailable')} {co2_unit} | "
        f"Latest Recorded: {kpis.get('latest_emissions', 'Unavailable')} {co2_unit}."
    )
    add_body_line(
        f"Total Energy ({kpis.get('energy_column') or 'N/A'}): {kpis.get('total_energy', 'Unavailable')} | "
        f"Average Renewable Share ({kpis.get('renewable_column') or 'N/A'}): {kpis.get('avg_renewable_share', 'Unavailable')}%."
    )

    # 5. Historical Emissions Analysis
    add_section_header("5. Historical Emissions Analysis")
    yr_list = kpis.get("emissions_by_year", [])
    if yr_list:
        yr_summary = " | ".join([f"{y['period']}: {y['total_emissions']} {co2_unit} (avg {y['avg_emissions']})" for y in yr_list[:8]])
        add_body_line(f"Historical Trend by Period: {yr_summary}.")
    else:
        add_body_line("Historical Trend by Period: Unavailable (no temporal breakdown present).")

    # 6. Machine Learning Methodology & 7. Model Comparison
    add_section_header("6 & 7. Machine Learning Methodology and Model Comparison")
    if latest_model_run:
        add_body_line(
            f"Methodology: Zero-leakage preprocessing pipeline fit strictly on training data ({latest_model_run.get('split_strategy')} split, "
            f"test_size={latest_model_run.get('test_size')}). Selection Criterion: {latest_model_run.get('selection_criterion')}."
        )
        for m in latest_model_run.get("models_summary", {}).values():
            best_tag = " [BEST MODEL]" if m.get("is_best") else ""
            add_body_line(
                f"MAE={m['mae']} | RMSE={m['rmse']} | R2={m['r2']} | MAPE={m['mape'] if m['mape'] is not None else 'Omitted'} | "
                f"Time={m['training_duration_ms']}ms{best_tag}",
                bold_prefix=f"- {m['model_name']}:",
            )
    else:
        add_body_line("Model Evaluation: Unavailable (No models have been trained for this dataset yet).")

    # 8. Emissions Forecasts
    add_section_header("8. Emissions Forecasts")
    if forecast_data and forecast_data.get("forecast_series"):
        f_txt = " | ".join([f"{f['period']}: {f['predicted_emissions']} {co2_unit}" for f in forecast_data["forecast_series"]])
        add_body_line(f"Projected Emissions ({forecast_data.get('model_name')}): {f_txt}.")
    else:
        add_body_line("Emissions Forecast: Unavailable (Forecast not yet executed or temporal column absent).")

    # 9. Optimization Methodology & Results & 10. What-If Scenarios
    add_section_header("9 & 10. Constrained Carbon Optimization (SLSQP) and What-If Scenarios")
    if latest_opt_run:
        add_body_line(
            f"SLSQP Optimization ({latest_opt_run['model_name']}): Baseline = {latest_opt_run['baseline_emissions']} {co2_unit} -> "
            f"Optimized = {latest_opt_run['optimized_emissions']} {co2_unit} | Absolute Reduction = {latest_opt_run['absolute_reduction']} {co2_unit} "
            f"({latest_opt_run['percentage_reduction']}%). Status: {latest_opt_run['status']}."
        )
        for vc in latest_opt_run.get("result", {}).get("variable_changes", [])[:6]:
            add_body_line(
                f"{vc['baseline_value']} -> {vc['optimized_value']} ({vc['pct_change']:+.1f}%, bounds [{vc['bound_min']}, {vc['bound_max']}])",
                bold_prefix=f"  * {vc['variable']}:",
            )
    else:
        add_body_line("Optimization Results: Unavailable (No optimization run executed yet).")

    if scenarios_data and scenarios_data.get("scenarios"):
        for sc in scenarios_data["scenarios"]:
            add_body_line(
                f"Predicted: {sc['scenario_emissions']} {co2_unit} (Reduction: {sc['absolute_reduction']} {co2_unit} / {sc['percentage_reduction']}%)",
                bold_prefix=f"  * Scenario '{sc['scenario_name']}':",
            )

    # 11. Recommendations & 12. Decarbonization Action Plan
    add_section_header("11 & 12. Evidence-Based Recommendations and Decarbonization Action Plan")
    for rec in recommendations[:5]:
        add_body_line(
            f"{rec['proposed_action']} | Impact: {rec['expected_environmental_impact']} | Cost: {rec['estimated_cost']}",
            bold_prefix=f"- [{rec['category']} - {rec['priority']}]:",
        )
    for act in action_plan:
        add_body_line(
            f"{act['description']} (Owner: {act['responsible_role']} | Target: {act['target_completion_date']} | KPI: {act['kpi']} | Status: {act['progress_status']})",
            bold_prefix=f"- {act['horizon']}:",
        )

    # 13. Assumptions & Limitations & 14. Data Sources & References
    add_section_header("13 & 14. Assumptions, Limitations, Data Sources, and References")
    add_body_line(
        "Limitations: (1) SciPy SLSQP identifies local constrained optima on smooth surrogate surfaces; operational engineering validation is required. "
        "(2) Tree models do not extrapolate beyond training feature bounds. (3) Financial ROI requires user-supplied marginal unit costs. "
        "(4) This document is an analytical decision-support output and is NOT a certified regulatory ESG or ISO 14064 verification statement."
    )
    add_body_line(
        f"Data Sources & References: Active dataset '{dataset_record['filename']}' (ID: {dataset_id}), WRI/WBCSD GHG Protocol Corporate Standard, "
        "IPCC 2019 Refinement Emission Factor Guidelines, ISO 50001 Energy Management, and SBTi Corporate Net-Zero Standard v1.2."
    )

    pdf.output(report_path)
    return report_path, "application/pdf"
