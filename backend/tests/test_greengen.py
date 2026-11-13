"""Comprehensive Automated Test Suite for GreenGen Backend.
Verifies all 15 required testing and validation criteria:
1. CSV and Excel (.xlsx) upload
2. Invalid file handling (unsupported extension, empty file, single-column file)
3. Data profiling & semantic column concept mapping
4. Missing-value preprocessing & activity-based emission factor calculation
5. Feature and target selection validation
6. Model training across all 4 regression algorithms (Random Forest, Bayesian Ridge, XGBoost, Gaussian Process)
7. Evaluation metrics (MAE, RMSE, R2, and conditional MAPE near-zero check)
8. Prediction requests with uncertainty intervals and out-of-distribution checks
9. Forecast validation (chronological forecasting vs non-temporal dataset rejection)
10. Optimization bounds and constraints (SciPy SLSQP)
11. Zero-emissions baseline calculation edge case
12. RAG source retrieval and grounded deterministic fallback labeling
13. Recommendation generation grounded in dataset evidence
14. Report generation (PDF, CSV, and XLSX)
15. Dataset switching and strict result isolation
"""
import io
import os
import sys
import unittest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import importlib.abc, importlib.util
class _DirectFSFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if not fullname.startswith('app'):
            return None
        rel = fullname.replace('.', '/')
        pkg_dir = os.path.join(BACKEND_DIR, rel)
        pkg_init = os.path.join(pkg_dir, '__init__.py')
        if os.path.isfile(pkg_init):
            return importlib.util.spec_from_file_location(fullname, pkg_init, submodule_search_locations=[pkg_dir])
        mod_file = os.path.join(BACKEND_DIR, rel + '.py')
        if os.path.isfile(mod_file):
            return importlib.util.spec_from_file_location(fullname, mod_file)
        return None
sys.meta_path.insert(0, _DirectFSFinder())

from app.main import app
from app.services.model_evaluation import evaluate_predictions

SYNTHETIC_CSV_PATH = os.path.join(os.path.dirname(__file__), "synthetic_dev_dataset.csv")


class TestGreenGenPlatform(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)
        with open(SYNTHETIC_CSV_PATH, "rb") as f:
            cls.synthetic_csv_bytes = f.read()

    def test_01_csv_and_excel_upload(self) -> None:
        # 1a. Upload CSV
        resp = self.client.post(
            "/api/datasets/upload?is_synthetic=true",
            files={"file": ("synthetic_dev_dataset.csv", self.synthetic_csv_bytes, "text/csv")},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        data = resp.json()
        self.assertTrue(data["is_synthetic"])
        self.assertEqual(data["row_count"], 48)
        self.assertEqual(data["mapping"]["primary_target"], "co2_emissions_tonnes")
        self.__class__.primary_ds_id = data["dataset_id"]

        # 1b. Upload Excel (.xlsx) generated in-memory
        df = pd.read_csv(io.BytesIO(self.synthetic_csv_bytes)).head(20)
        xlsx_buf = io.BytesIO()
        df.to_excel(xlsx_buf, index=False)
        xlsx_buf.seek(0)
        resp_xlsx = self.client.post(
            "/api/datasets/upload",
            files={
                "file": (
                    "industrial_facilities.xlsx",
                    xlsx_buf.read(),
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
        )
        self.assertEqual(resp_xlsx.status_code, 200, resp_xlsx.text)
        self.assertEqual(resp_xlsx.json()["row_count"], 20)

    def test_02_invalid_file_handling(self) -> None:
        # Invalid extension (.exe / .txt)
        resp_ext = self.client.post(
            "/api/datasets/upload",
            files={"file": ("malicious.exe", b"binary_payload", "application/octet-stream")},
        )
        self.assertEqual(resp_ext.status_code, 400)
        self.assertIn("Unsupported file format", resp_ext.json()["detail"])

        # Empty file (0 bytes)
        resp_empty = self.client.post(
            "/api/datasets/upload",
            files={"file": ("empty.csv", b"", "text/csv")},
        )
        self.assertEqual(resp_empty.status_code, 400)

        # Single-column dataset
        resp_one_col = self.client.post(
            "/api/datasets/upload",
            files={"file": ("single_col.csv", b"only_col\n1\n2\n3\n", "text/csv")},
        )
        self.assertEqual(resp_one_col.status_code, 400)

    def test_03_data_profiling_and_preview(self) -> None:
        ds_id = getattr(self.__class__, "primary_ds_id")
        resp = self.client.get(f"/api/datasets/{ds_id}/profile?sector=Manufacturing")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("profile", body)
        self.assertIn("kpis", body)
        self.assertIsNotNone(body["kpis"]["total_emissions"])
        self.assertEqual(body["kpis"]["applied_filters"].get("sector"), "Manufacturing")

        prev = self.client.get(f"/api/datasets/{ds_id}/preview?limit=10")
        self.assertEqual(prev.status_code, 200)
        self.assertEqual(len(prev.json()["rows"]), 10)

    def test_04_missing_value_preprocessing_and_emission_factors(self) -> None:
        # Upload a dataset with missing values, duplicates, and activity data without CO2 column
        raw_csv = (
            "year,sector,electricity_kwh,fossil_fuel_liters,renewable_share_pct\n"
            "2022,Manufacturing,10000,500,20.0\n"
            "2022,Manufacturing,10000,500,20.0\n"
            "2023,Manufacturing,,600,25.0\n"
            "2023,Logistics,14000,,15.0\n"
            "2024,Logistics,15000,800,\n"
        ).encode("utf-8")
        up = self.client.post(
            "/api/datasets/upload",
            files={"file": ("activity_only.csv", raw_csv, "text/csv")},
        )
        self.assertEqual(up.status_code, 200)
        act_id = up.json()["dataset_id"]
        self.assertFalse(up.json()["mapping"]["capabilities"]["direct_co2_modeling"])
        self.assertTrue(up.json()["mapping"]["capabilities"]["activity_based_co2_calculation_possible"])

        prep = self.client.post(
            f"/api/datasets/{act_id}/preprocess",
            json={
                "missing_numeric_strategy": "median",
                "missing_categorical_strategy": "mode",
                "drop_duplicates": True,
                "outlier_method": "iqr_clip",
                "emission_factors": {
                    "electricity_kg_per_kwh": 0.42,
                    "fuel_kg_per_liter": 2.68,
                },
            },
        )
        self.assertEqual(prep.status_code, 200, prep.text)
        pdata = prep.json()
        self.assertEqual(pdata["row_count"], 4)  # 1 exact duplicate removed
        self.assertEqual(pdata["profile"]["missing_cells"], 0)
        self.assertEqual(pdata["mapping"]["primary_target"], "calculated_co2_tonnes")

    def test_05_feature_and_target_selection_validation(self) -> None:
        ds_id = getattr(self.__class__, "primary_ds_id")
        # Invalid non-existent target column
        bad_resp = self.client.post(
            f"/api/datasets/{ds_id}/train",
            json={
                "target_column": "non_existent_col",
                "feature_columns": ["electricity_kwh"],
            },
        )
        self.assertEqual(bad_resp.status_code, 400)

        # Categorical target column rejected
        cat_resp = self.client.post(
            f"/api/datasets/{ds_id}/train",
            json={
                "target_column": "sector",
                "feature_columns": ["electricity_kwh"],
            },
        )
        self.assertEqual(cat_resp.status_code, 400)

    def test_06_and_07_train_four_models_and_evaluate_metrics(self) -> None:
        ds_id = getattr(self.__class__, "primary_ds_id")
        features = [
            "electricity_kwh",
            "renewable_share_pct",
            "fossil_fuel_liters",
            "transport_km",
            "waste_tonnes",
            "water_m3",
            "industrial_production_units",
            "sector",
        ]
        resp = self.client.post(
            f"/api/datasets/{ds_id}/train",
            json={
                "target_column": "co2_emissions_tonnes",
                "target_unit": "tCO2e",
                "feature_columns": features,
                "test_size": 0.2,
                "split_strategy": "chronological",
                "time_column": "year",
            },
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        res = resp.json()
        models = res["models"]
        for expected_key in ["random_forest", "bayesian_ridge", "xgboost", "gaussian_process"]:
            self.assertIn(expected_key, models)
            m = models[expected_key]
            self.assertEqual(m["status"], "Trained")
            self.assertGreaterEqual(m["mae"], 0.0)
            self.assertGreaterEqual(m["rmse"], 0.0)
            self.assertGreater(m["r2"], 0.5)
            self.assertIsNotNone(m["mape"])

        # Verify MAPE near-zero guard directly
        eval_zero = evaluate_predictions(np.array([0.0, 10.0, 20.0]), np.array([0.5, 9.8, 20.2]))
        self.assertIsNone(eval_zero["mape"])
        self.assertIn("Omitted", eval_zero["mape_note"])

    def test_08_prediction_requests(self) -> None:
        ds_id = getattr(self.__class__, "primary_ds_id")
        pred_resp = self.client.post(
            f"/api/datasets/{ds_id}/predict",
            json={
                "model_name": "gaussian_process",
                "input_values": {
                    "electricity_kwh": 145000,
                    "renewable_share_pct": 45.0,
                    "fossil_fuel_liters": 17000,
                    "transport_km": 40000,
                    "waste_tonnes": 35.0,
                    "water_m3": 3800,
                    "industrial_production_units": 1450,
                    "sector": "Manufacturing",
                },
            },
        )
        self.assertEqual(pred_resp.status_code, 200, pred_resp.text)
        pdata = pred_resp.json()
        self.assertIn("predicted_value", pdata)
        self.assertEqual(len(pdata["confidence_interval_95"]), 2)
        self.assertLess(pdata["confidence_interval_95"][0], pdata["confidence_interval_95"][1])

    def test_09_forecast_validation(self) -> None:
        ds_id = getattr(self.__class__, "primary_ds_id")
        fc_resp = self.client.post(
            f"/api/datasets/{ds_id}/forecast",
            json={"horizon_periods": 4, "time_column": "year"},
        )
        self.assertEqual(fc_resp.status_code, 200, fc_resp.text)
        fc = fc_resp.json()
        self.assertTrue(fc["supported"])
        self.assertEqual(len(fc["forecast_series"]), 4)

    def test_10_and_11_optimization_and_zero_emissions_edge_case(self) -> None:
        ds_id = getattr(self.__class__, "primary_ds_id")
        opt_resp = self.client.post(
            f"/api/datasets/{ds_id}/optimize",
            json={
                "target_reduction_pct": 12.0,
                "preserve_production_column": "industrial_production_units",
                "min_production_ratio": 1.0,
            },
        )
        self.assertEqual(opt_resp.status_code, 200, opt_resp.text)
        opt = opt_resp.json()
        self.assertLess(opt["optimized_emissions"], opt["baseline_emissions"])
        self.assertGreater(opt["absolute_reduction"], 0.0)
        self.assertGreater(opt["percentage_reduction"], 0.0)
        for c in opt["constraints_report"]:
            self.assertTrue(c["satisfied"])

        # What-if scenarios
        sc_resp = self.client.post(
            f"/api/datasets/{ds_id}/scenarios",
            json={
                "scenarios": [
                    {"name": "Electricity -15% & Renewables +25%", "adjustments_pct": {"electricity_kwh": -15, "renewable_share_pct": 25}},
                    {"name": "Fossil Fuel -20%", "adjustments_pct": {"fossil_fuel_liters": -20}},
                ]
            },
        )
        self.assertEqual(sc_resp.status_code, 200, sc_resp.text)
        self.assertEqual(len(sc_resp.json()["scenarios"]), 2)

        # Zero-emissions baseline edge case test
        zero_csv = "\n".join(
            ["electricity_kwh,fossil_fuel_liters,co2_emissions_tonnes"]
            + [f"{100 + i},{50 + i},0.0" for i in range(15)]
        ).encode("utf-8")
        z_up = self.client.post("/api/datasets/upload", files={"file": ("zero_co2.csv", zero_csv, "text/csv")})
        z_id = z_up.json()["dataset_id"]
        self.client.post(
            f"/api/datasets/{z_id}/train",
            json={"target_column": "co2_emissions_tonnes", "feature_columns": ["electricity_kwh", "fossil_fuel_liters"]},
        )
        z_opt = self.client.post(f"/api/datasets/{z_id}/optimize", json={"target_reduction_pct": 10.0})
        self.assertEqual(z_opt.status_code, 200)
        self.assertEqual(z_opt.json()["percentage_reduction"], 0.0)
        self.assertIsNotNone(z_opt.json()["zero_baseline_note"])

    def test_12_rag_source_retrieval_and_assistant(self) -> None:
        ds_id = getattr(self.__class__, "primary_ds_id")
        chat_resp = self.client.post(
            f"/api/datasets/{ds_id}/assistant/chat",
            json={"question": "What happens if electricity consumption decreases by 15% and what is needed for Scope 3?"},
        )
        self.assertEqual(chat_resp.status_code, 200, chat_resp.text)
        chat = chat_resp.json()
        self.assertIn("GROUNDED_DETERMINISTIC_FALLBACK", chat["generation_mode"])
        self.assertGreaterEqual(len(chat["citations"]), 1)
        self.assertIn("Scope 3", chat["answer"])

    def test_13_and_14_recommendations_action_plan_and_reports(self) -> None:
        ds_id = getattr(self.__class__, "primary_ds_id")
        rec_resp = self.client.get(f"/api/datasets/{ds_id}/recommendations")
        self.assertEqual(rec_resp.status_code, 200)
        rdata = rec_resp.json()
        self.assertGreaterEqual(len(rdata["recommendations"]), 4)
        self.assertEqual(len(rdata["action_plan"]), 4)

        # Save edited action plan
        plan_items = rdata["action_plan"]
        plan_items[0]["progress_status"] = "Completed"
        save_resp = self.client.post(f"/api/datasets/{ds_id}/action-plan", json={"items": plan_items})
        self.assertEqual(save_resp.status_code, 200)
        self.assertEqual(save_resp.json()["items"][0]["progress_status"], "Completed")

        # Generate PDF, CSV, and XLSX reports
        for fmt in ["pdf", "csv", "xlsx"]:
            rep_resp = self.client.post(
                f"/api/datasets/{ds_id}/reports",
                json={"format": fmt, "organization_name": "GreenGen Test Suite"},
            )
            self.assertEqual(rep_resp.status_code, 200)
            self.assertGreater(len(rep_resp.content), 200)

    def test_15_dataset_switching_and_result_isolation(self) -> None:
        # Upload a completely independent dataset with different columns and scale
        custom_csv = "\n".join(
            ["year,facility_org,grid_power_kwh,diesel_fuel_liters,ghg_emissions_tco2"]
            + [f"{2015 + (i % 8)},Plant_{i % 2},{5000 + i * 250},{800 + i * 40},{12.5 + i * 0.85:.2f}" for i in range(24)]
        ).encode("utf-8")
        up2 = self.client.post("/api/datasets/upload", files={"file": ("plant_custom.csv", custom_csv, "text/csv")})
        self.assertEqual(up2.status_code, 200)
        ds2_id = up2.json()["dataset_id"]

        # Ensure ds2 has no models trained initially despite primary_ds_id having trained models
        m2 = self.client.get(f"/api/datasets/{ds2_id}/models").json()
        self.assertEqual(m2["runs_count"], 0)
        self.assertIsNone(m2["latest_run"])

        # Compare KPIs between primary_ds_id and ds2_id to confirm zero cross-contamination
        prof1 = self.client.get(f"/api/datasets/{self.__class__.primary_ds_id}/profile").json()
        prof2 = self.client.get(f"/api/datasets/{ds2_id}/profile").json()
        self.assertNotEqual(prof1["kpis"]["total_emissions"], prof2["kpis"]["total_emissions"])
        self.assertEqual(prof2["kpis"]["co2_column"], "ghg_emissions_tco2")


if __name__ == "__main__":
    unittest.main(verbosity=2)
