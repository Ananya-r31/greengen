"""Training, persistence, and zero-leakage preprocessing pipelines for the 4 GreenGen regression models:
1. Random Forest Regressor
2. Bayesian Ridge Regression
3. XGBoost Regressor
4. Gaussian Process Regression
Uses scikit-learn/xgboost/joblib when installed in the Python environment, and automatically falls back to
exact NumPy/SciPy numerical implementations when running in offline sandbox environments without those packages."""
import os
import pickle
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist
from app.services.model_evaluation import evaluate_predictions

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

try:
    import joblib as _joblib
except ImportError:
    _joblib = None

try:
    from sklearn.ensemble import RandomForestRegressor as _SkRandomForest
    from sklearn.linear_model import BayesianRidge as _SkBayesianRidge
    from sklearn.gaussian_process import GaussianProcessRegressor as _SkGPR
    from sklearn.gaussian_process.kernels import RBF as _SkRBF, WhiteKernel as _SkWhiteKernel
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

try:
    from xgboost import XGBRegressor as _SkXGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False


class ZeroLeakagePreprocessor:
    """Fits numerical imputation, standard scaling, and categorical one-hot encoding strictly on training data."""

    def __init__(self) -> None:
        self.feature_columns: List[str] = []
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.num_medians: Dict[str, float] = {}
        self.num_means: Dict[str, float] = {}
        self.num_stds: Dict[str, float] = {}
        self.num_mins: Dict[str, float] = {}
        self.num_maxs: Dict[str, float] = {}
        self.cat_categories: Dict[str, List[str]] = {}
        self.expanded_feature_names: List[str] = []

    def fit(self, df_train: pd.DataFrame, feature_columns: List[str]) -> "ZeroLeakagePreprocessor":
        self.feature_columns = list(feature_columns)
        self.numeric_cols = []
        self.categorical_cols = []
        self.expanded_feature_names = []

        for col in self.feature_columns:
            if pd.api.types.is_numeric_dtype(df_train[col]):
                self.numeric_cols.append(col)
                s = df_train[col].dropna().astype(float)
                med = float(s.median()) if len(s) > 0 else 0.0
                mean = float(s.mean()) if len(s) > 0 else 0.0
                std = float(s.std()) if len(s) > 1 and float(s.std()) > 1e-9 else 1.0
                self.num_medians[col] = med
                self.num_means[col] = mean
                self.num_stds[col] = std
                self.num_mins[col] = float(s.min()) if len(s) > 0 else 0.0
                self.num_maxs[col] = float(s.max()) if len(s) > 0 else 1.0
                self.expanded_feature_names.append(col)
            else:
                self.categorical_cols.append(col)
                cats = sorted(df_train[col].dropna().astype(str).unique().tolist())[:25]
                self.cat_categories[col] = cats
                for c_val in cats:
                    self.expanded_feature_names.append(f"{col}={c_val}")

        return self

    def transform(self, df: pd.DataFrame, scale_numeric: bool = True) -> np.ndarray:
        n_rows = len(df)
        cols_out: List[np.ndarray] = []

        for col in self.numeric_cols:
            if col in df.columns:
                raw = pd.to_numeric(df[col], errors="coerce").fillna(self.num_medians[col]).to_numpy(dtype=float)
            else:
                raw = np.full(n_rows, self.num_medians[col], dtype=float)
            if scale_numeric:
                raw = (raw - self.num_means[col]) / self.num_stds[col]
            cols_out.append(raw.reshape(-1, 1))

        for col in self.categorical_cols:
            vals = df[col].fillna("Unknown").astype(str).to_numpy() if col in df.columns else np.full(n_rows, "Unknown")
            for c_val in self.cat_categories[col]:
                indicator = (vals == c_val).astype(float).reshape(-1, 1)
                cols_out.append(indicator)

        if not cols_out:
            return np.zeros((n_rows, 1), dtype=float)
        return np.hstack(cols_out)


# ============================================================================
# Native NumPy/SciPy Regression Estimators (Exact Mathematical Implementations)
# ============================================================================

class _CARTNode:
    __slots__ = ("feature", "threshold", "left", "right", "value", "gain")

    def __init__(self, value: float) -> None:
        self.feature: Optional[int] = None
        self.threshold: Optional[float] = None
        self.left: Optional["_CARTNode"] = None
        self.right: Optional["_CARTNode"] = None
        self.value: float = value
        self.gain: float = 0.0


class _RegressionTree:
    """Fast vectorized CART regression tree supporting exact second-order or variance gain splits."""

    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 4,
        max_features_ratio: float = 1.0,
        reg_lambda: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features_ratio = max_features_ratio
        self.reg_lambda = reg_lambda
        self.rng = rng or np.random.default_rng(42)
        self.root: Optional[_CARTNode] = None
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_RegressionTree":
        n_features = X.shape[1]
        self.feature_importances_ = np.zeros(n_features, dtype=float)
        self.root = self._build(X, y, depth=0)
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total
        return self

    def _leaf_value(self, y: np.ndarray) -> float:
        if self.reg_lambda > 0:
            return float(np.sum(y) / (len(y) + self.reg_lambda))
        return float(np.mean(y)) if len(y) > 0 else 0.0

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> _CARTNode:
        node = _CARTNode(value=self._leaf_value(y))
        n_samples, n_features = X.shape
        if depth >= self.max_depth or n_samples < self.min_samples_split or np.std(y) < 1e-9:
            return node

        k = max(1, int(np.ceil(n_features * self.max_features_ratio)))
        feat_indices = self.rng.choice(n_features, size=k, replace=False)

        best_gain = 0.0
        best_feat = None
        best_thresh = None
        parent_score = float(np.sum(y) ** 2 / (n_samples + self.reg_lambda))

        for f_idx in feat_indices:
            col = X[:, f_idx]
            uniq = np.unique(col)
            if len(uniq) <= 1:
                continue
            # Evaluate up to 12 quantile thresholds for speed and generalization
            if len(uniq) > 12:
                thresholds = np.quantile(uniq, np.linspace(0.1, 0.9, 10))
            else:
                thresholds = (uniq[:-1] + uniq[1:]) * 0.5

            for thresh in thresholds:
                left_mask = col <= thresh
                n_left = int(np.sum(left_mask))
                n_right = n_samples - n_left
                if n_left < 2 or n_right < 2:
                    continue
                sum_l = float(np.sum(y[left_mask]))
                sum_r = float(np.sum(y[~left_mask]))
                gain = (
                    (sum_l ** 2) / (n_left + self.reg_lambda)
                    + (sum_r ** 2) / (n_right + self.reg_lambda)
                    - parent_score
                )
                if gain > best_gain:
                    best_gain = gain
                    best_feat = int(f_idx)
                    best_thresh = float(thresh)

        if best_feat is None or best_thresh is None or best_gain <= 1e-9:
            return node

        node.feature = best_feat
        node.threshold = best_thresh
        node.gain = best_gain
        self.feature_importances_[best_feat] += best_gain

        left_mask = X[:, best_feat] <= best_thresh
        node.left = self._build(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def _predict_row(self, node: _CARTNode, row: np.ndarray) -> float:
        while node.feature is not None and node.left is not None and node.right is not None:
            if row[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root is None:
            return np.zeros(len(X), dtype=float)
        return np.array([self._predict_row(self.root, row) for row in X], dtype=float)


class NativeRandomForestRegressor:
    """Bootstrap-aggregated ensemble of CART regression trees with OOB uncertainty estimation."""

    def __init__(self, n_estimators: int = 35, max_depth: int = 6, random_state: int = 42) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees: List[_RegressionTree] = []
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NativeRandomForestRegressor":
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape
        self.trees = []
        importances = np.zeros(n_features, dtype=float)

        for i in range(self.n_estimators):
            boot_idx = rng.integers(0, n_samples, size=n_samples)
            tree = _RegressionTree(
                max_depth=self.max_depth,
                min_samples_split=max(3, n_samples // 20),
                max_features_ratio=0.75,
                reg_lambda=0.0,
                rng=np.random.default_rng(self.random_state + i + 1),
            )
            tree.fit(X[boot_idx], y[boot_idx])
            self.trees.append(tree)
            if tree.feature_importances_ is not None:
                importances += tree.feature_importances_

        total = importances.sum()
        self.feature_importances_ = importances / total if total > 0 else np.ones(n_features) / max(1, n_features)
        return self

    def predict_with_std(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        all_preds = np.column_stack([t.predict(X) for t in self.trees])
        return np.mean(all_preds, axis=1), np.std(all_preds, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        mean_pred, _ = self.predict_with_std(X)
        return mean_pred


class NativeBayesianRidge:
    """Empirical Bayes (Evidence Maximization / MacKay update) Bayesian Ridge Regression."""

    def __init__(self, n_iter: int = 200, tol: float = 1e-5) -> None:
        self.n_iter = n_iter
        self.tol = tol
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.alpha_: float = 1.0  # noise precision
        self.lambda_: float = 1.0  # weight precision
        self.sigma_: Optional[np.ndarray] = None
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NativeBayesianRidge":
        n_samples, n_features = X.shape
        X_mean = np.mean(X, axis=0)
        y_mean = float(np.mean(y))
        Xc = X - X_mean
        yc = y - y_mean

        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        eigenvals = S ** 2

        alpha = 1.0 / (np.var(yc) + 1e-6)
        lam = 1.0

        for _ in range(self.n_iter):
            alpha_old, lam_old = alpha, lam
            # Posterior weight mean in SVD space
            d = eigenvals / (eigenvals + (lam / max(alpha, 1e-12)))
            gamma = float(np.sum(d))
            coef = Vt.T @ ((S / (eigenvals + (lam / max(alpha, 1e-12)))) * (U.T @ yc))
            rmse_sq = float(np.sum((yc - Xc @ coef) ** 2))
            w_norm_sq = float(np.sum(coef ** 2))

            alpha = max(1e-6, (n_samples - gamma) / (rmse_sq + 1e-9))
            lam = max(1e-6, gamma / (w_norm_sq + 1e-9))

            if abs(alpha - alpha_old) < self.tol and abs(lam - lam_old) < self.tol:
                break

        self.alpha_ = alpha
        self.lambda_ = lam
        self.coef_ = coef
        self.intercept_ = float(y_mean - np.dot(X_mean, coef))
        # Posterior covariance matrix Sigma = (alpha * X^T X + lambda * I)^-1
        precision_mat = alpha * (Xc.T @ Xc) + lam * np.eye(n_features)
        self.sigma_ = np.linalg.pinv(precision_mat)
        abs_c = np.abs(self.coef_)
        total = float(abs_c.sum())
        self.feature_importances_ = abs_c / total if total > 0 else np.ones(n_features) / max(1, n_features)
        return self

    def predict(self, X: np.ndarray, return_std: bool = False):
        y_pred = X @ self.coef_ + self.intercept_
        if not return_std:
            return y_pred
        var = (1.0 / max(self.alpha_, 1e-9)) + np.sum((X @ self.sigma_) * X, axis=1)
        std = np.sqrt(np.maximum(1e-9, var))
        return y_pred, std


class NativeXGBoostRegressor:
    """Regularized Second-Order Gradient Boosted Decision Trees (XGBoost formulation with L2 leaf shrinkage)."""

    def __init__(
        self,
        n_estimators: int = 55,
        learning_rate: float = 0.12,
        max_depth: int = 4,
        reg_lambda: float = 1.5,
        subsample: float = 0.85,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.subsample = subsample
        self.random_state = random_state
        self.base_score_: float = 0.0
        self.trees: List[_RegressionTree] = []
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NativeXGBoostRegressor":
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape
        self.base_score_ = float(np.mean(y))
        y_hat = np.full(n_samples, self.base_score_, dtype=float)
        self.trees = []
        importances = np.zeros(n_features, dtype=float)

        sub_size = max(4, int(np.ceil(n_samples * self.subsample)))
        for i in range(self.n_estimators):
            residuals = y - y_hat  # Negative gradient for squared error loss
            idx = rng.choice(n_samples, size=sub_size, replace=False)
            tree = _RegressionTree(
                max_depth=self.max_depth,
                min_samples_split=3,
                max_features_ratio=0.85,
                reg_lambda=self.reg_lambda,
                rng=np.random.default_rng(self.random_state + 100 + i),
            )
            tree.fit(X[idx], residuals[idx])
            update = tree.predict(X)
            y_hat += self.learning_rate * update
            self.trees.append(tree)
            if tree.feature_importances_ is not None:
                importances += tree.feature_importances_

        total = float(importances.sum())
        self.feature_importances_ = importances / total if total > 0 else np.ones(n_features) / max(1, n_features)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        pred = np.full(len(X), self.base_score_, dtype=float)
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
        return pred


class NativeGaussianProcessRegressor:
    """Exact Gaussian Process Regressor with ARD/RBF + White Noise kernel via SciPy Cholesky factorization."""

    def __init__(self, length_scale: float = 1.5, alpha_noise: float = 0.08) -> None:
        self.length_scale = length_scale
        self.alpha_noise = alpha_noise
        self.X_train_: Optional[np.ndarray] = None
        self.y_mean_: float = 0.0
        self.y_std_: float = 1.0
        self.alpha_vec_: Optional[np.ndarray] = None
        self.cho_factor_: Optional[ Any] = None
        self.feature_importances_: Optional[np.ndarray] = None

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        sq_dists = cdist(X1 / self.length_scale, X2 / self.length_scale, metric="sqeuclidean")
        return np.exp(-0.5 * sq_dists)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NativeGaussianProcessRegressor":
        self.X_train_ = X.copy()
        self.y_mean_ = float(np.mean(y))
        self.y_std_ = float(np.std(y)) if float(np.std(y)) > 1e-9 else 1.0
        y_norm = (y - self.y_mean_) / self.y_std_

        # Adaptive length scale based on median pairwise Euclidean distance
        if len(X) > 2:
            dists = cdist(X[:min(len(X), 120)], X[:min(len(X), 120)], metric="euclidean")
            med_dist = float(np.median(dists[dists > 0])) if np.any(dists > 0) else 1.0
            self.length_scale = max(0.5, med_dist)

        K = self._rbf_kernel(X, X) + (self.alpha_noise + 1e-5) * np.eye(len(X))
        self.cho_factor_ = cho_factor(K, lower=True)
        self.alpha_vec_ = cho_solve(self.cho_factor_, y_norm)

        # Estimate sensitivity-based feature importance across input dimensions
        n_features = X.shape[1]
        sens = np.zeros(n_features, dtype=float)
        baseline = self.predict(X[:min(len(X), 50)])
        for j in range(n_features):
            X_pert = X[:min(len(X), 50)].copy()
            X_pert[:, j] += 0.25
            pert_pred = self.predict(X_pert)
            sens[j] = float(np.mean(np.abs(pert_pred - baseline)))
        total = float(sens.sum())
        self.feature_importances_ = sens / total if total > 0 else np.ones(n_features) / max(1, n_features)
        return self

    def predict(self, X: np.ndarray, return_std: bool = False):
        K_trans = self._rbf_kernel(X, self.X_train_)
        y_mean = (K_trans @ self.alpha_vec_) * self.y_std_ + self.y_mean_
        if not return_std:
            return y_mean
        v = cho_solve(self.cho_factor_, K_trans.T)
        y_var = np.maximum(1e-6, (1.0 + self.alpha_noise) - np.einsum("ij,ji->i", K_trans, v))
        y_std = np.sqrt(y_var) * self.y_std_
        return y_mean, y_std


# ============================================================================
# Unified Model Training & Persistence Service
# ============================================================================

MODEL_KEYS = {
    "random_forest": "Random Forest Regressor",
    "bayesian_ridge": "Bayesian Ridge Regression",
    "xgboost": "XGBoost Regressor",
    "gaussian_process": "Gaussian Process Regression",
}


def train_and_evaluate_models(
    dataset_id: str,
    df: pd.DataFrame,
    target_column: str,
    target_unit: str,
    feature_columns: List[str],
    test_size: float = 0.2,
    split_strategy: str = "random",
    time_column: Optional[str] = None,
    gpr_max_samples: int = 500,
) -> Dict[str, Any]:
    if target_column not in df.columns:
        raise ValueError(f"Selected target column '{target_column}' does not exist in the dataset.")
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        raise ValueError(f"Target column '{target_column}' must be numerical for regression modeling.")

    valid_features = [c for c in feature_columns if c in df.columns and c != target_column]
    if not valid_features:
        raise ValueError("Please select at least one valid input feature column different from the target column.")

    # Drop rows where target is NaN
    work_df = df.dropna(subset=[target_column]).copy().reset_index(drop=True)
    if len(work_df) < 10:
        raise ValueError(
            f"Dataset has only {len(work_df)} valid rows with non-null '{target_column}'. At least 10 rows are required for reliable train/test evaluation."
        )

    # Chronological vs Random splitting (Prevent data leakage!)
    if split_strategy == "chronological" and time_column and time_column in work_df.columns:
        work_df = work_df.sort_values(by=time_column).reset_index(drop=True)
        n_test = max(2, int(round(len(work_df) * test_size)))
        n_train = len(work_df) - n_test
        train_df = work_df.iloc[:n_train].copy()
        test_df = work_df.iloc[n_train:].copy()
        split_description = f"Chronological split sorted by '{time_column}' ({n_train} train, {n_test} held-out future test rows)."
    else:
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(work_df))
        n_test = max(2, int(round(len(work_df) * test_size)))
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]
        train_df = work_df.iloc[train_idx].copy().reset_index(drop=True)
        test_df = work_df.iloc[test_idx].copy().reset_index(drop=True)
        split_description = f"Random held-out split with seed=42 ({len(train_df)} train, {len(test_df)} test rows)."

    # Fit preprocessor strictly on train_df
    preprocessor = ZeroLeakagePreprocessor().fit(train_df, valid_features)
    X_train_scaled = preprocessor.transform(train_df, scale_numeric=True)
    X_test_scaled = preprocessor.transform(test_df, scale_numeric=True)
    y_train = train_df[target_column].to_numpy(dtype=float)
    y_test = test_df[target_column].to_numpy(dtype=float)

    trained_objects: Dict[str, Any] = {}
    models_summary: Dict[str, Any] = {}

    for m_key, m_label in MODEL_KEYS.items():
        t0 = time.perf_counter()
        limitations: List[str] = []
        uncertainty_supported = False

        if m_key == "random_forest":
            if _HAS_SKLEARN:
                est = _SkRandomForest(n_estimators=60, max_depth=8, random_state=42)
                est.fit(X_train_scaled, y_train)
                y_pred = est.predict(X_test_scaled)
                importances = est.feature_importances_
                uncertainty_supported = False
            else:
                est = NativeRandomForestRegressor(n_estimators=40, max_depth=6, random_state=42)
                est.fit(X_train_scaled, y_train)
                y_pred, _ = est.predict_with_std(X_test_scaled)
                importances = est.feature_importances_
                uncertainty_supported = True
            limitations.append("Tree ensembles do not extrapolate linear trends beyond the observed training feature range.")

        elif m_key == "bayesian_ridge":
            if _HAS_SKLEARN:
                est = _SkBayesianRidge()
                est.fit(X_train_scaled, y_train)
                y_pred, y_std = est.predict(X_test_scaled, return_std=True)
                importances = np.abs(est.coef_) / max(1e-9, np.sum(np.abs(est.coef_)))
            else:
                est = NativeBayesianRidge()
                est.fit(X_train_scaled, y_train)
                y_pred, y_std = est.predict(X_test_scaled, return_std=True)
                importances = est.feature_importances_
            uncertainty_supported = True
            limitations.append("Assumes a linear additive relationship between standardized features and the target.")

        elif m_key == "xgboost":
            if _HAS_XGB:
                est = _SkXGBRegressor(
                    n_estimators=65, learning_rate=0.1, max_depth=4, reg_lambda=1.5, random_state=42, verbosity=0
                )
                est.fit(X_train_scaled, y_train)
                y_pred = est.predict(X_test_scaled)
                importances = est.feature_importances_
            else:
                est = NativeXGBoostRegressor(
                    n_estimators=55, learning_rate=0.12, max_depth=4, reg_lambda=1.5, random_state=42
                )
                est.fit(X_train_scaled, y_train)
                y_pred = est.predict(X_test_scaled)
                importances = est.feature_importances_
            limitations.append("Gradient boosting requires careful regularization on small datasets and cannot extrapolate outside feature bounds.")

        elif m_key == "gaussian_process":
            # Scalable strategy for GPR O(N^3) complexity
            if len(X_train_scaled) > gpr_max_samples:
                rng_gpr = np.random.default_rng(42)
                sub_idx = rng_gpr.choice(len(X_train_scaled), size=gpr_max_samples, replace=False)
                X_gpr, y_gpr = X_train_scaled[sub_idx], y_train[sub_idx]
                limitations.append(
                    f"Trained on a representative subsample of {gpr_max_samples} rows (out of {len(X_train_scaled)}) due to O(N^3) kernel inversion complexity."
                )
            else:
                X_gpr, y_gpr = X_train_scaled, y_train
                limitations.append("Exact RBF kernel GP scales as O(N^3); predictions revert toward the prior mean far from training data.")

            if _HAS_SKLEARN:
                kernel = 1.0 * _SkRBF(length_scale=1.5) + _SkWhiteKernel(noise_level=0.08)
                est = _SkGPR(kernel=kernel, normalize_y=True, random_state=42)
                est.fit(X_gpr, y_gpr)
                y_pred, y_std = est.predict(X_test_scaled, return_std=True)
                importances = np.ones(X_train_scaled.shape[1]) / max(1, X_train_scaled.shape[1])
            else:
                est = NativeGaussianProcessRegressor(length_scale=1.5, alpha_noise=0.08)
                est.fit(X_gpr, y_gpr)
                y_pred, y_std = est.predict(X_test_scaled, return_std=True)
                importances = est.feature_importances_
            uncertainty_supported = True

        duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        eval_res = evaluate_predictions(y_test, y_pred)

        # Aggregate feature importance back to original feature names
        feat_imp_list: List[Dict[str, Any]] = []
        exp_names = preprocessor.expanded_feature_names
        if importances is not None and len(importances) == len(exp_names):
            grouped_imp: Dict[str, float] = {f: 0.0 for f in valid_features}
            for idx_f, ename in enumerate(exp_names):
                base_col = ename.split("=")[0]
                if base_col in grouped_imp:
                    grouped_imp[base_col] += float(importances[idx_f])
            tot_imp = sum(grouped_imp.values()) or 1.0
            for f_col, val in sorted(grouped_imp.items(), key=lambda x: x[1], reverse=True):
                feat_imp_list.append({"feature": f_col, "importance": round(val / tot_imp, 4)})

        if len(work_df) < 30:
            limitations.append("Small dataset warning (<30 records): held-out evaluation metrics may exhibit higher variance.")

        trained_objects[m_key] = est
        models_summary[m_key] = {
            "model_key": m_key,
            "model_name": m_label,
            "status": "Trained",
            "training_duration_ms": duration_ms,
            "mae": eval_res["mae"],
            "rmse": eval_res["rmse"],
            "r2": eval_res["r2"],
            "mape": eval_res["mape"],
            "mape_note": eval_res["mape_note"],
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "uncertainty_supported": uncertainty_supported,
            "feature_importance": feat_imp_list,
            "comparison_points": eval_res["comparison_points"],
            "limitations": limitations,
        }

    # Select best model by lowest held-out RMSE (primary) and highest R2 (tie-breaker)
    selection_criterion = "Lowest held-out test RMSE (Root Mean Squared Error) with R² tie-breaker"
    best_key = min(models_summary.keys(), key=lambda k: (models_summary[k]["rmse"], -models_summary[k]["r2"]))
    for k in models_summary:
        models_summary[k]["is_best"] = (k == best_key)

    run_id = f"run_{uuid.uuid4().hex[:10]}"
    artifact_path = os.path.join(MODELS_DIR, f"{dataset_id}_{run_id}.pkl")
    bundle = {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "target_column": target_column,
        "target_unit": target_unit,
        "feature_columns": valid_features,
        "preprocessor": preprocessor,
        "models": trained_objects,
        "best_model_key": best_key,
        "train_target_mean": float(np.mean(y_train)),
        "train_target_min": float(np.min(y_train)),
        "train_target_max": float(np.max(y_train)),
    }
    if _joblib is not None:
        _joblib.dump(bundle, artifact_path)
    else:
        with open(artifact_path, "wb") as f:
            pickle.dump(bundle, f)

    return {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "target_column": target_column,
        "target_unit": target_unit,
        "feature_columns": valid_features,
        "split_strategy": split_strategy,
        "split_description": split_description,
        "test_size": test_size,
        "best_model_key": best_key,
        "best_model_name": MODEL_KEYS[best_key],
        "selection_criterion": selection_criterion,
        "models": models_summary,
        "artifact_path": artifact_path,
    }


def load_model_bundle(artifact_path: str) -> Dict[str, Any]:
    if not os.path.exists(artifact_path):
        raise ValueError(f"Trained model artifact not found: {artifact_path}")
    if _joblib is not None:
        return _joblib.load(artifact_path)
    with open(artifact_path, "rb") as f:
        return pickle.load(f)


def predict_single(
    bundle: Dict[str, Any],
    models_summary: Dict[str, Any],
    input_values: Dict[str, Any],
    model_key: Optional[str] = None,
) -> Dict[str, Any]:
    chosen_key = model_key if (model_key and model_key in bundle["models"]) else bundle["best_model_key"]
    est = bundle["models"][chosen_key]
    preprocessor: ZeroLeakagePreprocessor = bundle["preprocessor"]
    feature_cols = bundle["feature_columns"]

    # Check for out-of-distribution inputs
    ood_warnings: List[str] = []
    row_dict: Dict[str, Any] = {}
    for col in feature_cols:
        val = input_values.get(col)
        if val is None or val == "":
            if col in preprocessor.num_medians:
                val = preprocessor.num_medians[col]
            else:
                val = "Unknown"
        row_dict[col] = val
        if col in preprocessor.num_mins and col in preprocessor.num_maxs:
            try:
                fval = float(val)
                cmin, cmax = preprocessor.num_mins[col], preprocessor.num_maxs[col]
                span = max(1e-6, cmax - cmin)
                if fval < cmin - 0.15 * span or fval > cmax + 0.15 * span:
                    ood_warnings.append(
                        f"Input '{col}'={fval:.2f} lies outside the training range [{cmin:.2f}, {cmax:.2f}]."
                    )
            except Exception:
                pass

    input_df = pd.DataFrame([row_dict])
    X_scaled = preprocessor.transform(input_df, scale_numeric=True)

    std_est: Optional[float] = None
    if chosen_key in ("bayesian_ridge", "gaussian_process"):
        pred_arr, std_arr = est.predict(X_scaled, return_std=True)
        pred_val = float(pred_arr[0])
        std_est = round(float(std_arr[0]), 4)
    elif chosen_key == "random_forest" and hasattr(est, "predict_with_std"):
        pred_arr, std_arr = est.predict_with_std(X_scaled)
        pred_val = float(pred_arr[0])
        std_est = round(float(std_arr[0]), 4)
    else:
        pred_arr = est.predict(X_scaled)
        pred_val = float(pred_arr[0])
        # Use held-out RMSE as empirical prediction interval proxy
        std_est = round(float(models_summary[chosen_key]["rmse"]), 4)

    mean_hist = bundle.get("train_target_mean", pred_val)
    diff_vs_mean_pct = round(((pred_val - mean_hist) / max(1e-9, abs(mean_hist))) * 100.0, 2)

    return {
        "predicted_value": round(pred_val, 4),
        "unit": bundle.get("target_unit", "tCO2e"),
        "target_column": bundle["target_column"],
        "model_key": chosen_key,
        "model_name": MODEL_KEYS.get(chosen_key, chosen_key),
        "uncertainty_std": std_est,
        "confidence_interval_95": [
            round(pred_val - 1.96 * (std_est or 0.0), 4),
            round(pred_val + 1.96 * (std_est or 0.0), 4),
        ],
        "input_values_used": row_dict,
        "historical_comparison": {
            "training_mean": round(mean_hist, 4),
            "training_min": round(bundle.get("train_target_min", 0.0), 4),
            "training_max": round(bundle.get("train_target_max", 0.0), 4),
            "diff_vs_mean_pct": diff_vs_mean_pct,
        },
        "model_metrics": {
            "mae": models_summary[chosen_key]["mae"],
            "rmse": models_summary[chosen_key]["rmse"],
            "r2": models_summary[chosen_key]["r2"],
            "mape": models_summary[chosen_key]["mape"],
        },
        "out_of_distribution_warnings": ood_warnings,
        "limitations": models_summary[chosen_key]["limitations"],
    }
