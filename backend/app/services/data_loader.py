"""Safe file loading and validation for CSV and Excel datasets."""
import io
import os
import re
import uuid
from typing import Tuple
import pandas as pd

MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB limit
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


class DatasetLoadError(ValueError):
    """Raised when an uploaded dataset file is invalid, empty, or unsafe."""


def sanitize_filename(filename: str) -> str:
    base = os.path.basename(filename or "dataset.csv")
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    return cleaned[:120]


def validate_and_save_upload(filename: str, content: bytes) -> Tuple[str, str, pd.DataFrame]:
    if not filename:
        raise DatasetLoadError("Filename is missing.")
    safe_name = sanitize_filename(filename)
    _, ext = os.path.splitext(safe_name.lower())
    if ext not in ALLOWED_EXTENSIONS:
        raise DatasetLoadError(
            f"Unsupported file format '{ext}'. Please upload a valid .csv or .xlsx/.xls file."
        )
    if len(content) == 0:
        raise DatasetLoadError("The uploaded file is empty (0 bytes).")
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise DatasetLoadError(
            f"File size ({len(content) / (1024 * 1024):.2f} MB) exceeds the 25 MB limit."
        )

    try:
        if ext == ".csv":
            df = pd.read_csv(io.BytesIO(content))
        else:
            df = pd.read_excel(io.BytesIO(content))
    except Exception as exc:
        raise DatasetLoadError(f"Failed to parse {ext} dataset: {str(exc)}") from exc

    if df is None or df.empty or len(df.columns) == 0:
        raise DatasetLoadError("The uploaded dataset contains no rows or columns.")
    if len(df.columns) < 2:
        raise DatasetLoadError("The dataset must contain at least 2 columns (features and/or target).")

    # Normalize column headers (strip whitespace)
    df.columns = [str(c).strip() for c in df.columns]

    dataset_id = f"ds_{uuid.uuid4().hex[:10]}"
    saved_path = os.path.join(UPLOAD_DIR, f"{dataset_id}_{safe_name}")
    if ext == ".csv":
        df.to_csv(saved_path, index=False)
    else:
        csv_equiv = os.path.join(UPLOAD_DIR, f"{dataset_id}.csv")
        df.to_csv(csv_equiv, index=False)
        saved_path = csv_equiv

    return dataset_id, saved_path, df


def load_dataframe(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise DatasetLoadError(f"Dataset file not found on disk: {file_path}")
    _, ext = os.path.splitext(file_path.lower())
    if ext == ".csv":
        return pd.read_csv(file_path)
    return pd.read_excel(file_path)
