from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_DATASET_PATH = Path(
    os.getenv("SMARTMETER_DATASET_PATH", str(RAW_DATA_DIR / "household_power_consumption.txt"))
)
DEFAULT_PROCESSED_DATASET_PATH = Path(
    os.getenv(
        "SMARTMETER_PROCESSED_DATASET_PATH",
        str(PROCESSED_DATA_DIR / "household_power_consumption.parquet"),
    )
)
DEFAULT_LOG_LEVEL = os.getenv("SMARTMETER_LOG_LEVEL", "INFO")

if not DEFAULT_DATASET_PATH.is_absolute():
    DEFAULT_DATASET_PATH = PROJECT_ROOT / DEFAULT_DATASET_PATH

if not DEFAULT_PROCESSED_DATASET_PATH.is_absolute():
    DEFAULT_PROCESSED_DATASET_PATH = PROJECT_ROOT / DEFAULT_PROCESSED_DATASET_PATH
