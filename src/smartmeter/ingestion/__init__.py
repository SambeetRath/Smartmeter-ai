"""Dataset ingestion helpers for SmartMeter AI."""

__all__ = [
    "DEFAULT_EXPECTED_COLUMNS",
    "load_dataset",
    "load_and_validate_dataset",
    "normalize_column_name",
]


def __getattr__(name: str):
    if name == "DEFAULT_EXPECTED_COLUMNS":
        from .loader import DEFAULT_EXPECTED_COLUMNS

        return DEFAULT_EXPECTED_COLUMNS
    if name == "load_dataset":
        from .loader import load_dataset

        return load_dataset
    if name == "load_and_validate_dataset":
        from .loader import load_and_validate_dataset

        return load_and_validate_dataset
    if name == "normalize_column_name":
        from .loader import normalize_column_name

        return normalize_column_name
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
