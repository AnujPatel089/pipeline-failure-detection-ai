"""Dataset discovery and loading."""
from pathlib import Path
import pandas as pd
from src.config import RAW_DATA_DIR, REQUIRED_COLUMNS


def discover_csv(raw_data_dir: Path = RAW_DATA_DIR) -> Path:
    """Find the sole CSV with the required schema, ignoring macOS metadata."""
    if not raw_data_dir.is_dir():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_data_dir}")
    csv_files = sorted(p for p in raw_data_dir.rglob("*.csv") if not p.name.startswith("._"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {raw_data_dir}")
    matching, unreadable = [], []
    for path in csv_files:
        try:
            columns = set(pd.read_csv(path, nrows=0).columns)
        except Exception as exc:
            unreadable.append(f"{path}: {exc}")
            continue
        if set(REQUIRED_COLUMNS).issubset(columns):
            matching.append(path)
    if len(matching) == 1:
        return matching[0]
    if len(matching) > 1:
        raise RuntimeError(f"Multiple CSV files match the required schema: {matching}")
    details = f" Unreadable files: {'; '.join(unreadable)}" if unreadable else ""
    raise RuntimeError(f"No CSV under {raw_data_dir} has the required schema.{details}")


def load_dataset(raw_data_dir: Path = RAW_DATA_DIR) -> tuple[pd.DataFrame, Path]:
    """Discover and load the SCADA CSV."""
    path = discover_csv(raw_data_dir)
    return pd.read_csv(path), path
