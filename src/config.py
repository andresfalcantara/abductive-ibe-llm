from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATASET_PATH = PROJECT_ROOT / "data/raw/art_validation.jsonl"
PROCESSED_DATASET_PATH = PROJECT_ROOT / "data/processed/art_scoring_units.parquet"
