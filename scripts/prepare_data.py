from pathlib import Path
from datasets import load_dataset, Dataset
from typing import Optional
import pandas as pd
from src.config import RAW_DATASET_PATH, PROCESSED_DATASET_PATH

DATASET_NAME = "allenai/art"
EXPECTED_ROWS = 1532


def download_art() -> Dataset:
    try:
        
        print(f"Loading {DATASET_NAME}...")
        ds = load_dataset(DATASET_NAME, split="validation")

        assert len(ds) == EXPECTED_ROWS, (
            f"Expected {EXPECTED_ROWS} rows, but got {len(ds)}"    
        )

        ds = ds.add_column("instance_id", list(range(len(ds))))
        ordered_cols = ["instance_id"] + [col for col in ds.column_names if col != "instance_id"]
        ds = ds.select_columns(ordered_cols)
        
        save_dataset(ds, RAW_DATASET_PATH)

        return ds
    
    except Exception as e:
        raise RuntimeError(f"There was an error downloading {DATASET_NAME}: {e}.")
    
def process_art_scoring_units(raw_dataset: Dataset) -> None:

    df = pd.DataFrame(raw_dataset)
    rows = []

    for _, r in df.iterrows():
        rows.append({
            "instance_id": r["instance_id"],
            "hyp_slot": "H1",
            "E": r["observation_1"],
            "O": r["observation_2"],
            "H": r["hypothesis_1"],
            "H_alt": r["hypothesis_2"],
            "is_correct": r["label"] == 1,
        })

        rows.append({
            "instance_id": r["instance_id"],
            "hyp_slot": "H2",
            "E": r["observation_1"],
            "O": r["observation_2"],
            "H": r["hypothesis_2"],
            "H_alt": r["hypothesis_1"],
            "is_correct": r["label"] == 2,
        })

    processed_ds = Dataset.from_list(rows)

    assert len(processed_ds) == 2*EXPECTED_ROWS, (
        f"Expected {2*EXPECTED_ROWS} rows, but got {len(processed_ds)}"    
    )

    save_dataset(processed_ds, PROCESSED_DATASET_PATH)

def save_dataset(ds: Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".parquet":
        ds.to_parquet(path)
    else:
        ds.to_json(path)

    print(f"Saved {len(ds)} records to {path}") 

def main() -> None:

    raw_dataset = download_art()
    process_art_scoring_units(raw_dataset)



if __name__ == "__main__":
    main()