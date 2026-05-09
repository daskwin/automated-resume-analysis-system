from pathlib import Path

import pandas as pd
from datasets import load_dataset


BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

RAW_OUTPUT_PATH = RAW_DIR / "resume_dataset_hf_raw.csv"
PROCESSED_OUTPUT_PATH = PROCESSED_DIR / "resume_dataset.csv"


DATASET_NAME = "Darshan-04/Resume-classification"


def main() -> None:
    print(f"Loading dataset: {DATASET_NAME}")

    dataset = load_dataset(DATASET_NAME)
    df_raw = dataset["train"].to_pandas()

    print("Raw shape:", df_raw.shape)
    print("Raw columns:", df_raw.columns.tolist())

    expected_columns = {"Resume_str", "Category"}
    missing_columns = expected_columns - set(df_raw.columns)

    if missing_columns:
        raise ValueError(f"Missing columns in dataset: {missing_columns}")

    df = df_raw.rename(
        columns={
            "Resume_str": "resume_text",
            "Category": "target_role",
        }
    )

    df = df[["resume_text", "target_role"]].copy()

    df["resume_text"] = df["resume_text"].astype(str).str.strip()
    df["target_role"] = df["target_role"].astype(str).str.strip()

    df = df.dropna()
    df = df[df["resume_text"].str.len() > 100]
    df = df.drop_duplicates(subset=["resume_text", "target_role"])

    df["source"] = "huggingface_darshan_04_resume_classification"

    df_raw.to_csv(RAW_OUTPUT_PATH, index=False)
    df.to_csv(PROCESSED_OUTPUT_PATH, index=False)

    print()
    print("Processed shape:", df.shape)

    print()
    print("Class distribution:")
    print(df["target_role"].value_counts())

    print()
    print(f"Raw dataset saved to: {RAW_OUTPUT_PATH}")
    print(f"Processed dataset saved to: {PROCESSED_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
