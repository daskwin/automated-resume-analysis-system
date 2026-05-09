from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = BASE_DIR / "data" / "processed" / "resume_dataset.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "resume_dataset_general_grouped.csv"


TEXT_COLUMN = "resume_text"
TARGET_COLUMN = "target_role"


CATEGORY_TO_DOMAIN = {
    "ACCOUNTANT": "FINANCE",
    "FINANCE": "FINANCE",
    "BANKING": "FINANCE",
    "BUSINESS-DEVELOPMENT": "BUSINESS",
    "CONSULTANT": "BUSINESS",
    "SALES": "BUSINESS",
    "PUBLIC-RELATIONS": "BUSINESS",
    "INFORMATION-TECHNOLOGY": "TECHNICAL",
    "ENGINEERING": "TECHNICAL",
    "DESIGNER": "CREATIVE",
    "ARTS": "CREATIVE",
    "DIGITAL-MEDIA": "CREATIVE",
    "APPAREL": "CREATIVE",
    "CONSTRUCTION": "OPERATIONS",
    "AVIATION": "OPERATIONS",
    "AUTOMOBILE": "OPERATIONS",
    "AGRICULTURE": "OPERATIONS",
    "HR": "PEOPLE",
    "TEACHER": "PEOPLE",
    "HEALTHCARE": "PEOPLE",
    "FITNESS": "PEOPLE",
    "CHEF": "SERVICE",
    "BPO": "SERVICE",
    "ADVOCATE": "LEGAL",
}


def clean_text(text: str) -> str:
    text = str(text)
    text = text.replace("\r", "\n")
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    print("Raw shape:", df.shape)
    print("Raw columns:", df.columns.tolist())

    df = df[[TEXT_COLUMN, TARGET_COLUMN, "source"]].dropna()

    df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(clean_text)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.strip()
    df["source"] = df["source"].astype(str).str.strip()

    df = df[df[TEXT_COLUMN].str.len() > 100]
    df = df.drop_duplicates(subset=[TEXT_COLUMN, TARGET_COLUMN])

    df["original_category"] = df[TARGET_COLUMN]
    df["target_role"] = df["original_category"].map(CATEGORY_TO_DOMAIN).fillna("OTHER")
    df["source"] = df["source"] + "_grouped"

    output_df = df[
        [
            "resume_text",
            "target_role",
            "source",
            "original_category",
        ]
    ].copy()

    output_df = output_df.sample(frac=1, random_state=42).reset_index(drop=True)

    output_df.to_csv(OUTPUT_PATH, index=False)

    print()
    print("Grouped dataset shape:", output_df.shape)

    print()
    print("Original category distribution:")
    print(df["original_category"].value_counts())

    print()
    print("Grouped domain distribution:")
    print(output_df["target_role"].value_counts())

    print()
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
