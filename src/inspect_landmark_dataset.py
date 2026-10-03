import pandas as pd
from pathlib import Path


PROJECT_PATH = Path(__file__).resolve().parent.parent
DATASET_FILE = PROJECT_PATH / "outputs" / "landmark_dataset.csv"


def main():

    print("Inspecting structured landmark dataset...\n")

    if not DATASET_FILE.exists():
        print("ERROR: Dataset file not found:")
        print(DATASET_FILE)
        return

    data = pd.read_csv(DATASET_FILE)

    print("Dataset loaded successfully.\n")

    print(f"Total rows: {len(data)}")
    print(f"Total columns: {len(data.columns)}")

    print("\nColumns:")
    print(data.columns.tolist())

    print("\nSign distribution:")
    print(data["sign"].value_counts())

    print("\nSubset distribution:")
    print(data["subset"].value_counts())

    print("\nFirst 5 rows:")
    print(data.head())

    print("\nMissing values:")
    print(data.isnull().sum().sum())

    print("\nDataset inspection completed.")


if __name__ == "__main__":
    main()