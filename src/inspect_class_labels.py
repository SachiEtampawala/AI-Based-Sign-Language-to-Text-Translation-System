from pathlib import Path

DATASET_PATH = Path.home() / "Downloads" / "archive"

CLASS_FILE = DATASET_PATH / "wlasl_class_list.txt"

def main():
    print("Reading WLASL class labels...\n")

    if not CLASS_FILE.exists():
        print("ERROR : Class list file not found.")
        print(CLASS_FILE)
        return

    with open(CLASS_FILE, "r", encoding="utf-8") as file:
        labels = [line.strip() for line in file if line.strip()]

    print(f"Total class labels : {len(labels)}\n")

    print("First 20 class labels :\n")

    for index, label in enumerate(labels[:20]):
        print(f"Action ID : {index}")
        print(f"Word : {label}")
        print("-" * 40)

if __name__ == "__main__":
    main()