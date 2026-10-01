import json
from pathlib import Path

DATASET_PATH = Path.home() / "Downloads" / "archive"

JSON_FILE = DATASET_PATH / "nslt_100.json"


def main():
    print("Reading WLASL labels...\n")

    with open(JSON_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    print(f"Total entries: {len(data)}\n")

    # Show the first 10 entries
    print("First 10 entries:\n")

    for i, (video_id, information) in enumerate(data.items()):
        print(f"Video ID: {video_id}")
        print(f"Subset: {information['subset']}")
        print(f"Action: {information['action']}")
        print("-" * 40)

        if i == 9:
            break


if __name__ == "__main__":
    main()