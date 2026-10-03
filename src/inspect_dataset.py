import json
from pathlib import Path

# Location of the downloaded WLASL dataset

DATASET_PATH = Path.home() / "Downloads" / "archive"

JSON_FILE = DATASET_PATH / "nslt_100.json"
VIDEOS_FOLDER = DATASET_PATH / "videos"

def main():
    print("Checking WLASL dataset...\n")

    # Check JSON file

    if not JSON_FILE.exists():
        print("ERROR : JSON file not found.")
        print(JSON_FILE)
        return

    # Check videos folder

    if not VIDEOS_FOLDER.exists():
        print("ERROR : Videos folder not found.")
        print(VIDEOS_FOLDER)
        return

    print("✓ JSON file found")
    print("✓ Videos folder found\n")

    # Load JSON

    with open(JSON_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    print(f"Number of entries : {len(data)}")

    # Show the first entry

    first_key = next(iter(data))
    first_entry = data[first_key]

    print("\nFirst dataset entry :")
    print(f"Key : {first_key}")
    print(f"Data : {first_entry}")

    print("\n--------")

    # Count available videos

    video_files = list(VIDEOS_FOLDER.glob("*.mp4"))

    print(f"\nMP4 videos found : {len(video_files)}")
    print("\nDataset inspection completed.")

if __name__ == "__main__":
    main()