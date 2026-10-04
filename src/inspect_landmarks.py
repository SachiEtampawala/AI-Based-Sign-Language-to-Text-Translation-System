from pathlib import Path

PROJECT_PATH = Path(__file__).resolve().parent.parent

LANDMARK_FILE = PROJECT_PATH / "outputs" / "hand_landmarks.txt"

def main():
    print("Inspecting extracted hand landmarks...\n")

    if not LANDMARK_FILE.exists():
        print("ERROR : Landmark file not found.")
        print(LANDMARK_FILE)
        return

    with open(LANDMARK_FILE, "r", encoding="utf-8") as file:
        lines = file.readlines()

    frame_count = 0
    hand_count = 0
    landmark_count = 0

    for line in lines:

        line = line.strip()

        if line.startswith("Frame :"):
            frame_count += 1

        elif line.startswith("Hand :"):
            hand_count += 1

        elif line[:1].isdigit() and ":" in line:
            landmark_count += 1

    print(f"Frames with landmarks : {frame_count}")
    print(f"Hands detected : {hand_count}")
    print(f"Total landmark points : {landmark_count}")

    print("\nFirst 20 lines of landmark data :\n")

    for line in lines[:20]:
        print(line.strip())

    print("\nLandmark inspection completed.")

if __name__ == "__main__":
    main()