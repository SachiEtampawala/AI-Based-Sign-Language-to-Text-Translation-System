import json
import csv
import cv2
import mediapipe as mp
from pathlib import Path

# Paths

DATASET_PATH = Path.home() / "Downloads" / "archive"

JSON_FILE = DATASET_PATH / "nslt_100.json"
CLASS_FILE = DATASET_PATH / "wlasl_class_list.txt"
VIDEOS_FOLDER = DATASET_PATH / "videos"

PROJECT_PATH = Path(__file__).resolve().parent.parent
OUTPUT_FOLDER = PROJECT_PATH / "outputs"

DATASET_FILE = OUTPUT_FOLDER / "landmark_dataset.csv"
MODEL_FILE = PROJECT_PATH / "models" / "hand_landmarker.task"

# Settings

TARGET_VIDEOS = 100

# Load class labels

def load_class_labels():

    labels = {}

    with open(CLASS_FILE, "r", encoding="utf-8") as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            parts = line.split(maxsplit=1)

            if len(parts) == 2:

                action_id = int(parts[0])
                word = parts[1].strip()

                labels[action_id] = word

    return labels

# Get existing video IDs

def load_existing_video_ids():

    existing_ids = set()

    if not DATASET_FILE.exists():
        return existing_ids

    with open(
        DATASET_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            existing_ids.add(row["video_id"])

    return existing_ids

# Extract landmarks

def extract_video_landmarks(video_file, landmarker):

    cap = cv2.VideoCapture(str(video_file))

    if not cap.isOpened():
        return []

    video_landmarks = []

    while True:

        success, frame = cap.read()

        if not success:
            break

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = landmarker.detect(mp_image)

        frame_landmarks = []

        if result.hand_landmarks:

            # Use the first detected hand

            hand = result.hand_landmarks[0]

            for landmark in hand:

                frame_landmarks.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

        # 21 landmarks × 3 coordinates

        if len(frame_landmarks) != 63:

            frame_landmarks = [0.0] * 63

        video_landmarks.append(frame_landmarks)

    cap.release()

    return video_landmarks

# Main

def main():

    print("Expanding landmark dataset...\n")

    if not JSON_FILE.exists():

        print("ERROR : JSON file not found.")
        print(JSON_FILE)
        return

    if not CLASS_FILE.exists():

        print("ERROR : Class label file not found.")
        print(CLASS_FILE)
        return

    if not VIDEOS_FOLDER.exists():

        print("ERROR : Videos folder not found.")
        print(VIDEOS_FOLDER)
        return

    if not MODEL_FILE.exists():

        print("ERROR : MediaPipe model not found.")
        print(MODEL_FILE)
        return

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load metadata

    with open(
        JSON_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    labels = load_class_labels()

    existing_ids = load_existing_video_ids()

    print(f"Dataset entries: {len(data)}")
    print(f"Existing videos: {len(existing_ids)}")
    print(f"Target additional videos: {TARGET_VIDEOS}\n")

    # Check whether the CSV already exists

    file_exists = DATASET_FILE.exists()

    # MediaPipe setup

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(

        base_options=BaseOptions(
            model_asset_path=str(MODEL_FILE)
        ),

        running_mode=VisionRunningMode.IMAGE,

        num_hands=1,

        min_hand_detection_confidence=0.5,

        min_hand_presence_confidence=0.5,

        min_tracking_confidence=0.5
    )

    processed = 0
    skipped = 0

    with HandLandmarker.create_from_options(options) as landmarker:

        with open(
            DATASET_FILE,
            "a",
            newline="",
            encoding="utf-8"
        ) as csv_file:

            writer = csv.writer(csv_file)

            # Add header only if file is new

            if not file_exists:

                header = [
                    "video_id",
                    "sign",
                    "subset"
                ]

                for i in range(63):
                    header.append(
                        f"landmark_{i}"
                    )

                writer.writerow(header)

            for video_id, information in data.items():

                if processed >= TARGET_VIDEOS:
                    break

                # Skip videos already processed

                if video_id in existing_ids:
                    continue

                video_file = (
                    VIDEOS_FOLDER /
                    f"{video_id}.mp4"
                )

                # Skip missing videos

                if not video_file.exists():

                    skipped += 1
                    continue

                action = information["action"]

                action_id = action[0]

                sign = labels.get(
                    action_id,
                    "unknown"
                )

                subset = information["subset"]

                print(
                    f"Processing {video_id} "
                    f"→ {sign} "
                    f"({subset})"
                )

                landmarks = extract_video_landmarks(
                    video_file,
                    landmarker
                )

                if not landmarks:

                    print("No frames found. Skipping.")
                    skipped += 1
                    continue

                for frame_landmarks in landmarks:

                    writer.writerow([
                        video_id,
                        sign,
                        subset,
                        *frame_landmarks
                    ])

                processed += 1

                print(
                    f"  Frames processed: "
                    f"{len(landmarks)}"
                )

    print("\n----------")
    print("Dataset expansion completed.")
    print(f"Additional videos processed : {processed}")
    print(f"Videos skipped : {skipped}")
    print(f"Dataset saved to :")
    print(DATASET_FILE)


if __name__ == "__main__":
    main()