import json
import csv
import cv2
import mediapipe as mp
from pathlib import Path

# Project paths

DATASET_PATH = Path.home() / "Downloads" / "archive"

JSON_FILE = DATASET_PATH / "nslt_100.json"
CLASS_FILE = DATASET_PATH / "wlasl_class_list.txt"
VIDEOS_FOLDER = DATASET_PATH / "videos"

PROJECT_PATH = Path(__file__).resolve().parent.parent
OUTPUT_FOLDER = PROJECT_PATH / "outputs"

OUTPUT_FILE = OUTPUT_FOLDER / "landmark_dataset.csv"
MODEL_FILE = PROJECT_PATH / "models" / "hand_landmarker.task"

# Settings

MAX_VIDEOS = 20

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

# Extract landmarks from video

def extract_video_landmarks(video_file, landmarker):
    cap = cv2.VideoCapture(str(video_file))

    if not cap.isOpened():
        return []

    video_landmarks = []

    while True:
        success, frame = cap.read()

        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = landmarker.detect(mp_image)

        frame_landmarks = []

        if result.hand_landmarks:
            for hand in result.hand_landmarks:

                hand_values = []

                for landmark in hand:
                    hand_values.extend([
                        landmark.x,
                        landmark.y,
                        landmark.z
                    ])

                frame_landmarks.extend(hand_values)

        # If no hand is detected, use zeros

        if not frame_landmarks:
            frame_landmarks = [0.0] * 63

        # Currently using one hand for the dataset
        
        if len(frame_landmarks) > 63:
            frame_landmarks = frame_landmarks[:63]

        while len(frame_landmarks) < 63:
            frame_landmarks.append(0.0)

        video_landmarks.append(frame_landmarks)

    cap.release()

    return video_landmarks

# Main

def main():

    print("Creating structured landmark dataset...\n")

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

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    # Load dataset information

    with open(JSON_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    labels = load_class_labels()

    print(f"Dataset entries : {len(data)}")
    print(f"Class labels : {len(labels)}")
    print(f"Processing first {MAX_VIDEOS} available videos...\n")

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

    processed_videos = 0
    skipped_videos = 0

    with HandLandmarker.create_from_options(options) as landmarker:

        with open(
            OUTPUT_FILE,
            "w",
            newline="",
            encoding="utf-8"
        ) as csv_file:

            writer = csv.writer(csv_file)

            # CSV header

            header = [
                "video_id",
                "sign",
                "subset"
            ]

            for i in range(63):
                header.append(f"landmark_{i}")

            writer.writerow(header)

            for video_id, information in data.items():

                if processed_videos >= MAX_VIDEOS:
                    break

                video_file = VIDEOS_FOLDER / f"{video_id}.mp4"

                # Skip missing videos

                if not video_file.exists():
                    skipped_videos += 1
                    continue

                action = information["action"]

                # First value represents the class ID

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

                video_landmarks = extract_video_landmarks(
                    video_file,
                    landmarker
                )

                if not video_landmarks:
                    print("No frames found. Skipping.")
                    skipped_videos += 1
                    continue

                # Save every frame as one CSV row

                for frame_landmarks in video_landmarks:

                    writer.writerow([
                        video_id,
                        sign,
                        subset,
                        *frame_landmarks
                    ])

                processed_videos += 1

                print(
                    f"  Frames processed : "
                    f"{len(video_landmarks)}"
                )

    print("\n----------")
    print("Dataset creation completed.")
    print(f"Videos processed : {processed_videos}")
    print(f"Videos skipped : {skipped_videos}")
    print(f"Saved to : {OUTPUT_FILE}")

if __name__ == "__main__":
    main()