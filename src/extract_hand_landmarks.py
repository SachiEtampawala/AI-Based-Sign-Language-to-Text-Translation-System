import cv2
import mediapipe as mp
from pathlib import Path


DATASET_PATH = Path.home() / "Downloads" / "archive"

VIDEO_FILE = DATASET_PATH / "videos" / "69422.mp4"

PROJECT_PATH = Path(__file__).resolve().parent.parent

MODEL_FILE = PROJECT_PATH / "models" / "hand_landmarker.task"

OUTPUT_FOLDER = PROJECT_PATH / "outputs"

OUTPUT_FILE = OUTPUT_FOLDER / "hand_landmarks.txt"


def main():
    print("Extracting hand landmarks...\n")

    if not VIDEO_FILE.exists():
        print("ERROR: Video not found:")
        print(VIDEO_FILE)
        return

    if not MODEL_FILE.exists():
        print("ERROR: MediaPipe model not found:")
        print(MODEL_FILE)
        return

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    # Create MediaPipe Hand Landmarker
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(MODEL_FILE)
        ),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(str(VIDEO_FILE))

    if not cap.isOpened():
        print("ERROR: Could not open the video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_number = 0
    frames_with_hands = 0

    with HandLandmarker.create_from_options(options) as landmarker:

        with open(OUTPUT_FILE, "w", encoding="utf-8") as output:

            while True:
                success, frame = cap.read()

                if not success:
                    break

                frame_number += 1

                # Convert OpenCV BGR image to RGB
                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                # Convert image to MediaPipe format
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                # Detect hands
                result = landmarker.detect(mp_image)

                if result.hand_landmarks:

                    frames_with_hands += 1

                    output.write(
                        f"Frame: {frame_number}\n"
                    )

                    for hand_number, hand_landmarks in enumerate(
                        result.hand_landmarks
                    ):

                        output.write(
                            f"Hand: {hand_number + 1}\n"
                        )

                        for landmark_number, landmark in enumerate(
                            hand_landmarks
                        ):

                            output.write(
                                f"{landmark_number}: "
                                f"x={landmark.x:.6f}, "
                                f"y={landmark.y:.6f}, "
                                f"z={landmark.z:.6f}\n"
                            )

                    output.write("\n")

    cap.release()

    print(f"Video: 69422.mp4")
    print(f"Total frames: {total_frames}")
    print(f"Frames with detected hands: {frames_with_hands}")
    print(f"Saved landmarks to: {OUTPUT_FILE}")

    print("\nHand landmark extraction completed.")


if __name__ == "__main__":
    main()