import cv2
from pathlib import Path

DATASET_PATH = Path.home() / "Downloads" / "archive"

VIDEO_FILE = DATASET_PATH / "videos" / "69422.mp4"

OUTPUT_FOLDER = Path(__file__).resolve().parent.parent / "outputs"

OUTPUT_FILE = OUTPUT_FOLDER / "sample_frame.jpg"

def main():
    print("Extracting frame from WLASL video...\n")

    if not VIDEO_FILE.exists():
        print("ERROR : Video not found:")
        print(VIDEO_FILE)
        return

    # Create outputs folder if it does not exist

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    # Open video

    cap = cv2.VideoCapture(str(VIDEO_FILE))

    if not cap.isOpened():
        print("ERROR : Could not open the video.")
        return

    # Move to the middle frame

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame = total_frames // 2

    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)

    success, frame = cap.read()

    if not success:
        print("ERROR : Could not read the frame.")
        cap.release()
        return

    # Save frame

    cv2.imwrite(str(OUTPUT_FILE), frame)

    cap.release()

    print(f"Video: 69422.mp4")
    print(f"Total frames : {total_frames}")
    print(f"Frame extracted: {middle_frame}")
    print(f"Saved to : {OUTPUT_FILE}")

    print("\nFrame extraction completed.")

if __name__ == "__main__":
    main()