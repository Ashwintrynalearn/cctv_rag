import argparse
import csv
import subprocess
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 1: Person detection + ByteTrack on CCTV video"
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Path to input CCTV video",
    )

    parser.add_argument(
        "--output",
        default="output/tracked.mp4",
        help="Path for final annotated MP4",
    )

    parser.add_argument(
        "--csv",
        default="output/detections.csv",
        help="Path for detection CSV",
    )

    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO model to use",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Number of video frames processed per second",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Width used for YOLO inference",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.40,
        help="Detection confidence threshold",
    )

    return parser.parse_args()


def resize_keep_aspect(frame, target_width):
    height, width = frame.shape[:2]

    if width <= target_width:
        return frame

    scale = target_width / width
    target_height = int(height * scale)

    return cv2.resize(
        frame,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )


def main():
    args = parse_args()

    input_path = Path(args.video)
    output_path = Path(args.output)
    csv_path = Path(args.csv)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Video not found: {input_path}"
        )

    # Make sure output directories exist.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Temporary AVI file.
    raw_output_path = output_path.with_name(
        output_path.stem + "_raw.avi"
    )

    print(f"Loading video: {input_path}")

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {input_path}"
        )

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )
    original_height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if original_fps <= 0:
        raise RuntimeError(
            "Could not determine video FPS."
        )

    duration = total_frames / original_fps

    print()
    print("Video information")
    print("-----------------")
    print(f"Resolution : {original_width}x{original_height}")
    print(f"FPS        : {original_fps:.2f}")
    print(f"Frames     : {total_frames}")
    print(f"Duration   : {duration:.1f} seconds")
    print()
    print(f"Processing FPS : {args.fps}")
    print(f"Inference width: {args.width}")
    print(f"Confidence     : {args.conf}")

    # Calculate how many original frames to skip.
    frame_step = max(
        1,
        round(original_fps / args.fps)
    )

    print(
        f"Processing approximately every "
        f"{frame_step} original frames."
    )

    print()
    print(f"Temporary video: {raw_output_path}")
    print(f"Final video    : {output_path}")
    print(f"CSV            : {csv_path}")
    print()

    # Load YOLO.
    print("Loading YOLO model...")
    model = YOLO(args.model)
    print("YOLO model loaded.")

    # COCO class 0 = person.
    PERSON_CLASS_ID = 0

    writer = None

    csv_file = open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    )

    csv_writer = csv.writer(csv_file)

    csv_writer.writerow(
        [
            "frame",
            "timestamp",
            "track_id",
            "class",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
        ]
    )

    frame_number = 0
    processed_count = 0

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Skip frames to reduce processing load.
            if frame_number % frame_step != 0:
                frame_number += 1
                continue

            timestamp = frame_number / original_fps

            # Resize frame while preserving aspect ratio.
            resized_frame = resize_keep_aspect(
                frame,
                args.width,
            )

            # Run YOLO + ByteTrack.
            results = model.track(
                source=resized_frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[PERSON_CLASS_ID],
                conf=args.conf,
                imgsz=args.width,
                verbose=False,
            )

            result = results[0]

            # Draw bounding boxes + tracking IDs.
            annotated_frame = result.plot()

            # Create the temporary AVI writer once.
            if writer is None:
                output_height, output_width = (
                    annotated_frame.shape[:2]
                )

                fourcc = cv2.VideoWriter_fourcc(
                    *"MJPG"
                )

                writer = cv2.VideoWriter(
                    str(raw_output_path),
                    fourcc,
                    args.fps,
                    (output_width, output_height),
                )

                if not writer.isOpened():
                    raise RuntimeError(
                        f"Could not create temporary video: "
                        f"{raw_output_path}"
                    )

            writer.write(annotated_frame)

            # Save detection information.
            if result.boxes is not None:
                boxes = result.boxes

                xyxy = boxes.xyxy.cpu().numpy()
                confidences = (
                    boxes.conf.cpu().numpy()
                )
                classes = boxes.cls.cpu().numpy()

                if boxes.id is not None:
                    track_ids = (
                        boxes.id
                        .cpu()
                        .numpy()
                        .astype(int)
                    )
                else:
                    track_ids = [
                        -1
                    ] * len(xyxy)

                for (
                    box,
                    confidence,
                    class_id,
                    track_id,
                ) in zip(
                    xyxy,
                    confidences,
                    classes,
                    track_ids,
                ):
                    x1, y1, x2, y2 = box

                    csv_writer.writerow(
                        [
                            frame_number,
                            f"{timestamp:.2f}",
                            track_id,
                            "person",
                            f"{confidence:.4f}",
                            f"{x1:.2f}",
                            f"{y1:.2f}",
                            f"{x2:.2f}",
                            f"{y2:.2f}",
                        ]
                    )

            processed_count += 1

            if processed_count % 10 == 0:
                print(
                    f"Processed {processed_count} frames "
                    f"(video time: {timestamp:.1f}s)"
                )

            frame_number += 1

    finally:
        cap.release()

        if writer is not None:
            writer.release()

        csv_file.close()

    print()
    print("YOLO processing complete.")

    # Convert temporary AVI to H.264 MP4.
    print("Converting output video to H.264 MP4...")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(raw_output_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        check=True,
    )

    # Remove temporary AVI.
    if raw_output_path.exists():
        raw_output_path.unlink()

    print()
    print("================================")
    print("Stage 1 completed successfully!")
    print("================================")
    print(f"Tracked video : {output_path}")
    print(f"Detection CSV : {csv_path}")
    print(f"Processed     : {processed_count} frames")


if __name__ == "__main__":
    main()