import csv
from pathlib import Path

import cv2


VIDEO_PATH = "data/The CCTV People Demo 1 - CCTV People (1080p).mp4"
DETECTIONS_FILE = "output/detections_v2.csv"
LINE_FILE = "output/entrance_line.txt"

OUTPUT_RAW = "output/event_validation_raw.avi"
OUTPUT_FINAL = "output/event_validation.mp4"

LINE_SELECTION_WIDTH = 1200
DISPLAY_WIDTH = 640

# How long to show the CROSSING message
EVENT_DISPLAY_SECONDS = 1.5


def load_line():
    with open(LINE_FILE, "r") as f:
        lines = f.readlines()

    x1, y1 = map(int, lines[0].strip().split(","))
    x2, y2 = map(int, lines[1].strip().split(","))

    scale = DISPLAY_WIDTH / LINE_SELECTION_WIDTH

    return (
        (int(x1 * scale), int(y1 * scale)),
        (int(x2 * scale), int(y2 * scale)),
    )


def resize_keep_aspect(frame, target_width):
    h, w = frame.shape[:2]

    scale = target_width / w
    new_h = int(h * scale)

    return cv2.resize(
        frame,
        (target_width, new_h),
        interpolation=cv2.INTER_AREA,
    )


def line_side(point, p1, p2):
    x, y = point
    x1, y1 = p1
    x2, y2 = p2

    value = (
        (x2 - x1) * (y - y1)
        - (y2 - y1) * (x - x1)
    )

    return 1 if value > 0 else -1


def load_detections():
    detections = {}

    with open(DETECTIONS_FILE, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            frame = int(row["frame"])

            if frame not in detections:
                detections[frame] = []

            detections[frame].append(
                {
                    "track_id": int(row["track_id"]),
                    "x1": float(row["x1"]),
                    "y1": float(row["y1"]),
                    "x2": float(row["x2"]),
                    "y2": float(row["y2"]),
                }
            )

    return detections


def main():

    p1, p2 = load_line()

    print(f"Entrance line: {p1} -> {p2}")

    detections = load_detections()

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    original_fps = cap.get(cv2.CAP_PROP_FPS)

    frame_step = max(
        1,
        round(original_fps / 2.0)
    )

    writer = None

    previous_side = {}

    # Keeps event text visible temporarily.
    active_events = {}

    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_number % frame_step != 0:
            frame_number += 1
            continue

        timestamp = frame_number / original_fps

        frame = resize_keep_aspect(
            frame,
            DISPLAY_WIDTH
        )

        # -------------------------
        # DRAW ENTRANCE LINE
        # -------------------------

        cv2.line(
            frame,
            p1,
            p2,
            (0, 255, 255),
            3
        )

        cv2.putText(
            frame,
            "ENTRANCE LINE",
            (p1[0], max(25, p1[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        # -------------------------
        # PROCESS PEOPLE
        # -------------------------

        if frame_number in detections:

            for person in detections[frame_number]:

                track_id = person["track_id"]

                if track_id == -1:
                    continue

                x1 = int(person["x1"])
                y1 = int(person["y1"])
                x2 = int(person["x2"])
                y2 = int(person["y2"])

                center_x = int((x1 + x2) / 2)
                bottom_y = y2

                point = (
                    center_x,
                    bottom_y
                )

                side = line_side(
                    point,
                    p1,
                    p2
                )

                # Bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Feet / bottom-center point
                cv2.circle(
                    frame,
                    point,
                    6,
                    (0, 0, 255),
                    -1
                )

                label = (
                    f"ID {track_id} | side {side}"
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

                # -------------------------
                # CROSSING CHECK
                # -------------------------

                if track_id in previous_side:

                    old_side = previous_side[
                        track_id
                    ]

                    if old_side != side:

                        active_events[track_id] = {
                            "time": timestamp,
                            "text": (
                                f"PERSON #{track_id} "
                                f"CROSSING "
                                f"{old_side} -> {side}"
                            )
                        }

                        print(
                            f"{timestamp:.2f}s | "
                            f"Person #{track_id} | "
                            f"{old_side} -> {side}"
                        )

                previous_side[track_id] = side

        # -------------------------
        # DISPLAY RECENT EVENTS
        # -------------------------

        y_position = 40

        expired = []

        for track_id, event in active_events.items():

            age = timestamp - event["time"]

            if age > EVENT_DISPLAY_SECONDS:
                expired.append(track_id)
                continue

            cv2.putText(
                frame,
                event["text"],
                (20, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2
            )

            y_position += 30

        for track_id in expired:
            del active_events[track_id]

        # Timestamp
        cv2.putText(
            frame,
            f"Time: {timestamp:.2f}s",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # -------------------------
        # VIDEO WRITER
        # -------------------------

        if writer is None:

            h, w = frame.shape[:2]

            fourcc = cv2.VideoWriter_fourcc(
                *"MJPG"
            )

            writer = cv2.VideoWriter(
                OUTPUT_RAW,
                fourcc,
                2.0,
                (w, h)
            )

        writer.write(frame)

        frame_number += 1

    cap.release()

    if writer is not None:
        writer.release()

    print()
    print("Raw validation video created.")
    print("Converting to H.264...")

    import subprocess

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            OUTPUT_RAW,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            OUTPUT_FINAL,
        ],
        check=True
    )

    Path(OUTPUT_RAW).unlink(
        missing_ok=True
    )

    print()
    print("Done!")
    print(f"Open: {OUTPUT_FINAL}")


if __name__ == "__main__":
    main()