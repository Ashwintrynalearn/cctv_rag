import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO


# COCO class IDs
PERSON = 0
BACKPACK = 24
UMBRELLA = 25
HANDBAG = 26
SUITCASE = 28

OBJECT_CLASSES = {
    BACKPACK: "backpack",
    UMBRELLA: "umbrella",
    HANDBAG: "handbag",
    SUITCASE: "suitcase",
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        required=True
    )

    parser.add_argument(
        "--model",
        default="yolov8n.pt"
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=2.0
    )

    parser.add_argument(
        "--width",
        type=int,
        default=640
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25
    )

    parser.add_argument(
        "--output",
        default="output/person_objects.csv"
    )

    return parser.parse_args()


def resize_keep_aspect(frame, target_width):

    h, w = frame.shape[:2]

    scale = target_width / w

    new_h = int(h * scale)

    return cv2.resize(
        frame,
        (target_width, new_h),
        interpolation=cv2.INTER_AREA
    )


def center(box):

    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2
    )


def object_near_person(person_box, object_box):
    """
    Determines whether an object is reasonably
    associated with a person's bounding box.

    We intentionally expand the person's bounding
    box because bags often sit partially outside it.
    """

    px1, py1, px2, py2 = person_box

    ox, oy = center(object_box)

    person_width = px2 - px1
    person_height = py2 - py1

    margin_x = person_width * 0.35
    margin_y = person_height * 0.15

    expanded_x1 = px1 - margin_x
    expanded_x2 = px2 + margin_x

    expanded_y1 = py1 - margin_y
    expanded_y2 = py2 + margin_y

    return (
        expanded_x1 <= ox <= expanded_x2
        and
        expanded_y1 <= oy <= expanded_y2
    )


def main():

    args = parse_args()

    video_path = Path(args.video)

    if not video_path.exists():
        raise FileNotFoundError(video_path)

    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open video"
        )

    original_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frame_step = max(
        1,
        round(
            original_fps / args.fps
        )
    )

    print(
        f"Video FPS: {original_fps:.2f}"
    )

    print(
        f"Processing every "
        f"{frame_step} frames"
    )

    model = YOLO(args.model)

    # Stores evidence across frames.
    #
    # Example:
    #
    # associations[17]["backpack"] = 5
    #
    # means Person 17 was associated with a
    # backpack in five processed frames.

    associations = defaultdict(
        lambda: defaultdict(int)
    )

    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_number % frame_step != 0:

            frame_number += 1
            continue

        timestamp = (
            frame_number / original_fps
        )

        frame = resize_keep_aspect(
            frame,
            args.width
        )

        # Detect and track persons + portable objects.
        results = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[
                PERSON,
                BACKPACK,
                UMBRELLA,
                HANDBAG,
                SUITCASE,
            ],
            conf=args.conf,
            imgsz=args.width,
            verbose=False,
        )

        result = results[0]

        people = []
        objects = []

        if result.boxes is not None:

            boxes = result.boxes

            xyxy = (
                boxes.xyxy
                .cpu()
                .numpy()
            )

            classes = (
                boxes.cls
                .cpu()
                .numpy()
                .astype(int)
            )

            confidences = (
                boxes.conf
                .cpu()
                .numpy()
            )

            if boxes.id is not None:

                track_ids = (
                    boxes.id
                    .cpu()
                    .numpy()
                    .astype(int)
                )

            else:

                track_ids = (
                    [-1] * len(xyxy)
                )

            for (
                box,
                class_id,
                confidence,
                track_id
            ) in zip(
                xyxy,
                classes,
                confidences,
                track_ids
            ):

                if class_id == PERSON:

                    people.append(
                        {
                            "track_id": track_id,
                            "box": box,
                            "confidence": confidence,
                        }
                    )

                elif class_id in OBJECT_CLASSES:

                    objects.append(
                        {
                            "class_id": class_id,
                            "name":
                                OBJECT_CLASSES[
                                    class_id
                                ],
                            "box": box,
                            "confidence": confidence,
                        }
                    )

        # --------------------------------
        # ASSOCIATE OBJECTS WITH PEOPLE
        # --------------------------------

        for person in people:

            person_id = (
                person["track_id"]
            )

            if person_id == -1:
                continue

            for obj in objects:

                if object_near_person(
                    person["box"],
                    obj["box"]
                ):

                    object_name = (
                        obj["name"]
                    )

                    associations[
                        person_id
                    ][
                        object_name
                    ] += 1

                    print(
                        f"{timestamp:6.2f}s | "
                        f"Person #{person_id} | "
                        f"{object_name}"
                    )

        frame_number += 1

    cap.release()

    # --------------------------------
    # SAVE RESULTS
    # --------------------------------

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "person_id",
                "object",
                "frames_detected"
            ]
        )

        for (
            person_id,
            objects
        ) in associations.items():

            for (
                object_name,
                count
            ) in objects.items():

                writer.writerow(
                    [
                        person_id,
                        object_name,
                        count
                    ]
                )

    print()
    print("======================")
    print("Stage 3A Complete")
    print("======================")

    print(
        f"Saved to: {output_path}"
    )


if __name__ == "__main__":
    main()