import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import cv2

from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config


# ============================================================
# CONFIG
# ============================================================

VIDEO_PATH = "data/The CCTV People Demo 1 - CCTV People (1080p).mp4"
DETECTIONS_FILE = "output/detections_v2.csv"

CROPS_DIR = Path("output/person_crops")
OBSERVATIONS_FILE = Path(
    "output/person_attribute_observations.csv"
)
PROFILES_FILE = Path(
    "output/person_attributes.csv"
)

MODEL_NAME = (
    "mlx-community/SmolVLM2-2.2B-Instruct-mlx"
)

# Width used in Stage 1 tracking.
TRACKING_WIDTH = 640

# Ignore extremely small detections.
MIN_CROP_AREA = 3000

# Extra space around person bounding box.
PADDING_X = 0.20
PADDING_Y = 0.15


# ============================================================
# LOAD DETECTIONS
# ============================================================

def load_detections():
    """
    Load detections from Stage 1 CSV.

    Returns:

        {
            person_id: [
                {
                    "frame": int,
                    "timestamp": float,
                    "x1": float,
                    "y1": float,
                    "x2": float,
                    "y2": float,
                    "area": float
                }
            ]
        }
    """

    detections = defaultdict(list)

    with open(
        DETECTIONS_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            person_id = int(row["track_id"])

            if person_id == -1:
                continue

            x1 = float(row["x1"])
            y1 = float(row["y1"])
            x2 = float(row["x2"])
            y2 = float(row["y2"])

            width = x2 - x1
            height = y2 - y1

            area = width * height

            if area < MIN_CROP_AREA:
                continue

            detections[person_id].append(
                {
                    "frame": int(row["frame"]),
                    "timestamp": float(
                        row["timestamp"]
                    ),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "area": area,
                }
            )

    return detections


# ============================================================
# SELECT BEST CROP
# ============================================================

def select_crops(person_detections):
    """
    Select the largest detected crop for a person.

    A larger bounding box generally gives the VLM more
    useful visual information.
    """

    if not person_detections:
        return []

    best = max(
        person_detections,
        key=lambda x: x["area"],
    )

    return [best]


# ============================================================
# EXTRACT PERSON CROP
# ============================================================

def extract_crop(
    cap,
    detection,
    original_width,
    original_height,
    person_id,
    crop_index,
):
    """
    Extract person crop from the original video.
    """

    frame_number = detection["frame"]

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_number,
    )

    success, frame = cap.read()

    if not success:
        return None

    # Stage 1 detections were generated from
    # 640px-wide frames.
    scale = original_width / TRACKING_WIDTH

    x1 = int(detection["x1"] * scale)
    y1 = int(detection["y1"] * scale)
    x2 = int(detection["x2"] * scale)
    y2 = int(detection["y2"] * scale)

    box_width = x2 - x1
    box_height = y2 - y1

    padding_x = int(
        box_width * PADDING_X
    )

    padding_y = int(
        box_height * PADDING_Y
    )

    x1 = max(
        0,
        x1 - padding_x,
    )

    y1 = max(
        0,
        y1 - padding_y,
    )

    x2 = min(
        original_width,
        x2 + padding_x,
    )

    y2 = min(
        original_height,
        y2 + padding_y,
    )

    crop = frame[
        y1:y2,
        x1:x2,
    ]

    if crop.size == 0:
        return None

    person_dir = (
        CROPS_DIR /
        f"person_{person_id}"
    )

    person_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        person_dir /
        f"crop_{crop_index}.jpg"
    )

    cv2.imwrite(
        str(output_path),
        crop,
    )

    return output_path


# ============================================================
# RUN VLM
# ============================================================

def analyze_person(
    model,
    processor,
    config,
    image_paths,
):
    """
    Analyze the selected person image(s).
    """

    prompt = """
Analyze this CCTV image of ONE person.

Return exactly these seven lines and nothing else:

description: <one concise description of clearly visible appearance>
upper_clothing: <value>
lower_clothing: <value>
bag_present: <yes or no>
bag_type: <value>
bag_color: <value>
helmet_present: <yes or no>

Rules:
- Only describe clearly visible information.
- Do not guess.
- If something is not visible, write: unknown
- Use simple clothing descriptions.
- For bag_present, use only yes or no.
- For helmet_present, use only yes or no.
"""
    image_paths = [
        str(path)
        for path in image_paths
    ]

    formatted_prompt = apply_chat_template(
        processor,
        config,
        prompt,
        num_images=len(image_paths),
    )

    output = generate(
        model,
        processor,
        formatted_prompt,
        image_paths,
        max_tokens=80,
        temp=0.0,
    )

    return output.strip()


# ============================================================
# JSON PARSER
# ============================================================

def parse_attributes(text):
    attributes = {
        "description": "",
        "upper_clothing": "unknown",
        "lower_clothing": "unknown",
        "bag_present": "unknown",
        "bag_type": "unknown",
        "bag_color": "unknown",
        "helmet_present": "unknown",
    }

    for line in text.splitlines():

        line = line.strip()

        if ":" not in line:
            continue

        key, value = line.split(":", 1)

        key = key.strip().lower()
        value = value.strip()

        if key in attributes:
            attributes[key] = value

    return attributes


def read_existing_csv(path, key_field):
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {row[key_field]: row for row in csv.DictReader(f)}


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(description="Describe one clear crop for every tracked person")
    parser.add_argument("--force", action="store_true", help="Reanalyze people that already have a successful profile")
    args = parser.parse_args()

    CROPS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    Path("output").mkdir(
        parents=True,
        exist_ok=True,
    )

    detections = load_detections()

    print(
        f"Found {len(detections)} tracked people."
    )

    old_profiles = read_existing_csv(PROFILES_FILE, "person_id")
    old_observations = read_existing_csv(OBSERVATIONS_FILE, "person_id")
    pending_ids = [
        pid for pid in sorted(detections)
        if args.force or old_profiles.get(str(pid), {}).get("parse_status") != "success"
    ]
    print(f"People to describe: {len(pending_ids)}; already complete: {len(detections) - len(pending_ids)}")
    if not pending_ids:
        print("All tracked people already have successful profiles.")
        return

    # --------------------------------------------------------
    # Open original video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        VIDEO_PATH
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: "
            f"{VIDEO_PATH}"
        )

    original_width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    original_height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    print(
        f"Original video: "
        f"{original_width}x{original_height}"
    )

    # --------------------------------------------------------
    # Load model ONCE
    # --------------------------------------------------------

    print()
    print("Loading SmolVLM2...")

    model, processor = load(
        MODEL_NAME
    )

    config = load_config(
        MODEL_NAME
    )

    print("SmolVLM2 loaded.")
    print()

    observation_rows = dict(old_observations) if not args.force else {}
    profile_rows = dict(old_profiles) if not args.force else {}

    # --------------------------------------------------------
    # Analyze selected people
    # --------------------------------------------------------

    for person_id in pending_ids:

        selected = select_crops(
            detections[person_id]
        )

        if not selected:

            print(
                f"Person #{person_id} "
                f"has no usable crop."
            )

            continue

        print(
            f"Person #{person_id}: "
            f"selecting best crop..."
        )

        image_paths = []

        for crop_index, detection in enumerate(
            selected,
            start=1,
        ):

            crop_path = extract_crop(
                cap,
                detection,
                original_width,
                original_height,
                person_id,
                crop_index,
            )

            if crop_path is not None:
                image_paths.append(
                    crop_path
                )

        if not image_paths:

            print(
                f"  No valid images for "
                f"Person #{person_id}"
            )

            continue

        print(
            f"  Crop: {image_paths[0]}"
        )

        print(
            "  Running VLM..."
        )

        start_time = time.time()

        try:

            raw_output = analyze_person(
                model,
                processor,
                config,
                image_paths,
            )

        except Exception as exc:

            print(
                f"  ERROR: {exc}"
            )

            continue

        elapsed = (
            time.time() - start_time
        )

        print(
            f"  VLM time: "
            f"{elapsed:.1f} seconds"
        )

        print(
            f"  VLM output: "
            f"{raw_output}"
        )

        parsed = parse_attributes(
            raw_output
        )

        # ----------------------------------------------------
        # Save raw observation
        # ----------------------------------------------------

        observation_rows[str(person_id)] = {
                "person_id": person_id,
                "num_crops": len(
                    image_paths
                ),
                "crop_paths": ";".join(
                    str(path)
                    for path in image_paths
                ),
                "raw_output": raw_output,
                "time_seconds": round(
                    elapsed,
                    2,
                ),
            }

        # ----------------------------------------------------
        # Save structured profile
        # ----------------------------------------------------

        profile_rows[str(person_id)] = {
                    "person_id": person_id,
                    **parsed,
                    "parse_status": "success",
                }

        # Save after each person so an interrupted full-video pass can resume.
        write_csv(OBSERVATIONS_FILE,
                  ["person_id", "num_crops", "crop_paths", "raw_output", "time_seconds"],
                  observation_rows.values())
        write_csv(PROFILES_FILE,
                  ["person_id", "description", "upper_clothing", "lower_clothing", "bag_present", "bag_type",
                   "bag_color", "helmet_present", "parse_status"],
                  profile_rows.values())

        print()

    cap.release()

    # --------------------------------------------------------
    # Save raw observations
    # --------------------------------------------------------

    write_csv(OBSERVATIONS_FILE,
              ["person_id", "num_crops", "crop_paths", "raw_output", "time_seconds"],
              observation_rows.values())
    write_csv(PROFILES_FILE,
              ["person_id", "description", "upper_clothing", "lower_clothing", "bag_present", "bag_type",
               "bag_color", "helmet_present", "parse_status"],
              profile_rows.values())

    print()
    print("==============================")
    print("Stage 3 complete")
    print("==============================")
    print(
        f"Observations: "
        f"{OBSERVATIONS_FILE}"
    )
    print(
        f"Profiles: "
        f"{PROFILES_FILE}"
    )
    print(
        f"Crops: "
        f"{CROPS_DIR}"
    )


if __name__ == "__main__":
    main()
