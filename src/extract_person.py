import csv
import cv2
from pathlib import Path

VIDEO = "data/The CCTV People Demo 1 - CCTV People (1080p).mp4"
DETECTIONS = "output/detections_v2.csv"

PERSON_ID = 33
OUTPUT = "output/person_33.jpg"

# Pick the largest detected crop for this person.
best = None
best_area = 0

with open(DETECTIONS, "r") as f:
    reader = csv.DictReader(f)

    for row in reader:
        if int(row["track_id"]) != PERSON_ID:
            continue

        x1 = float(row["x1"])
        y1 = float(row["y1"])
        x2 = float(row["x2"])
        y2 = float(row["y2"])

        area = (x2 - x1) * (y2 - y1)

        if area > best_area:
            best_area = area
            best = row

if best is None:
    raise RuntimeError(f"Person {PERSON_ID} not found")

frame_number = int(best["frame"])

cap = cv2.VideoCapture(VIDEO)

cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    frame_number
)

ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Could not read video frame")

# IMPORTANT:
# detections_v2.csv coordinates are from our
# 640px-wide Stage-1 frames.

original_h, original_w = frame.shape[:2]

scale = 640 / original_w

small_h = int(original_h * scale)

frame = cv2.resize(
    frame,
    (640, small_h)
)

x1 = int(float(best["x1"]))
y1 = int(float(best["y1"]))
x2 = int(float(best["x2"]))
y2 = int(float(best["y2"]))

# Add some padding so the VLM can see bags/clothing
padding_x = int((x2 - x1) * 0.20)
padding_y = int((y2 - y1) * 0.10)

x1 = max(0, x1 - padding_x)
x2 = min(frame.shape[1], x2 + padding_x)

y1 = max(0, y1 - padding_y)
y2 = min(frame.shape[0], y2 + padding_y)

crop = frame[y1:y2, x1:x2]

Path("output").mkdir(exist_ok=True)

cv2.imwrite(OUTPUT, crop)

print(f"Person: {PERSON_ID}")
print(f"Frame: {frame_number}")
print(f"Timestamp: {best['timestamp']}s")
print(f"Saved: {OUTPUT}")