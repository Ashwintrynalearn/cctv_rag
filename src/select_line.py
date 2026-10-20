import cv2
from pathlib import Path

VIDEO_PATH = "data/The CCTV People Demo 1 - CCTV People (1080p).mp4"

points = []


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
        points.append((x, y))
        print(f"Point {len(points)}: ({x}, {y})")


cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")
cap.set(cv2.CAP_PROP_POS_MSEC, 5000)
ret, frame = cap.read()

cap.release()

if not ret or frame is None:
    raise RuntimeError("Could not read the first frame")

# Resize the frame to a size that fits comfortably on a laptop screen.
original_h, original_w = frame.shape[:2]

display_w = 1200
scale = display_w / original_w
display_h = int(original_h * scale)

frame = cv2.resize(
    frame,
    (display_w, display_h),
    interpolation=cv2.INTER_AREA,
)

window = "Select Entrance Line"

cv2.namedWindow(window, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window, display_w, display_h)
cv2.setMouseCallback(window, mouse_callback)

print()
print("Click two points across the entrance.")
print("Then press ENTER.")
print("Press ESC to cancel.")
print()

while True:

    canvas = frame.copy()

    # Draw selected points
    for i, (x, y) in enumerate(points):
        cv2.circle(
            canvas,
            (x, y),
            8,
            (0, 255, 0),
            -1,
        )

        cv2.putText(
            canvas,
            f"P{i + 1}",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    # Draw entrance line
    if len(points) == 2:
        cv2.line(
            canvas,
            points[0],
            points[1],
            (0, 255, 0),
            4,
        )

    cv2.imshow(window, canvas)

    key = cv2.waitKey(30) & 0xFF

    if key == 13:  # ENTER
        if len(points) == 2:
            break

    elif key == 27:  # ESC
        points.clear()
        break

cv2.destroyAllWindows()

if len(points) != 2:
    print("Selection cancelled.")
    raise SystemExit

print()
print("Selected entrance line:")
print(f"Point 1: {points[0]}")
print(f"Point 2: {points[1]}")

Path("output").mkdir(exist_ok=True)

with open("output/entrance_line.txt", "w") as f:
    f.write(f"{points[0][0]},{points[0][1]}\n")
    f.write(f"{points[1][0]},{points[1][1]}\n")

print()
print("Saved to:")
print("output/entrance_line.txt")