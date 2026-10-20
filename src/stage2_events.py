import csv
from pathlib import Path


DETECTIONS_FILE = "output/detections_v2.csv"
LINE_FILE = "output/entrance_line.txt"
EVENTS_FILE = "output/events.csv"

LINE_SELECTION_WIDTH = 1200
TRACKING_WIDTH = 640

# Prevent tiny movements around the line from immediately
# generating repeated events.
COOLDOWN_SECONDS = 2.0


def load_line():
    with open(LINE_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    x1, y1 = map(int, lines[0].strip().split(","))
    x2, y2 = map(int, lines[1].strip().split(","))

    scale = TRACKING_WIDTH / LINE_SELECTION_WIDTH

    return (
        (int(x1 * scale), int(y1 * scale)),
        (int(x2 * scale), int(y2 * scale)),
    )


def get_side(point, p1, p2):
    x, y = point
    x1, y1 = p1
    x2, y2 = p2

    value = (
        (x2 - x1) * (y - y1)
        - (y2 - y1) * (x - x1)
    )

    if value > 0:
        return 1

    if value < 0:
        return -1

    return 0


def main():
    if not Path(DETECTIONS_FILE).exists():
        raise FileNotFoundError(DETECTIONS_FILE)

    if not Path(LINE_FILE).exists():
        raise FileNotFoundError(LINE_FILE)

    p1, p2 = load_line()

    print(f"Entrance line: {p1} -> {p2}")
    print()
    print("Camera mapping:")
    print("+1 = INSIDE")
    print("-1 = OUTSIDE")
    print("+1 -> -1 = EXIT")
    print("-1 -> +1 = ENTER")
    print()

    # Previous side for every tracked person.
    previous_side = {}

    # Last event timestamp for every person.
    last_event_time = {}

    events = []

    with open(DETECTIONS_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            track_id = int(row["track_id"])

            if track_id == -1:
                continue

            timestamp = float(row["timestamp"])

            x1 = float(row["x1"])
            y1 = float(row["y1"])
            x2 = float(row["x2"])
            y2 = float(row["y2"])

            # Bottom-center of bounding box.
            # This roughly represents the person's feet.
            center_x = (x1 + x2) / 2
            bottom_y = y2

            point = (
                int(center_x),
                int(bottom_y),
            )

            side = get_side(point, p1, p2)

            # Ignore points directly on the line.
            if side == 0:
                continue

            if track_id not in previous_side:
                previous_side[track_id] = side
                continue

            old_side = previous_side[track_id]

            if old_side != side:

                # Cooldown to avoid rapid repeated crossings.
                last_time = last_event_time.get(track_id)

                if (
                    last_time is None
                    or timestamp - last_time >= COOLDOWN_SECONDS
                ):

                    if old_side == -1 and side == 1:
                        event_type = "ENTER"

                    elif old_side == 1 and side == -1:
                        event_type = "EXIT"

                    else:
                        event_type = "CROSSING"

                    event = {
                        "event_id": len(events) + 1,
                        "timestamp": timestamp,
                        "person_id": track_id,
                        "event_type": event_type,
                        "location": "Main Entrance",
                        "description": (
                            f"Person {track_id} "
                            f"{event_type.lower()}ed "
                            f"through the main entrance"
                        ),
                    }

                    events.append(event)
                    last_event_time[track_id] = timestamp

                    print(
                        f"{timestamp:7.2f}s | "
                        f"Person #{track_id:3d} | "
                        f"{event_type}"
                    )

            previous_side[track_id] = side

    Path(EVENTS_FILE).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        EVENTS_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "event_id",
                "timestamp",
                "person_id",
                "event_type",
                "location",
                "description",
            ],
        )

        writer.writeheader()
        writer.writerows(events)

    print()
    print("==========================")
    print("Stage 2 complete")
    print("==========================")
    print(f"Events detected : {len(events)}")
    print(f"Saved to        : {EVENTS_FILE}")


if __name__ == "__main__":
    main()