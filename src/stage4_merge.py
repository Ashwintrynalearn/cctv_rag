import csv
from pathlib import Path


EVENTS_FILE = "output/events.csv"
ATTRIBUTES_FILE = "output/person_attributes.csv"
OUTPUT_FILE = "output/person_events.csv"


def load_events():
    events = {}

    with open(
        EVENTS_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            person_id = int(row["person_id"])

            if person_id not in events:
                events[person_id] = []

            events[person_id].append(row)

    return events


def load_attributes():
    attributes = {}

    with open(
        ATTRIBUTES_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            person_id = int(row["person_id"])

            attributes[person_id] = row

    return attributes


def main():

    if not Path(EVENTS_FILE).exists():
        raise FileNotFoundError(
            f"Missing: {EVENTS_FILE}"
        )

    if not Path(ATTRIBUTES_FILE).exists():
        raise FileNotFoundError(
            f"Missing: {ATTRIBUTES_FILE}"
        )

    events = load_events()
    attributes = load_attributes()

    merged_rows = []

    for person_id, person_events in events.items():

        attr = attributes.get(
            person_id,
            {}
        )

        for event in person_events:

            merged_rows.append(
                {
                    "event_id": event["event_id"],
                    "person_id": person_id,
                    "timestamp": event["timestamp"],
                    "event_type": event["event_type"],
                    "location": event["location"],

                    "upper_clothing": attr.get(
                        "upper_clothing",
                        ""
                    ),

                    "lower_clothing": attr.get(
                        "lower_clothing",
                        ""
                    ),

                    "bag_present": attr.get(
                        "bag_present",
                        ""
                    ),

                    "bag_type": attr.get(
                        "bag_type",
                        ""
                    ),

                    "bag_color": attr.get(
                        "bag_color",
                        ""
                    ),

                    "helmet_present": attr.get(
                        "helmet_present",
                        ""
                    ),

                    "other_visible_objects":
                        attr.get(
                            "other_visible_objects",
                            ""
                        ),
                }
            )

    Path(OUTPUT_FILE).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "event_id",
                "person_id",
                "timestamp",
                "event_type",
                "location",
                "upper_clothing",
                "lower_clothing",
                "bag_present",
                "bag_type",
                "bag_color",
                "helmet_present",
                "other_visible_objects",
            ]
        )

        writer.writeheader()
        writer.writerows(
            merged_rows
        )

    print()
    print("==============================")
    print("Stage 4 Complete")
    print("==============================")

    print(
        f"Merged events: {len(merged_rows)}"
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()