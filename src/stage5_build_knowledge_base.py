"""Build SQLite facts and a FAISS index for person descriptions and crossings."""
import csv
import sqlite3
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ATTRIBUTES_FILE = Path("output/person_attributes.csv")
EVENTS_FILE = Path("output/events.csv")
DATABASE_FILE = Path("output/cctv_knowledge.sqlite")
INDEX_FILE = Path("output/knowledge.faiss")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def read_rows(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run the earlier pipeline stages first")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def describe_person(row):
    facts = []
    visual_description = (row.get("description") or "").strip()
    for key, label in (("upper_clothing", "upper clothing"), ("lower_clothing", "lower clothing"),
                       ("bag_type", "bag type"), ("bag_color", "bag color"),
                       ("helmet_present", "helmet")):
        value = (row.get(key) or "").strip()
        if value and value.lower() not in {"unknown", "none"}:
            facts.append(f"{label}: {value}")
    bag = (row.get("bag_present") or "").strip().lower()
    if bag in {"yes", "no"}:
        facts.append(f"bag present: {bag}")
    parts = [f"Person {row['person_id']}"]
    if visual_description:
        parts.append(visual_description)
    if facts:
        parts.append("; ".join(facts))
    if len(parts) == 1:
        parts.append("no clear appearance attributes recorded")
    return ". ".join(parts) + "."


def main():
    people = read_rows(ATTRIBUTES_FILE)
    events = read_rows(EVENTS_FILE)
    people_by_id = {int(row["person_id"]): row for row in people}
    DATABASE_FILE.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DATABASE_FILE)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS persons (
            person_id INTEGER PRIMARY KEY,
            upper_clothing TEXT,
            lower_clothing TEXT,
            bag_present TEXT,
            bag_type TEXT,
            bag_color TEXT,
            helmet_present TEXT,
            parse_status TEXT,
            description TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS events (
            event_id INTEGER PRIMARY KEY,
            person_id INTEGER NOT NULL REFERENCES persons(person_id),
            timestamp REAL NOT NULL,
            event_type TEXT NOT NULL,
            location TEXT,
            description TEXT
        );
        CREATE TABLE IF NOT EXISTS search_records (
            record_id INTEGER PRIMARY KEY AUTOINCREMENT,
            record_type TEXT NOT NULL,
            person_id INTEGER NOT NULL REFERENCES persons(person_id),
            event_id INTEGER,
            text TEXT NOT NULL,
            UNIQUE(record_type, person_id, event_id)
        );
        CREATE INDEX IF NOT EXISTS idx_events_person ON events(person_id);
        CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
    """)
    connection.execute("DELETE FROM search_records")
    connection.execute("DELETE FROM events")
    connection.execute("DELETE FROM persons")

    for row in people:
        pid = int(row["person_id"])
        connection.execute("""
            INSERT INTO persons VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (pid, row.get("upper_clothing", ""), row.get("lower_clothing", ""),
              row.get("bag_present", ""), row.get("bag_type", ""), row.get("bag_color", ""),
              row.get("helmet_present", ""), row.get("parse_status", ""), describe_person(row)))
        connection.execute("INSERT INTO search_records(record_type, person_id, text) VALUES (?, ?, ?)",
                           ("person", pid, describe_person(row)))

    for row in events:
        pid = int(row["person_id"])
        if pid not in people_by_id:
            # Keep the event queryable even if that person's VLM profile failed.
            missing = {"person_id": pid, "parse_status": "missing"}
            people_by_id[pid] = missing
            connection.execute("INSERT INTO persons(person_id, parse_status, description) VALUES (?, ?, ?)",
                               (pid, "missing", describe_person(missing)))
            connection.execute("INSERT INTO search_records(record_type, person_id, text) VALUES (?, ?, ?)",
                               ("person", pid, describe_person(missing)))
        event_id = int(row["event_id"])
        timestamp = float(row["timestamp"])
        event_type = row.get("event_type", "").upper()
        location = row.get("location", "")
        event_description = row.get("description", "")
        connection.execute("INSERT INTO events VALUES (?, ?, ?, ?, ?, ?)",
                           (event_id, pid, timestamp, event_type, location, event_description))
        profile_text = describe_person(people_by_id[pid])
        text = f"{event_description}. {profile_text}. At {timestamp:.2f} seconds, {event_type.lower()} at {location}."
        connection.execute("INSERT INTO search_records(record_type, person_id, event_id, text) VALUES (?, ?, ?, ?)",
                           ("event", pid, event_id, text))

    records = connection.execute("SELECT record_id, text FROM search_records ORDER BY record_id").fetchall()
    connection.commit()
    connection.close()

    if not records:
        raise RuntimeError("No person or event records found to index")
    print(f"Loading text embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    vectors = model.encode([row[1] for row in records], convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=True)
    vectors = np.asarray(vectors, dtype="float32")
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(vectors.shape[1]))
    ids = np.asarray([row[0] for row in records], dtype="int64")
    index.add_with_ids(vectors, ids)
    faiss.write_index(index, str(INDEX_FILE))
    print(f"Indexed {len(people)} people and {len(events)} events.")
    print(f"SQLite facts: {DATABASE_FILE}")
    print(f"FAISS vectors: {INDEX_FILE}")


if __name__ == "__main__":
    main()
