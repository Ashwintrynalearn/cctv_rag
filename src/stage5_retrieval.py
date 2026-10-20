import csv
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "output/person_events.csv"

INDEX_FILE = "output/events.faiss"
METADATA_FILE = "output/events_metadata.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# CREATE SEARCHABLE TEXT
# ============================================================

def event_to_text(row):
    """
    Convert one structured event into a natural-language
    representation suitable for semantic embedding.
    """

    person_id = row["person_id"]
    timestamp = row["timestamp"]
    event_type = row["event_type"]
    location = row["location"]

    upper = row["upper_clothing"]
    lower = row["lower_clothing"]

    bag_present = row["bag_present"]
    bag_type = row["bag_type"]
    bag_color = row["bag_color"]

    helmet = row["helmet_present"]

    parts = [
        f"Person {person_id}",
        f"{event_type.lower()} at {location}",
        f"at {timestamp} seconds",
    ]

    if upper and upper != "unknown":
        parts.append(
            f"wearing {upper}"
        )

    if lower and lower != "unknown":
        parts.append(
            f"wearing {lower}"
        )

    if bag_present == "yes":
        if (
            bag_type
            and bag_type != "unknown"
        ):
            parts.append(
                f"carrying a {bag_type}"
            )
        else:
            parts.append(
                "carrying a bag"
            )

        if (
            bag_color
            and bag_color != "unknown"
        ):
            parts.append(
                f"that is {bag_color}"
            )

    if helmet == "yes":
        parts.append(
            "wearing a helmet"
        )
    elif helmet == "no":
        parts.append(
            "not wearing a helmet"
        )

    return ". ".join(parts) + "."


# ============================================================
# LOAD EVENTS
# ============================================================

def load_events():
    if not Path(INPUT_FILE).exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_FILE}"
        )

    rows = []

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    return rows


# ============================================================
# BUILD INDEX
# ============================================================

def build_index(rows, model):

    texts = []

    metadata = []

    print()
    print("Creating searchable event text...")
    print()

    for row in rows:

        text = event_to_text(row)

        texts.append(text)

        metadata.append(
            {
                "event_id": row["event_id"],
                "person_id": row["person_id"],
                "timestamp": row["timestamp"],
                "event_type": row["event_type"],
                "location": row["location"],
                "text": text,
                "upper_clothing": row[
                    "upper_clothing"
                ],
                "lower_clothing": row[
                    "lower_clothing"
                ],
                "bag_present": row[
                    "bag_present"
                ],
                "bag_type": row[
                    "bag_type"
                ],
                "bag_color": row[
                    "bag_color"
                ],
                "helmet_present": row[
                    "helmet_present"
                ],
            }
        )

        print(
            f"{row['event_id']}: {text}"
        )

    print()
    print(
        f"Generating embeddings for "
        f"{len(texts)} events..."
    )

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    # Because vectors are normalized, inner product
    # is equivalent to cosine similarity.
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    return index, metadata


# ============================================================
# SAVE
# ============================================================

def save_index(index, metadata):

    Path("output").mkdir(
        parents=True,
        exist_ok=True
    )

    faiss.write_index(
        index,
        INDEX_FILE
    )

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    print()
    print(
        f"FAISS index saved to: "
        f"{INDEX_FILE}"
    )

    print(
        f"Metadata saved to: "
        f"{METADATA_FILE}"
    )


# ============================================================
# SEARCH
# ============================================================

def search(query, model, index, metadata, top_k=5):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    scores, indices = index.search(
        query_embedding,
        min(top_k, index.ntotal)
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0:
            continue

        item = metadata[idx].copy()

        item["similarity"] = float(
            score
        )

        results.append(item)

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("==============================")
    print("Stage 5: Semantic Retrieval")
    print("==============================")

    rows = load_events()

    print(
        f"Loaded {len(rows)} events."
    )

    print()
    print(
        f"Loading embedding model: "
        f"{MODEL_NAME}"
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    print("Embedding model loaded.")

    index, metadata = build_index(
        rows,
        model
    )

    save_index(
        index,
        metadata
    )

    # --------------------------------------------------------
    # Interactive search
    # --------------------------------------------------------

    print()
    print("==============================")
    print("Interactive Search")
    print("==============================")

    print(
        "Type a question or search phrase."
    )
    print(
        "Type 'exit' to stop."
    )
    print()

    while True:

        query = input(
            "Query: "
        ).strip()

        if not query:
            continue

        if query.lower() == "exit":
            break

        results = search(
            query,
            model,
            index,
            metadata,
            top_k=5
        )

        print()

        for i, result in enumerate(
            results,
            start=1
        ):

            print(
                f"[{i}] "
                f"similarity={result['similarity']:.3f}"
            )

            print(
                f"    Event: "
                f"{result['event_id']}"
            )

            print(
                f"    Person: "
                f"{result['person_id']}"
            )

            print(
                f"    Time: "
                f"{result['timestamp']}s"
            )

            print(
                f"    Type: "
                f"{result['event_type']}"
            )

            print(
                f"    Text: "
                f"{result['text']}"
            )

            print()

        print("-" * 60)
        print()


if __name__ == "__main__":
    main()