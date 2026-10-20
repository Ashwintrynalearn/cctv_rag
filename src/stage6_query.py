"""Grounded natural-language queries over SQLite facts and FAISS search."""
import argparse
import json
import os
import re
import sqlite3

import faiss
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

DATABASE_FILE = "output/cctv_knowledge.sqlite"
INDEX_FILE = "output/knowledge.faiss"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"


def detect_event_type(query):
    q = query.lower()
    if re.search(r"\b(enter(?:s|ed|ing)?|entr(?:y|ies)|came in|go(?:es)? in)\b", q):
        return "ENTER"
    if re.search(r"\b(exit(?:s|ed|ing)?|left|went out)\b", q):
        return "EXIT"
    return None


def get_evidence(query, model, index, connection, top_k=8):
    event_type = detect_event_type(query)
    is_count = bool(re.search(r"\b(how many|count|number of)\b", query.lower()))
    has_attribute_filter = bool(re.search(
        r"\b(wearing|shirt|jacket|coat|pants|trousers|bag|backpack|helmet|blue|red|black|white|green|yellow|purple|brown|gray|grey)\b",
        query.lower(),
    ))
    if event_type and is_count and not has_attribute_filter:
        count = connection.execute(
            "SELECT COUNT(DISTINCT person_id) FROM events WHERE event_type = ?", (event_type,)
        ).fetchone()[0]
        rows = connection.execute(
            "SELECT e.person_id, e.timestamp, e.event_type, e.location, e.description, p.description "
            "FROM events e JOIN persons p USING(person_id) WHERE e.event_type = ? ORDER BY e.timestamp",
            (event_type,),
        ).fetchall()
        evidence = [{"person_id": r[0], "timestamp_seconds": r[1], "event_type": r[2],
                     "location": r[3], "event": r[4], "appearance": r[5]} for r in rows[:10]]
        return {"kind": "exact_count", "count": count, "event_type": event_type,
                "matching_record_count": len(rows), "examples": evidence}
    if event_type and is_count and has_attribute_filter:
        return {"kind": "unsupported_filtered_count", "event_type": event_type}
    if is_count and not event_type:
        return {"kind": "unsupported_count"}

    vector = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, ids = index.search(vector.astype("float32"), min(top_k, index.ntotal))
    results = []
    for score, record_id in zip(scores[0], ids[0]):
        if record_id < 0:
            continue
        row = connection.execute(
            "SELECT record_type, person_id, event_id, text FROM search_records WHERE record_id = ?",
            (int(record_id),),
        ).fetchone()
        if row:
            results.append({"similarity": round(float(score), 4), "record_type": row[0],
                            "person_id": row[1], "event_id": row[2], "text": row[3]})
    return {"kind": "retrieval", "records": results}


def answer_with_gemini(client, model_name, query, evidence):
    if evidence["kind"] == "unsupported_filtered_count":
        return "This version can count all entering or exiting people, but it cannot yet filter those counts by clothing or carried objects."
    if evidence["kind"] == "unsupported_count":
        return "This version can answer exact counts for all entrance and exit events. Other count filters are not implemented yet."
    if evidence["kind"] == "exact_count" and not evidence["matching_record_count"]:
        return f"No {evidence['event_type'].lower()} events are recorded in the database."
    prompt = f"""You answer questions about one CCTV video using only the evidence below.
Do not infer facts that are not present. If evidence is insufficient, say so.
For event claims, include the person's ID and timestamp when available.
If an exact_count is supplied, preserve that number exactly. Keep the answer concise.

Question: {query}
Evidence (JSON): {json.dumps(evidence, ensure_ascii=False, indent=2)}
"""
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.1),
    )
    return (response.text or "I couldn't form an answer from the available records.").strip()


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Ask questions about tracked people and entrance events")
    parser.add_argument("query", nargs="?", help="Question; omit to use interactive mode")
    parser.add_argument("--database", default=DATABASE_FILE)
    parser.add_argument("--index", default=INDEX_FILE)
    args = parser.parse_args()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY before running. See README.md.")
    if not os.path.exists(args.database) or not os.path.exists(args.index):
        raise SystemExit("Knowledge base is missing. Run: python src/stage5_build_knowledge_base.py")

    model = SentenceTransformer(EMBEDDING_MODEL)
    index = faiss.read_index(args.index)
    connection = sqlite3.connect(args.database)
    client = genai.Client(api_key=api_key)
    gemini_model = os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)

    def run(query):
        evidence = get_evidence(query, model, index, connection)
        print(answer_with_gemini(client, gemini_model, query, evidence))

    try:
        if args.query:
            run(args.query)
        else:
            while True:
                try:
                    query = input("Question (or 'exit'): ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if query.lower() in {"exit", "quit"}:
                    break
                if query:
                    run(query)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
