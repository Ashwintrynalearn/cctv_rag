# CCTV Incident Investigation Assistant

A single-camera prototype that detects and tracks people, describes each person's appearance from one clear crop, records entrance/exit events, and answers questions from those records.

The current knowledge base covers visible person attributes and ENTER/EXIT events. Action recognition is out of scope for this version.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Put your Gemini API key in `.env` as `GEMINI_API_KEY=...`. The query stage uses Gemini's `gemini-3.5-flash-lite` by default; you can change it with `GEMINI_MODEL`. Gemini's free tier has request limits, so it is not unlimited. The key stays local in `.env` and should not be committed.

The appearance-description stage uses MLX and is intended for Apple Silicon. Install `ffmpeg` to write annotated MP4 outputs.

## Run the full workflow

### 1. Track people through the video

```bash
python src/stage1_track.py \
  --video "data/The CCTV People Demo 1 - CCTV People (1080p).mp4" \
  --output output/tracked_v2.mp4 \
  --csv output/detections_v2.csv
```

This processes the full video at the configured sampling rate (default 2 frames per second). Use the same detection CSV in later steps so track IDs remain consistent.

### 2. Detect entrance and exit events

Choose the entrance line, then process and review the crossings:

```bash
python src/select_line.py
python src/stage2_events.py
python src/validate_events.py
```

Review `output/event_validation.mp4`; line-side orientation and tracker errors can affect event accuracy.

### 3. Describe every tracked person

```bash
python src/stage3_attributes.py
```

The script selects one largest usable crop for each tracked person ID, then asks the VLM for a concise appearance description plus visible clothing, bag, and helmet attributes. It saves progress after every person and skips profiles already marked successful if rerun. Use `--force` to reanalyze everyone. The resulting files are `output/person_attributes.csv`, `output/person_attribute_observations.csv`, and crops under `output/person_crops/`.

Each track ID gets one description. If tracking assigns a new ID after an occlusion, it is treated as a separate person record; these IDs are camera-track identities, not verified real-world identities.

### 4. Build SQLite facts and the FAISS search index

```bash
python src/stage5_build_knowledge_base.py
```

This reads every person profile and every entrance/exit event, stores structured records in `output/cctv_knowledge.sqlite`, and embeds person/event descriptions into `output/knowledge.faiss`. SQLite is the source of truth for exact event facts; FAISS finds descriptions that are semantically related to a question.

### 5. Ask questions with Gemini

```bash
python src/stage6_query.py "How many people exited?"
python src/stage6_query.py "Who was carrying a blue bag?"
```

Omit the question to enter interactive mode:

```bash
python src/stage6_query.py
```

For an unqualified ENTER/EXIT count, SQLite computes the distinct-person count and Gemini phrases the answer from that exact result. Other non-count questions retrieve matching person/event records from FAISS and ask Gemini to answer using only those records. The answer includes person IDs and timestamps when the evidence contains them. Attribute-filtered counts such as “how many people wearing blue exited?” are not supported yet; this prototype will say so rather than guess.

The Gemini query contains text descriptions and event metadata, not video frames or crops. Never put the API key in source code or commit the `.env` file.

## Existing similarity search

`python src/stage5_retrieval.py` remains available as the original interactive FAISS demo. The SQLite + Gemini flow is the recommended query path.

## Current limits

- VLM descriptions are based on one crop and may miss attributes not visible in that frame.
- ByteTrack IDs can split or switch when people are occluded; one track ID is not proof of a unique real-world person.
- Entrance/exit event logic is geometric and needs review against the validation video.
- Exact count answers are supported for all ENTER or EXIT records. Counts filtered by appearance, and new event types, can be added after this baseline is checked.
