# Offline Transcriber MVP

Fully offline, free MVP web app for audio transcription + speaker labeling.

## What it does
- Upload `.mp3`, `.wav`, or `.m4a` and receive a `job_id`.
- Runs offline pipeline:
  1. Converts input audio to mono 16k WAV with `ffmpeg`.
  2. Transcribes with `faster-whisper` (segment timestamps enabled).
  3. Performs diarization-like speaker labeling with:
     - `silero-vad` speech regions
     - overlapping 2-5 second chunks
     - `speechbrain` ECAPA speaker embeddings
     - agglomerative clustering
     - alignment of clustered speakers back to transcript segments by time overlap
- Outputs:
  - Web transcript view: `[time] Speaker N: text`
  - `result.json` with `start`, `end`, `speaker`, `text`
  - `result.srt` with speaker label in subtitle lines
- Stores all artifacts under `./data/jobs/<job_id>/`.

## 3-step quickstart
1. Build and run:
   ```bash
   docker compose up --build
   ```
2. Open: `http://localhost:8000`
3. Upload audio and wait for status to become `completed`, then download JSON/SRT.

## Notes
- CPU-only and offline once model files are cached locally.
- First run may take longer while models are downloaded.
- Basic error handling is included and surfaced in job status.

## API
- `POST /api/jobs` (multipart file upload) → `{ "job_id": "..." }`
- `GET /api/jobs/{job_id}` → status + message + segments when complete
- `GET /api/jobs/{job_id}/result.json`
- `GET /api/jobs/{job_id}/result.srt`

## Tests
Run unit tests for alignment logic:
```bash
pytest
```
