from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "jobs"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a"}

WHISPER_MODEL_SIZE = "small"
CHUNK_SECONDS = 3.0
CHUNK_OVERLAP_SECONDS = 0.5
MAX_SPEAKERS = 6
SPEAKER_FALLBACK = "Speaker 1"
