from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.config import MAX_SPEAKERS, SPEAKER_FALLBACK

try:
    import numpy as np
except Exception:  # pragma: no cover - local env fallback
    np = None


def _to_matrix(embeddings: list[Any]):
    if np is None:
        return embeddings
    return np.vstack(embeddings)


@dataclass
class SpeechChunk:
    start: float
    end: float


def split_regions_into_chunks(
    speech_regions: list[tuple[float, float]],
    chunk_seconds: float,
    overlap_seconds: float,
) -> list[SpeechChunk]:
    chunks: list[SpeechChunk] = []
    step = max(chunk_seconds - overlap_seconds, 0.1)
    for start, end in speech_regions:
        if end <= start:
            continue
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + chunk_seconds, end)
            chunks.append(SpeechChunk(start=cursor, end=chunk_end))
            if chunk_end >= end:
                break
            cursor += step
    return chunks


def cluster_embeddings(embeddings: list[Any]):
    if not embeddings:
        return []

    sample_count = len(embeddings)
    n_clusters = min(MAX_SPEAKERS, max(1, sample_count // 3 + 1))
    if sample_count == 1:
        return [0]

    try:
        from sklearn.cluster import AgglomerativeClustering

        model = AgglomerativeClustering(n_clusters=n_clusters)
        matrix = _to_matrix(embeddings)
        return model.fit_predict(matrix).tolist()
    except Exception:
        return [i % n_clusters for i in range(sample_count)]


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speakers_to_segments(
    transcript_segments: list[dict],
    chunks: list[SpeechChunk],
    labels: list[int] | Any,
) -> list[dict]:
    label_list = list(labels)
    if len(chunks) != len(label_list):
        raise ValueError("chunks and labels must have equal lengths")

    output: list[dict] = []
    for segment in transcript_segments:
        seg_start = float(segment["start"])
        seg_end = float(segment["end"])
        votes: dict[int, float] = {}

        for chunk, label in zip(chunks, label_list):
            overlap = _overlap(seg_start, seg_end, chunk.start, chunk.end)
            if overlap > 0:
                label_int = int(label)
                votes[label_int] = votes.get(label_int, 0.0) + overlap

        if votes:
            top_speaker = max(votes.items(), key=lambda x: x[1])[0]
            speaker = f"Speaker {top_speaker + 1}"
        else:
            speaker = SPEAKER_FALLBACK

        output.append(
            {
                "start": seg_start,
                "end": seg_end,
                "speaker": speaker,
                "text": segment.get("text", "").strip(),
            }
        )
    return output
