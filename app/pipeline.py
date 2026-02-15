from __future__ import annotations

import json
import subprocess
import wave
from pathlib import Path

import numpy as np
import torch
from faster_whisper import WhisperModel
from silero_vad import get_speech_timestamps, load_silero_vad
from speechbrain.inference.speaker import EncoderClassifier

from app.config import CHUNK_OVERLAP_SECONDS, CHUNK_SECONDS
from app.diarization import (
    assign_speakers_to_segments,
    cluster_embeddings,
    split_regions_into_chunks,
)

_whisper_model: WhisperModel | None = None
_vad_model = None
_speaker_model: EncoderClassifier | None = None


def load_wav_mono_16k(wav_file: Path) -> np.ndarray:
    with wave.open(str(wav_file), "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frame_count = wf.getnframes()
        pcm = wf.readframes(frame_count)

    if sample_rate != 16000:
        raise RuntimeError(f"Expected 16kHz WAV, got {sample_rate}Hz")
    if channels != 1:
        raise RuntimeError(f"Expected mono WAV, got {channels} channels")
    if sample_width != 2:
        raise RuntimeError("Expected 16-bit PCM WAV input")

    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    return audio


def _get_whisper_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
    return _whisper_model


def _get_vad_model():
    global _vad_model
    if _vad_model is None:
        _vad_model = load_silero_vad()
    return _vad_model


def _get_speaker_model() -> EncoderClassifier:
    global _speaker_model
    if _speaker_model is None:
        _speaker_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"},
        )
    return _speaker_model


def convert_to_wav(input_file: Path, output_file: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_file),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_file),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")


def transcribe(wav_file: Path) -> list[dict]:
    model = _get_whisper_model()
    segments, _ = model.transcribe(str(wav_file), word_timestamps=True, vad_filter=False)
    return [
        {
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
        }
        for seg in segments
    ]


def diarize(wav_file: Path) -> tuple[list, list[int]]:
    # Read wav directly to avoid torchaudio backend requirements on local hosts.
    audio = load_wav_mono_16k(wav_file)
    vad_model = _get_vad_model()
    speech_timestamps = get_speech_timestamps(audio, vad_model, return_seconds=True)
    speech_regions = [(float(s["start"]), float(s["end"])) for s in speech_timestamps]
    chunks = split_regions_into_chunks(speech_regions, CHUNK_SECONDS, CHUNK_OVERLAP_SECONDS)

    if not chunks:
        return [], []

    speaker_model = _get_speaker_model()
    embeddings: list = []

    for chunk in chunks:
        start_idx = int(chunk.start * 16000)
        end_idx = int(chunk.end * 16000)
        snippet = audio[start_idx:end_idx]
        if snippet.shape[0] < 1600:
            continue
        tensor = torch.tensor(snippet).float().unsqueeze(0)
        emb = speaker_model.encode_batch(tensor).detach().cpu().numpy().reshape(-1)
        embeddings.append(emb)

    if not embeddings:
        return [], []

    labels = cluster_embeddings(embeddings)
    usable_chunks = chunks[: len(labels)]
    return usable_chunks, labels


def format_srt_time(seconds: float) -> str:
    ms = int(seconds * 1000)
    hours, rem = divmod(ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(segments: list[dict], path: Path) -> None:
    lines = []
    for idx, seg in enumerate(segments, start=1):
        lines.append(str(idx))
        lines.append(f"{format_srt_time(seg['start'])} --> {format_srt_time(seg['end'])}")
        lines.append(f"{seg['speaker']}: {seg['text']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_pipeline(input_audio: Path, job_dir: Path) -> list[dict]:
    wav_file = job_dir / "audio_16k_mono.wav"
    convert_to_wav(input_audio, wav_file)

    transcript_segments = transcribe(wav_file)
    try:
        chunks, labels = diarize(wav_file)
    except Exception:
        chunks, labels = [], []
    diarized_segments = assign_speakers_to_segments(transcript_segments, chunks, labels)

    json_path = job_dir / "result.json"
    srt_path = job_dir / "result.srt"
    json_path.write_text(json.dumps({"segments": diarized_segments}, indent=2), encoding="utf-8")
    write_srt(diarized_segments, srt_path)

    return diarized_segments
