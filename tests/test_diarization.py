from app.diarization import SpeechChunk, assign_speakers_to_segments


def test_assign_speaker_by_largest_overlap():
    segments = [{"start": 0.0, "end": 4.0, "text": "hello"}]
    chunks = [SpeechChunk(0.0, 1.0), SpeechChunk(1.0, 4.0)]
    labels = [1, 0]

    output = assign_speakers_to_segments(segments, chunks, labels)

    assert output[0]["speaker"] == "Speaker 1"


def test_assign_speaker_fallback_when_no_overlap():
    segments = [{"start": 5.0, "end": 6.0, "text": "silence"}]
    chunks = [SpeechChunk(0.0, 1.0)]
    labels = [0]

    output = assign_speakers_to_segments(segments, chunks, labels)

    assert output[0]["speaker"] == "Speaker 1"
