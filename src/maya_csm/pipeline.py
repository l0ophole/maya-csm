"""Synthesis pipeline: tags -> sanitize -> chunk -> generate -> join -> WAV."""

from __future__ import annotations

from .audio import join_with_gaps, to_wav_bytes, trim_trailing_noise
from .chunking import split_text
from .config import Settings
from .sanitize import sanitize
from .tags import parse_tags

SAMPLE_RATE = 24000


def synthesize(text: str, settings: Settings, model) -> bytes:
    chunks_audio = []
    for segment in parse_tags(text, settings.tag_map):
        clean = sanitize(segment.text)
        for chunk in split_text(clean, settings.max_chunk_chars):
            audio = model.generate_chunk(
                chunk, segment.gen_params, context=tuple(settings.reference_clips)
            )
            chunks_audio.append(trim_trailing_noise(audio, sr=SAMPLE_RATE))
    if not chunks_audio:
        raise ValueError("no speakable text after tag and sanitization processing")
    return to_wav_bytes(join_with_gaps(chunks_audio, sr=SAMPLE_RATE, gap_ms=settings.gap_ms), sr=SAMPLE_RATE)
