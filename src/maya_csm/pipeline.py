"""Synthesis pipeline: tags -> sanitize -> chunk -> generate -> join -> WAV.

Chunks are independent (same static context, no carry-over), so when the engine
exposes more than one replica (`num_replicas`) they are generated concurrently,
one per GPU. `ThreadPoolExecutor.map` keeps results in chunk order.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from .audio import join_with_gaps, to_wav_bytes, trim_trailing_noise
from .chunking import split_text
from .config import Settings
from .sanitize import sanitize
from .tags import parse_tags

SAMPLE_RATE = 24000


def synthesize(text: str, settings: Settings, model) -> bytes:
    jobs = [
        (chunk, segment.gen_params)
        for segment in parse_tags(text, settings.tag_map)
        for chunk in split_text(sanitize(segment.text), settings.max_chunk_chars)
    ]
    if not jobs:
        raise ValueError("no speakable text after tag and sanitization processing")

    context = tuple(settings.reference_clips)

    def render(job):
        chunk, gen_params = job
        audio = model.generate_chunk(chunk, gen_params, context=context)
        return trim_trailing_noise(audio, sr=SAMPLE_RATE)

    workers = min(len(jobs), getattr(model, "num_replicas", 1))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            chunks_audio = list(pool.map(render, jobs))
    else:
        chunks_audio = [render(job) for job in jobs]

    joined = join_with_gaps(chunks_audio, sr=SAMPLE_RATE, gap_ms=settings.gap_ms)
    return to_wav_bytes(joined, sr=SAMPLE_RATE)
