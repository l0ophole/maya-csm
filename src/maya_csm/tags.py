"""Expressive-tag layer.

csm-maya-exp2 has no [laughs]-style tokens — the tokenizer is stock Llama-3 and
the model garbles bracket characters. Expressiveness must come from prose cues
the model saw in training (interjections like "Haha!", "Shh...") plus sampling
tweaks. This module translates SillyTavern-style [tag] markers into cue-rewritten
text segments with per-segment generation overrides, and must run BEFORE
sanitization (which strips brackets).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

_TAG_RE = re.compile(r"\[(\w[\w\s-]*)\]")


@dataclass(frozen=True)
class TagSpec:
    cue: str
    position: str = "prepend"  # prepend | replace | append
    gen_params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Segment:
    text: str
    gen_params: dict = field(default_factory=dict)


# Cues are heuristic — tune by ear via a MAYA_TAG_MAP JSON override.
DEFAULT_TAG_MAP: dict[str, TagSpec] = {
    "laughing": TagSpec("Haha!", gen_params={"temperature": 0.9}),
    "laughs": TagSpec("Haha!", gen_params={"temperature": 0.9}),
    "giggling": TagSpec("Hee hee.", gen_params={"temperature": 0.9}),
    "giggles": TagSpec("Hee hee.", gen_params={"temperature": 0.9}),
    "chuckling": TagSpec("Heh."),
    "chuckles": TagSpec("Heh."),
    "whispering": TagSpec("Shh...", gen_params={"temperature": 0.6}),
    "whispers": TagSpec("Shh...", gen_params={"temperature": 0.6}),
    "sighing": TagSpec("Hhh..."),
    "sighs": TagSpec("Hhh..."),
    "gasping": TagSpec("Oh!"),
    "gasps": TagSpec("Oh!"),
    "crying": TagSpec("Oh no... sniff...", gen_params={"temperature": 0.8}),
    "excited": TagSpec("Oh wow!", gen_params={"temperature": 0.9}),
    "nervous": TagSpec("Um... uh..."),
    "pause": TagSpec("...", position="replace"),
}


def load_tag_map(path: str | Path) -> dict[str, TagSpec]:
    """DEFAULT_TAG_MAP overlaid with entries from a JSON file."""
    data = json.loads(Path(path).read_text())
    mapping = dict(DEFAULT_TAG_MAP)
    for tag, spec in data.items():
        mapping[tag.lower()] = TagSpec(
            cue=spec["cue"],
            position=spec.get("position", "prepend"),
            gen_params=spec.get("gen_params", {}),
        )
    return mapping


def _make_segment(tag: str | None, text: str, mapping: dict[str, TagSpec]) -> Segment | None:
    spec = mapping.get(tag.lower()) if tag else None
    if spec is None:  # untagged leading text, or unknown tag: strip silently
        return Segment(text=text) if text else None
    if spec.position == "append":
        combined = f"{text} {spec.cue}".strip()
    else:  # prepend and replace both emit the cue where the tag stood
        combined = f"{spec.cue} {text}".strip()
    return Segment(text=combined, gen_params=spec.gen_params) if combined else None


def parse_tags(text: str, mapping: dict[str, TagSpec] | None = None) -> list[Segment]:
    """Split text into tag-scoped segments; each tag applies until the next tag."""
    mapping = DEFAULT_TAG_MAP if mapping is None else mapping
    parts = _TAG_RE.split(text)  # [leading, tag1, text1, tag2, text2, ...]
    pairs = [(None, parts[0])] + list(zip(parts[1::2], parts[2::2]))
    segments = []
    for tag, chunk in pairs:
        seg = _make_segment(tag, chunk.strip(), mapping)
        if seg is not None:
            segments.append(seg)
    return segments
