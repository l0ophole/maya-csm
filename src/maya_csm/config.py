"""Environment-driven settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .tags import DEFAULT_TAG_MAP, TagSpec, load_tag_map

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VENDORED_ADAPTER = _REPO_ROOT / "references" / "csm-maya-exp2"


@dataclass(frozen=True)
class RefClip:
    """Phase-2 hook: a Maya reference clip conditioning generation via audio context."""

    audio_path: str
    transcript: str


@dataclass(frozen=True)
class Settings:
    base_model: str = "sesame/csm-1b"
    adapter_path: str = str(_VENDORED_ADAPTER)
    speaker_id: str = "4"  # the LoRA was trained only on speaker ID 4
    max_chunk_chars: int = 150
    gap_ms: int = 120
    sampling: bool = True
    preload: bool = False
    hf_token: str | None = None
    tag_map: dict[str, TagSpec] = field(default_factory=lambda: dict(DEFAULT_TAG_MAP))
    reference_clips: list[RefClip] = field(default_factory=list)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        default_adapter = str(_VENDORED_ADAPTER) if _VENDORED_ADAPTER.is_dir() else "shb777/csm-maya-exp2"
        tag_map_path = env.get("MAYA_TAG_MAP")
        return cls(
            base_model=env.get("MAYA_BASE_MODEL", "sesame/csm-1b"),
            adapter_path=env.get("MAYA_ADAPTER", default_adapter),
            max_chunk_chars=int(env.get("MAYA_MAX_CHUNK_CHARS", "150")),
            gap_ms=int(env.get("MAYA_GAP_MS", "120")),
            sampling=env.get("MAYA_SAMPLING", "1") not in ("0", "false", "no"),
            preload=env.get("MAYA_PRELOAD", "0") in ("1", "true", "yes"),
            hf_token=env.get("HF_TOKEN"),
            tag_map=load_tag_map(tag_map_path) if tag_map_path else dict(DEFAULT_TAG_MAP),
        )
