"""CSM-1B + csm-maya-exp2 LoRA wrapper.

Load pattern follows the model README / TinkerSpace demo. Heavy deps (torch,
transformers, peft) are imported inside methods so the unit suite and server
module import cleanly without the [model] extra installed.
"""

from __future__ import annotations

import math

import numpy as np

from .config import RefClip, Settings

# 375 audio tokens covers roughly 200 chars of speech (author's demo settings).
_TOKENS_PER_CHAR = 375 / 200
_MIN_NEW_TOKENS = 125
_MAX_NEW_TOKENS = 750

_SAMPLING_DEFAULTS = {
    "do_sample": True,
    "temperature": 0.7,
    "depth_decoder_do_sample": True,
    "depth_decoder_temperature": 0.7,
    "depth_decoder_top_k": 20,
    "depth_decoder_top_p": 0.95,
}


class MayaModel:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.processor = None
        self.model = None
        self.device = None

    def load(self) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, CsmForConditionalGeneration

        s = self.settings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        try:
            self.processor = AutoProcessor.from_pretrained(s.base_model, token=s.hf_token)
            base = CsmForConditionalGeneration.from_pretrained(
                s.base_model, device_map=self.device, torch_dtype=dtype, token=s.hf_token
            )
        except OSError as exc:
            raise RuntimeError(
                f"Could not load base model '{s.base_model}'. It is gated on Hugging Face: "
                "accept the terms at https://huggingface.co/sesame/csm-1b and set HF_TOKEN."
            ) from exc
        self.model = PeftModel.from_pretrained(base, s.adapter_path)
        self.model.eval()

    def generate_chunk(
        self, text: str, gen_params: dict, context: tuple[RefClip, ...] = ()
    ) -> np.ndarray:
        import torch

        if self.model is None:
            raise RuntimeError("model not loaded; call load() first")
        conversation = [self._context_turn(clip) for clip in context]
        conversation.append(
            {"role": self.settings.speaker_id, "content": [{"type": "text", "text": text}]}
        )
        inputs = self.processor.apply_chat_template(
            conversation, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device)
        max_new = min(_MAX_NEW_TOKENS, max(_MIN_NEW_TOKENS, math.ceil(len(text) * _TOKENS_PER_CHAR)))
        params = {**_SAMPLING_DEFAULTS, **gen_params} if self.settings.sampling else dict(gen_params)
        with torch.no_grad():
            audio = self.model.generate(
                **inputs, max_new_tokens=max_new, output_audio=True, **params
            )
        return audio[0].to(torch.float32).cpu().numpy()

    def _context_turn(self, clip: RefClip) -> dict:
        import soundfile as sf

        audio, _sr = sf.read(clip.audio_path, dtype="float32")
        return {
            "role": self.settings.speaker_id,
            "content": [
                {"type": "text", "text": clip.transcript},
                {"type": "audio", "audio": audio},
            ],
        }
