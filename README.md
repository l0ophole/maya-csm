# maya-csm

A SillyTavern-compatible text-to-speech server for the **Maya** voice (Sesame's
voice-chat agent), built on [`sesame/csm-1b`](https://huggingface.co/sesame/csm-1b)
with the community LoRA fine-tune
[`shb777/csm-maya-exp2`](https://huggingface.co/shb777/csm-maya-exp2).

- OpenAI-compatible API: `POST /v1/audio/speech` returning WAV (24 kHz mono)
- Expressive tags in input text: `[laughing]`, `[giggling]`, `[whispering]`, `[sighing]`, …
- Runs on a free Colab/Kaggle GPU (via a Cloudflare tunnel) or locally on CPU (slow)

> **License note:** the csm-maya-exp2 adapter is CC-BY-NC-SA-4.0 — personal and
> research use only. The base model `sesame/csm-1b` is gated: accept its terms on
> Hugging Face and use an `HF_TOKEN`.

## Quick start (Colab / Kaggle — recommended)

Open `notebooks/colab_server.ipynb` or `notebooks/kaggle_server.ipynb`, follow the
setup notes in the first cell (GPU runtime, `HF_TOKEN` secret, repo URL), run all
cells, and copy the printed `https://….trycloudflare.com` URL into SillyTavern
(see [docs/SILLYTAVERN.md](docs/SILLYTAVERN.md)).

## Local install

```bash
uv venv --python 3.12 && uv pip install -e '.[model]'
export HF_TOKEN=hf_...   # token with access to sesame/csm-1b
python -m maya_csm --host 0.0.0.0 --port 8000
```

Without a CUDA GPU (CSM-1B needs ~4 GB VRAM) inference falls back to CPU at
roughly minutes per sentence — fine for testing, painful for chat. Raise
SillyTavern's request timeout accordingly, or use the notebooks.

The vendored adapter at `references/csm-maya-exp2/` is used automatically when
present; set `MAYA_ADAPTER=shb777/csm-maya-exp2` to pull from the Hub instead.

## API

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model": "maya-csm", "input": "[laughing] That is hilarious.", "voice": "maya"}' \
  -o out.wav
```

Also serves `GET /health`, `GET /v1/models`, and `GET /v1/audio/voices`.
`response_format` is accepted but output is always WAV.

## Expressive tags

The model has **no native tag tokens** — tags are translated into prose cues the
model learned (plus per-tag sampling tweaks) before synthesis. A tag applies
from where it appears until the next tag.

| Tag | Spoken cue |
|---|---|
| `[laughing]` / `[laughs]` | "Haha!" |
| `[giggling]` / `[giggles]` | "Hee hee." |
| `[chuckling]` / `[chuckles]` | "Heh." |
| `[whispering]` / `[whispers]` | "Shh..." |
| `[sighing]` / `[sighs]` | "Hhh..." |
| `[gasping]` / `[gasps]` | "Oh!" |
| `[crying]` | "Oh no... sniff..." |
| `[excited]` | "Oh wow!" |
| `[nervous]` | "Um... uh..." |
| `[pause]` | "..." |

Unknown tags are removed silently. Cues are heuristic — sometimes a cue is
spoken literally rather than performed; tune the mapping by ear with a JSON
file (same shape as `DEFAULT_TAG_MAP` in `src/maya_csm/tags.py`) and
`MAYA_TAG_MAP=/path/to/tags.json`.

## Configuration (env vars)

| Variable | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | — | Hugging Face token for the gated base model |
| `MAYA_BASE_MODEL` | `sesame/csm-1b` | Base model id |
| `MAYA_ADAPTER` | vendored copy, else `shb777/csm-maya-exp2` | LoRA path or Hub id |
| `MAYA_MAX_CHUNK_CHARS` | `150` | Per-generation text chunk size |
| `MAYA_GAP_MS` | `120` | Silence inserted between chunks |
| `MAYA_SAMPLING` | `1` | `0` = greedy decoding (the fine-tune author's baseline) |
| `MAYA_PRELOAD` | `0` | `1` = load model at startup instead of first request |
| `MAYA_TAG_MAP` | — | JSON file overriding/extending the tag table |

## Known limitations

- Voice can drift slightly between chunks/generations (model limitation); keep
  utterances short. Reference-clip conditioning (a `reference_clips` hook in
  `config.py`) is designed in but not yet populated — a future phase will
  condition every request on curated Maya clips for consistency.
- The model garbles `( ) " ; ! [ ] /` — the server strips/replaces them.
- Occasional trailing noise is trimmed heuristically; tune the threshold in
  `audio.py` if clips end abruptly.

## Development

```bash
uv pip install -e '.[dev]'
pytest            # unit suite, no model/GPU needed
pytest -m model   # real-generation smoke test (needs [model] extra + HF_TOKEN)
```
