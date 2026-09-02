# Performance improvement report — reducing SillyTavern → audio latency

**Scope:** cut the delay between SillyTavern's TTS extension (OpenAI‑Compatible provider)
firing a request and audio coming back, for the `maya-csm` server.
**Constraints for this report** (from the person who asked):

| Constraint | Value | Consequence |
|---|---|---|
| Budget | **up to ~$15 / month** | Free options ranked first; one clearly‑worth‑it paid tier is in scope; premium APIs are context only. |
| Voice | **Maya (`sesame/csm-1b` + `csm-maya-exp2` LoRA) is a hard requirement** | No swapping to Kokoro / Orpheus / ElevenLabs / Cartesia etc. Alternative *models* appear only as "not applicable, here's why". |
| Current numbers | **none measured — "feels slow"** | §1 is a 15‑minute benchmark you should run first; all impact figures below are estimates until you do. |
| Usage | **most days, 1–3 h/session** | ~30–90 GPU‑h/month. Within Kaggle's ~30 h/week, but with little headroom, and Kaggle allows only **one** GPU session at a time. |

> Each lever links to the code it would touch (`file:line`).

## Status

**2026-09-02 — shipped:**

- **§A1** `_COMPILE_MAX_LENGTH` 1024 → **384** (`model.py`).
- **§A2** warm-up generation in `MayaModel.load()` (GPU only, never blocks startup) —
  cuDNN autotune / allocator growth move to load; skipped in compile mode (the graph
  capture is thread-affine and must land on the serving thread — the notebook's warm-up
  cell does it).
- **§A3** per-chunk token budget → `ceil(chars × 1.4)` clamped `[80, 750]`, extracted to
  the unit-tested `_estimate_max_new_tokens()`.
- **§E1** SillyTavern "Narrate by paragraphs" tip, a warm-up/timing cell, an optional
  benchmark cell, and an **ngrok** tunnel option above the Cloudflare block are now in
  both `notebooks/kaggle_server.ipynb` and `notebooks/colab_server.ipynb`.

**2026-09-02 — corrected (this had been wrong):**

- **§B1 the P100 recommendation is withdrawn.** Kaggle's PyTorch is **2.10.0+cu128**,
  minimum compute capability **sm_70**; the P100 is **sm_60** and the model fails to load
  on it (`CUDA error: no kernel image is available`). **T4 x2 is the only working Kaggle
  accelerator** and stays the default.
- **§B2 `MAYA_COMPILE=1` is not a blind default.** It uses `mode="reduce-overhead"`, and
  torch 2.10 has a [reported regression](https://github.com/pytorch/pytorch/issues/174575)
  there. The notebooks ship with `MAYA_COMPILE` unset (matching `config.py`); it's a
  one-line opt-in to A/B with the benchmark cell.

Still open: §A4 (Opus output), §A5 (break the compile⇄multi-GPU split), §A6 (streaming +
custom ST provider), §A10 (reference-clip prefix cache), §B4 (named Cloudflare tunnel),
§C1 (Modal).

---

## 0. TL;DR — what to do, in order

| # | Action | Type | Est. effect on time‑to‑first‑audio (TTFA) | $/mo | Effort |
|---|---|---|---|---|---|
| 1 | **Benchmark** (§1) so every later change is measured, not guessed — *now the last cell of both notebooks* | — | — | 0 | 15 min |
| 2 | **SillyTavern: "Narrate by paragraphs" + short first sentence** (§E1) — *documented in both notebooks* | config | Perceived TTFA **~15–25 s → ~3–6 s**; the rest generates during playback | 0 | 2 min |
| 3 | ~~Switch Kaggle accelerator T4×2 → P100~~ (§B1) — ❌ *withdrawn: P100 is sm_60, Kaggle's torch 2.10 needs sm_70+*. **Stay on T4 x2.** | infra | — | 0 | — |
| 4 | **Startup warm‑up generation** (§A2) — ✅ *shipped in `model.py`* | code | Removes the first‑request cliff (first reply of a session no longer 2–20× slower) | 0 | done |
| 5 | **Benchmark `MAYA_COMPILE=1` vs the multi‑GPU default** (§B2, §A1) — ✅ *`_COMPILE_MAX_LENGTH=384` shipped; compile stays opt-in* | config | Maybe **~1.5–2×** on the depth‑decoder loop — but torch 2.10 has a `reduce-overhead` regression, so **measure** | 0 | 1 h on Kaggle |
| 6 | **Move hosting to Modal free tier** (§C1) — stable URL, L4/T4, per‑second billing | infra | Kills tunnel‑URL churn and the 12 h/quota juggling; L4 ≈ 1.2–1.5× a T4 | **$0** (within $30/mo free credit) | ~half a day |
| 7 | If still not enough: **rent a cheap Ampere GPU part‑time** (RTX 3090 / 4090, Vast or RunPod) (§C3) | infra | **~3–5×** vs a T4; RTF ≈ 0.28 documented on a 4090 | ~$3–15 | ~half a day |
| 8 | Long game: **true audio streaming + a custom ST provider** (§A6) | project | TTFA **~1–2 s** and roughly constant regardless of reply length | 0 | multi‑day |

Steps 1–5 are free. With the P100 off the table, the big free lever is **§E1**
(narrate by paragraphs — the first short sentence plays while the rest generate), which
cuts the *perceived* wait regardless of raw speed; **§B2** (compile) may add a per‑stream
1.5–2× on top *if* it doesn't hit the torch 2.10 regression — benchmark it. Steps 6–8 are
the paid / bigger‑effort tier, and a rented Ampere GPU (§C3, ~$3–15/mo) is now the main
route to a real hardware speedup.

---

## 1. Measure first (you have no numbers)

CSM is autoregressive: it emits one 80 ms audio frame at a time, and each frame is
**one backbone transformer step + a 31‑step loop through the tiny "depth decoder"**
(4 layers, 1024‑wide). Public reports put a mid‑length sentence at **5–10 s** on a
typical GPU and note that streaming can drop *perceived* latency to **1–2 s**. A T4 is
near the slow end. That's almost certainly what "feels slow" is.

Before changing anything, capture a baseline.

### 1a. Add timing to the server (temporary)

In `src/maya_csm/pipeline.py` `synthesize()` (`pipeline.py:32`), wrap `render`:

```python
import time
def render(job):
    chunk, gen_params = job
    t0 = time.perf_counter()
    audio = model.generate_chunk(chunk, gen_params, context=context)
    audio = trim_trailing_noise(audio, sr=SAMPLE_RATE)
    dur = len(audio) / SAMPLE_RATE
    dt = time.perf_counter() - t0
    print(f"[timing] {len(chunk):3d} chars -> {dur:5.1f}s audio in {dt:5.1f}s  RTF={dt/max(dur,1e-3):.2f}")
    return audio
```

`RTF` (real‑time factor) = generate time ÷ audio seconds. **RTF < 1** means generation
outruns playback (streaming will feel instant); **RTF > 1** means even streaming will
stutter on long replies.

### 1b. Client‑side wall time

```bash
for n in 1 2 4; do
  text=$(python - <<PY
print(" ".join(["This is sentence number %d and it is a fairly normal length." % i for i in range($n)]))
PY
)
  curl -s -o /dev/null -w "sentences=$n  total=%{time_total}s\n" \
    -X POST "$URL/v1/audio/speech" \
    -H 'Content-Type: application/json' \
    -d "{\"input\": \"$text\", \"voice\": \"maya\"}"
done
```

Run it twice (first call may include lazy model load / compile).

### 1c. Confirm the GPU picture

```bash
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv -l 1
```

during a multi‑sentence request. On T4×2 both rows should spike; if only one does,
the reply wasn't long enough to split and the second GPU is dead weight.

### 1d. A/B knobs already in the codebase

| Command | Isolates |
|---|---|
| `MAYA_DEVICES=cuda:0` vs unset | multi‑GPU benefit |
| `MAYA_COMPILE=1` vs `0` | `torch.compile` benefit |
| `MAYA_SAMPLING=0` vs `1` | greedy vs sampled cost |
| `MAYA_MAX_CHUNK_CHARS` 150 vs 100 | chunk size vs parallelism/drift |

Record `total` for 1 / 2 / 4 sentences under each. That table is the real version of
this report.

---

## 2. Where the time goes (the latency budget)

For a typical 2–3 sentence chat reply on the current setup (Kaggle T4×2, sampled, no
compile, model already loaded):

| Stage | Rough share | Notes |
|---|---|---|
| **Autoregressive generation** | **85–95 %** | The whole ballgame. Backbone + 31‑step depth loop per 80 ms frame. Memory‑bandwidth‑ and kernel‑launch‑bound on a T4. |
| Chunk join + WAV encode | <1 % | `audio.py` — numpy + `libsndfile`, milliseconds. |
| Cloudflare quick tunnel + network | 1–8 % | Uncompressed WAV (~384 kbit/s) over a shared free tunnel. Bigger on long replies. |
| Tag parse / sanitize / split | ~0 % | `tags.py`, `sanitize.py`, `chunking.py` — trivial. |
| Model load | 0 % *per request* | …but **minutes** on the first request of a session if not preloaded, and `MAYA_PRELOAD=1` is already set in the notebooks. |
| `torch.compile` capture | 0 % normally | **1–3 min on the first request** when `MAYA_COMPILE=1`. |

**Implication:** every worthwhile lever either (a) makes each frame cheaper
(faster GPU, `torch.compile`, fewer tokens), (b) does more frames in parallel
(multi‑GPU, batching), or (c) starts returning audio before the whole reply is done
(streaming / shorter units). Nothing else moves the needle.

### The two speedups that currently fight each other

`build_engine()` (`model.py:187`) gives you **either**:

- **Multi‑GPU** — sampled generation, chunks spread across both T4s via a thread pool; or
- **`MAYA_COMPILE=1`** — one GPU, greedy, `torch.compile` + CUDA graphs.

They're mutually exclusive because a captured CUDA graph can't be replayed from the
`ThreadPoolExecutor` worker threads (`model.py:191`). Breaking that trade‑off (§A5) is
one of the larger structural wins available.

---

## A. Code / architecture changes (all $0)

Ranked by impact per unit effort.

### A1. ✅ 🟡 Tighten `_COMPILE_MAX_LENGTH` — *1 line, ~10–30 % on the compiled path*

**Done** — `model.py` now sets `_COMPILE_MAX_LENGTH = 384` (was 1024). The transformers
CSM "go brrr" recipe uses `max_length = 250`. The static KV cache is allocated to this
length and every attention step pays for its size. `MAYA_MAX_CHUNK_CHARS=150` → ~210
audio tokens (at the new 1.4 ratio) + prompt ≈ **~320 needed**, so 384 leaves headroom:

```python
_COMPILE_MAX_LENGTH = 384   # was 1024; static cache = this long, keep it near actual need
```

- **Effect:** smaller static cache → less memory traffic per step → faster compiled
  inference and lower VRAM. Only matters when `MAYA_COMPILE=1`.
- **Risk:** if a chunk ever exceeds the budget you get a recompile (slow) or a
  truncation. Guard by also lowering the token estimate (A3) or keeping
  `MAYA_MAX_CHUNK_CHARS ≤ 150`.
- **Validate:** `torch._logging.set_logs(recompiles=True)` — no sustained recompiles.

### A2. ✅ 🟡 Warm‑up generation at startup — *removes the first‑request cliff*

**Done** — `MayaModel.load()` now calls `_warmup()`: one throwaway `generate_chunk`
on GPU (skipped on CPU; never propagates an exception). With `MAYA_PRELOAD=1` this
runs at startup, moving cuDNN autotuning and allocator growth off the first request.

It's **skipped in compile mode** on purpose: `mode="reduce-overhead"` captures the
CUDA graph on the calling thread and it must be captured on the uvicorn worker
thread that replays it, so the graph capture stays on the first real request. The
notebooks' warm‑up cell hits the HTTP endpoint before you connect SillyTavern, so
the ~1–3 min compile is absorbed there.

- **Effect:** first user‑visible request is as fast as steady state (non‑compile);
  the compile wait moves to the notebook, not the chat (compile).

### A3. ✅ 🔵 Right‑size `max_new_tokens` — *trims worst‑case latency*

**Done** — the per‑chunk cap was `chars × 1.875` clamped `[125, 750]`; for 150 chars
that's 281 frames ≈ **22.5 s of audio ceiling** for a chunk that should speak ~8–10 s.
Now `ceil(chars × 1.4)` clamped `[80, 750]`, extracted to `_estimate_max_new_tokens()`
with unit tests. CSM stops early on its own (`codebook_eos_token_id = 0`), so this only
bites the **bad** generations (drift / rambling) — now cut at ~9 s instead of ~22 s.

- **Risk:** a legitimately dense 150‑char chunk could clip. Low (chunks are short);
  raise the 1.4 ratio if you hear clipped endings. Don't go below ~1.1.

### A4. 🔵 Honour `response_format` — return Opus/MP3, not always WAV — *~20 LoC*

`server.py:26` accepts `response_format` then ignores it; `audio.py:45` always emits
PCM‑16 WAV. **SillyTavern's OpenAI‑Compatible provider sends `response_format: "mp3"`**
and just wraps whatever blob it gets in an object URL, so the browser decodes WAV
fine today — but WAV at 24 kHz/16‑bit is ~**384 kbit/s**; Opus at 24–32 kbit/s is
**~12–16× smaller**.

```python
# audio.py — needs libsndfile with Opus (or `av`/ffmpeg)
def to_bytes(audio, sr=24000, fmt="wav"):
    if fmt in ("opus", "ogg"):
        buf = io.BytesIO(); sf.write(buf, audio, sr, format="OGG", subtype="OPUS"); return buf.getvalue(), "audio/ogg"
    if fmt == "mp3":
        buf = io.BytesIO(); sf.write(buf, audio, sr, format="MP3"); return buf.getvalue(), "audio/mpeg"
    buf = io.BytesIO(); sf.write(buf, audio, sr, format="WAV", subtype="PCM_16"); return buf.getvalue(), "audio/wav"
```

- **Effect:** on a 10 s reply, ~480 KB → ~30 KB. Saves ~0.2–1.5 s of transfer over a
  congested free tunnel, and makes the **ngrok 1 GB/month** cap (§B4) usable
  (~40 min of WAV → ~10 h of Opus).
- **Risk:** low. Encoding adds a few ms. Keep WAV as the fallback.
- **Not** a big win on a fast link — it's a bandwidth/robustness fix more than a
  latency one.

### A5. 🟢 Break the compile ⇄ multi‑GPU trade‑off — *~half day, multiplicative win*

Today `MAYA_COMPILE=1` forces a single replica because CUDA graphs can't be driven
from `ModelPool`'s threads. Two ways out:

**A5a — two compiled single‑GPU server processes behind a proxy (recommended).**
Run the server twice, `CUDA_VISIBLE_DEVICES=0` and `=1`, ports 8000/8001, each with
`MAYA_COMPILE=1`; put a ~20‑line round‑robin `httpx` proxy (or nginx) on 8080 and
point the tunnel at that. Each process compiles its own graph in its own CUDA
context. Multi‑chunk replies and concurrent requests fan out across both cards.

- **Effect:** `torch.compile` (~1.5–2×) **and** 2‑GPU parallelism (~1.8× on
  multi‑chunk replies) at once. On a 2‑chunk reply that's roughly **3×** vs today's
  best single mode.
- **Cost:** 2× model in VRAM (fine — ~4 GB each on a 16 GB T4), 2× compile time at
  startup (mitigated by A2), a tiny proxy to maintain.

**A5b — replace the thread pool with `multiprocessing`/subprocess replicas.**
Same idea, integrated into `build_engine()` instead of an external proxy. More code
(IPC of numpy arrays, lifecycle), cleaner deployment. Do A5a first; graduate to A5b
if it proves out.

> Note: this is single‑GPU‑per‑*stream*, so it only pays off on Kaggle's **T4 x2**
> (the P100 doesn't run — §B1). On a single‑GPU host it collapses to plain
> `MAYA_COMPILE=1`.

### A6. 🟢 (perceived) True audio streaming + a custom ST provider — *multi‑day, biggest perceived win*

CSM's `generate()` already accepts a `streamer=` (it streams codebook frames as they
are sampled). The missing pieces:

1. **Server:** a streamer that buffers ~8–16 frames, runs the Mimi codec decode on
   that slice, and yields PCM to a `StreamingResponse` — so audio leaves the box
   ~300–500 ms after generation starts instead of after it finishes. The community
   fork **`davidbrowne17/csm-streaming`** already does exactly this (frame batching +
   incremental decode) and reports **40–60 % lower total generation time** and
   **RTF ≈ 0.28 on a 4090**; it supports LoRA fine‑tunes, so the `csm-maya-exp2`
   adapter should drop in.
2. **Client:** here's the catch — **SillyTavern's OpenAI‑Compatible provider does
   `await response.blob()`**, i.e. it waits for the *entire* body before playback, so
   a streaming server buys nothing *through that provider*. You need either:
   - a small **custom ST TTS extension** that plays via `MediaSource`/chunked
     `<audio>` (there's prior art — AllTalk and XTTS providers stream), or
   - a local patch to `openai-compatible.js`.

- **Effect:** TTFA ~**1–2 s** and roughly **independent of reply length** — the
  single largest perceived improvement on the table.
- **Cost:** real engineering + an ST extension you now maintain across ST updates.
  This is the "phase 3" item, not a quick win. Lift the streaming core from
  `csm-streaming` rather than writing it from scratch.

### A7. ⚪ Smaller, GPU‑count‑aware chunking — *skip for now (YAGNI)*

`chunking.py:25` splits at a fixed 150 chars. You could target `ceil(total/N_replicas)`
so a 2‑GPU box always splits a 2‑sentence reply into exactly 2 balanced chunks. In
practice the gain is small, smaller chunks drift in voice and add 120 ms join gaps
(`MAYA_GAP_MS`), and "narrate by paragraphs" (§E1) already gives ST‑level
parallelism. Not worth the complexity unless benchmarking says otherwise.

### A8. ⚪ `use_kernels=True` / `flash_attention_2` — *situational, test‑only*

- `CsmForConditionalGeneration.from_pretrained(..., use_kernels=True)` pulls optimised
  layers from the HF `kernels` hub. One kwarg in `model.py:87`. Possible 5–20 % free;
  **must be tested on the T4 (Turing)** — kernel availability varies by arch.
- `attn_implementation="sdpa"` (`model.py:91`) is already correct for the T4 (Turing
  has no FlashAttention‑2). If you move to L4 / A10G / Ampere+ (§C), switch to
  `"flash_attention_2"` — worth ~1.2–1.4× there. Make it a setting.

### A9. ⚪ Quantization (int8/int4) — *not recommended here*

int8 dynamic quant helps **compute‑bound** models; CSM at batch 1 is
**memory‑bandwidth‑bound**, and `bitsandbytes` int8 is often *slower* than fp16 for a
1 B model. int4 *weight‑only* would cut weight bandwidth ~2× (the one angle that could
help) but tooling on the RVQ/codebook heads is immature and quality risk is real.
VRAM isn't your constraint (T4 has 16 GB, model ~4 GB). **Low priority / experimental.**

### A10. 🟡 (when phase‑2 lands) Cache the reference‑clip prefix — *do it right the first time*

`config.py:16` already has the `RefClip` hook and `pipeline.py:30` threads
`settings.reference_clips` into every `generate_chunk`. Once you populate it, **every
request** will re‑run the Mimi encoder on the clips and re‑prefill that (potentially
long) audio context through the backbone — pure repeated work, and it grows the prompt
that the static cache (§A1) has to hold.

Build it so the conditioning context is encoded **once at load** and its backbone KV
is reused as a frozen prefix for every request (transformers `Cache` supports this
pattern; the depth decoder is per‑frame and unaffected). Same output, and you don't
pay for the clips on the hot path.

- **Effect:** keeps per‑request latency flat as you add reference clips for voice
  consistency, instead of each clip adding hundreds of ms of prefill.
- **Effort:** moderate, but far cheaper to design in now than to retrofit.

---

## B. Free infrastructure

### B1. ❌ Kaggle P100 — *does not work; use T4 x2*

Frame‑by‑frame decoding is bound by **memory bandwidth**, and Kaggle's P100
(732 GB/s) would be ~2.3× a T4 (320 GB/s) — the original version of this report led
with it. **It doesn't run.** Kaggle's PyTorch is **2.10.0+cu128**, whose minimum
compute capability is **sm_70**; the P100 is **sm_60** (Pascal), so `from_pretrained`
dies at weight init:

```
AcceleratorError: CUDA error: no kernel image is available for execution on the device
```

Confirm on any image with:

```python
import torch
print(torch.cuda.get_device_capability(0), torch.cuda.get_arch_list())
# P100 -> (6, 0) and no 'sm_60' in the list  ->  unusable
```

Downgrading torch to a Pascal‑supporting build (≤ 2.7 cu126) fights the
`transformers` ≥ 4.52 requirement for CSM and re‑downloads ~2.5 GB each session —
not worth it.

**So on Kaggle it's T4 x2 or nothing.** For that runtime:

- Multi‑sentence replies split into chunks and run **one chunk per T4** — roughly
  halved wall‑time for a 2‑chunk reply. Single‑chunk replies use one T4 (the other
  idles).
- `MAYA_COMPILE=1` (§B2) is the only per‑stream lever left — measure it.
- A single T4 stream is ~RTF 1.2–1.6 (roughly real‑time). If that's still too slow,
  the fix is a faster GPU off Kaggle (§C3), not a Kaggle setting.

### B2. 🟡 `MAYA_COMPILE=1` — *$0, but benchmark it; not a blind default*

Covered in `docs/PERFORMANCE.md` and §A above. The recipe removes per‑kernel launch
overhead from the 31‑step depth loop — the existing docs call it *"the largest single
speedup."* Constraints: single GPU, greedy decoding (per‑tag `temperature` nudges
ignored), ~1–3 min first‑request compile (the notebook's warm‑up cell absorbs it).

**Caveat that turned real:** it uses `torch.compile(mode="reduce-overhead")`, and
Kaggle's **torch 2.10** has a
[reported `reduce-overhead` throughput regression](https://github.com/pytorch/pytorch/issues/174575).
So compile is shipped **off** in the notebooks (matching `config.py`); flip
`MAYA_COMPILE=1`, run the benchmark cell, and keep it only if it actually beats the
T4 x2 multi‑GPU default for your typical reply length. If it's slower or unstable,
pin an older torch or just leave it off.

### B3. 🔵 "Colab **in addition to** Kaggle" — *useful, but not a latency play*

You asked specifically about running a Colab instance alongside Kaggle. Straight
answer:

- **You can** — they're different platforms. (Kaggle itself now allows only **one**
  GPU session, so you can't run two Kaggle GPU notebooks.)
- **It does not reduce per‑reply latency for a single user.** SillyTavern sends TTS
  requests **sequentially** (it `await`s each response before the next), so a second
  backend sits idle while the first works. Two instances only help when requests run
  *concurrently* — multiple paragraphs in flight, or multiple people.
- **What it does buy:**
  - **Redundancy / failover** — when Kaggle drops your session mid‑chat (12 h cap,
    weekly quota, maintenance), flip SillyTavern's endpoint to the Colab URL.
  - **Quota headroom** — ~30 h/week Kaggle + ~15–30 h/week Colab free ≈ enough for
    daily 1–3 h use without running dry.
  - **A pick‑the‑faster‑box option** — e.g. Kaggle T4 x2 for parallel narration; a
    Colab **Pro L4** (§C2) instance for FlashAttention‑2 + compile on short replies.
    But you'd *use one at a time*, not both.
- **Verdict:** run a second instance for **uptime**, not speed. If you want a latency
  win from spending money, put it into a faster single box (§C), not a second free one.

A minimal "failover" setup: a 15‑line local reverse proxy that health‑checks both
tunnels and routes to whichever is up, so SillyTavern keeps one stable URL.

### B4. 🔵 Stop the tunnel URL from churning — *$0, reliability*

The notebooks use a **`trycloudflare` quick tunnel** — new random URL every session,
which you paste into SillyTavern each time. Two better options, both free:

| Option | Stable URL | Bandwidth cap | Setup |
|---|---|---|---|
| **Cloudflare *named* tunnel** | yes, your own hostname | **unmetered** (Cloudflare Tunnel is free for everyone as of 2026) | needs a domain on a free Cloudflare zone; `cloudflared` with a token |
| **ngrok free** | yes — one `*.ngrok-free.dev` dev domain | **1 GB/month**, 20k req/month | drop‑in; the notebook already has a commented `pyngrok` fallback |

- Named Cloudflare tunnel is the better fit (no bandwidth cap → WAV is fine). ngrok's
  1 GB cap is ~40 min of WAV audio/month — only viable **with Opus output (§A4)**.
- **Latency effect:** negligible directly; big **reliability/quality‑of‑life** effect
  (no re‑pasting URLs, fewer dead sessions). A named tunnel also tends to have
  steadier throughput than the shared `trycloudflare` pool.

### B5. ⚪ Other free GPU pools — *marginal for this constraint set*

| Service | Free allowance | Useful GPU | Catch |
|---|---|---|---|
| **Lightning AI** | ~15 credits ≈ **80 GPU‑h/month**, no CC | L4 / T4 (interruptible) | always‑on Studio restarts every 4 h; still a tunnel |
| **HF ZeroGPU Space** | free (more with $9 Pro) | **H200 slice** (very fast) | per‑call spin‑up, 120 s function cap, quota; built for Gradio not as an always‑on API backend for an external client |
| **Google Colab (free)** | ~15–30 h/week | T4 | 90‑min idle disconnect, preemption |

Lightning's 80 h/month is the most "server‑like" of these and stacks with Kaggle for
quota. ZeroGPU's H200 is tempting on raw speed but the call model fights an
always‑on chat use case — revisit only if someone ports maya‑csm to a ZeroGPU Space.

---

## C. Paid infrastructure within ~$15/month

The Maya voice is a hard requirement, so this is all "**run the same model on better
iron / more reliably**", not "switch services".

### C1. 🟢 **Modal free tier** — stable URL + better GPU + per‑second billing — *effectively $0*

**Why it's the best single move for the money (which is none):**

- `@modal.asgi_app()` serves your **existing FastAPI app** (`server.py`) at a
  permanent `https://<you>--maya-csm.modal.run` URL. No tunnel, no URL churn, no 12 h
  cap, no weekly quota.
- **Per‑second billing, scale‑to‑zero.** You pay only for seconds actually spent
  generating. A 2 h chat session might be ~15–25 min of real GPU time → you're billed
  for ~15–25 min.
- **Free tier = $30/month of credits, auto‑renewing, no card required.**

| GPU on Modal | Rate | $30 credit ≈ | vs your T4 | Notes |
|---|---|---|---|---|
| T4 | ~$0.59/h | ~50 h | 1.0× | same speed, but stable URL + no session juggling |
| **L4** | ~$0.80/h | ~37 h | ~1.2–1.5× | **same ~300 GB/s bandwidth as a T4** — the win is Ada bf16 + FlashAttention‑2 + cleaner `torch.compile`, not raw decode speed |
| A10G | ~$1.10/h | ~27 h | ~1.9× | 600 GB/s — a real bandwidth jump |
| A100‑40 | ~$2.10/h | ~14 h | ~4–5× | overkill for 1 B, but per‑second so a session is cents |

- **Math for your usage:** 1–3 h/session, most days, but only ~30–40 % of that is
  actual generation → ~20–40 GPU‑h/month of *billed* time. On **T4 that's inside the
  free $30**; on **L4 it's borderline** (~30–37 h) — fine if you let it scale to zero
  between sessions and eat the occasional cold start.
- **Cold start:** ~2–4 s Modal infra + ~5–15 s to load 4 GB of weights from a Modal
  Volume = **~10–20 s on the first request after idle**. Mitigations: `scaledown_window`
  ~600 s so it stays warm through a session; `min_containers=1` *only while you're
  actively chatting* (burns ~$0.80/h). Snapshotting/`@enter` loads weights once per
  container, not per request.
- **Effort:** ~half a day. Wrap `create_app()` in a Modal ASGI function, bake deps
  into the image, put weights on a Volume, set `HF_TOKEN` as a Modal Secret.
- **This also subsumes §A2/§B4** (warm containers, stable URL) and gives you L4‑class
  hardware for $0.

### C2. 🟡 **Colab Pro** ($9.99/mo) — L4 + compile + FlashAttention‑2

- $9.99/mo = 100 compute units; **L4 burns ~1.71 u/h (~$0.17/h effective)** → ~**58 h
  of L4/month** on the base units, more purchasable at the same rate.
- L4 vs T4 for this workload: bandwidth is **the same** (~300 vs 320 GB/s), so raw
  frame‑by‑frame decode is barely faster. The gain (**~1.2–1.5×**, a bit more with
  compile) comes from bf16, FlashAttention‑2, and cleaner `torch.compile` codegen on
  Ada. Don't switch to L4 *for the bandwidth* — switch for FA2 + the platform.
- **Downsides:** no true background execution (that's Pro+ at $49.99); ~90‑min idle
  disconnect (you're active during a chat, so mostly fine); still need a tunnel
  (§B4); still a notebook, not a service.
- **vs Modal:** Modal is $0, gives a stable URL, and bills per second. Colab Pro is
  simpler (you already run notebooks) and predictable. If the Modal setup effort is
  unappealing, Colab Pro on L4 is the low‑friction paid pick.

### C3. 🟡 Rent a cheap Ampere/Ada GPU part‑time — *~$3–15/mo, biggest raw speedup in budget*

For 1 B‑param frame‑by‑frame decode, a consumer 3090/4090 is transformational:

| GPU | Mem BW | vs T4 | Vast.ai (on‑demand) | RunPod (community) | $15/mo buys |
|---|---|---|---|---|---|
| RTX 3090 | 936 GB/s | ~2.9× | **~$0.10–0.22/h** | ~$0.22/h | ~70–150 h |
| RTX 4090 | 1008 GB/s + Ada | **~3–5× real** (RTF ≈ 0.28 documented) | ~$0.20–0.34/h | ~$0.34/h | ~45–75 h |
| A5000 | 768 GB/s | ~2.4× | ~$0.12–0.20/h | ~$0.16/h | ~75–125 h |

- At **RTF ≈ 0.28** a 4090 generates a 2‑sentence (~10 s) reply in **~3 s**, and with
  §A6 streaming, TTFA ~**1 s**.
- **$15/mo covers ~45–75 h on a 4090** — comfortably more than "1–3 h most days" *if*
  you **start/stop the instance per session** (or script it). Leaving it 24/7 is
  ~$180–250/mo — out of budget.
- **Friction:** you manage a pod (SSH, `git pull`, start the server, start the
  tunnel) or bake a template/Docker image. Vast is cheapest and has spot pricing
  (~$0.04–0.14/h on a 3090) but instances can be reclaimed. RunPod community is
  pricier but steadier, and its **serverless** mode (FlashBoot, ~10–15 s cold start,
  per‑second) is an option — though its endpoint shape isn't OpenAI‑compatible
  without a shim, so Modal (§C1) is the better serverless fit.
- **Best "spend the $15" pick:** a **RunPod community RTX 3090 or 4090**, started per
  session from a saved template, tunneled the same way as Kaggle. Or push the whole
  thing to **Modal L4** and stay at $0.

### C4. HF Inference Endpoints — *priced out for always‑on, fine scale‑to‑zero*

T4 $0.50/h, L4 $0.70–0.80/h, scale‑to‑zero (≥15 min idle), per‑minute billing. No
free GPU credits. A T4 endpoint at 8 h/day ≈ **$144/mo** — over budget. Only sensible
with aggressive scale‑to‑zero, at which point Modal's free credits win.

---

## D. Out of scope, and why (recorded so it's not re‑litigated)

| Option | Why excluded |
|---|---|
| ElevenLabs v3 / Cartesia Sonic / Inworld / OpenAI `gpt-4o-mini-tts` / Hume Octave / MiniMax | **Not the Maya voice.** All are OpenAI‑compatible or close, several are cheaper *per character* than a rented GPU, and latency is far better (Cartesia ~90 ms, ElevenLabs Flash ~75 ms). If the voice requirement ever softens, Cartesia (voice‑clone + ~40 ms model latency) or an expressive ElevenLabs voice would beat self‑hosting on every axis except "it's literally Maya". Filed for later. |
| Orpheus‑TTS, Maya1, Chatterbox, Dia, XTTSv2, Kokoro | Faster, streamable, tag‑capable open models — but **not this fine‑tune's voice.** |
| DeepInfra / Chutes / OpenRouter hosted `csm-1b` | Base `csm-1b` only — **no `csm-maya-exp2` adapter**, so not the Maya voice, and no expressive‑tag behaviour this project relies on. |
| vLLM / TensorRT‑LLM for CSM | No stable native CSM support in vLLM as of now (the `csm-streaming` demo uses a custom path). Potentially a **large** win — continuous batching, paged KV, CUDA graphs — but not a dependable route today. **Watch `davidbrowne17/csm-streaming` and transformers CSM issues.** |
| Always‑on rented GPU (24/7) | ~$150–250/mo. Way over the $15 budget. |
| Tensor/pipeline parallelism across the 2 T4s | PCIe‑linked, no NVLink; cross‑GPU traffic for a 1 B model costs more than it saves. (Already rejected in `docs/PERFORMANCE.md`.) |

---

## E. SillyTavern‑side settings (all $0, do these regardless)

### E1. 🟢 Turn on "Narrate by paragraphs" — *the biggest free perceived‑latency win*

SillyTavern's TTS extension, with **"Narrate by paragraphs (when not streaming)"**
enabled, splits the reply on newlines and pushes each paragraph as a **separate**
provider request into a queue. Generation and playback are **separate queues**, so:

- audio for paragraph 1 starts **playing** as soon as it's ready;
- paragraphs 2…N **generate while paragraph 1 plays**.

If generation RTF < ~1 (a T4 with compile, or any rented Ampere GPU — a plain T4 is
right at the line), every paragraph after the first is ready before playback reaches
it → the *only* wait you feel is the first paragraph.

- **Make the first sentence short** (prompt the character card for it, or rely on
  natural dialogue openers). TTFA ≈ generation time of ~1 short sentence ≈ **3–6 s**
  today, **~1–2 s** on faster hardware.
- **Known bug:** ST issue #4228 — "Narrate by paragraphs" can be ignored when
  response **streaming** is also on. Verify in the browser Network tab that you see
  **multiple** `/v1/audio/speech` POSTs per reply, not one.
- Also enable **"Narrate quoted text only"** if you don't want `*actions*` spoken —
  less text = less generation.

### E2. 🔵 Client‑side cache is already working for you

ST hashes narration text and caches the audio blob — repeated/regenerated lines are
instant. Nothing to do; just know that re‑rolls of *identical* text won't re‑hit the
server.

### E3. 🔵 Request timeout

Only relevant so long replies don't error out mid‑generation — it doesn't affect
speed. Keep SillyTavern's `requestTimeout` comfortably above your worst‑case reply
time (measure in §1).

---

## Master ranking

### By potential performance improvement (TTFA for a typical 2–3 sentence reply)

| Rank | Lever | Mechanism | Est. improvement | $/mo | Effort |
|---|---|---|---|---|---|
| 1 | **A6** True streaming + custom ST provider | return audio mid‑generation | TTFA → **~1–2 s**, length‑independent | 0 | multi‑day |
| 2 | **E1** Narrate by paragraphs | overlap generation with playback | perceived **~15–25 s → ~3–6 s** | 0 | 2 min |
| 3 | **C3** Rent RTX 3090/4090 part‑time | ~3–5× memory bandwidth + Ada | **~3–5×** on generation | ~$3–15 | half day |
| 4 | **A5** Break compile⇄multi‑GPU | CUDA graphs + 2 GPUs together | **~3×** on multi‑chunk replies (T4×2 only) | 0 | half day |
| 5 | **B2** `MAYA_COMPILE=1` (+A1) | kill launch overhead in depth loop | **~1.5–2×** *if* torch 2.10 doesn't regress it — measure | 0 | 1 h validate |
| — | ~~**B1** Kaggle P100~~ | ~~~2.3× memory bandwidth~~ | ❌ P100 is sm_60; Kaggle torch 2.10 needs sm_70+ | 0 | — |
| 7 | **C1** Modal L4 / **C2** Colab Pro L4 | bf16 + FA2 + better compile (bandwidth ≈ T4) | **~1.2–1.5×** (+ stable URL, no session limits) | $0 / $10 | half day / low |
| 8 | **A2** Startup warm‑up | move one‑time costs off the hot path | first reply **2–20× → 1×** | 0 | 15 min |
| 9 | **A1** Shrink `_COMPILE_MAX_LENGTH` | smaller static cache | ~10–30 % on compiled path | 0 | 1 line |
| 10 | **A3** Right‑size `max_new_tokens` | cap the bad generations | tail latency ~22 s → ~9 s | 0 | 2 lines |
| 11 | **A4** Opus output | 12–16× less to transfer | ~0.2–1.5 s on a slow tunnel | 0 | ~20 LoC |
| 12 | **A8** `use_kernels` / FA2 | optimised layers | 5–20 %, hardware‑dependent | 0 | test |
| — | **A10** Cache reference‑clip prefix | avoid re‑prefilling clips each request | prevents a **regression** when phase‑2 lands | 0 | moderate |
| — | **B3** Colab *alongside* Kaggle | (concurrency only) | **~0** for one user | 0 | — |
| — | **A9** Quantization | — | ~0 or negative on T4 | 0 | — |

### By cost (cheapest first)

| Tier | Levers | One‑liner |
|---|---|---|
| **$0, minutes** | E1, A1 | Narrate by paragraphs; `_COMPILE_MAX_LENGTH=384` (shipped). **Do today.** |
| **$0, ~1 h** | B2 | Benchmark `MAYA_COMPILE=1` vs the T4 x2 default; keep whichever wins. |
| **$0, hours** | A2, A3, A4, B4, A8 | Warm‑up (shipped); token cap (shipped); Opus; named tunnel; kernel flags. |
| **$0, days** | A5, A6 | Break the compile/multi‑GPU split; real streaming + ST provider. |
| **$0, "free tier"** | C1 (Modal), B5 (Lightning) | Per‑second billing inside $30/mo credit; 80 GPU‑h/mo. |
| **~$10/mo** | C2 (Colab Pro L4) | Low‑friction L4 with compile + FA2. |
| **~$3–15/mo** | C3 (Vast/RunPod 3090‑4090, per session) | Biggest raw speedup available in budget. |
| **>$15/mo** | C4, always‑on pods, commercial APIs | Out of budget and/or wrong voice — see §D. |

---

## Recommended sequence (what I'd actually do)

1. **§1 benchmark** on **T4 x2** (the only Kaggle option). Run the notebook's
   benchmark cell for 1/2/4 sentences, `MAYA_COMPILE=0` then `=1`.
2. **§E1** — narrate by paragraphs, short opener. Free, instant, biggest *felt*
   change.
3. **§B2** — keep `MAYA_COMPILE` at whichever setting won the benchmark.
   (`_COMPILE_MAX_LENGTH=384` and the §A2 warm‑up are already shipped.)
4. **Re‑benchmark** end‑to‑end from SillyTavern with paragraphs on.
5. If per‑reply speed is still the problem: the P100 is gone, so a hardware win means
   leaving Kaggle — **§C1 Modal free tier** (L4, stable URL, $0) or **§C3** a
   per‑session rented RTX 3090/4090 (~$3–15/mo, ~3–5×).
6. If session‑juggling / URL‑churn is the real pain: **§C1 Modal** or **§B4** a named
   Cloudflare tunnel.
7. If you need "instant": **§C3** a rented 4090, and/or invest the days in **§A6**
   streaming.
8. Independently, whenever: **§A4** Opus output, small permanent wins.

---

## Appendix

### GPU memory bandwidth (the number that matters for CSM decode)

| GPU | Mem BW | Arch | Native bf16 | FA2 | Rel. speed¹ | Where |
|---|---|---|---|---|---|---|
| Tesla T4 | 320 GB/s | Turing (sm_75) | ✗ | ✗ | 1.0× | Kaggle ×2, Colab free, Modal, HF |
| L4 | 300 GB/s | Ada | ✓ | ✓ | ~1.2–1.5× (bandwidth ≈ T4; FA2/compile win) | Colab Pro, Modal, HF, Lightning |
| A10 / A10G | 600 GB/s | Ampere | ✓ | ✓ | ~1.9× | Modal, AWS |
| ~~Tesla P100~~ | ~~732 GB/s~~ | Pascal (**sm_60**) | ✗ | ✗ | — | ❌ **PyTorch 2.10+cu128 (Kaggle) needs sm_70+; the P100 does not run** |
| RTX A5000 | 768 GB/s | Ampere | ✓ | ✓ | ~2.4× | Vast, RunPod |
| V100 | 900 GB/s | Volta (sm_70) | ✗ | ✗ | ~2.8× | still supported by torch 2.10+cu128 (sm_70 is the floor) |
| RTX 3090 | 936 GB/s | Ampere | ✓ | ✓ | ~2.9× | Vast, RunPod |
| RTX 4090 | 1008 GB/s | Ada | ✓ | ✓ | ~3–5× (RTF≈0.28) | Vast, RunPod |
| A100‑40/80 | 1555 / 2039 GB/s | Ampere | ✓ | ✓ | ~4–6× | Modal, Colab Pro, clouds |
| H100 / H200 | 3350 / 4800 GB/s | Hopper | ✓ | ✓ | ~8–12× | HF ZeroGPU (sliced), clouds |

¹ Rough, for batch‑1 autoregressive TTS decode; combines bandwidth with
arch/kernel/`compile` effects. Measure to confirm.

### Key facts this report rests on

- **CSM architecture:** backbone (16 layers, 2048‑wide) predicts codebook 0; depth
  decoder (**4 layers, 1024‑wide, 31 steps/frame**) predicts codebooks 1–31; Mimi
  codec → 24 kHz audio at 12.5 frames/s (80 ms/frame). The 31‑step loop is
  launch‑overhead‑bound → `torch.compile`/CUDA graphs help it most.
- **CSM stops early** on `codebook_eos_token_id = 0`; `max_new_tokens` is a safety cap,
  not the normal amount of work.
- **`generate()` takes a `streamer=`** — streaming codebook frames out is supported;
  incremental Mimi decode is the part you'd add.
- **SillyTavern OpenAI‑Compatible provider** POSTs `{input, model, voice,
  response_format:"mp3", speed}` and does `await response.blob()` — **no streaming
  consumption**, WAV tolerated. "Narrate by paragraphs" splits on `\n` into separate
  queued requests.
- **Kaggle:** ~30 GPU‑h/week, 12 h/session, **1 GPU session at a time**. Offers P100
  or T4×2, but its **PyTorch is 2.10.0+cu128 (min sm_70)** so the **P100 (sm_60) is
  unusable** — T4×2 is the only working choice.
- **`torch.compile(mode="reduce-overhead")`** has a
  [reported throughput regression in torch 2.10](https://github.com/pytorch/pytorch/issues/174575);
  `MAYA_COMPILE=1` must be benchmarked on Kaggle's build, not assumed.
- **Modal:** $30/mo free credits (renewing, no card), per‑second, scale‑to‑zero,
  `@modal.asgi_app()` serves FastAPI at a stable URL; cold start ~2–4 s + weight load.
- **Prices** (2026, change often — verify before committing): Modal T4 $0.59/L4
  $0.80/A10G $1.10/h; RunPod community RTX 3090 $0.22 / 4090 $0.34/h, serverless 4090
  $1.10/h flex; Vast RTX 3090 $0.10–0.22 / 4090 $0.20–0.34/h on‑demand; Colab Pro
  $9.99/mo (L4 ≈ $0.17/h‑equiv); HF Endpoints T4 $0.50 / L4 $0.70–0.80/h.

### Sources

- Transformers CSM model docs (architecture, "go brrr" compile recipe, batching,
  `streamer=`, `codebook_eos_token_id`): <https://huggingface.co/docs/transformers/main/model_doc/csm>
- `torch.compile` `reduce-overhead` / CUDA graphs (70–100 % on top of base compile;
  PyTorch 2.10 regression note): <https://huggingface.co/docs/transformers/en/perf_torch_compile>, <https://github.com/pytorch/pytorch/issues/174575>
- CSM streaming fork (RTF 0.28 on 4090, 40–60 % faster total, LoRA support, frame
  batching): <https://github.com/davidbrowne17/csm-streaming>
- OpenAI‑compatible CSM server prior art: <https://github.com/phildougherty/sesame_csm_openai>
- CSM latency ("5–10 s / sentence", "streaming → 1–2 s perceived"): <https://www.cerebrium.ai/articles/deploying-sesame-csm-the-most-realistic-voice-model>, <https://github.com/SesameAILabs/csm/issues/80>
- SillyTavern TTS extension (OpenAI‑Compatible provider, narrate by paragraphs, #4228): <https://github.com/SillyTavern/SillyTavern-Docs/blob/main/extensions/TTS.md>, <https://github.com/SillyTavern/SillyTavern/issues/4228>
- Kaggle quota / single GPU session: <https://www.kaggle.com/general/105509>, <https://www.kaggle.com/discussions/questions-and-answers/477091>
- Modal pricing + free credits + web endpoints: <https://modal.com/pricing>, <https://modal.com/docs/guide/webhooks>, <https://modal.com/docs/guide/cold-start>, <https://www.spheron.network/blog/modal-gpu-pricing-2026-per-second-billing/>
- RunPod pricing / FlashBoot: <https://www.runpod.io/pricing>, <https://www.runpod.io/blog/introducing-flashboot-serverless-cold-start>
- Vast.ai pricing: <https://getdeploying.com/gpus/nvidia-rtx-4090>, <https://computeprices.com/providers/vast>
- Colab pricing / limits: <https://cloud.google.com/colab/pricing>, <https://www.hivenet.com/post/google-colaboratory-gpu-complete-guide-to-free-cloud-gpu-access-and-limitations>
- HF Inference Endpoints pricing: <https://www.spheron.network/blog/hugging-face-inference-endpoints-pricing-2026/>
- Lightning AI free tier: <https://www.gmicloud.ai/en/blog/where-can-i-get-free-gpu-cloud-trials-in-2026-a-complete-guide>
- ngrok free static domain / Cloudflare Tunnel free: <https://ngrok.com/blog/free-static-domains-ngrok-users>, <https://ngrok.com/docs/pricing-limits/free-plan-limits>
- Quantization tradeoffs (memory‑ vs compute‑bound; bnb int8 slower): <https://pytorch.org/blog/pytorch-native-architecture-optimization/>

_Report generated 2026‑09‑02. Prices and free‑tier terms move fast — re‑verify anything you're about to pay for._
