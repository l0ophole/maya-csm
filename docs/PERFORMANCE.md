# Performance & multi-GPU

CSM generates audio one 80 ms frame at a time (a backbone step plus a 31-step
depth-decoder loop), so a single request is latency-bound. There are two speedup
strategies and they are **mutually exclusive** — pick one:

- **Multi-GPU** (default): sampled generation, chunks spread across every GPU.
- **`MAYA_COMPILE=1`**: one GPU, greedy decoding, CUDA graphs. Faster per stream,
  but CUDA graphs can't be driven from the multi-GPU thread pool, so it runs on a
  single GPU.

fp16 and the LoRA merge apply to both. For the wider menu of options (hardware,
hosting, streaming, SillyTavern settings) see
[PERFORMANCE-IMPROVEMENTS.md](PERFORMANCE-IMPROVEMENTS.md).

## Multi-GPU (data parallelism)

`synthesize()` splits text into ~150-char chunks (`MAYA_MAX_CHUNK_CHARS`). Chunks
are independent, so when more than one GPU is visible the server loads **one model
replica per GPU** and generates the chunks concurrently — a 4-sentence reply on a
2-GPU box finishes in about half the time. Short one-chunk replies are unaffected.

- Automatic on Kaggle's **T4 x2**. Confirm with
  `nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv` —
  both rows should show resident memory, and utilization should spike on both
  during a multi-sentence request.
- `MAYA_DEVICES=cuda:0` pins to one GPU (useful for A/B timing).
- Concurrent HTTP requests share the same replica pool, so replicas are never
  oversubscribed.

**Not** used: tensor / pipeline parallelism. Kaggle's T4s are linked over PCIe
(no NVLink); splitting one `generate()` call across both cards pays more in
cross-GPU traffic than it saves for a 1B model.

## Always on (help every request)

| Knob | Effect |
|---|---|
| `float16` (default `MAYA_DTYPE`) | Turing/T4 has no native bf16; fp16 avoids the emulation path. Set `MAYA_DTYPE=bfloat16` only if fp16 produces audible artifacts. |
| LoRA merge | The adapter is folded into the base weights at load (`merge_and_unload`) — identical output, no per-forward adapter cost. |
| Warm-up | `load()` runs one throwaway generation on GPU so cuDNN autotuning and allocator growth happen at load (with `MAYA_PRELOAD=1`, at startup), not on the first real request. Skipped on CPU, and skipped in compile mode (the CUDA-graph capture must happen on the serving thread — see below); a warm-up failure never blocks startup. |
| Token budget | The per-chunk `max_new_tokens` cap is `ceil(chars × 1.4)` clamped to `[80, 750]` (`_estimate_max_new_tokens`). CSM stops early on its own EOS; this just bounds a drifting generation (~9 s worst case for a 150-char chunk, was ~22 s). |

## `MAYA_COMPILE=1` (opt-in, single GPU)

`torch.compile` + static KV cache, following the transformers CSM "go brrr" recipe.
Removes per-kernel launch overhead, which dominates the 31-step depth-decoder loop
— on paper the largest single speedup.

> **Benchmark it before relying on it.** It uses `mode="reduce-overhead"`, and
> PyTorch 2.10 (Kaggle's current build) has a
> [reported `reduce-overhead` throughput regression](https://github.com/pytorch/pytorch/issues/174575).
> The notebooks ship with `MAYA_COMPILE` unset; flip it on, run the benchmark cell,
> and keep it only if it beats the multi-GPU default for your reply lengths.

Constraints, all enforced automatically when the flag is set:

- **One GPU only.** The static cache makes transformers compile with CUDA graphs
  (`mode="reduce-overhead"`); a captured graph can't be replayed from the
  multi-GPU worker threads, so `build_engine` drops to a single replica.
- **Greedy decoding.** `torch.multinomial` sampling advances the RNG offset
  outside the graph (`RuntimeError: Offset increment outside graph capture`), so
  `do_sample` is forced off — which matches the recipe and the fine-tune author's
  baseline anyway. Per-tag `temperature` nudges are ignored in this mode.
- **Compilation happens on the first request** (~1–3 min), not at load, so the
  graph is captured on the uvicorn worker thread that will replay it (this is why
  the `load()` warm-up is skipped in compile mode). The notebooks' warm-up cell
  hits the HTTP endpoint before you connect SillyTavern, so that first request —
  and the compile — is yours, not the chat's. Recompiles each fresh Kaggle session.

To check compilation is stable, run once with:

```python
import torch._logging
torch._logging.set_logs(recompiles=True, graph_breaks=True, cudagraphs=True)
```

Sustained `recompiles` lines mean an input shape is varying more than expected —
raise `_COMPILE_MAX_LENGTH` in `model.py` (384, sized for `MAYA_MAX_CHUNK_CHARS`
≈ 150) or lower `MAYA_MAX_CHUNK_CHARS`.

## Not an option: the P100

Kaggle's P100 has ~2× a T4's memory bandwidth and would help frame-by-frame decoding
— but **it no longer runs**. Kaggle's PyTorch is `2.10.0+cu128`, whose minimum
compute capability is `sm_70`; the P100 is `sm_60` (Pascal), so loading the model
fails with `CUDA error: no kernel image is available for execution on the device`.
Use **T4 x2**. Check any environment with:

```python
import torch
print(torch.cuda.get_device_capability(0), torch.cuda.get_arch_list())
```
