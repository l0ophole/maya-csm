# Performance & multi-GPU

CSM generates audio one 80 ms frame at a time (a backbone step plus a 31-step
depth-decoder loop), so a single request is latency-bound. The server speeds
things up in two independent ways.

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

## Single-GPU knobs (help every request)

| Knob | Effect |
|---|---|
| `float16` (default `MAYA_DTYPE`) | Turing/T4 has no native bf16; fp16 avoids the emulation path. Set `MAYA_DTYPE=bfloat16` only if fp16 produces audible artifacts. |
| LoRA merge | The adapter is folded into the base weights at load (`merge_and_unload`) — identical output, no per-forward adapter cost. Always on. |
| `MAYA_COMPILE=1` | `torch.compile` + static KV cache (per the transformers CSM docs). Removes per-kernel launch overhead, which dominates the 31-step depth-decoder loop — the largest single speedup. Costs 1–3 min of compilation at startup (done in a warmup call, not on the first user request) and recompiles each fresh Kaggle session. Pairs best with `MAYA_SAMPLING=0`. |

To check compilation is stable, run once with:

```python
import torch._logging
torch._logging.set_logs(recompiles=True, graph_breaks=True, cudagraphs=True)
```

Sustained `recompiles` lines mean an input shape is varying more than expected —
raise `_COMPILE_MAX_LENGTH` in `model.py` or lower `MAYA_MAX_CHUNK_CHARS`.

## Alternative: single P100

Kaggle's P100 has ~2× the memory bandwidth of one T4, which is the bottleneck for
frame-by-frame decoding. For short (single-chunk) replies a P100 can beat a T4 x2
with no code change — worth trying if your replies are usually one or two
sentences.
