# gpt-engine — a GPT and its inference engine, built from scratch on AMD RDNA4

| | |
|---|---|
| **Author** | Pratyush Mohit |
| **Status** | Draft v0.1 |
| **Last updated** | 2026-10-28 |
| **Working name** | `gpt-engine` (rename freely) |
| **Hardware** | AMD Radeon RX 9070 XT (RDNA4, gfx1201), 16 GB GDDR6, 640 GB/s |
| **Timeline** | ~8–10 weeks at 15–20 h/week; one tagged release every 1–2 weeks |

> **This is a learning project.** The goal is to deeply understand how an LLM is built, trained, and served, not just to have working code. See §3 for what the learner writes vs. what an AI assistant may write.

---

## 1. TL;DR

Build a GPT in PyTorch from a blank file, train a small one, load GPT-2 weights into it, and serve it on vLLM as a baseline. Then rebuild vLLM's core ideas one release at a time: sampling and KV cache → GPU kernels (HIP + Triton) → FlashAttention and quantization → a serving engine with paged KV cache and continuous batching → an instruction-tuned model served end to end. Every release is measured against the previous one and against vLLM, on an AMD consumer GPU, where few people have done this.

## 2. Goals and non-goals

### Goals
- **G1 Understanding:** be able to explain, from memory, every concept in the inference workbook: attention shapes, KV cache maths, prefill vs decode, rooflines, coalescing, fusion, FlashAttention, paged KV, continuous batching, prefix caching, quantization trade-offs.
- **G2 Building:** write the core of each piece yourself (§3).
- **G3 Measuring:** every change ships with a number (tokens/s, TTFT, TPOT, % of peak bandwidth) and an explanation of *why* it changed.
- **G4 Portfolio:** one repo with tagged releases, a README that tells the story with numbers, and a blog post per release.
- **G5 Upstream:** use the vLLM-on-RDNA4 setup as the start of an upstream contribution (§9).

### Non-goals
- Training a large or state-of-the-art model. Use small models; correctness is checked against GPT-2.
- Beating vLLM. The goal is to understand and explain the gap.
- Multi-GPU (later project).
- Production hardening (auth, multi-tenant, autoscaling).

## 3. Rules for AI assistants (Claude Code, etc.)

**The learner writes (assistants explain, hint, and review, but do not write this code):**
- Attention, transformer blocks, the GPT model, the training loop
- The GPT-2 weight loader and the export converter
- Sampling, the KV cache, the generation loop
- All HIP and Triton kernels
- FlashAttention
- Quantization
- The scheduler, the block manager (paged KV), continuous batching, prefix caching

**Assistants may write:**
- Environment setup scripts, Dockerfiles, Makefile, CI
- Benchmark harness plumbing, plotting, result tables
- Test scaffolding (the learner writes the key assertions)
- The HTTP/API layer boilerplate in v0.5
- README formatting (the learner writes the analysis)

**How assistants should help:** when asked, explain the concept, point to the reference (§5), and give a hint or a smaller sub-problem before any code. Review the learner's code for correctness and performance. Ask "what do you predict?" before measurements.

## 4. Environment

| Item | Choice |
|---|---|
| OS | Ubuntu 24.04 LTS (dual boot); AMD's best-supported setup for ROCm on RDNA4 |
| GPU stack | ROCm 7.x |
| Framework | PyTorch (ROCm build); `torch.cuda` APIs map to HIP |
| Kernels | HIP (`hipcc`, `torch.utils.cpp_extension.load_inline`), Triton (AMD backend) |
| Profiling | `torch.profiler`, `rocprofv3`, ROCm Compute Profiler, Radeon GPU Profiler |
| Baselines | vLLM (bf16 on ROCm; AMD Docker image as fallback), llama.cpp (HIP or Vulkan), Hugging Face `transformers` |
| Python | 3.12, `uv` |
| Tokenizer | GPT-2 BPE via `tiktoken` (learn BPE from Raschka ch 2, reuse the GPT-2 tokenizer for compatibility) |

**Record in `docs/hardware.md`:** exact GPU, driver, ROCm, PyTorch, Triton, and vLLM versions; the card's published peak FP16/bf16 throughput, memory bandwidth, CU count, LDS size, and wavefront size; and your computed **ridge point** (peak FLOPs ÷ bandwidth). These are the numbers your rooflines use.

## 5. References

| Ref | What |
|---|---|
| **Raschka** | *Build a Large Language Model (From Scratch)*, Sebastian Raschka, plus repo `rasbt/LLMs-from-scratch` |
| **Syllabus** | Inference Engineering fast-track (12 weeks), referenced as W1–W12 |
| **Workbook** | Inference workbook questions, referenced as Q1.1 etc. |
| **Karpathy** | "Let's reproduce GPT-2 (124M)" video |
| **PMPP** | *Programming Massively Parallel Processors* |
| **Scaling book** | "How to Scale Your Model" (rooflines, transformer maths, inference chapters) |
| **Papers** | FlashAttention 1/2, Orca (continuous batching), vLLM (PagedAttention), SGLang (RadixAttention) |
| **Code to read** | `nano-vllm`, Triton tutorials, vLLM source, Mini-SGLang |

## 6. Releases

Each release = a git tag, a short write-up (`docs/releases/vX.Y.md`), updated README numbers, and that release's workbook questions answered from memory.

### v0.1: GPT from scratch, trained and served on vLLM
**Concepts:** tensors, autograd, BPE, embeddings, causal multi-head attention, layer norm, GELU, residuals, the training loop, bf16, LR warmup and cosine decay, gradient clipping, weight tying.

| Scope | Reference |
|---|---|
| PyTorch refresher | Raschka App. A; Syllabus W1 |
| Data loader with `tiktoken` | Raschka ch 2 |
| Causal multi-head attention from a blank file, tested against `F.scaled_dot_product_attention` | Raschka ch 3; W1 |
| GPT model + generation (greedy) | Raschka ch 4; W2 |
| Train a small GPT (10–30M params) on TinyStories, bf16 on the 9070 XT | Raschka ch 5 + App. D |
| Load OpenAI GPT-2 (124M) weights into your model | Raschka ch 5; Karpathy |
| Export your trained model to HF `GPT2LMHeadModel` format (write the name-mapping converter) | — |
| Serve both models on vLLM (OpenAI-compatible API); call from a script | vLLM docs |
| Baseline benchmark with `vllm bench` at 1 / 4 / 16 concurrent requests | — |

**Exit criteria**
- Your GPT-2 logits match HF to **≤ 1e-4** max abs diff on a fixed prompt set
- Your trained model's validation loss curve is in the README; samples are coherent short stories
- vLLM serves your model; baseline TTFT / TPOT / throughput table recorded
- Workbook Q1.1–Q1.12, Q2.1–Q2.10 answered from memory
- Anything that broke in the vLLM-on-RDNA4 setup is logged in `docs/vllm-rdna4-notes.md`

### v0.2: Fast generation
**Concepts:** sampling (temperature, top-k, top-p), TTFT / TPOT / throughput, KV cache, prefill vs decode, compute- vs memory- vs overhead-bound, CPU launch overhead, CUDA/HIP graphs, `torch.compile`, static KV cache.

| Scope | Reference |
|---|---|
| Top-k, top-p, temperature sampling | Raschka ch 5; W2; Q2.3 |
| Correct GPU timing (synchronize, warmup, events) | W2; Q2.1 |
| Measure time per token vs sequence length *without* a cache (explain the growth) | W2 |
| KV cache: preallocated, per layer; logits match the no-cache path to 1e-4 | W3; Raschka KV cache article |
| KV cache memory: compute by hand, compare with `torch.cuda.memory_allocated` | W3; Q3.2 |
| Batched generation with padding and masks | Q3.7 |
| Profile with `torch.profiler`; identify one compute-, one memory-, one overhead-bound region | W3; Q3.4 |
| `torch.compile`, static KV cache, HIP graphs (`reduce-overhead`) | PyTorch "GPT, Fast" blog; W3 |

**Exit criteria**
- A table of tokens/s after each change (no cache → cache → compile → graphs), each with a one-line *why*
- TTFT and TPOT of your engine (batch 1) vs vLLM (batch 1), same model and prompts
- Decode tokens/s as % of the bandwidth-bound ceiling (bytes per token ÷ 640 GB/s)
- Workbook Q3.1–Q3.12 answered from memory

### v0.3: GPU kernels (HIP + Triton)
**Concepts:** threads, wavefronts, workgroups, CUs, occupancy, coalescing, LDS (shared memory), bank conflicts, tiling, arithmetic intensity, rooflines, reductions, wave-level shuffles, numerically stable softmax, kernel fusion.

| Scope | Reference |
|---|---|
| HIP: vector add, GELU (vs torch), via `load_inline` | PMPP ch 2–4; W4 |
| Transpose: naive → LDS-tiled → padded (bank conflicts) | PMPP ch 5–6; W5; Q5.5 |
| Matmul: naive → coalesced → tiled; compare with hipBLASLt | PMPP 5.4; W5 |
| Reduction: naive → LDS → wave shuffle; vs `torch.sum` | PMPP ch 10; W6 |
| Softmax kernel (row-wise, stable) and layer norm | W6; PMPP 20.3 |
| Triton: softmax, layer norm, a fused op (e.g. bias + GELU) | Triton tutorials 01–05; W8 |
| Plug your kernels into your GPT; outputs still match | W6, W8 |
| Roofline plot on *your* card: predicted vs measured for every kernel and one decode step | Scaling book ch 1, 4, 12; W7 |

**Exit criteria**
- Every kernel: correctness test + GB/s or GFLOP/s + % of peak, in one table (HIP vs Triton vs torch)
- Roofline chart with each kernel placed and explained
- A short "CUDA concepts → RDNA4" mapping in the write-up (warp → wavefront, shared memory → LDS, Tensor Cores → WMMA, Nsight → rocprof)
- Workbook Q4.x–Q8.x answered from memory

> Risk: Triton's AMD backend is tuned mainly for datacenter GPUs. If a Triton feature fails on gfx1201, document it (possible upstream issue) and continue in HIP.

### v0.4: FlashAttention and quantization
**Concepts:** online softmax, tiling Q/K/V through on-chip memory, why FlashAttention is exact, prefill vs decode attention shapes, split-K / Flash-Decoding, int8 weight-only quantization, per-tensor vs per-channel scales, the accuracy vs speed trade-off.

| Scope | Reference |
|---|---|
| Online softmax recurrence, by hand then in code | W9; Q9.2 |
| FlashAttention forward in Triton: non-causal, then causal | PMPP 20.5; FA paper; Triton tutorial 06; W9 |
| Benchmark vs SDPA for prefill shapes and decode shapes (q = 1); explain the decode result | W9; Q9.x |
| int8 weight-only quantization (per-channel, symmetric), dequant in the matmul | Grootendorst's quantization guide; W8; Q11.8 |
| Predict the decode speedup from int8 before measuring; explain the gap | Q8 "check yourself" |
| Accuracy check: perplexity on a held-out set, bf16 vs int8 | — |

**Exit criteria**
- FlashAttention output matches SDPA within tolerance; speed table for prefill and decode shapes
- int8 table: memory, tokens/s, perplexity, predicted vs measured speedup
- Workbook Q9.x, Q11.5, Q11.8, Q11.10 answered from memory

### v0.5: The serving engine
**Concepts:** static vs continuous (iteration-level) batching, PagedAttention and the block table, KV fragmentation, admission control, prefix caching, chunked prefill, streaming, TTFT vs TPOT trade-offs under load, why p99 rises while utilisation is low.

| Scope | Reference |
|---|---|
| Request queue + scheduler loop (static batching first) | Orca paper; W11 |
| Continuous batching: add/remove sequences every step | Orca; W12 |
| Block manager: fixed-size KV blocks (e.g. 16 tokens), block table per sequence, gather + SDPA (no custom kernel needed) | vLLM paper; `nano-vllm`; W12 |
| Prefix caching: hash full blocks; reuse on match | SGLang RadixAttention; W11 |
| Behaviour when KV memory runs out (reject, queue, or preempt; document which) | Q12.3 |
| OpenAI-compatible `/v1/completions` with streaming (SSE) | — |
| Load generator: Poisson arrivals at several request rates; record per-request TTFT/TPOT | — |
| Stretch: chunked prefill; preemption with recompute | W11; Q11.4 |

**Exit criteria**
- Under load: TTFT and TPOT **p50 / p95 / p99** and throughput vs request rate, for your engine and vLLM (same model, prompts, hardware)
- KV memory utilisation: contiguous allocation vs paged (show the fragmentation saving)
- Prefix-cache hit rate and TTFT saving on a shared-prefix workload
- A write-up: "the three biggest gaps between my engine and vLLM, and why"
- Workbook Q10.x–Q12.x answered from memory

### v1.0: Your own chat model, end to end
**Concepts:** instruction fine-tuning, prompt formatting, LoRA (low-rank adapters), evaluating instruction-following, serving a fine-tuned model.

| Scope | Reference |
|---|---|
| Implement LoRA as a PyTorch module yourself | Raschka App. E |
| Instruction-tune GPT-2 medium (355M) with LoRA | Raschka ch 7 |
| Evaluate with a fixed prompt set + a simple LLM-as-judge rubric | Raschka ch 7 |
| Merge the adapter and serve through **your engine** | — |
| Call it from a LangGraph agent through the OpenAI-compatible API | — |
| Final benchmark: your engine vs vLLM on the fine-tuned model | — |

**Exit criteria**
- Fine-tuned model beats the base model on the eval set (scores in README)
- Served by your engine, callable from LangGraph
- README: one table of key numbers from v0.1 → v1.0, plus the final engine-vs-vLLM comparison
- Blog post: the full journey, AMD-specific lessons included

## 7. Metrics and measurement protocol

### 7.1 Correctness (P0: every release)
| Check | Threshold |
|---|---|
| Your GPT-2 vs HF logits | max abs diff ≤ 1e-4 (fp32), documented tolerance for bf16 |
| Cached vs uncached generation | identical greedy tokens; logits ≤ 1e-4 |
| Every custom kernel vs PyTorch reference | documented `atol`/`rtol` per dtype |
| Engine output vs plain generation (greedy) | identical tokens |

### 7.2 Performance
| Metric | Definition |
|---|---|
| **TTFT** | request arrival → first token (queueing + prefill) |
| **TPOT** | mean time between subsequent tokens (one decode step) |
| **End-to-end latency** | ≈ TTFT + (N − 1) × TPOT |
| **Throughput** | output tokens/s summed over all requests |
| **% of bandwidth ceiling** | measured decode tokens/s ÷ (640 GB/s ÷ bytes read per token) |
| **% of peak (kernels)** | achieved GB/s or FLOP/s ÷ card peak |
| **KV memory utilisation** | used KV bytes ÷ allocated KV bytes |
| **Prefix-cache hit rate** | reused blocks ÷ total prompt blocks |

Report latency as **p50 / p95 / p99**, never as averages alone.

### 7.3 Protocol
- Fixed prompt sets in `bench/prompts/` (short, long, shared-prefix).
- Warm up before timing; synchronize the GPU; use GPU events for kernel timing.
- At least 5 runs per configuration; report median and spread.
- Every result row records the git commit and the versions from `docs/hardware.md`.
- **Predict first:** before each measurement, write the expected number and why in the release notes, then explain the difference.

## 8. Repo structure (suggested)

```
gpt_engine/
  model/        # attention, blocks, GPT, weight loading, export converter
  train/        # data, training loop, schedules, checkpoints
  generate/     # sampling, KV cache, generation loop
  kernels/      # hip/, triton/, tests and benchmarks per kernel
  attention/    # flash attention (triton)
  quant/        # int8 weight-only
  engine/       # scheduler, block manager, prefix cache, server (OpenAI API)
  lora/         # LoRA module, instruction tuning
bench/          # prompts, load generator, harness, plots
docs/           # hardware.md, releases/, vllm-rdna4-notes.md, roofline.md
tests/
README.md  ENGINE_REQUIREMENTS.md  pyproject.toml  Makefile
```

## 9. Upstream contribution track (runs in parallel)

| When | Step |
|---|---|
| v0.1 | Log every vLLM-on-RDNA4 setup issue in `docs/vllm-rdna4-notes.md` |
| After v0.3 | Reproduce open RDNA4 issues (e.g. vLLM #28649); post a clean reproduction with versions and numbers |
| After v0.4 | Docs PR (tested RDNA4 setup) and/or tuned gfx1201 kernel configs PR using vLLM's tuning scripts |
| Ongoing | Comment on the issue before coding to avoid duplicate work; keep PRs small with benchmarks |

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| ROCm/PyTorch setup issues on RDNA4 | Use the officially supported Ubuntu 24.04 + ROCm 7 combo; pin versions in `docs/hardware.md` |
| vLLM doesn't run well on gfx1201 | bf16 only (avoid FP8 paths); fall back to AMD's vLLM Docker image or a rented NVIDIA GPU for baseline numbers |
| Triton features unsupported on gfx1201 | Continue in HIP; document as a potential upstream issue |
| Profiler counters unavailable on consumer cards | Use timing + rooflines; note limitations |
| Scope creep / losing weeks | Strict exit criteria per release; stretch items only after exit criteria pass |
| Copying instead of learning | §3 rules; write core code from a blank file; workbook questions from memory |

## 11. Definition of done (v1.0)

All release exit criteria met; README shows the v0.1 → v1.0 numbers and the final comparison with vLLM; one blog post per release (or one combined post); every workbook question answered from memory at least once.
