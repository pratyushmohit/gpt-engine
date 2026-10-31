# gpt-engine

**A GPT and its inference engine, built from scratch on an AMD RDNA4 consumer GPU.**

Build a GPT in PyTorch from a blank file, train a small one, load GPT-2 weights into it, and serve it on vLLM as a baseline. Then rebuild vLLM's core ideas one release at a time: sampling and KV cache → GPU kernels (HIP + Triton) → FlashAttention and quantization → a serving engine with paged KV cache and continuous batching → an instruction-tuned model served end to end.

Every release is measured against the previous one and against vLLM, on an AMD Radeon RX 9070 XT.

> This is a learning project. The goal is to deeply understand how an LLM is built, trained, and served, not only to have working code. The full design doc is in [`ENGINE_REQUIREMENTS.md`](ENGINE_REQUIREMENTS.md).

## Hardware and stack

| | |
|---|---|
| GPU | AMD Radeon RX 9070 XT (RDNA4, gfx1201), 16 GB GDDR6, 640 GB/s |
| OS | Ubuntu 24.04 LTS |
| GPU stack | ROCm 7.x, PyTorch (ROCm build) |
| Kernels | HIP, Triton (AMD backend) |
| Baselines | vLLM, llama.cpp, Hugging Face `transformers` |
| Tooling | Python, `uv`, `tiktoken` |

Exact versions are recorded in `docs/hardware.md`.

## Roadmap

| Release | Focus | Status |
|---|---|---|
| **v0.1** | GPT from scratch: attention, model, training on TinyStories, GPT-2 weight loading, HF export, vLLM baseline | 🚧 In progress |
| **v0.2** | Fast generation: sampling, KV cache, profiling, `torch.compile`, HIP graphs | ⏳ Planned |
| **v0.3** | GPU kernels in HIP and Triton: transpose, matmul, reductions, softmax, layer norm, fusion, rooflines | ⏳ Planned |
| **v0.4** | FlashAttention (Triton) and int8 weight-only quantization | ⏳ Planned |
| **v0.5** | Serving engine: continuous batching, paged KV cache, prefix caching, OpenAI-compatible streaming API | ⏳ Planned |
| **v1.0** | LoRA instruction-tuned GPT-2 medium, served end to end through this engine | ⏳ Planned |

Each release ships as a git tag with a write-up in `docs/releases/`.

## Results

_Numbers are added as each release lands._

| Metric | v0.1 | v0.2 | v0.3 | v0.4 | v0.5 | v1.0 | vLLM |
|---|---|---|---|---|---|---|---|
| Decode tokens/s (batch 1) | | | | | | | |
| TTFT (ms) | | | | | | | |
| TPOT (ms) | | | | | | | |
| % of bandwidth ceiling | | | | | | | |

## Measurement protocol

- Correctness first: GPT-2 logits match HF to ≤ 1e-4 (fp32); cached and uncached generation produce identical greedy tokens; every custom kernel is tested against a PyTorch reference.
- Warm up, synchronize, and time with GPU events. At least 5 runs per configuration; report median and spread.
- Latency is reported as p50 / p95 / p99, never averages alone.
- Every result records the git commit and software versions.
- Predict first: the expected number and the reasoning are written down before each measurement.

## Repo layout (planned)

```
gpt_engine/
  model/       attention, blocks, GPT, weight loading, export converter
  train/       data, training loop, schedules, checkpoints
  generate/    sampling, KV cache, generation loop
  kernels/     hip/, triton/, tests and benchmarks per kernel
  attention/   FlashAttention (Triton)
  quant/       int8 weight-only
  engine/      scheduler, block manager, prefix cache, OpenAI-compatible server
  lora/        LoRA module, instruction tuning
bench/         prompts, load generator, harness, plots
docs/          hardware.md, releases/, vllm-rdna4-notes.md, roofline.md
tests/
```

## Setup

```bash
uv sync
uv run main.py
```

## References

- Sebastian Raschka, *Build a Large Language Model (From Scratch)*, and [`rasbt/LLMs-from-scratch`](https://github.com/rasbt/LLMs-from-scratch)
- Andrej Karpathy, "Let's reproduce GPT-2 (124M)"
- Hwu, Kirk, El Hajj, *Programming Massively Parallel Processors*
- "How to Scale Your Model" (the scaling book)
- Papers: FlashAttention 1/2, Orca, vLLM (PagedAttention), SGLang (RadixAttention)
- Code: `nano-vllm`, Triton tutorials, vLLM, Mini-SGLang
