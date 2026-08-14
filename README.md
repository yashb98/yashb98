# Yash Bishnoi

ML engineer in Edinburgh. I reproduce language models bit-exact, then run the ablations the papers skip.

Most of my work is reproduction under a strict evidence gate: a claim doesn't count until the number traces back to a file and survives an adversarial check. Currently working through inference-time kernels, quantisation, and `torch.compile` internals.

---

### Now

- **[BuildFromScratch](https://github.com/yashb98/BuildFromScratch)** — single-file PyTorch reproductions of SmolLM2-135M and Qwen3-0.6B-Base, both at `max|Δlogits| = 0.0` against the reference weights. Includes a three-way architecture study (faithful / modernised / partial-RoPE) trained at matched compute.
- **[KaggleLab](https://github.com/yashb98/KaggleLab)** — small-scale paper reproductions with a blocking traceability gate: every published number is parsed from an attached evidence dataset, every headline claim gets an independent adversarial pass.
- **[multi-agent-patterns](https://github.com/yashb98/multi-agent-patterns)** — orchestration patterns for long-running agent systems.

### Open source

Patches to the libraries I use daily — kernels, quantisation dispatch, and compiler internals.

| Project | Contribution | Status |
| --- | --- | --- |
| [pytorch/pytorch](https://github.com/pytorch/pytorch) | `[dynamo]` Split `FrozensetVariable` off `SetVariable` to match CPython semantics (+421 / −279) | In review |
| [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel) | Fix FLCE `torch.compile` failure from the `aten.addmm.dtype_out` overload | In review |
| [bitsandbytes-foundation/bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) | Add an sm121 (GB10) tier to the 4-bit GEMM dispatch heuristic | In review |

### Stack

`PyTorch` · `Triton` · `CUDA` · `torch.compile` / Dynamo · `Transformers` · `TRL` · `LoRA` / Unsloth · `lm-evaluation-harness` · `TypeScript`

### Elsewhere

[yashbishnoi.io](https://yashbishnoi.io/) · [LinkedIn](https://linkedin.com/in/yash-bishnoi)
