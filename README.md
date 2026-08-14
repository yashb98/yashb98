<p align="center">
  <img src="https://raw.githubusercontent.com/yashb98/yashb98/main/assets/header.svg" alt="Yash Bishnoi — ML Engineer, Edinburgh" width="100%">
</p>

Most of my work is reproduction under a strict evidence gate: a claim doesn't count until the number traces back to a file and survives an adversarial check. Currently working through inference-time kernels, quantisation, and `torch.compile` internals.

---

### Now

<details open>
<summary><b>BuildFromScratch</b> — single-file LM reproductions at bit-exact parity</summary>

<br>

SmolLM2-135M and Qwen3-0.6B-Base rebuilt as single-file PyTorch models, verified against the reference weights, then used as a controlled base for architecture ablations at matched compute.

| Check | Result |
| --- | --- |
| SmolLM2-135M parity vs HuggingFace | `max\|Δlogits\| = 0.0` |
| Qwen3-0.6B-Base parity vs HuggingFace | `max\|Δlogits\| = 0.0` |
| Qwen3 faithful baseline | ~2.14× reference, ~275,000× less training data |
| Modernised IMU-1 bundle @ matched compute | **+18%** over the faithful baseline (23.52 vs 28.65) |

Three builds — faithful, modernised, exploratory partial-RoPE — each with parity scripts, training pipelines and an `lm-evaluation-harness` wrapper.

**→ [github.com/yashb98/BuildFromScratch](https://github.com/yashb98/BuildFromScratch)**

</details>

<details>
<summary><b>KaggleLab</b> — paper reproductions behind a traceability gate</summary>

<br>

Small-scale reproductions published on a fixed cadence, with three rules that don't bend:

- **Traceability** — every published number is parsed from a file in an attached evidence dataset. No number without a source.
- **Adversarial re-check** — every headline claim gets an independent pass that tries to break it.
- **Visible scorecard** — each post states its claim class, seeds, comparability, eval status, and *what it does not show*.

**→ [github.com/yashb98/KaggleLab](https://github.com/yashb98/KaggleLab)**

</details>

<details>
<summary><b>multi-agent-patterns</b> — orchestration for long-running agent systems</summary>

<br>

Patterns for agent systems that run for hours rather than turns: JobPulse daily automation and MindGraph knowledge visualisation as the working examples.

**→ [github.com/yashb98/multi-agent-patterns](https://github.com/yashb98/multi-agent-patterns)**

</details>

### Open source

Patches to the libraries I use daily — kernels, quantisation dispatch, and compiler internals.

| Project | Contribution | Status |
| --- | --- | --- |
| [pytorch/pytorch](https://github.com/pytorch/pytorch) | `[dynamo]` Split `FrozensetVariable` off `SetVariable` to match CPython semantics (+421 / −279) | In review |
| [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel) | Fix FLCE `torch.compile` failure from the `aten.addmm.dtype_out` overload | In review |
| [bitsandbytes-foundation/bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) | Add an sm121 (GB10) tier to the 4-bit GEMM dispatch heuristic | In review |

### Stack

<img src="https://raw.githubusercontent.com/yashb98/yashb98/main/assets/stack.svg" alt="PyTorch, Triton, CUDA, torch.compile, Dynamo, Transformers, TRL, LoRA, Unsloth, lm-eval-harness, TypeScript" width="100%">

<details>
<summary><b>Also built</b></summary>

<br>

| Project | What it is |
| --- | --- |
| [LetsFineTune](https://github.com/yashb98/LetsFineTune) | Fine-tuned LMs with LoRA, Unsloth and Transformers — configs, scripts, evals |
| [Velox_AI](https://github.com/yashb98/Velox_AI) | Voice-agent platform for real phone calls: Twilio + Gemini + Deepgram, hybrid RAG |
| [Prism](https://github.com/yashb98/Prism) | Research agent decomposing million-token corpora into verifiable reasoning chains |
| [90-days-of-ml](https://github.com/yashb98/90-days-of-ml) | 30+ projects across core ML, MLOps on AWS, and production capstones |
| [Deep-Learning-for-Facial-3D-Reconstruction](https://github.com/yashb98/Deep-Learning-for-Facial-3D-Reconstruction---Simulator) | MSc project: PCA normalisation, 1,200-image rotation sweep, SixDRepNet pose estimation |

</details>

### Elsewhere

[yashbishnoi.io](https://yashbishnoi.io/) · [LinkedIn](https://linkedin.com/in/yash-bishnoi)
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
