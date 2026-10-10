<div align="center">

# LoVRR: Reinforcement Learning-Enhanced Multi-Agent Video Localization and Reasoning

<p>
  <a href="#license"><img src="https://img.shields.io/badge/License-BSD--3--Clause-green" alt="License"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11+-blue?logo=python" alt="Python"></a>
</p>

[**Project Page**](https://jasperh901.github.io/LoVRR/)

</div>


## Abstract

Complex video localization and reasoning requires cross-modal semantic alignment alongside precise temporal grounding over highly sparse timelines. Existing large language models often rely on superficial one-shot visual judgments and lack deep reasoning capabilities for complex temporal tasks, leading to localization and reasoning errors. In this paper, we propose **LoVRR** (**Lo**calize–**V**erify–**R**efine–**R**eason), a reinforcement learning-enhanced multimodal video reasoning agent. LoVRR makes two core contributions. First, we innovatively introduce temporal reward-based reinforcement learning into a multi-agent system, endowing lightweight Chain-of-Thought models with deliberative reasoning capabilities for precise temporal localization and inference in complex videos. Second, we propose **VG-MSR**, a novel LLM-guided self-refinement algorithm that performs multi-scale re-localization under semantic grounding accuracy rewards, effectively improving localization precision and robustness while reducing computational overhead. Extensive experiments on Grounded VideoQA and other benchmarks demonstrate that LoVRR achieves state-of-the-art performance on complex video understanding tasks.

## Overview

LoVRR follows a progressive *localize first, then think deeply* paradigm. The system first narrows the temporal search space to query-relevant evidence clips, and only then applies expensive deep reasoning to refined evidence. This design reduces distraction from long-video noise, makes temporal evidence selection explicit, and concentrates reasoning capacity on the most informative parts of the video.

<p align="center">
  <img src="lovrr/figs/lovrr_moti.png" width="80%" alt="Motivation">
</p>

### Architecture

The framework consists of four specialized roles built on a unified Qwen2-VL backbone with **Chain-of-LoRA** adapters:

<p align="center">
  <img src="lovrr/figs/LoVRR_archi.png" width="95%" alt="LoVRR Architecture">
</p>

| Role | Function |
|------|----------|
| **Planner** | Structured planning and query rewriting; routes to grounding or direct-answering paths |
| **Grounder** | Generates top-*k* candidate temporal segments via a timestamp decoder |
| **Refiner** | Binary confidence scoring, candidate reranking, and VG-MSR multi-scale boundary refinement |
| **DeepThinker** | Reinforcement learning-enhanced deep reasoning|

### Key Components

- **Chain-of-LoRA (CoL):** Role-specific LoRA adapters on a single backbone; enables seamless role switching with minimal parameter overhead.
- **VG-MSR:** Refiner-Guided Multi-Scale Search Refinement — treats the Refiner as a queryable critic and performs iterative multi-scale local search (8 directions × 3 step sizes) around initial proposals with explicit budget and risk control.
- **T-GRPO:** Temporally-aware Group Relative Policy Optimization — contrasts temporally ordered vs. shuffled frame sequences to reward temporal-order dependency and suppress shortcut reasoning.

## Project Structure

```
LoVRR/
├── lovrr/
│   ├── model/                      # AgentQwen2VL backbone and LoRA adapters
│   ├── inference/                  # Planner, Grounder, Refiner, DeepThinker
│   │   ├── roles.py                # Shared model and role switching
│   │   ├── pipeline.py             # Localize, verify, refine, reason
│   │   ├── deepthinker.py           # Reasoning and answer generation
│   │   ├── temporal.py             # Temporal localization and scoring
│   │   └── refinement.py           # VG-MSR refinement
│   ├── dataset/                    # Dataset loaders and collators
│   ├── train/                      # Training scripts
│   ├── utils/
│   ├── serve.py                    # HTTP API
│   ├── infer.py                    # Command-line inference
│   └── constants.py
├── eval/                           # Benchmark evaluation
├── config.yaml                     # Model, adapters, inference, server
├── requirements.txt                # Dependencies for all roles
├── start_all.sh
└── start_lovrr.sh
```

## Installation

### Prerequisites

- Python 3.11
- CUDA 12.0+
- A GPU with sufficient memory for the Qwen2-VL backbone and video inputs

### Environment Setup

All four roles run in `lovrr_env` and use the same dependencies.

```bash
git clone https://github.com/jasperh901/LoVRR.git
cd LoVRR
conda create -n lovrr_env python=3.11 -y
conda activate lovrr_env
pip install -r requirements.txt
pip install -e .
```

## Quick Start

Set the checkpoint and adapter paths in `config.yaml`. The model directory contains the Qwen2-VL processor, LoVRR configuration and localization weights, with role adapters in `planner/`, `grounder/`, `refiner/`, and `deepthinker/`.

```yaml
model:
  model_path: "/path/to/LoVRR-7B"
  base_model_path: "/path/to/Qwen2-VL-7B-Instruct"
  device: "auto"
  adapters:
    planner: "/path/to/LoVRR-7B/planner"
    grounder: "/path/to/LoVRR-7B/grounder"
    refiner: "/path/to/LoVRR-7B/refiner"
    deepthinker: "/path/to/LoVRR-7B/deepthinker"

server:
  host: "127.0.0.1"
  port: 8000
```

Planner, Grounder, Refiner, and DeepThinker switch adapters on one loaded Qwen2-VL model and share its processor and video preprocessing. The pipeline localizes relevant clips, verifies and refines their boundaries, then generates reasoning and an answer.

### Command-Line Inference

```bash
conda activate lovrr_env
python -m lovrr.infer --config config.yaml \
    --video /path/to/video.mp4 \
    --question "What happens after the person enters the room?"
```

### Service

```bash
./start_all.sh --config config.yaml
```

The same service can be started with `./start_lovrr.sh` or `python -m lovrr.serve --config config.yaml`.

```bash
curl http://127.0.0.1:8000/answer \
    -H 'Content-Type: application/json' \
    -d '{"video_path":"/path/to/video.mp4","question":"What happens after the person enters the room?"}'
```

Responses include the evidence span, reasoning, and final answer. `/health` reports model readiness, and `/infer` accepts the same requests as `/answer`.

## Training

### Stage 1: Chain-of-LoRA SFT (Planner + Grounder + Refiner)

```bash
conda activate lovrr_env
deepspeed lovrr/train/train.py \
    --model_name_or_path /path/to/Qwen2-VL-7B-Instruct \
    --datasets /path/to/sft_data \
    --role planner \
    --output_dir outputs/planner_lora \
    --deepspeed ds_config.json
```

Repeat with `--role grounder` and `--role refiner` to train each adapter.

### Stage 2: DeepThinker (SFT Cold Start → T-GRPO RL)

DeepThinker training follows a two-stage recipe:
1. **SFT Cold Start:** Fine-tune on long-chain reasoning supervision data.
2. **T-GRPO Reinforcement Learning:** Optimize with temporally-aware group relative policy optimization using temporal-order and answer-correctness rewards.

Please refer to the training scripts in `lovrr/train/` for detailed configurations.

## Demo

<p align="center">
  <img src="lovrr/figs/lovrr_demo.png" width="90%" alt="LoVRR Demo">
</p>

## Evaluation

Run benchmarks from the repository root in `lovrr_env`. Each worker shares one model across its roles.

### CG-Bench (Long Video, Grounded VideoQA)

```bash
conda activate lovrr_env
python -m eval.infer_cgbench_enhanced \
    --model_path /path/to/LoVRR-7B \
    --worker_id 0 --num_workers 1 --gpu 0 \
    --output_dir outputs/cgbench
python -m eval.aggregate_cgbench_results --output_dir outputs/cgbench
```

### MVBench (General Video QA)

```bash
python -m eval.infer_mvbench_enhanced \
    --model_path /path/to/LoVRR-7B \
    --base_model_path /path/to/Qwen2-VL-7B-Instruct \
    --pred_path outputs/mvbench \
    --use_planner --use_refinement --use_deepthinker
```

### Other Benchmarks

Evaluation on Video-MME, LVBench, NExT-GQA, and Charades-STA follows a similar pattern. See the `eval/` directory for details.

## VG-MSR Refinement Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `refine_iters` | 3 | Maximum refinement iterations |
| `budget` | 32 | Maximum Refiner evaluations per sample |
| `step_ratio` | 0.2 | Initial step size = segment_length × 0.2 |
| `step_min` | 1.0 s | Minimum step size |
| `step_decay` | 0.7 | Step decay factor per iteration |
| `skip_high` | 0.85 | Skip refinement if confidence ≥ 0.85 |
| `skip_low` | 0.15 | Skip refinement if confidence ≤ 0.15 |
| `gain` | 0.01 | Minimum improvement threshold to accept |

## Main Results

We evaluate LoVRR on **seven benchmarks** across three task categories.

### General Video QA

<table>
<tr><td>

**MVBench**

| Method | Size | Avg. |
|--------|------|------|
| *GPT-4V* | *—* | *43.5* |
| VideoChat2 | 7B | 60.4 |
| VideoMind | 7B | 64.8 |
| LLaVA-ST | 7B | 64.2 |
| **LoVRR** | **7B** | **67.6** |

</td><td>

**Video-MME**

| Method | Size | All | Long |
|--------|------|-----|------|
| *GPT-4o* | *—* | *71.9* | *65.3* |
| LongVA | 7B | 52.6 | 46.2 |
| VideoMind | 7B | 58.2 | 49.2 |
| Video-R1 | 7B | 59.3 | — |
| Apollo | 7B | 61.3 | — |
| **LoVRR** | **7B** | **63.0** | **53.7** |

</td><td>

**LVBench**

| Method | Size | Overall |
|--------|------|---------|
| *GPT-4o* | *—* | *30.8* |
| PLLaVA | 34B | 26.1 |
| **LoVRR** | **7B** | **40.0** |

</td></tr>
</table>

### Grounded VideoQA

<table>
<tr><td>

**CG-Bench**

| Method | Size | long-acc. | mIoU |
|--------|------|-----------|------|
| *GPT-4o* | *—* | *45.2* | *5.62* |
| VideoMind | 7B | 37.5 | 6.56 |
| Qwen2-VL | 72B | 41.3 | 3.58 |
| InternVL2 | 78B | 42.2 | 3.91 |
| **LoVRR** | **7B** | **37.8** | **6.65** |

</td><td>

**NExT-GQA**

| Method | Size | R@0.3 | mIoU | Acc@GQA |
|--------|------|-------|------|---------|
| HawkEye | 7B | 37.0 | 25.7 | — |
| VChat-TPO | 7B | 41.2 | 27.7 | 25.5 |
| **LoVRR** | **7B** | **47.6** | **30.1** | **24.6** |

</td></tr>
</table>

### Video Temporal Grounding — Charades-STA (Zero-Shot)

| Method | Size | FT | R@0.3 | R@0.5 | R@0.7 | mIoU |
|--------|------|----|-------|-------|-------|------|
| *R²-Tuning* | *—* | *✓* | *70.9* | *59.8* | *37.0* | *50.9* |
| E.T. Chat | 4B | ✗ | 65.7 | 45.9 | 20.0 | 42.3 |
| VideoChat-Flash | 7B | ✗ | — | — | — | 47.2 |
| VideoMind | 7B | ✗ | 73.5 | 58.4 | 29.8 | 49.6 |
| **LoVRR** | **7B** | **✗** | **74.8** | **59.0** | **31.7** | **50.6** |

## License

This project is released under the [BSD 3-Clause License](LICENSE).

## Acknowledgments

This project is built upon several excellent open-source works. We sincerely thank the authors for their contributions:

- **[VideoMind](https://github.com/yeliudev/VideoMind)**
- **[Video-R1](https://github.com/tulerfeng/Video-R1)**
- **[Qwen2-VL](https://github.com/QwenLM/Qwen2-VL)**
- **[PEFT](https://github.com/huggingface/peft)**
- **[DeepSpeed](https://github.com/microsoft/DeepSpeed)**
