# Copyright (c) 2025. LoVRR: LoRA-based Video Reasoning and Refinement
# Licensed under the BSD-3-Clause License.

"""
LoVRR: A multi-agent video understanding system with Chain-of-LoRA architecture.

Components:
- Planner: Analyzes questions and creates execution plans
- Grounder: Locates relevant video segments
- Refiner: Scores and refines temporal boundaries
- DeepThinker: Reasons over video evidence and generates final answers

Architecture:
- Single base model (Qwen2-VL) with dynamically switched LoRA adapters
- Multi-scale refinement for precise temporal grounding
- Planner, Grounder, Refiner, and DeepThinker share the model and processor
"""

__version__ = "1.0.0"
__author__ = "Anonymous"

from .constants import (
    PLANNER_PROMPT,
    GROUNDER_PROMPT,
    REFINER_PROMPT,
    REG_TOKEN,
    SEG_S_TOKEN,
    SEG_E_TOKEN,
)
