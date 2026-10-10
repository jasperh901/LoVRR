import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Optional

import torch

from lovrr.dataset.utils import process_vision_info


logger = logging.getLogger(__name__)


@dataclass
class DeepThinkerResponse:
    success: bool
    thinking: Optional[str] = None
    answer: Optional[str] = None
    raw_output: Optional[str] = None
    error: Optional[str] = None
    latency: float = 0.0


class DeepThinker:
    def __init__(self, roles, temperature=0.1, top_p=0.001):
        self.roles = roles
        self.model = roles.model
        self.processor = roles.processor
        self.temperature = temperature
        self.top_p = top_p
        self.question_template = (
            "{Question}\n"
            "Please think about this question as if you were a human pondering deeply. "
            "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
            "It's encouraged to include self-reflection or verification in the reasoning process. "
            "Provide your detailed reasoning between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags."
        )
        self.type_templates = {
            "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
            "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
            "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
            "free-form": " Please provide your text answer within the <answer> </answer> tags.",
            "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags."
        }

    def parse_response(self, output):
        thinking = None
        think_match = re.search(r'<think>(.*?)</think>', output, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
        answer_match = re.search(r'<answer>(.*?)</answer>', output, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
        elif '</think>' in output:
            answer = output.split('</think>')[-1].strip()
        else:
            answer = output.strip()
        return thinking, answer

    def answer(self, video_path, question, start_time=None, end_time=None,
               problem_type="free-form", max_frames=32, max_new_tokens=1024):
        started = time.monotonic()
        try:
            paths = video_path if isinstance(video_path, (list, tuple)) else [video_path]
            if not paths or not all(os.path.isfile(path) for path in paths):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            video = {
                "type": "video",
                "video": video_path,
                "max_pixels": 200704,
                "nframes": max_frames,
            }
            if start_time is not None:
                video["video_start"] = start_time
            if end_time is not None:
                video["video_end"] = end_time
            prompt = self.question_template.format(Question=question)
            prompt += self.type_templates.get(problem_type, self.type_templates["free-form"])
            messages = [{"role": "user", "content": [video, {"type": "text", "text": prompt}]}]
            with self.roles.activate("deepthinker"), torch.inference_mode():
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                images, videos = process_vision_info(messages)
                inputs = self.processor(text=[text], images=images, videos=videos, return_tensors="pt", padding=True)
                inputs = inputs.to(self.roles.device)
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                output_ids = output_ids[:, inputs.input_ids.shape[1]:]
                output = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            thinking, answer = self.parse_response(output)
            return DeepThinkerResponse(True, thinking, answer, output, latency=time.monotonic() - started)
        except Exception as error:
            logger.exception("Inference failed")
            return DeepThinkerResponse(False, error=str(error), latency=time.monotonic() - started)

    def infer(self, **kwargs):
        return self.answer(**kwargs)
