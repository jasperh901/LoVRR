import json
import logging

import torch

from lovrr.constants import PLANNER_PROMPT
from lovrr.dataset.utils import process_vision_info
from lovrr.utils.io import get_duration
from lovrr.utils.parser import parse_query, parse_span

from .deepthinker import DeepThinker
from .refinement import refine_span_with_multiscale
from .roles import RoleModel
from .temporal import run_grounder, run_refiner_rerank_top5


logger = logging.getLogger(__name__)


class LoVRR:
    def __init__(self, roles, inference=None, refinement=None):
        self.roles = roles
        self.model = roles.model
        self.processor = roles.processor
        self.inference = inference or {}
        self.refinement = refinement or {}
        self.deepthinker = DeepThinker(
            roles,
            temperature=self.inference.get("temperature", 0.1),
            top_p=self.inference.get("top_p", 0.001),
        )

    @classmethod
    def from_config(cls, config):
        model = config["model"]
        roles = RoleModel.from_pretrained(
            model["model_path"],
            role_paths=model.get("adapters"),
            device=model.get("device", "auto"),
            base_model_path=model.get("base_model_path"),
        )
        return cls(roles, config.get("inference"), config.get("refinement"))

    def plan(self, video_path, question):
        messages = [{"role": "user", "content": [{
            "type": "video", "video": video_path,
            "min_pixels": 36 * 28 * 28, "max_pixels": 64 * 28 * 28,
            "max_frames": 100, "fps": 1.0,
        }, {"type": "text", "text": PLANNER_PROMPT.format(question)}]}]
        with self.roles.activate("planner"), torch.inference_mode():
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            images, videos = process_vision_info(messages)
            inputs = self.processor(text=[text], images=images, videos=videos, return_tensors="pt")
            inputs = inputs.to(self.roles.device)
            output = self.model.generate(
                **inputs, do_sample=False, temperature=None, top_p=None,
                top_k=None, repetition_penalty=None, max_new_tokens=256,
            )
            output = output[0, inputs.input_ids.shape[1]:]
            response = self.processor.decode(output, skip_special_tokens=True)
        query, do_grounding = question, True
        try:
            plan = json.loads(response)
            action = plan[0] if isinstance(plan, list) else plan
            if action["type"].lower() == "grounder" and action.get("value"):
                query = action["value"]
            elif action["type"].lower() == "answerer":
                do_grounding = False
        except (ValueError, KeyError, IndexError, TypeError):
            logger.warning("Failed to parse planner response")
        return query, do_grounding, response

    def answer(self, video_path, question, problem_type=None, max_frames=None, max_new_tokens=None):
        with self.roles.lock:
            duration = get_duration(video_path)
            query, do_grounding, planner_response = self.plan(video_path, question)
            span = [0, duration]
            details = {"planner_response": planner_response, "query": query}
            if do_grounding:
                query = parse_query(query)
                with self.roles.activate("grounder"):
                    preds, confs, response, _ = run_grounder(
                        self.model, self.processor, self.roles.device,
                        video_path, query, duration, logger,
                    )
                details["grounder_response"] = response
                with self.roles.activate("refiner"):
                    ranked, _, scores, sorted_scores, _ = run_refiner_rerank_top5(
                        self.model, self.processor, self.roles.device,
                        video_path, query, duration, preds, confs, logger,
                    )
                    span, score = ranked[0], sorted_scores[0]
                    info = refine_span_with_multiscale(
                        self.model, self.processor, self.roles.device,
                        video_path, query, duration, span, base_score=score,
                        logger=logger, **self.refinement,
                    )
                span = info["final_span"]
                details.update(refiner_scores=scores, refinement=info)
            span = parse_span(span, duration, 32)
            response = self.deepthinker.answer(
                video_path, question, start_time=span[0], end_time=span[1],
                problem_type=problem_type or self.inference.get("problem_type", "free-form"),
                max_frames=max_frames if max_frames is not None else self.inference.get("max_frames", 32),
                max_new_tokens=max_new_tokens if max_new_tokens is not None else self.inference.get("max_new_tokens", 2048),
            )
            return {
                "success": response.success, "span": list(span), "thinking": response.thinking,
                "answer": response.answer, "raw_output": response.raw_output,
                "error": response.error, **details,
            }
