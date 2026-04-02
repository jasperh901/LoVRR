#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepThinker FastAPI Service

This service wraps DeepThinker model for inference, accepting video clips and questions,
returning reasoning process and answers.

Usage:
    conda activate deepthinker
    python deepthinker_service.py --config config.yaml
"""

import argparse
import logging
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Optional

import torch
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DeepThinkerService")


class InferenceRequest(BaseModel):
    """Request model for video inference"""
    video_path: str = Field(..., description="Absolute path to the video file")
    question: str = Field(..., description="Question about the video")
    start_time: Optional[float] = Field(None, description="Start time in seconds (optional)")
    end_time: Optional[float] = Field(None, description="End time in seconds (optional)")
    problem_type: str = Field("free-form", description="Type: multiple choice, numerical, OCR, free-form, regression")
    max_frames: int = Field(32, description="Maximum number of frames to process")
    max_new_tokens: int = Field(1024, description="Maximum new tokens to generate")


class InferenceResponse(BaseModel):
    """Response model for video inference"""
    success: bool
    thinking: Optional[str] = None
    answer: Optional[str] = None
    raw_output: Optional[str] = None
    error: Optional[str] = None


class DeepThinkerService:
    """DeepThinker inference service wrapper"""
    
    def __init__(self, config: dict):
        self.config = config
        self.model_path = config.get("model_path", "DeepThinker/DeepThinker-7B")
        self.device = config.get("device", "cuda")
        self.use_vllm = config.get("use_vllm", True)
        self.tensor_parallel_size = config.get("tensor_parallel_size", 1)
        self.gpu_memory_utilization = config.get("gpu_memory_utilization", 0.95)
        self.max_model_len = config.get("max_model_len", 8192)
        self.enforce_eager = config.get("enforce_eager", True)
        
        # Sampling parameters
        self.temperature = config.get("temperature", 0.1)
        self.top_p = config.get("top_p", 0.001)
        
        self.llm = None
        self.processor = None
        self.tokenizer = None
        
        # Prompt templates
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
        
    def initialize(self):
        """Initialize the model and processor"""
        logger.info(f"Initializing DeepThinker model from: {self.model_path}")
        logger.info(f"Using vLLM: {self.use_vllm}")
        
        if self.use_vllm:
            self._init_vllm()
        else:
            self._init_transformers()
            
        logger.info("DeepThinker model initialized successfully")
        
    def _init_vllm(self):
        """Initialize using vLLM for faster inference"""
        from vllm import LLM, SamplingParams
        from transformers import AutoProcessor, AutoTokenizer
        
        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            enforce_eager=self.enforce_eager,
            limit_mm_per_prompt={"video": 1, "image": 1},
            trust_remote_code=True,
        )
        
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=1024,
        )
        
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        self.processor.tokenizer = self.tokenizer
        
    def _init_transformers(self):
        """Initialize using transformers for flexible inference"""
        from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration
        
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        self.processor.tokenizer = self.tokenizer
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        
    def parse_response(self, output: str) -> tuple[Optional[str], Optional[str]]:
        """Parse thinking and answer from model output"""
        thinking = None
        answer = None
        
        # Extract thinking
        think_match = re.search(r'<think>(.*?)</think>', output, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
            
        # Extract answer
        answer_match = re.search(r'<answer>(.*?)</answer>', output, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
        else:
            # Fallback: if no answer tags, use the text after </think>
            if '</think>' in output:
                answer = output.split('</think>')[-1].strip()
            else:
                answer = output.strip()
                
        return thinking, answer
    
    def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Run inference on a video"""
        try:
            # Validate video path
            if not os.path.exists(request.video_path):
                return InferenceResponse(
                    success=False,
                    error=f"Video file not found: {request.video_path}"
                )
            
            # Build message content
            video_content = {
                "type": "video",
                "video": request.video_path,
                "max_pixels": 200704,  # max pixels per frame
                "nframes": request.max_frames,
            }
            
            # Add time range if specified
            if request.start_time is not None:
                video_content["video_start"] = request.start_time
            if request.end_time is not None:
                video_content["video_end"] = request.end_time
                
            # Build prompt
            prompt_text = self.question_template.format(Question=request.question)
            prompt_text += self.type_templates.get(request.problem_type, self.type_templates["free-form"])
            
            messages = [{
                "role": "user",
                "content": [
                    video_content,
                    {"type": "text", "text": prompt_text}
                ]
            }]
            
            # Generate response
            if self.use_vllm:
                output = self._infer_vllm(messages, request.max_new_tokens)
            else:
                output = self._infer_transformers(messages, request.max_new_tokens)
                
            # Parse response
            thinking, answer = self.parse_response(output)
            
            return InferenceResponse(
                success=True,
                thinking=thinking,
                answer=answer,
                raw_output=output
            )
            
        except Exception as e:
            logger.error(f"Inference error: {str(e)}", exc_info=True)
            return InferenceResponse(
                success=False,
                error=str(e)
            )
    
    def _infer_vllm(self, messages: list, max_new_tokens: int) -> str:
        """Run inference using vLLM"""
        from qwen_vl_utils import process_vision_info
        
        # Convert to prompt string
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Process video input
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        
        # Prepare vllm input - only use supported kwargs
        # vLLM doesn't support all kwargs from qwen_vl_utils, filter them
        SUPPORTED_MM_KWARGS = {"fps", "min_pixels", "max_pixels", "nframes", "video_start", "video_end"}
        mm_kwargs = {}
        if video_kwargs:
            for key, val in video_kwargs.items():
                if key not in SUPPORTED_MM_KWARGS:
                    continue
                # Handle both list and non-list values
                if isinstance(val, (list, tuple)) and len(val) > 0:
                    mm_kwargs[key] = val[0]
                else:
                    mm_kwargs[key] = val
        
        llm_inputs = [{
            "prompt": prompt,
            "multi_modal_data": {"video": video_inputs[0]} if video_inputs else {},
            "mm_processor_kwargs": mm_kwargs,
        }]
        
        # Update sampling params with max_new_tokens
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=max_new_tokens,
        )
        
        # Run inference
        outputs = self.llm.generate(llm_inputs, sampling_params=sampling_params)
        output_text = outputs[0].outputs[0].text
        
        return output_text
    
    def _infer_transformers(self, messages: list, max_new_tokens: int) -> str:
        """Run inference using transformers"""
        from qwen_vl_utils import process_vision_info
        
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(self.model.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
            )
        
        output_ids = output_ids[:, inputs.input_ids.shape[1]:]
        output_text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        
        return output_text


# Global service instance
service: Optional[DeepThinkerService] = None

# FastAPI app
app = FastAPI(
    title="DeepThinker Inference Service",
    description="Video reasoning service powered by DeepThinker model",
    version="1.0.0"
)


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global service
    if service is not None:
        service.initialize()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": service is not None and service.llm is not None}


@app.post("/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest):
    """Run inference on a video"""
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return service.infer(request)


@app.post("/answer", response_model=InferenceResponse)
async def answer(request: InferenceRequest):
    """Alias for /infer endpoint for compatibility"""
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return service.infer(request)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="DeepThinker Inference Service")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--host", type=str, default=None, help="Host to bind (overrides config)")
    parser.add_argument("--port", type=int, default=None, help="Port to bind (overrides config)")
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        # Use default config
        config = {
            "model_path": "models/DeepThinker-7B",
            "use_vllm": True,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.90,
            "max_model_len": 32768,
            "enforce_eager": False,
            "temperature": 0.1,
            "top_p": 0.001,
            "host": "127.0.0.1",
            "port": 8002,
        }
        logger.warning(f"Config file not found: {config_path}, using default config")
    else:
        config = load_config(str(config_path))
        config = config.get("deepthinker_service", config)
    
    # Override with command line args
    host = args.host or config.get("host", "127.0.0.1")
    port = args.port or config.get("port", 8002)
    
    # Initialize service
    global service
    service = DeepThinkerService(config)
    
    logger.info(f"Starting DeepThinker service on {host}:{port}")
    
    # Run server
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
