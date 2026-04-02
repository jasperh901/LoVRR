#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepThinker Client

Client library for communicating with DeepThinker service from LoVRR.

Usage:
    from deepthinker_client import DeepThinkerClient
    
    client = DeepThinkerClient("http://127.0.0.1:8002")
    result = client.answer(
        video_path="/path/to/video.mp4",
        question="What is happening in the video?",
        start_time=10.0,
        end_time=30.0
    )
    print(result.answer)
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger("DeepThinkerClient")


@dataclass
class DeepThinkerResponse:
    """Response from DeepThinker service"""
    success: bool
    thinking: Optional[str] = None
    answer: Optional[str] = None
    raw_output: Optional[str] = None
    error: Optional[str] = None
    latency: float = 0.0


class DeepThinkerClient:
    """Client for DeepThinker inference service"""
    
    def __init__(
        self,
        service_url: str = "http://127.0.0.1:8002",
        timeout: float = 300.0,  # 5 minutes default timeout for video inference
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize DeepThinker client.
        
        Args:
            service_url: URL of the DeepThinker service
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries on failure
            retry_delay: Delay between retries in seconds
        """
        self.service_url = service_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
    def health_check(self) -> bool:
        """Check if the service is healthy"""
        try:
            response = requests.get(
                f"{self.service_url}/health",
                timeout=5.0
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False
    
    def wait_for_service(self, timeout: float = 60.0, interval: float = 2.0) -> bool:
        """
        Wait for the service to become available.
        
        Args:
            timeout: Maximum time to wait in seconds
            interval: Check interval in seconds
            
        Returns:
            True if service is available, False if timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.health_check():
                logger.info("DeepThinker service is available")
                return True
            time.sleep(interval)
        logger.error(f"DeepThinker service not available after {timeout}s")
        return False
    
    def answer(
        self,
        video_path: str,
        question: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        problem_type: str = "free-form",
        max_frames: int = 32,
        max_new_tokens: int = 1024,
    ) -> DeepThinkerResponse:
        """
        Get answer from DeepThinker for a video question.
        
        Args:
            video_path: Absolute path to the video file
            question: Question about the video
            start_time: Start time in seconds (optional, for clip)
            end_time: End time in seconds (optional, for clip)
            problem_type: Type of problem (free-form, multiple choice, etc.)
            max_frames: Maximum frames to process
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            DeepThinkerResponse with thinking, answer, and metadata
        """
        request_data = {
            "video_path": video_path,
            "question": question,
            "problem_type": problem_type,
            "max_frames": max_frames,
            "max_new_tokens": max_new_tokens,
        }
        
        if start_time is not None:
            request_data["start_time"] = start_time
        if end_time is not None:
            request_data["end_time"] = end_time
            
        start = time.time()
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.service_url}/answer",
                    json=request_data,
                    timeout=self.timeout
                )
                
                latency = time.time() - start
                
                if response.status_code == 200:
                    data = response.json()
                    return DeepThinkerResponse(
                        success=data.get("success", False),
                        thinking=data.get("thinking"),
                        answer=data.get("answer"),
                        raw_output=data.get("raw_output"),
                        error=data.get("error"),
                        latency=latency
                    )
                else:
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    logger.warning(f"Request failed (attempt {attempt + 1}): {error_msg}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout (attempt {attempt + 1})")
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error (attempt {attempt + 1}): {e}")
            except Exception as e:
                logger.error(f"Unexpected error (attempt {attempt + 1}): {e}")
                
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay * (attempt + 1))
                
        latency = time.time() - start
        return DeepThinkerResponse(
            success=False,
            error=f"Failed after {self.max_retries} attempts",
            latency=latency
        )
    
    def infer(self, **kwargs) -> DeepThinkerResponse:
        """Alias for answer() method"""
        return self.answer(**kwargs)


# Singleton client instance for easy import
_default_client: Optional[DeepThinkerClient] = None


def get_client(service_url: str = "http://127.0.0.1:8002") -> DeepThinkerClient:
    """Get or create a singleton client instance"""
    global _default_client
    if _default_client is None or _default_client.service_url != service_url:
        _default_client = DeepThinkerClient(service_url)
    return _default_client


def answer(
    video_path: str,
    question: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    service_url: str = "http://127.0.0.1:8002",
    **kwargs
) -> DeepThinkerResponse:
    """
    Convenience function to get answer from DeepThinker.
    
    Args:
        video_path: Absolute path to the video file
        question: Question about the video
        start_time: Start time in seconds (optional)
        end_time: End time in seconds (optional)
        service_url: URL of the DeepThinker service
        **kwargs: Additional arguments passed to DeepThinkerClient.answer()
        
    Returns:
        DeepThinkerResponse with answer and metadata
    """
    client = get_client(service_url)
    return client.answer(
        video_path=video_path,
        question=question,
        start_time=start_time,
        end_time=end_time,
        **kwargs
    )


if __name__ == "__main__":
    # Test the client
    import argparse
    
    parser = argparse.ArgumentParser(description="Test DeepThinker Client")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8002", help="Service URL")
    parser.add_argument("--video", type=str, required=True, help="Video path")
    parser.add_argument("--question", type=str, required=True, help="Question")
    parser.add_argument("--start", type=float, default=None, help="Start time")
    parser.add_argument("--end", type=float, default=None, help="End time")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    client = DeepThinkerClient(args.url)
    
    print("Checking service health...")
    if not client.health_check():
        print("Service not available, waiting...")
        if not client.wait_for_service(timeout=30):
            print("Service unavailable, exiting")
            exit(1)
    
    print(f"Sending request for video: {args.video}")
    print(f"Question: {args.question}")
    
    result = client.answer(
        video_path=args.video,
        question=args.question,
        start_time=args.start,
        end_time=args.end
    )
    
    print(f"\n{'='*50}")
    print(f"Success: {result.success}")
    print(f"Latency: {result.latency:.2f}s")
    
    if result.success:
        print(f"\n--- Thinking ---")
        print(result.thinking or "(no thinking)")
        print(f"\n--- Answer ---")
        print(result.answer or "(no answer)")
    else:
        print(f"\nError: {result.error}")
