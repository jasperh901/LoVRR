#!/usr/bin/env python3
"""
DeepThinker Answer Worker for CGBench Evaluation

This worker picks up pending tasks from LoVRR workers and provides DeepThinker answers.
It runs independently and processes tasks as they become available.

Usage:
    python deepthinker_answer_worker.py --worker_id 0 --deepthinker_url http://127.0.0.1:8006 --output_dir outputs/cgbench_xxx
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from deepthinker_client import DeepThinkerClient


def setup_logger(output_dir, worker_id):
    """Setup logger for this worker."""
    logger = logging.getLogger(f'deepthinker_worker_{worker_id}')
    logger.setLevel(logging.INFO)
    
    log_file = os.path.join(output_dir, f'deepthinker_worker_{worker_id}.log')
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def get_pending_tasks(pending_dir):
    """Get list of pending task files."""
    if not os.path.exists(pending_dir):
        return []
    return [f for f in os.listdir(pending_dir) if f.endswith('.json')]


def load_task(pending_dir, task_file):
    """Load a pending task."""
    task_path = os.path.join(pending_dir, task_file)
    with open(task_path) as f:
        return json.load(f)


def mark_task_processing(pending_dir, done_dir, task_file, worker_id):
    """Move task to processing state."""
    src = os.path.join(pending_dir, task_file)
    # Add worker_id to indicate who is processing
    processing_dir = os.path.join(os.path.dirname(pending_dir), 'deepthinker_processing')
    os.makedirs(processing_dir, exist_ok=True)
    dst = os.path.join(processing_dir, f"w{worker_id}_{task_file}")
    
    try:
        os.rename(src, dst)
        return dst
    except FileNotFoundError:
        # Another worker already picked it up
        return None


def mark_task_done(processing_path, done_dir, task_data):
    """Move task to done state with result."""
    os.makedirs(done_dir, exist_ok=True)
    
    qid = task_data['qid']
    done_path = os.path.join(done_dir, f"{qid}.json")
    
    with open(done_path, 'w') as f:
        json.dump(task_data, f, indent=2)
    
    # Remove processing file
    if os.path.exists(processing_path):
        os.remove(processing_path)


def process_task(client, task_data, logger, max_frames=32):
    """Process a single task with DeepThinker."""
    qid = task_data['qid']
    video_path = task_data['video_path']
    question = task_data['question']
    choices = task_data['choices']
    final_span = task_data['final_span']
    
    s, e = final_span
    logger.info(f"Processing qid={qid}, segment=[{s:.2f}, {e:.2f}]")
    
    # Format question with choices
    choice_str = '\n'.join([f"({chr(65+i)}) {c}" for i, c in enumerate(choices)])
    prompt = f"{question}\n{choice_str}\nAnswer with the option letter only."
    
    try:
        start_time = time.time()
        result = client.answer(
            video_path=video_path,
            question=prompt,
            start_time=s,
            end_time=e,
            max_frames=max_frames,
            max_new_tokens=2048,
            problem_type="multiple choice"
        )
        elapsed = time.time() - start_time
        
        if result.success and result.answer:
            task_data['deepthinker_answer'] = result.answer
            task_data['deepthinker_thinking'] = result.thinking
            task_data['deepthinker_time'] = elapsed
            task_data['status'] = 'done'
            logger.info(f"DeepThinker answer: {result.answer} (took {elapsed:.2f}s)")
        else:
            task_data['deepthinker_error'] = result.error
            task_data['status'] = 'error'
            logger.warning(f"DeepThinker failed: {result.error}")
            
    except Exception as ex:
        task_data['deepthinker_error'] = str(ex)
        task_data['status'] = 'error'
        logger.error(f"DeepThinker exception: {ex}")
    
    return task_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker_id', type=int, required=True, help='Worker ID (0 or 1)')
    parser.add_argument('--deepthinker_url', type=str, required=True, help='DeepThinker service URL')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory (same as LoVRR workers)')
    parser.add_argument('--max_frames', type=int, default=32, help='Max frames for DeepThinker')
    parser.add_argument('--poll_interval', type=float, default=2.0, help='Seconds between polling for new tasks')
    parser.add_argument('--timeout', type=float, default=300.0, help='Timeout for DeepThinker requests')
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(args.output_dir, f'r1_{args.worker_id}')
    
    logger.info(f"Starting DeepThinker answer worker {args.worker_id}")
    logger.info(f"DeepThinker URL: {args.deepthinker_url}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Initialize client
    client = DeepThinkerClient(service_url=args.deepthinker_url, timeout=args.timeout)
    
    # Check health
    if not client.health_check():
        logger.error(f"DeepThinker service at {args.deepthinker_url} is not healthy!")
        return
    logger.info("DeepThinker service is healthy")
    
    # Directories
    pending_dir = os.path.join(args.output_dir, 'deepthinker_pending')
    done_dir = os.path.join(args.output_dir, 'deepthinker_done')
    
    processed_count = 0
    idle_count = 0
    max_idle = 300  # Exit after 5 minutes of no tasks
    
    logger.info("Starting task processing loop...")
    
    while True:
        # Get pending tasks
        pending_tasks = get_pending_tasks(pending_dir)
        
        if not pending_tasks:
            idle_count += 1
            if idle_count >= max_idle:
                logger.info(f"No tasks for {max_idle * args.poll_interval}s, exiting...")
                break
            time.sleep(args.poll_interval)
            continue
        
        idle_count = 0
        
        # Process first available task
        for task_file in pending_tasks:
            # Try to claim this task
            processing_path = mark_task_processing(pending_dir, done_dir, task_file, args.worker_id)
            
            if processing_path is None:
                # Another worker got it
                continue
            
            # Load and process task
            try:
                task_data = load_task(os.path.dirname(processing_path), os.path.basename(processing_path))
                task_data = process_task(client, task_data, logger, args.max_frames)
                mark_task_done(processing_path, done_dir, task_data)
                processed_count += 1
                logger.info(f"Completed task {task_data['qid']} (total: {processed_count})")
            except Exception as ex:
                logger.error(f"Error processing task {task_file}: {ex}")
                # Move back to pending
                if os.path.exists(processing_path):
                    os.rename(processing_path, os.path.join(pending_dir, task_file))
            
            break  # Process one task per loop iteration
    
    logger.info(f"DeepThinker worker {args.worker_id} finished. Processed {processed_count} tasks.")


if __name__ == '__main__':
    main()
