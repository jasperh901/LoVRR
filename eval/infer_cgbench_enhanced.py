#!/usr/bin/env python3
"""
CGBench (mini) Enhanced Evaluation Script
Pipeline: Grounder -> Refiner (1 iter refinement) -> Dual Answer (LoVRR + DeepThinker)

Each worker runs all roles on one Qwen2-VL model.

Usage:
    python infer_cgbench_enhanced.py --worker_id 0 --num_workers 3 --gpu 2
"""

import argparse
import json
import logging
import os
import sys
import time
import filelock
from datetime import datetime

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lovrr.inference import DeepThinker, RoleModel
from lovrr.inference.temporal import (
    run_grounder, run_refiner_with_segment_tokens, run_refiner_rerank_top5,
    refine_span_simple, run_lovrr_answerer,
)
from lovrr.utils.io import get_duration


# ==================== Constants ====================
CGBENCH_DATA_PATH = 'data/cgbench/cgbench_mini.json'
CGBENCH_VIDEO_DIR = 'data/cgbench/videos_3fps_480_noaudio'

# DeepThinker parameters
DEEPTHINKER_MAX_FRAMES = 32


def setup_logger(output_dir, worker_id):
    """Setup logger for this worker."""
    logger = logging.getLogger(f'cgbench_worker_{worker_id}')
    logger.setLevel(logging.INFO)
    
    # File handler
    log_file = os.path.join(output_dir, f'worker_{worker_id}.log')
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def load_cgbench_data():
    """Load CGBench mini dataset."""
    with open(CGBENCH_DATA_PATH) as f:
        data = json.load(f)
    return data


def get_video_path(video_uid):
    """Get video path from video_uid."""
    return os.path.join(CGBENCH_VIDEO_DIR, f'{video_uid}.mp4')


def save_result(output_dir, result, worker_id, logger):
    """Save result to worker's output file with file locking."""
    output_file = os.path.join(output_dir, f'results_worker_{worker_id}.json')
    lock_file = output_file + '.lock'
    
    with filelock.FileLock(lock_file):
        results = []
        if os.path.exists(output_file):
            with open(output_file) as f:
                results = json.load(f)
        results.append(result)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)


def load_checkpoint(output_dir, worker_id):
    """Load checkpoint for this worker."""
    ckpt_file = os.path.join(output_dir, f'checkpoint_worker_{worker_id}.json')
    if os.path.exists(ckpt_file):
        with open(ckpt_file) as f:
            return json.load(f)
    return {'processed_qids': []}


def save_checkpoint(output_dir, worker_id, processed_qids):
    """Save checkpoint for this worker."""
    ckpt_file = os.path.join(output_dir, f'checkpoint_worker_{worker_id}.json')
    with open(ckpt_file, 'w') as f:
        json.dump({'processed_qids': processed_qids}, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker_id', type=int, required=True, help='Worker ID (0, 1, 2)')
    parser.add_argument('--num_workers', type=int, default=3, help='Total number of workers')
    parser.add_argument('--gpu', type=int, required=True, help='GPU ID to use')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--model_path', type=str, default='models/LoVRR-7B')
    parser.add_argument('--deepthinker_adapter_path', default=None)
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    args = parser.parse_args()
    
    # Setup output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = os.path.join(os.path.dirname(__file__), 'outputs', f'cgbench_enhanced_{timestamp}')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logger
    logger = setup_logger(args.output_dir, args.worker_id)
    logger.info(f"Starting CGBench worker {args.worker_id}/{args.num_workers} on GPU {args.gpu}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda')
    
    # Load model
    logger.info(f"Loading model from {args.model_path}...")
    roles = RoleModel.from_pretrained(
        args.model_path,
        role_paths={'deepthinker': args.deepthinker_adapter_path},
        device='cuda',
        required_roles=('grounder', 'refiner', 'deepthinker'),
    )
    model, processor, device = roles.model, roles.processor, roles.device
    deepthinker = DeepThinker(roles)
    logger.info("Model loaded successfully")

    # Load dataset
    data = load_cgbench_data()
    logger.info(f"Loaded {len(data)} samples from CGBench mini")
    
    # Split data for this worker
    worker_data = [d for i, d in enumerate(data) if i % args.num_workers == args.worker_id]
    logger.info(f"Worker {args.worker_id} will process {len(worker_data)} samples")
    
    # Load checkpoint
    checkpoint = load_checkpoint(args.output_dir, args.worker_id)
    processed_qids = set(checkpoint['processed_qids']) if args.resume else set()
    logger.info(f"Resuming with {len(processed_qids)} already processed samples")
    
    # Process samples
    for idx, sample in enumerate(worker_data):
        qid = sample['qid']
        
        if qid in processed_qids:
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing [{idx+1}/{len(worker_data)}] qid={qid}")
        logger.info(f"Question: {sample['question']}")
        
        video_path = get_video_path(sample['video_uid'])
        if not os.path.exists(video_path):
            logger.error(f"Video not found: {video_path}")
            continue
        
        try:
            # Get video duration
            duration = get_duration(video_path)
            logger.info(f"Video duration: {duration:.2f}s")
            
            result = {
                'qid': qid,
                'video_uid': sample['video_uid'],
                'question': sample['question'],
                'choices': sample['choices'],
                'ground_truth': sample['right_answer'],
                'clue_intervals': sample['clue_intervals'],
                'duration': duration,
                'agents': []
            }
            
            # ========== Grounder (top-5 predictions) ==========
            result['agents'].append('grounder')
            grounder_start = time.time()
            preds, confs, grounder_response, grounder_success = run_grounder(
                model, processor, device, video_path, sample['question'], duration, logger
            )
            grounder_time = time.time() - grounder_start
            
            result['grounder_output'] = grounder_response
            result['grounder_preds'] = preds
            result['grounder_confs'] = confs
            result['grounder_success'] = grounder_success
            result['grounder_time'] = grounder_time
            
            # ========== Refiner (score top-5, rerank, select best) ==========
            result['agents'].append('refiner')
            refiner_start = time.time()
            
            if len(preds) > 1:
                reranked_preds, _, probs, probs_sorted, ranks = run_refiner_rerank_top5(
                    model, processor, device, video_path, sample['question'], duration, preds, confs, logger
                )
                result['refiner_probs'] = probs
                result['refiner_probs_sorted'] = probs_sorted
                result['refiner_ranks'] = ranks
                result['pred_ori'] = preds
                result['conf_ori'] = confs
                
                # Best prediction after refiner reranking
                best_pred = reranked_preds[0]
                best_score = probs_sorted[0]
            else:
                best_pred = preds[0]
                best_score = run_refiner_with_segment_tokens(
                    model, processor, device, video_path, sample['question'], duration, preds[0], logger
                )
                reranked_preds = preds
                result['refiner_probs'] = [best_score]
                result['refiner_probs_sorted'] = [best_score]
                result['refiner_ranks'] = [0]
            
            refiner_time = time.time() - refiner_start
            result['refiner_time'] = refiner_time
            result['refiner_best_pred'] = best_pred
            result['refiner_best_score'] = best_score
            logger.info(f"Refiner selected: [{best_pred[0]:.2f}, {best_pred[1]:.2f}], score={best_score:.4f}")
            
            # ========== Refinement (refine the best from refiner) ==========
            result['agents'].append('refinement')
            refine_start = time.time()
            refined_span, refined_score, refine_info = refine_span_simple(
                model, processor, device, video_path, sample['question'], duration, best_pred, best_score, logger
            )
            refine_time = time.time() - refine_start
            
            result['refined_span'] = refined_span
            result['refined_score'] = refined_score
            result['refinement_info'] = refine_info
            result['refinement_time'] = refine_time
            
            # Use refined span for answering
            final_span = refined_span
            result['final_span'] = final_span
            result['pred'] = [final_span]  # For official eval compatibility
            result['conf'] = [refined_score]
            result['span'] = sample['clue_intervals']  # GT span for eval
            
            # ========== LoVRR Answerer ==========
            result['agents'].append('lovrr_answerer')
            vm_start = time.time()
            lovrr_answer = run_lovrr_answerer(
                model, processor, device, video_path, sample['question'], sample['choices'], final_span, duration, logger
            )
            vm_time = time.time() - vm_start
            
            result['lovrr_answer'] = lovrr_answer
            result['lovrr_time'] = vm_time
            
            result['agents'].append('deepthinker')
            choice_str = '\n'.join(f"({chr(65+i)}) {choice}" for i, choice in enumerate(sample['choices']))
            prompt = f"{sample['question']}\n{choice_str}\nAnswer with the option letter only."
            response = deepthinker.answer(
                video_path=video_path,
                question=prompt,
                start_time=final_span[0],
                end_time=final_span[1],
                max_frames=DEEPTHINKER_MAX_FRAMES,
                max_new_tokens=2048,
                problem_type='multiple choice',
            )
            result['deepthinker_answer'] = response.answer
            result['deepthinker_thinking'] = response.thinking
            result['deepthinker_time'] = response.latency
            result['deepthinker_status'] = 'done' if response.success else 'error'
            if not response.success:
                result['deepthinker_error'] = response.error

            # Save result
            save_result(args.output_dir, result, args.worker_id, logger)
            
            # Update checkpoint
            processed_qids.add(qid)
            save_checkpoint(args.output_dir, args.worker_id, list(processed_qids))
            
            total_time = grounder_time + refiner_time + refine_time + vm_time + response.latency
            logger.info(f"Sample completed in {total_time:.2f}s (G:{grounder_time:.1f}, V:{refiner_time:.1f}, R:{refine_time:.1f}, A:{vm_time:.1f})")
            
        except Exception as ex:
            logger.error(f"Error processing qid={qid}: {ex}", exc_info=True)
            continue
    
    logger.info(f"\nWorker {args.worker_id} completed. Processed {len(processed_qids)} samples.")


if __name__ == '__main__':
    main()
