#!/usr/bin/env python3
"""
CGBench (mini) Enhanced Evaluation Script
Pipeline: Grounder -> Refiner (1 iter refinement) -> Dual Answer (LoVRR + DeepThinker)

Decoupled design:
- LoVRR workers: Run grounder, refiner, LoVRR answerer, queue tasks for DeepThinker
- DeepThinker workers: Pick up pending tasks and provide answers

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
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'LoVRR-main'))

from lovrr.constants import GROUNDER_PROMPT, REFINER_PROMPT
from lovrr.dataset.utils import process_vision_info
from lovrr.model.builder import build_model
from lovrr.utils.io import get_duration
from lovrr.utils.parser import parse_span, parse_query


# ==================== Constants ====================
CGBENCH_DATA_PATH = 'data/cgbench/cgbench_mini.json'
CGBENCH_VIDEO_DIR = 'data/cgbench/videos_3fps_480_noaudio'

# Grounder parameters (official LoVRR config)
GROUNDER_MAX_FRAMES = 150
GROUNDER_FPS = 1.0
GROUNDER_MIN_PIXELS = 36 * 28 * 28
GROUNDER_MAX_PIXELS = 64 * 28 * 28

# Refiner parameters (official LoVRR config)
REFINER_MAX_FRAMES = 64
REFINER_FPS = 2.0
REFINER_MIN_PIXELS = 36 * 28 * 28
REFINER_MAX_PIXELS = 64 * 28 * 28

# Answerer parameters (official LoVRR config)
ANSWERER_MAX_FRAMES = 32
ANSWERER_FPS = 2.0
ANSWERER_MIN_PIXELS = 128 * 28 * 28
ANSWERER_MAX_PIXELS = 256 * 28 * 28

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


def run_grounder(model, processor, device, video_path, question, duration, logger):
    """Run grounder to locate relevant video segment (official LoVRR config)."""
    # Parse query (normalize whitespace, remove trailing periods)
    query = parse_query(question)
    logger.info(f"Running grounder for: {query[:50]}...")
    
    messages = [{
        'role': 'user',
        'content': [{
            'type': 'video',
            'video': video_path,
            'min_pixels': GROUNDER_MIN_PIXELS,
            'max_pixels': GROUNDER_MAX_PIXELS,
            'max_frames': GROUNDER_MAX_FRAMES,
            'fps': GROUNDER_FPS
        }, {
            'type': 'text',
            'text': GROUNDER_PROMPT.format(query)
        }]
    }]
    
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    data = processor(text=[text], images=images, videos=videos, return_tensors='pt')
    data = data.to(device)
    
    # Set grounder adapter
    model.base_model.disable_adapter_layers()
    model.base_model.enable_adapter_layers()
    model.set_adapter('grounder')
    
    with torch.inference_mode():
        output_ids = model.generate(
            **data,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=None,
            max_new_tokens=256
        )
    
    output_ids = output_ids[0, data.input_ids.size(1):]
    if output_ids[-1] == processor.tokenizer.eos_token_id:
        output_ids = output_ids[:-1]
    
    response = processor.decode(output_ids, clean_up_tokenization_spaces=False)
    
    # Get prediction from model.reg (grounder output) - return top-5 predictions
    grounder_success = len(model.reg) > 0
    if grounder_success:
        blob = model.reg[0].cpu().float()
        pred_tensor = blob[:, :2] * duration
        pred_tensor = pred_tensor.clamp(min=0, max=duration)
        conf = blob[:, -1].tolist()
        
        # Round timestamps to 0.001 unit (official CGBench setting)
        unit = 0.001
        pred_tensor = torch.round(pred_tensor / unit).long() * unit
        
        # Sort timestamps (ensure start < end)
        inds = (pred_tensor[:, 1] - pred_tensor[:, 0] < 0).nonzero()[:, 0]
        pred_tensor[inds] = pred_tensor[inds].roll(1)
        
        preds = pred_tensor.tolist()  # All top-5 predictions
        logger.info(f"Grounder output: {response} -> {len(preds)} preds, top=[{preds[0][0]:.2f}, {preds[0][1]:.2f}]")
    else:
        # Fallback: generate 5 segments covering the video
        logger.warning("Grounder failed, using fallback segments")
        preds = [[i * duration / 6, (i + 2) * duration / 6] for i in range(5)]
        conf = [0] * 5
    
    return preds, conf, response, grounder_success


def run_refiner_with_segment_tokens(model, processor, device, video_path, question, duration, cand, logger):
    """Run refiner with segment tokens (official LoVRR implementation)."""
    s0, e0 = parse_span(cand, duration, 2)
    offset = (e0 - s0) / 2
    s1, e1 = parse_span([s0 - offset, e0 + offset], duration)
    
    # Calculate percentage of s0, e0 within s1, e1
    s = (s0 - s1) / (e1 - s1) if (e1 - s1) > 0 else 0
    e = (e0 - s1) / (e1 - s1) if (e1 - s1) > 0 else 1
    
    messages = [{
        'role': 'user',
        'content': [{
            'type': 'video',
            'video': video_path,
            'video_start': s1,
            'video_end': e1,
            'min_pixels': REFINER_MIN_PIXELS,
            'max_pixels': REFINER_MAX_PIXELS,
            'max_frames': REFINER_MAX_FRAMES,
            'fps': REFINER_FPS
        }, {
            'type': 'text',
            'text': REFINER_PROMPT.format(question)
        }]
    }]
    
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    data = processor(text=[text], images=images, videos=videos, return_tensors='pt')
    
    # Insert segment start/end tokens (official implementation)
    video_grid_thw = data['video_grid_thw'][0]
    num_frames = int(video_grid_thw[0])
    window = int(video_grid_thw[1] * video_grid_thw[2] / 4)
    
    pos_s = round(s * num_frames)
    pos_e = round(e * num_frames)
    pos_s = min(max(0, pos_s), num_frames)
    pos_e = min(max(0, pos_e), num_frames)
    
    base_idx = torch.nonzero(data['input_ids'][0] == model.config.vision_start_token_id).item()
    pos_s_idx = pos_s * window + base_idx + 1
    pos_e_idx = pos_e * window + base_idx + 2
    
    input_ids = data['input_ids'][0].tolist()
    input_ids.insert(pos_s_idx, model.config.seg_s_token_id)
    input_ids.insert(pos_e_idx, model.config.seg_e_token_id)
    data['input_ids'] = torch.LongTensor([input_ids])
    data['attention_mask'] = torch.ones_like(data['input_ids'])
    
    data = data.to(device)
    
    # Set refiner adapter
    model.base_model.disable_adapter_layers()
    model.base_model.enable_adapter_layers()
    model.set_adapter('refiner')
    
    with torch.inference_mode():
        logits = model(**data).logits[0, -1].softmax(dim=-1)
    
    # Magic numbers from Qwen2-VL vocab: 9454 -> Yes, 2753 -> No
    score = (logits[9454] - logits[2753]).sigmoid().item()
    
    return score


def run_refiner_rerank_top5(model, processor, device, video_path, question, duration, preds, confs, logger):
    """Run refiner on top-5 predictions and rerank (official LoVRR workflow)."""
    logger.info(f"Refiner: scoring {len(preds[:5])} predictions...")
    
    probs = []
    for i, cand in enumerate(preds[:5]):
        try:
            score = run_refiner_with_segment_tokens(model, processor, device, video_path, question, duration, cand, logger)
            probs.append(score)
            logger.info(f"  Pred {i}: [{cand[0]:.2f}, {cand[1]:.2f}] -> score={score:.4f}")
        except Exception as ex:
            logger.warning(f"  Pred {i}: [{cand[0]:.2f}, {cand[1]:.2f}] failed: {ex}")
            probs.append(0.0)
    
    # Rerank by refiner scores (descending)
    ranks = torch.Tensor(probs).argsort(descending=True).tolist()
    
    # Reorder predictions by refiner scores
    reranked_preds = [preds[idx] for idx in ranks]
    reranked_confs = [confs[idx] for idx in ranks]
    probs_sorted = [probs[idx] for idx in ranks]
    
    logger.info(f"Refiner rerank: best=[{reranked_preds[0][0]:.2f}, {reranked_preds[0][1]:.2f}], score={probs_sorted[0]:.4f}")
    
    return reranked_preds, reranked_confs, probs, probs_sorted, ranks


def refine_span_simple(model, processor, device, video_path, question, duration, init_span, base_score, logger):
    """1-iteration refinement with fine scale (0.5x) and 8 degrees of freedom."""
    logger.info(f"Refinement: init_span=[{init_span[0]:.2f}, {init_span[1]:.2f}], base_score={base_score:.4f}")
    
    s, e = parse_span(init_span, duration, 2)
    cur_len = max(0.0, e - s)
    
    # Use fine scale: step = 0.5 * (0.2 * cur_len), minimum 1.0 second
    base_step = max(1.0, 0.2 * cur_len)
    step = base_step * 0.5  # Fine scale (0.5x)
    step = max(1.0, step)   # Ensure minimum step of 1 second
    
    logger.info(f"  Using fine scale: step={step:.2f}s (base={base_step:.2f}s)")
    
    # Generate 8 neighbor candidates (8 degrees of freedom)
    candidates = [
        [s - step, e - step],  # Shift left
        [s + step, e + step],  # Shift right
        [s - step, e + step],  # Expand both
        [s + step, e - step],  # Shrink both
        [s - step, e],         # Expand left only
        [s + step, e],         # Shrink left only
        [s, e - step],         # Shrink right only
        [s, e + step],         # Expand right only
    ]
    
    # Ensure valid candidates using parse_span
    valid_candidates = []
    seen = set()
    for cand in candidates:
        cs, ce = parse_span(cand, duration, 2)
        key = (round(cs, 3), round(ce, 3))
        if key not in seen:
            seen.add(key)
            valid_candidates.append([cs, ce])
    
    # Score all candidates using refiner with segment tokens
    best_span = init_span
    best_score = base_score
    
    for cand in valid_candidates:
        if cand == init_span:
            continue
        try:
            score = run_refiner_with_segment_tokens(model, processor, device, video_path, question, duration, cand, logger)
            logger.info(f"  Candidate [{cand[0]:.2f}, {cand[1]:.2f}] -> score={score:.4f}")
            if score > best_score:
                best_score = score
                best_span = cand
        except Exception as ex:
            logger.warning(f"  Candidate [{cand[0]:.2f}, {cand[1]:.2f}] failed: {ex}")
    
    accepted = (best_span != init_span)
    logger.info(f"Refinement result: [{best_span[0]:.2f}, {best_span[1]:.2f}], score={best_score:.4f}, accepted={accepted}")
    
    return best_span, best_score, {
        'init_span': init_span,
        'init_score': base_score,
        'final_span': best_span,
        'final_score': best_score,
        'accepted': accepted,
        'candidates_evaluated': len(valid_candidates)
    }


def run_lovrr_answerer(model, processor, device, video_path, question, choices, span, duration, logger):
    """Run LoVRR's internal answerer on the located segment (official config)."""
    # Use parse_span with min_len=32 (official setting)
    s, e = parse_span(span, duration, 32)
    logger.info(f"LoVRR answerer on segment [{s:.2f}, {e:.2f}]")
    
    # Format question with choices (official MCQ format)
    prompt = question + '\nOptions:'
    for idx, opt in enumerate(choices):
        prompt += f"\n({chr(ord('A') + idx)}) {opt[0].upper() + opt[1:]}"
    prompt += '\nPlease only give the best option.'
    
    messages = [{
        'role': 'user',
        'content': [{
            'type': 'video',
            'video': video_path,
            'video_start': s,
            'video_end': e,
            'min_pixels': ANSWERER_MIN_PIXELS,
            'max_pixels': ANSWERER_MAX_PIXELS,
            'max_frames': ANSWERER_MAX_FRAMES,
            'fps': ANSWERER_FPS
        }, {
            'type': 'text',
            'text': prompt
        }]
    }]
    
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    text += 'Best Option: ('
    images, videos = process_vision_info(messages)
    data = processor(text=[text], images=images, videos=videos, return_tensors='pt')
    data = data.to(device)
    
    with model.disable_adapter():
        with torch.inference_mode():
            output_ids = model.generate(
                **data,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                repetition_penalty=None,
                max_new_tokens=256
            )
    
    output_ids = output_ids[0, data.input_ids.size(1):]
    if output_ids[-1] == processor.tokenizer.eos_token_id:
        output_ids = output_ids[:-1]
    
    response = processor.decode(output_ids, clean_up_tokenization_spaces=False)
    logger.info(f"LoVRR answer: {response}")
    
    return response


def save_task_for_deepthinker(output_dir, task_data, logger):
    """Save task to pending queue for DeepThinker processing."""
    pending_dir = os.path.join(output_dir, 'deepthinker_pending')
    os.makedirs(pending_dir, exist_ok=True)
    
    task_file = os.path.join(pending_dir, f"{task_data['qid']}.json")
    with open(task_file, 'w') as f:
        json.dump(task_data, f, indent=2)
    
    logger.info(f"Queued task {task_data['qid']} for DeepThinker")


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
    parser.add_argument('--model_path', type=str, default='data/model_zoo/LoVRR-7B')
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
    model, processor = build_model(args.model_path, device='cuda')
    device = next(model.parameters()).device
    model.eval()
    logger.info("Model loaded successfully")
    
    # Load additional adapters (refiner)
    adapter_path = os.path.join(args.model_path, 'refiner')
    if os.path.isdir(adapter_path):
        logger.info('Loading refiner adapter...')
        model.load_adapter(adapter_path, adapter_name='refiner')
    else:
        logger.warning(f"Refiner adapter not found at {adapter_path}")
    
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
                reranked_preds, reranked_confs, probs, probs_sorted, ranks = run_refiner_rerank_top5(
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
                reranked_confs = confs
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
            
            # ========== Queue for DeepThinker ==========
            deepthinker_task = {
                'qid': qid,
                'video_path': video_path,
                'question': sample['question'],
                'choices': sample['choices'],
                'ground_truth': sample['right_answer'],
                'final_span': final_span,
                'duration': duration,
                'status': 'pending'
            }
            save_task_for_deepthinker(args.output_dir, deepthinker_task, logger)
            result['deepthinker_status'] = 'queued'
            
            # Save result
            save_result(args.output_dir, result, args.worker_id, logger)
            
            # Update checkpoint
            processed_qids.add(qid)
            save_checkpoint(args.output_dir, args.worker_id, list(processed_qids))
            
            total_time = grounder_time + refiner_time + refine_time + vm_time
            logger.info(f"Sample completed in {total_time:.2f}s (G:{grounder_time:.1f}, V:{refiner_time:.1f}, R:{refine_time:.1f}, A:{vm_time:.1f})")
            
        except Exception as ex:
            logger.error(f"Error processing qid={qid}: {ex}", exc_info=True)
            continue
    
    logger.info(f"\nWorker {args.worker_id} completed. Processed {len(processed_qids)} samples.")


if __name__ == '__main__':
    main()
