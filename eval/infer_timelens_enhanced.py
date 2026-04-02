#!/usr/bin/env python3
"""
LoVRR Enhanced Evaluation on TimeLens-Bench
VTG Mode: Grounder -> Refiner -> Refinement (no Planner, no Answerer)

Reuses functions from infer_mvbench_enhanced.py
"""

import argparse
import json
import os
import sys
import logging
from datetime import datetime
from pathlib import Path

import torch

# Add paths
SCRIPT_DIR = Path(__file__).parent.absolute()
BRIDGE_PATH = SCRIPT_DIR.parent
LOVRR_DIR = BRIDGE_PATH / "LoVRR-main"
sys.path.insert(0, str(BRIDGE_PATH))
sys.path.insert(0, str(LOVRR_DIR))

import nncore
from torch.utils.data import Dataset, DataLoader

from lovrr.constants import GROUNDER_PROMPT
from lovrr.dataset.utils import process_vision_info
from lovrr.model.builder import build_model
from lovrr.utils.parser import parse_span

# Import refine functions from infer_mvbench_enhanced
from eval.infer_mvbench_enhanced import (
    _refiner_score,
    refine_span_with_multiscale,
)


def setup_logging(output_path):
    """Setup logging."""
    log_path = Path(output_path) / "eval.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_timelens_annotations(anno_path: str):
    """Load TimeLens-Bench annotations."""
    with open(anno_path, 'r') as f:
        data = json.load(f)
    
    annos = []
    for vid, info in data.items():
        duration = info['duration']
        for i, (query, span) in enumerate(zip(info['queries'], info['spans'])):
            annos.append({
                'vid': vid,
                'qid': f"{vid}_{i}",
                'query': query,
                'span': [span],
                'duration': duration,
            })
    return annos


def run_grounder(model, processor, device, video_path, query, duration,
                 min_pixels=36*28*28, max_pixels=64*28*28, max_frames=150, fps=1.0,
                 num_threads=1):
    """Run grounder to get initial span predictions."""
    messages = [{
        'role': 'user',
        'content': [{
            'type': 'video',
            'video': video_path,
            'min_pixels': min_pixels,
            'max_pixels': max_pixels,
            'max_frames': max_frames,
            'fps': fps,
            'num_threads': num_threads
        }, {
            'type': 'text',
            'text': GROUNDER_PROMPT.format(query)
        }]
    }]
    
    try:
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        data = processor(text=[text], images=images, videos=videos, return_tensors='pt')
        data = data.to(device)
        
        model.base_model.disable_adapter_layers()
        model.base_model.enable_adapter_layers()
        model.set_adapter('grounder')
        
        with torch.inference_mode():
            output_ids = model.generate(
                **data,
                max_new_tokens=256,
                do_sample=False,
            )
        
        response = processor.batch_decode(
            output_ids[:, data['input_ids'].shape[1]:],
            skip_special_tokens=True
        )[0].strip()
        
        # Parse predictions
        preds = []
        import re
        pattern = r'<seg>(\d+\.?\d*),\s*(\d+\.?\d*)</seg>'
        matches = re.findall(pattern, response)
        
        if matches:
            for m in matches:
                start, end = float(m[0]) * duration, float(m[1]) * duration
                preds.append(parse_span([start, end], duration))
        else:
            # Fallback: try to extract any numbers
            nums = re.findall(r'(\d+\.?\d*)', response)
            if len(nums) >= 2:
                start, end = float(nums[0]), float(nums[1])
                # Check if normalized
                if start <= 1.0 and end <= 1.0:
                    start, end = start * duration, end * duration
                preds.append(parse_span([start, end], duration))
        
        if not preds:
            preds = [[0, duration]]
        
        return preds, response, True
        
    except Exception as e:
        logging.error(f"Grounder error: {e}")
        return [[0, duration]], str(e), False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='activitynet-timelens',
                        choices=['activitynet-timelens', 'charades-timelens', 'qvhighlights-timelens'])
    parser.add_argument('--anno_path', type=str, default=None)
    parser.add_argument('--video_root', type=str, default=None)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    # Official LoVRR Grounder params: max_frames=150, fps=1.0
    parser.add_argument('--grounder_max_frames', type=int, default=150)
    parser.add_argument('--grounder_fps', type=float, default=1.0)
    # Refiner params: max_frames=64, fps=2.0
    parser.add_argument('--refiner_max_frames', type=int, default=64)
    parser.add_argument('--refiner_fps', type=float, default=2.0)
    
    # Refinement params
    parser.add_argument('--use_refinement', action='store_true')
    parser.add_argument('--refine_iters', type=int, default=3)
    parser.add_argument('--refine_budget', type=int, default=32)
    parser.add_argument('--step_ratio', type=float, default=0.2)
    parser.add_argument('--step_decay', type=float, default=0.7)
    parser.add_argument('--gain', type=float, default=0.01)
    
    # Distributed
    parser.add_argument('--chunk', type=int, default=1)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    
    args = parser.parse_args()
    
    # Set paths
    timelens_root = Path('data/TimeLens-Bench')
    if args.anno_path is None:
        args.anno_path = str(timelens_root / f"{args.dataset}.json")
    if args.video_root is None:
        dataset_name = args.dataset.replace('-timelens', '')
        args.video_root = str(timelens_root / 'videos' / dataset_name)
    
    os.makedirs(args.output_path, exist_ok=True)
    logger = setup_logging(args.output_path)
    
    # Load annotations
    logger.info(f"Loading annotations from {args.anno_path}")
    annos = load_timelens_annotations(args.anno_path)
    logger.info(f"Total annotations: {len(annos)}")
    
    # Sort by duration (descending) for better GPU utilization
    annos.sort(key=lambda x: x['duration'], reverse=True)
    
    # Chunk for distributed
    annos = annos[args.index::args.chunk]
    logger.info(f"Processing {len(annos)} annotations (chunk {args.index}/{args.chunk})")
    
    # Checkpoint (with chunk index for multi-GPU)
    if args.chunk > 1:
        checkpoint_path = Path(args.output_path) / f"checkpoint_{args.index}.json"
        pred_path = Path(args.output_path) / f"predictions_{args.index}.jsonl"
    else:
        checkpoint_path = Path(args.output_path) / f"checkpoint.json"
        pred_path = Path(args.output_path) / f"predictions.jsonl"
    
    processed_ids = set()
    dumps = []
    if args.resume and checkpoint_path.exists():
        with open(checkpoint_path, 'r') as f:
            ckpt = json.load(f)
            dumps = ckpt.get('dumps', [])
            processed_ids = set(ckpt.get('processed_ids', []))
        logger.info(f"Resumed from checkpoint: {len(processed_ids)} processed")
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model, processor = build_model(args.model_path, device=args.device)
    device = next(model.parameters()).device
    logger.info(f"Model loaded on device: {device}")
    
    # Load refiner adapter (required for VTG mode)
    adapter_path = os.path.join(args.model_path, 'refiner')
    if os.path.isdir(adapter_path):
        logger.info('Loading refiner adapter...')
        model.load_adapter(adapter_path, adapter_name='refiner')
    else:
        logger.error(f"Refiner adapter not found at {adapter_path}")
        return
    
    # Process annotations
    for i, anno in enumerate(annos):
        qid = anno['qid']
        if qid in processed_ids:
            continue
        
        vid = anno['vid']
        query = anno['query']
        duration = anno['duration']
        gt_span = anno['span']
        
        # Find video file
        video_path = None
        for ext in ['.mp4', '.mkv', '.webm', '.avi']:
            p = Path(args.video_root) / f"{vid}{ext}"
            if p.exists():
                video_path = str(p)
                break
        
        if video_path is None:
            logger.warning(f"Video not found: {vid}")
            continue
        
        logger.info(f"[{i+1}/{len(annos)}] Processing {qid}")
        
        # Grounder
        logger.info("--- Grounder ---")
        grounder_preds, grounder_response, grounder_success = run_grounder(
            model, processor, device, video_path, query, duration,
            max_frames=args.grounder_max_frames,
            fps=args.grounder_fps,
        )
        logger.info(f"Grounder predictions: {grounder_preds}")
        
        # Refiner selection
        logger.info("--- Refiner ---")
        best_span = grounder_preds[0]
        best_score = _refiner_score(model, processor, device, video_path, query, duration, best_span)
        
        for pred in grounder_preds[1:5]:  # Check top 5
            score = _refiner_score(model, processor, device, video_path, query, duration, pred)
            if score > best_score:
                best_score = score
                best_span = pred
        
        logger.info(f"Best after verify: {best_span}, score={best_score:.3f}")
        
        # Refinement
        refinement_info = {}
        final_span = best_span
        if args.use_refinement:
            logger.info("--- Refinement ---")
            refinement_info = refine_span_with_multiscale(
                model, processor, device, video_path, query, duration,
                best_span, base_score=best_score,
                refine_iters=args.refine_iters,
                budget=args.refine_budget,
                step_ratio=args.step_ratio,
                step_decay=args.step_decay,
                gain=args.gain,
                logger=logger,
            )
            final_span = refinement_info.get('final_span', best_span)
            logger.info(f"Refinement result: {final_span} (accepted: {refinement_info.get('accepted', False)})")
        
        # Save result
        result = {
            'qid': qid,
            'vid': vid,
            'query': query,
            'duration': duration,
            'span': gt_span,
            'pred': [list(final_span)],
            'pred_ori': grounder_preds,
            'grounder_response': grounder_response,
            'grounder_success': grounder_success,
            'refiner_score': best_score,
            'refinement_info': refinement_info,
        }
        dumps.append(result)
        processed_ids.add(qid)
        
        # Save checkpoint periodically
        if len(dumps) % 10 == 0:
            with open(checkpoint_path, 'w') as f:
                json.dump({'dumps': dumps, 'processed_ids': list(processed_ids)}, f)
            logger.info(f"Checkpoint saved: {len(dumps)} processed")
    
    # Final save
    with open(checkpoint_path, 'w') as f:
        json.dump({'dumps': dumps, 'processed_ids': list(processed_ids)}, f)
    
    # Save predictions in TimeLens format
    with open(pred_path, 'w') as f:
        for item in dumps:
            vid = item['vid']
            query = item['query']
            gt_span = item['span'][0] if item['span'] else [0, 0]
            pred_span = item['pred'][0]
            
            # TimeLens format
            video_name = f"{vid}.mp4"
            key = f"{video_name}>>>{query}>>>{gt_span}"
            entry = {
                key: {
                    'timestamps': [pred_span],
                    'answers': item.get('grounder_response', ''),
                    'duration': item['duration'],
                }
            }
            f.write(json.dumps(entry) + '\n')
    
    # Calculate metrics
    logger.info("=== Final Metrics ===")
    compute_metrics(dumps, logger)
    
    logger.info(f"Done! Processed {len(dumps)} annotations")
    logger.info(f"Predictions saved to {pred_path}")


def compute_metrics(dumps, logger):
    """Compute R@0.3, R@0.5, R@0.7, mIoU."""
    def iou(pred, gt):
        p_start, p_end = pred
        g_start, g_end = gt
        inter_start = max(p_start, g_start)
        inter_end = min(p_end, g_end)
        inter = max(0, inter_end - inter_start)
        union = (p_end - p_start) + (g_end - g_start) - inter
        return inter / union if union > 0 else 0
    
    ious = []
    for item in dumps:
        gt = item['span'][0] if item['span'] else [0, 0]
        pred = item['pred'][0]
        ious.append(iou(pred, gt))
    
    n = len(ious)
    if n == 0:
        logger.info("No valid samples")
        return
    
    r03 = sum(1 for x in ious if x >= 0.3) / n * 100
    r05 = sum(1 for x in ious if x >= 0.5) / n * 100
    r07 = sum(1 for x in ious if x >= 0.7) / n * 100
    miou = sum(ious) / n * 100
    
    logger.info(f"R@0.3: {r03:.2f}%")
    logger.info(f"R@0.5: {r05:.2f}%")
    logger.info(f"R@0.7: {r07:.2f}%")
    logger.info(f"mIoU:  {miou:.2f}%")


if __name__ == '__main__':
    main()
