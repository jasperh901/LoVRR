# Copyright (c) 2025. Enhanced LoVRR + DeepThinker MVBench Evaluation
# Based on official LoVRR evaluation methodology with multi-scale refinement

import argparse
import copy
import json
import logging
import os
import sys
import time
from contextlib import nullcontext
from datetime import datetime

import nncore
import torch

# Add paths
BRIDGE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOVRR_PATH = os.path.join(BRIDGE_PATH, 'LoVRR-main')
sys.path.insert(0, BRIDGE_PATH)
sys.path.insert(0, LOVRR_PATH)

from lovrr.constants import GROUNDER_PROMPT, PLANNER_PROMPT, REFINER_PROMPT
from lovrr.dataset.hybrid import DATASETS
from lovrr.dataset.utils import process_vision_info
from lovrr.model.builder import build_model
from lovrr.utils.io import get_duration
from lovrr.utils.parser import parse_query, parse_span

# DeepThinker client
try:
    from deepthinker_client import DeepThinkerClient
    DEEPTHINKER_AVAILABLE = True
except ImportError:
    DEEPTHINKER_AVAILABLE = False
    print("Warning: deepthinker_client not found, DeepThinker integration disabled")

# Setup logging
def setup_logging(log_path):
    """Setup logging to file and console"""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# ============== Multi-scale Neighbor Refinement ==============

def _span_key(span, ndigits=3):
    return (round(float(span[0]), ndigits), round(float(span[1]), ndigits))


def _objective(score, span, duration, len_penalty=0.0):
    if len_penalty <= 0:
        return float(score)
    length = max(0.0, float(span[1]) - float(span[0]))
    norm = float(duration) if duration and duration > 0 else 1.0
    return float(score) - float(len_penalty) * (length / norm)


def _generate_multiscale_neighbors(span, duration, steps, min_len=2):
    """Generate neighbors at multiple step scales for broader exploration."""
    s, e = parse_span(span, duration, min_len)
    out = []
    seen = set()
    
    for step in steps:
        if step <= 0:
            continue
        cands = [
            [s - step, e - step],
            [s + step, e + step],
            [s - step, e + step],
            [s + step, e - step],
            [s - step, e],
            [s + step, e],
            [s, e - step],
            [s, e + step],
        ]
        for cand in cands:
            cs, ce = parse_span(cand, duration, min_len)
            key = _span_key([cs, ce])
            if key in seen:
                continue
            seen.add(key)
            out.append([cs, ce])
    return out


def _refiner_score(model, processor, device, video_path, query, duration, cand, num_threads=1):
    """Calculate refiner score for a candidate span."""
    s0, e0 = parse_span(cand, duration, 2)
    offset = (e0 - s0) / 2
    s1, e1 = parse_span([s0 - offset, e0 + offset], duration)

    s = (s0 - s1) / (e1 - s1)
    e = (e0 - s1) / (e1 - s1)

    messages = [{
        'role': 'user',
        'content': [{
            'type': 'video',
            'video': video_path,
            'num_threads': num_threads,
            'video_start': s1,
            'video_end': e1,
            'min_pixels': 36 * 28 * 28,
            'max_pixels': 64 * 28 * 28,
            'max_frames': 64,
            'fps': 2.0
        }, {
            'type': 'text',
            'text': REFINER_PROMPT.format(query)
        }]
    }]

    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    data = processor(text=[text], images=images, videos=videos, return_tensors='pt')

    video_grid_thw = data['video_grid_thw'][0]
    num_frames = int(video_grid_thw[0])
    window = int(video_grid_thw[1] * video_grid_thw[2] / 4)

    pos_s, pos_e = round(s * num_frames), round(e * num_frames)
    pos_s = min(max(0, pos_s), num_frames)
    pos_e = min(max(0, pos_e), num_frames)

    base_idx = torch.nonzero(data['input_ids'][0] == model.config.vision_start_token_id).item()
    pos_s_tok = pos_s * window + base_idx + 1
    pos_e_tok = pos_e * window + base_idx + 2

    input_ids = data['input_ids'][0].tolist()
    input_ids.insert(pos_s_tok, model.config.seg_s_token_id)
    input_ids.insert(pos_e_tok, model.config.seg_e_token_id)
    data['input_ids'] = torch.LongTensor([input_ids])
    data['attention_mask'] = torch.ones_like(data['input_ids'])
    data = data.to(device)

    model.base_model.disable_adapter_layers()
    model.base_model.enable_adapter_layers()
    model.set_adapter('refiner')

    with torch.inference_mode():
        logits = model(**data).logits[0, -1].softmax(dim=-1)

    score = (logits[9454] - logits[2753]).sigmoid().item()
    return score


def refine_span_with_multiscale(model, processor, device, video_path, query, duration, 
                                 init_span, base_score=None, num_threads=1,
                                 refine_iters=3, budget=32, step_ratio=0.2, step_min=1.0,
                                 step_decay=0.7, eps=0.005, skip_high=0.85, skip_low=0.15,
                                 gain=0.01, len_penalty=0.1, logger=None):
    """Multi-scale neighbor refinement (non-streaming version for batch eval)"""
    
    info = {
        'method': 'multiscale',
        'triggered': False,
        'accepted': False,
        'reason': None,
        'base_span': None,
        'base_score': None,
        'final_span': None,
        'final_score': None,
        'evals': 0,
        'iters': 0,
    }

    base_span = list(parse_span(init_span, duration, 2))
    if base_score is None:
        base_score = _refiner_score(model, processor, device, video_path, query, duration, base_span, num_threads)
        info['evals'] = 1
    info['base_span'] = base_span
    info['base_score'] = base_score

    if logger:
        logger.info(f"  Refinement init: span={base_span}, score={base_score:.4f}")

    if skip_high is not None and base_score >= skip_high:
        info['reason'] = 'skip_high'
        info['final_span'] = base_span
        info['final_score'] = base_score
        if logger:
            logger.info(f"  Refinement skipped (score >= {skip_high})")
        return info

    if skip_low is not None and base_score <= skip_low:
        info['reason'] = 'skip_low'
        info['final_span'] = base_span
        info['final_score'] = base_score
        if logger:
            logger.info(f"  Refinement skipped (score <= {skip_low})")
        return info

    info['triggered'] = True
    
    cache = {_span_key(base_span): base_score}
    base_obj = _objective(base_score, base_span, duration, len_penalty)

    best_span = base_span
    best_score = base_score
    best_obj = base_obj

    cur_len = max(0.0, float(base_span[1]) - float(base_span[0]))
    step = max(float(step_min), float(step_ratio) * cur_len)

    beams = [(best_obj, best_score, best_span)]

    reason = None
    for it in range(max(0, int(refine_iters))):
        if info['evals'] >= budget:
            reason = 'budget'
            break

        steps_to_use = [step * 0.5, step, step * 2.0]
        steps_to_use = [s for s in steps_to_use if s >= float(step_min)]

        cand_list = []
        all_candidates = []
        for _, _, span in beams:
            neighbors = _generate_multiscale_neighbors(span, duration, steps_to_use, 2)
            for cand in neighbors:
                all_candidates.append(cand)

        for cand in all_candidates:
            key = _span_key(cand)
            if key in cache:
                score = cache[key]
            else:
                if info['evals'] >= budget:
                    break
                score = _refiner_score(model, processor, device, video_path, query, duration, cand, num_threads)
                cache[key] = score
                info['evals'] += 1

            obj = _objective(score, cand, duration, len_penalty)
            cand_list.append((obj, score, cand))

        if not cand_list:
            reason = 'no_candidates'
            break

        cand_list.sort(key=lambda x: (x[0], x[1]), reverse=True)

        new_beams = []
        seen = set()
        for obj, score, span in cand_list:
            k = _span_key(span)
            if k in seen:
                continue
            seen.add(k)
            new_beams.append((obj, score, span))
            if len(new_beams) >= 1:  # beam=1
                break

        if not new_beams:
            reason = 'no_beams'
            break

        beams = new_beams
        top_obj, top_score, top_span = new_beams[0]

        prev_best_obj = best_obj
        if top_score > best_score or (top_score == best_score and top_obj > best_obj):
            best_span, best_score, best_obj = top_span, top_score, top_obj

        info['iters'] = it + 1

        if skip_high is not None and best_score >= skip_high:
            reason = 'reach_high'
            break

        obj_gain = best_obj - prev_best_obj
        if obj_gain < float(eps) and step <= float(step_min):
            reason = 'plateau'
            break

        step = max(float(step_min), step * step_decay)

    if reason is None:
        reason = 'max_iters'

    accept = best_score >= base_score + float(gain)
    info['accepted'] = bool(accept)
    info['reason'] = reason if accept else f'{reason}|no_gain'
    info['final_span'] = best_span if accept else base_span
    info['final_score'] = best_score if accept else base_score

    if logger:
        status = "ACCEPTED" if accept else "REJECTED"
        logger.info(f"  Refinement {status}: {info['base_span']} -> {info['final_span']}, "
                   f"score: {info['base_score']:.4f} -> {info['final_score']:.4f}, "
                   f"evals={info['evals']}, reason={info['reason']}")

    return info


# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='mvbench')
    parser.add_argument('--pred_path', required=True)
    parser.add_argument('--model_path', default='data/model_zoo/LoVRR-7B')
    parser.add_argument('--base_model_path', default='data/model_zoo/Qwen2-VL-7B-Instruct')
    parser.add_argument('--split', default='test')
    parser.add_argument('--style', default='mcq', choices=['mcq', 'options', 'direct'])
    
    # Enhancement options
    parser.add_argument('--use_refinement', action='store_true', help='Enable multi-scale refinement')
    parser.add_argument('--use_deepthinker', action='store_true', help='Use DeepThinker for answering')
    parser.add_argument('--deepthinker_url', default='http://127.0.0.1:8002')
    parser.add_argument('--deepthinker_no_fallback', action='store_true', help='Disable LoVRR fallback when DeepThinker fails')
    parser.add_argument('--deepthinker_max_frames', type=int, default=32, help='Max frames for DeepThinker (default 32)')
    parser.add_argument('--use_planner', action='store_true', help='Enable Planner for auto query rephrasing')
    
    # Refinement parameters
    parser.add_argument('--refine_iters', type=int, default=3)
    parser.add_argument('--refine_budget', type=int, default=32)
    parser.add_argument('--refine_step_ratio', type=float, default=0.2)
    parser.add_argument('--refine_step_min', type=float, default=1.0)
    parser.add_argument('--refine_step_decay', type=float, default=0.7)
    parser.add_argument('--refine_gain', type=float, default=0.01)
    
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--chunk', type=int, default=1)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--max_frames_override', type=int, default=32, help='Override max frames to reduce memory')
    
    args = parser.parse_args()
    return args


def load_checkpoint(checkpoint_path):
    """Load checkpoint for resume"""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return {'processed_ids': [], 'dumps': []}


def save_checkpoint(checkpoint_path, processed_ids, dumps):
    """Save checkpoint"""
    with open(checkpoint_path, 'w') as f:
        json.dump({'processed_ids': processed_ids, 'dumps': dumps}, f, indent=2)


if __name__ == '__main__':
    args = parse_args()

    # Setup paths
    os.makedirs(args.pred_path, exist_ok=True)
    
    if args.chunk > 1:
        pred_path = os.path.join(args.pred_path, f'output_{args.index}.json')
        log_path = os.path.join(args.pred_path, f'eval_{args.index}.log')
        checkpoint_path = os.path.join(args.pred_path, f'checkpoint_{args.index}.json')
    else:
        pred_path = os.path.join(args.pred_path, 'output.json')
        log_path = os.path.join(args.pred_path, 'eval.log')
        checkpoint_path = os.path.join(args.pred_path, 'checkpoint.json')
    
    logger = setup_logging(log_path)
    
    logger.info("=" * 60)
    logger.info("LoVRR + DeepThinker Enhanced MVBench Evaluation")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset}({args.split})")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Chunk: {args.index + 1}/{args.chunk}")
    logger.info(f"Output: {pred_path}")
    logger.info(f"Use Refinement: {args.use_refinement}")
    logger.info(f"Use DeepThinker: {args.use_deepthinker}")
    if args.use_refinement:
        logger.info(f"Refinement params: iters={args.refine_iters}, budget={args.refine_budget}, "
                   f"step_ratio={args.refine_step_ratio}, step_decay={args.refine_step_decay}, "
                   f"gain={args.refine_gain}")
    logger.info("=" * 60)

    # Initialize DeepThinker client if needed (skip health check, will retry per-sample)
    deepthinker_client = None
    if args.use_deepthinker and DEEPTHINKER_AVAILABLE:
        deepthinker_client = DeepThinkerClient(service_url=args.deepthinker_url, timeout=600.0)
        logger.info(f"DeepThinker client initialized: {args.deepthinker_url} (timeout=600s)")

    # Load checkpoint if resuming
    checkpoint = {'processed_ids': [], 'dumps': []}
    if args.resume:
        checkpoint = load_checkpoint(checkpoint_path)
        logger.info(f"Resuming from checkpoint: {len(checkpoint['processed_ids'])} samples processed")

    # Initialize model
    logger.info('Initializing role *grounder*')
    model, processor = build_model(args.model_path, device=args.device)
    device = next(model.parameters()).device
    logger.info(f"Model loaded on device: {device}")

    # Load adapters
    adapter_path = os.path.join(args.model_path, 'planner')
    if os.path.isdir(adapter_path):
        logger.info('Initializing role *planner*')
        model.load_adapter(adapter_path, adapter_name='planner')

    adapter_path = os.path.join(args.model_path, 'refiner')
    if os.path.isdir(adapter_path):
        logger.info('Initializing role *refiner*')
        model.load_adapter(adapter_path, adapter_name='refiner')

    # Load dataset
    dataset_cls = DATASETS.get(args.dataset)
    annos = dataset_cls.load_annos(split=args.split)
    
    # Split for multi-process
    annos = [annos[i::args.chunk] for i in range(args.chunk)][args.index]
    logger.info(f"Processing {len(annos)} samples")

    dumps = checkpoint['dumps']
    processed_ids = set(checkpoint['processed_ids'])
    
    start_time = time.time()
    
    for i in nncore.ProgressBar(range(len(annos))):
        anno = copy.deepcopy(annos[i])
        
        # Create unique ID for checkpoint
        sample_id = f"{anno.get('video_path', '')}_{anno.get('query', '')}_{i}"
        if sample_id in processed_ids:
            continue
            
        dump = copy.deepcopy(anno)
        
        video_path = anno['video_path']
        duration = anno.get('duration')
        
        if duration is None:
            duration = get_duration(video_path, num_threads=args.num_threads)
            dump['duration'] = duration

        logger.info(f"\n[{i+1}/{len(annos)}] {video_path}")
        logger.info(f"Duration: {duration:.2f}s")

        # Get question and options
        do_answering = all(k in anno for k in ('question', 'options'))
        
        if do_answering:
            question, options, ans = anno['question'], anno['options'], anno['ans']
            
            if args.style in ('mcq', 'options'):
                prompt = question + '\nOptions:'
                for idx, opt in enumerate(options):
                    prompt += f"\n({chr(ord('A') + idx)}) {opt[0].upper() + opt[1:]}"
                prompt += '\nPlease only give the best option.'
            else:
                prompt = question
                
            logger.info(f"Question: {question}")
            logger.info(f"Answer: {ans}")
        else:
            question = anno['query']
            logger.info(f"Query: {question}")

        do_grounding = True
        query = question
        dump['agents'] = []

        # ============ Planner ============
        if args.use_planner:
            logger.info('--- Planner ---')
            dump['agents'].append('planner')
            
            messages = [{
                'role': 'user',
                'content': [{
                    'type': 'video',
                    'video': video_path,
                    'num_threads': args.num_threads,
                    'min_pixels': 36 * 28 * 28,
                    'max_pixels': 64 * 28 * 28,
                    'max_frames': 100,
                    'fps': 1.0
                }, {
                    'type': 'text',
                    'text': PLANNER_PROMPT.format(question)
                }]
            }]

            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            images, videos = process_vision_info(messages)
            data = processor(text=[text], images=images, videos=videos, return_tensors='pt')
            data = data.to(device)

            model.base_model.disable_adapter_layers()
            model.base_model.enable_adapter_layers()
            model.set_adapter('planner')

            output_ids = model.generate(
                **data,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                repetition_penalty=None,
                max_new_tokens=256)

            output_ids = output_ids[0, data.input_ids.size(1):]
            if output_ids[-1] == processor.tokenizer.eos_token_id:
                output_ids = output_ids[:-1]
            response = processor.decode(output_ids, clean_up_tokenization_spaces=False)
            
            dump['planner_response'] = response
            logger.info(f"Planner response: {response}")

            try:
                parsed = json.loads(response)
                action = parsed[0] if isinstance(parsed, list) else parsed
                if action['type'].lower() == 'grounder' and action['value']:
                    query = action['value']
                    dump['planner_parsed_query'] = query
                    logger.info(f"Planner rephrased query: {query}")
                elif action['type'].lower() == 'answerer':
                    do_grounding = False
                    logger.info("Planner decided to skip grounding")
            except Exception as e:
                logger.warning(f'Failed to parse planner response: {e}')

        # ============ Grounder ============
        if do_grounding:
            logger.info('--- Grounder ---')
            dump['agents'].append('grounder')
            
            query = parse_query(query)
            
            messages = [{
                'role': 'user',
                'content': [{
                    'type': 'video',
                    'video': video_path,
                    'num_threads': args.num_threads,
                    'min_pixels': 36 * 28 * 28,
                    'max_pixels': 64 * 28 * 28,
                    'max_frames': 150,
                    'fps': 1.0
                }, {
                    'type': 'text',
                    'text': GROUNDER_PROMPT.format(query)
                }]
            }]

            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            images, videos = process_vision_info(messages)
            data = processor(text=[text], images=images, videos=videos, return_tensors='pt')
            data = data.to(device)

            model.base_model.disable_adapter_layers()
            model.base_model.enable_adapter_layers()
            model.set_adapter('grounder')

            output_ids = model.generate(
                **data,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                repetition_penalty=None,
                max_new_tokens=256)

            output_ids = output_ids[0, data.input_ids.size(1):]
            if output_ids[-1] == processor.tokenizer.eos_token_id:
                output_ids = output_ids[:-1]
            response = processor.decode(output_ids, clean_up_tokenization_spaces=False)

            dump['grounder_response'] = response
            dump['grounder_success'] = len(model.reg) > 0

            if dump['grounder_success']:
                blob = model.reg[0].cpu().float()
                pred, conf = blob[:, :2] * duration, blob[:, -1].tolist()
                pred = pred.clamp(min=0, max=duration)
                
                unit = getattr(dataset_cls, 'UNIT', 0.001)
                pred = torch.round(pred / unit).long() * unit
                
                inds = (pred[:, 1] - pred[:, 0] < 0).nonzero()[:, 0]
                pred[inds] = pred[inds].roll(1)
                pred = pred.tolist()
                
                logger.info(f"Grounder prediction: {pred[0]}")
            else:
                logger.warning('Grounder failed, using fallback')
                pred = [[i * duration / 6, (i + 2) * duration / 6] for i in range(5)]
                conf = [0] * 5

            dump['pred'] = pred
            dump['conf'] = conf

        # ============ Refiner ============
        if do_grounding and len(pred) > 1:
            logger.info('--- Refiner ---')
            dump['agents'].append('refiner')

            probs = []
            for cand in pred[:5]:
                s0, e0 = parse_span(cand, duration, 2)
                offset = (e0 - s0) / 2
                s1, e1 = parse_span([s0 - offset, e0 + offset], duration)

                s = (s0 - s1) / (e1 - s1)
                e = (e0 - s1) / (e1 - s1)

                messages = [{
                    'role': 'user',
                    'content': [{
                        'type': 'video',
                        'video': video_path,
                        'num_threads': args.num_threads,
                        'video_start': s1,
                        'video_end': e1,
                        'min_pixels': 36 * 28 * 28,
                        'max_pixels': 64 * 28 * 28,
                        'max_frames': 64,
                        'fps': 2.0
                    }, {
                        'type': 'text',
                        'text': REFINER_PROMPT.format(question)
                    }]
                }]

                text = processor.apply_chat_template(messages, add_generation_prompt=True)
                images, videos = process_vision_info(messages)
                data = processor(text=[text], images=images, videos=videos, return_tensors='pt')

                video_grid_thw = data['video_grid_thw'][0]
                num_frames, window = int(video_grid_thw[0]), int(video_grid_thw[1] * video_grid_thw[2] / 4)

                pos_s, pos_e = round(s * num_frames), round(e * num_frames)
                pos_s, pos_e = min(max(0, pos_s), num_frames), min(max(0, pos_e), num_frames)

                base_idx = torch.nonzero(data['input_ids'][0] == model.config.vision_start_token_id).item()
                pos_s, pos_e = pos_s * window + base_idx + 1, pos_e * window + base_idx + 2

                input_ids = data['input_ids'][0].tolist()
                input_ids.insert(pos_s, model.config.seg_s_token_id)
                input_ids.insert(pos_e, model.config.seg_e_token_id)
                data['input_ids'] = torch.LongTensor([input_ids])
                data['attention_mask'] = torch.ones_like(data['input_ids'])
                data = data.to(device)

                model.base_model.disable_adapter_layers()
                model.base_model.enable_adapter_layers()
                model.set_adapter('refiner')

                with torch.inference_mode():
                    logits = model(**data).logits[0, -1].softmax(dim=-1)

                score = (logits[9454] - logits[2753]).sigmoid().item()
                probs.append(score)

            ranks = torch.Tensor(probs).argsort(descending=True).tolist()
            
            pred = [pred[idx] for idx in ranks]
            conf = [conf[idx] for idx in ranks]
            probs_sorted = [probs[idx] for idx in ranks]

            logger.info(f"Refiner scores: {[f'{p:.3f}' for p in probs]}")
            logger.info(f"Best after verify: {pred[0]}, score={probs_sorted[0]:.3f}")

            dump['probs'] = probs
            dump['ranks'] = ranks
            dump['pred_ori'] = dump['pred']
            dump['conf_ori'] = dump['conf']
            dump['pred'] = pred
            dump['conf'] = conf

        # ============ Multi-scale Refinement ============
        if args.use_refinement and do_grounding and len(pred) > 0:
            logger.info('--- Refinement ---')
            dump['agents'].append('refinement')
            
            base_score = probs_sorted[0] if 'probs' in dump else None
            
            refine_info = refine_span_with_multiscale(
                model=model,
                processor=processor,
                device=device,
                video_path=video_path,
                query=query,
                duration=duration,
                init_span=pred[0],
                base_score=base_score,
                num_threads=args.num_threads,
                refine_iters=args.refine_iters,
                budget=args.refine_budget,
                step_ratio=args.refine_step_ratio,
                step_min=args.refine_step_min,
                step_decay=args.refine_step_decay,
                gain=args.refine_gain,
                logger=logger
            )
            
            dump['refinement_info'] = refine_info
            
            if refine_info['accepted']:
                dump['pred_before_refine'] = dump['pred']
                pred[0] = refine_info['final_span']
                dump['pred'] = pred

        # ============ Answerer ============
        if do_answering:
            logger.info('--- Answerer ---')
            dump['agents'].append('answerer')

            selected = pred[0] if 'pred' in dump else [0, duration]

            if hasattr(dataset_cls, 'MIN_RATIO'):
                min_len = duration * dataset_cls.MIN_RATIO
            else:
                min_len = getattr(dataset_cls, 'MIN_LEN', 32)

            s, e = parse_span(selected, duration, min_len)
            logger.info(f"Answering segment: [{s:.2f}, {e:.2f}]")

            # Answerer logic: DUAL mode for ALL samples (both LoVRR and DeepThinker)
            is_frame_sequence = isinstance(video_path, list)
            is_long_video = duration > 1800  # Flag for long videos (>30min)
            use_deepthinker_for_this = args.use_deepthinker and deepthinker_client is not None and not is_frame_sequence
            
            lovrr_response = None
            deepthinker_response = None
            
            # Use DUAL answerer mode for ALL samples (compare LoVRR vs DeepThinker)
            if use_deepthinker_for_this:
                dump['is_long_video'] = is_long_video
                dump['video_duration'] = duration
                logger.info(f"=== DUAL ANSWERER MODE ({duration:.0f}s, long={is_long_video}) ===")
                
                # Call LoVRR first
                logger.info("--- LoVRR Answerer ---")
                max_frames_ans = min(args.max_frames_override, getattr(dataset_cls, 'MAX_FRAMES', 64))
                messages = [{
                    'role': 'user',
                    'content': [{
                        'type': 'video',
                        'video': video_path,
                        'num_threads': args.num_threads,
                        'video_start': s,
                        'video_end': e,
                        'min_pixels': 64 * 28 * 28,
                        'max_pixels': 128 * 28 * 28,
                        'max_frames': max_frames_ans,
                        'fps': getattr(dataset_cls, 'FPS', 2.0)
                    }, {
                        'type': 'text',
                        'text': prompt
                    }]
                }]
                text = processor.apply_chat_template(messages, add_generation_prompt=True)
                text += 'Best Option: (' if args.style == 'mcq' else ''
                images, videos = process_vision_info(messages)
                data = processor(text=[text], images=images, videos=videos, return_tensors='pt')
                data = data.to(device)
                with model.disable_adapter():
                    output_ids = model.generate(
                        **data,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        top_k=None,
                        repetition_penalty=None,
                        max_new_tokens=256)
                output_ids = output_ids[0, data.input_ids.size(1):]
                if output_ids[-1] == processor.tokenizer.eos_token_id:
                    output_ids = output_ids[:-1]
                lovrr_response = processor.decode(output_ids, clean_up_tokenization_spaces=False)
                dump['lovrr_answer'] = lovrr_response
                logger.info(f"LoVRR answer: {lovrr_response}")
                
                # Call DeepThinker
                if use_deepthinker_for_this:
                    logger.info("--- DeepThinker Answerer ---")
                    try:
                        logger.info(f"Calling DeepThinker for long video ({duration:.1f}s)...")
                        result = deepthinker_client.answer(
                            video_path=video_path,
                            question=prompt,
                            start_time=s,
                            end_time=e,
                            max_frames=args.deepthinker_max_frames,
                            max_new_tokens=1024,
                            problem_type="multiple choice"
                        )
                        if result.success and result.answer:
                            deepthinker_response = result.answer
                            dump['deepthinker_answer'] = deepthinker_response
                            dump['deepthinker_thinking'] = result.thinking
                            logger.info(f"DeepThinker answer: {deepthinker_response}")
                        else:
                            dump['deepthinker_error'] = result.error
                            logger.warning(f"DeepThinker failed: {result.error}")
                    except Exception as ex:
                        logger.warning(f"DeepThinker error: {ex}")
                        dump['deepthinker_error'] = str(ex)
                
                # Use DeepThinker as primary if available
                if deepthinker_response:
                    response = deepthinker_response
                    dump['answerer_type'] = 'dual_deepthinker_primary'
                else:
                    response = lovrr_response
                    dump['answerer_type'] = 'dual_lovrr_only'
            
            # Fallback to LoVRR only (when DeepThinker not available)
            if 'answerer_type' not in dump or dump.get('answerer_type') not in ['deepthinker', 'dual_deepthinker_primary', 'dual_lovrr_only']:
                # Use reduced frames to prevent OOM
                max_frames_ans = min(args.max_frames_override, getattr(dataset_cls, 'MAX_FRAMES', 64))
                messages = [{
                    'role': 'user',
                    'content': [{
                        'type': 'video',
                        'video': video_path,
                        'num_threads': args.num_threads,
                        'video_start': s,
                        'video_end': e,
                        'min_pixels': 64 * 28 * 28,  # Reduced from 128
                        'max_pixels': 128 * 28 * 28,  # Reduced from 256
                        'max_frames': max_frames_ans,
                        'fps': getattr(dataset_cls, 'FPS', 2.0)
                    }, {
                        'type': 'text',
                        'text': prompt
                    }]
                }]

                text = processor.apply_chat_template(messages, add_generation_prompt=True)
                text += 'Best Option: (' if args.style == 'mcq' else ''
                images, videos = process_vision_info(messages)
                data = processor(text=[text], images=images, videos=videos, return_tensors='pt')
                data = data.to(device)

                with model.disable_adapter():
                    output_ids = model.generate(
                        **data,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        top_k=None,
                        repetition_penalty=None,
                        max_new_tokens=256)

                output_ids = output_ids[0, data.input_ids.size(1):]
                if output_ids[-1] == processor.tokenizer.eos_token_id:
                    output_ids = output_ids[:-1]
                response = processor.decode(output_ids, clean_up_tokenization_spaces=False)
                dump['answerer_type'] = 'lovrr'
                logger.info(f"LoVRR answer: {response}")

            dump['answerer_response'] = response
            dump['response'] = response

        dumps.append(dump)
        processed_ids.add(sample_id)
        
        # Save checkpoint every 10 samples
        if (i + 1) % 10 == 0:
            save_checkpoint(checkpoint_path, list(processed_ids), dumps)
            logger.info(f"Checkpoint saved: {len(dumps)} samples")

    # Final save
    nncore.dump(dumps, pred_path)
    save_checkpoint(checkpoint_path, list(processed_ids), dumps)
    
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Evaluation completed!")
    logger.info(f"Total samples: {len(dumps)}")
    logger.info(f"Time elapsed: {elapsed/60:.1f} minutes")
    logger.info(f"Output saved to: {pred_path}")
    logger.info("=" * 60)
