import torch

from lovrr.constants import REFINER_PROMPT
from lovrr.dataset.utils import process_vision_info
from lovrr.utils.parser import parse_span


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
