import torch

from lovrr.constants import GROUNDER_PROMPT, REFINER_PROMPT
from lovrr.dataset.utils import process_vision_info
from lovrr.utils.parser import parse_span, parse_query


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
