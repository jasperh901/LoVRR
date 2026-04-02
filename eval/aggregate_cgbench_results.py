#!/usr/bin/env python3
"""
Aggregate CGBench evaluation results and calculate accuracy.
"""

import argparse
import json
import os
from collections import defaultdict


def load_lovrr_results(output_dir):
    """Load results from all LoVRR workers."""
    results = []
    for i in range(3):
        result_file = os.path.join(output_dir, f'results_worker_{i}.json')
        if os.path.exists(result_file):
            with open(result_file) as f:
                results.extend(json.load(f))
    return {r['qid']: r for r in results}


def load_deepthinker_results(output_dir):
    """Load results from DeepThinker done directory."""
    done_dir = os.path.join(output_dir, 'deepthinker_done')
    results = {}
    if os.path.exists(done_dir):
        for f in os.listdir(done_dir):
            if f.endswith('.json'):
                with open(os.path.join(done_dir, f)) as fp:
                    data = json.load(fp)
                    results[data['qid']] = data
    return results


def extract_answer_letter(response):
    """Extract answer letter from response."""
    import re
    if not response:
        return None
    response = response.strip()
    
    # For LoVRR format: "A) xxx" or "(A) xxx" at start
    match = re.match(r'^\(?([A-H])\)?[\s\)]', response)
    if match:
        return match.group(1).upper()
    
    # For DeepThinker thinking format: remove <think>...</think> content first
    clean_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    if clean_response.strip():
        # Check for answer at start of cleaned response
        match = re.match(r'^\s*\(?([A-H])\)?[\s\.\)]', clean_response)
        if match:
            return match.group(1).upper()
    
    # Use last 800 chars for pattern matching (where conclusion usually is)
    tail = response[-800:] if len(response) > 800 else response
    
    # Pattern 1: "the answer is X" or "correct answer is X"
    match = re.search(r'(?:the\s+)?(?:correct\s+)?answer\s+(?:is|should\s+be|would\s+be)\s*:?\s*\(?([A-H])\)?', tail, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Pattern 2: "option X" with context
    match = re.search(r'(?:choose|select|pick|is)\s+option\s+\(?([A-H])\)?', tail, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Pattern 3: "X is correct" or "X is the answer"
    match = re.search(r'\b([A-H])\)?\s+is\s+(?:the\s+)?(?:correct|right|answer)', tail, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Pattern 4: Conclusion phrases
    match = re.search(r'(?:so|therefore|thus|hence|finally|conclusion)[,:\s]+(?:the\s+answer\s+is\s+)?(?:option\s+)?\(?([A-H])\)?', tail, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Pattern 5: "aligns with option X" or "matches option X"
    match = re.search(r'(?:aligns|matches|corresponds)\s+(?:with\s+)?(?:option\s+)?\(?([A-H])\)?', tail, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Pattern 6: Look for "Option X." or "(X)" standalone patterns in tail
    match = re.search(r'(?:^|\s)Option\s+([A-H])(?:\.|,|\s|$)', tail, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Pattern 7: "X." or "(X)" at end of sentence
    match = re.search(r'(?:answer|option|choice).*?\b([A-H])\b(?:\.|,|\)|$)', tail, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Pattern 8: "This matches" or "This is correct" after listing an option
    match = re.search(r'\(([A-H])\)[^(]*?(?:this\s+(?:matches|is\s+correct|fits|aligns))', tail, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Pattern 9: "X - This matches" pattern from option lists
    match = re.search(r'([A-H])[\.\)]\s*[^-]*-\s*This\s+(?:matches|is\s+correct|fits)', tail, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Pattern 10: Last mentioned option before "matches my count" or similar
    match = re.search(r'([A-H])[\.\)].*?(?:matches\s+my|is\s+what\s+I|aligns\s+with)', tail, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Pattern 11: "should be X" or "would be X" with option letter
    match = re.search(r'(?:should|would|must)\s+be\s+(?:option\s+)?\(?([A-H])\)?', tail, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Pattern 12: Look for the last (X) pattern before positive confirmation words
    matches = list(re.finditer(r'\(([A-H])\)', tail))
    if matches:
        # Return the last one that appears before any confirmation
        return matches[-1].group(1).upper()
    
    # NO fallback to first letter - that causes wrong extractions from thinking content
    return None


def calculate_accuracy(results, answer_key='lovrr_answer'):
    """Calculate accuracy for given answer key."""
    correct = 0
    total = 0
    by_category = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for qid, r in results.items():
        gt = r.get('ground_truth', '')
        pred = extract_answer_letter(r.get(answer_key, ''))
        
        if not gt or not pred:
            continue
        
        total += 1
        is_correct = (pred == gt.upper())
        if is_correct:
            correct += 1
        
        # By sub_category if available
        cat = r.get('sub_category', 'unknown')
        by_category[cat]['total'] += 1
        if is_correct:
            by_category[cat]['correct'] += 1
    
    return {
        'correct': correct,
        'total': total,
        'accuracy': correct / total if total > 0 else 0,
        'by_category': dict(by_category)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    
    print(f"Loading results from {args.output_dir}")
    
    # Load results
    vm_results = load_lovrr_results(args.output_dir)
    r1_results = load_deepthinker_results(args.output_dir)
    
    print(f"LoVRR results: {len(vm_results)}")
    print(f"DeepThinker results: {len(r1_results)}")
    
    # Merge results
    merged = {}
    for qid, r in vm_results.items():
        merged[qid] = r.copy()
        if qid in r1_results:
            merged[qid]['deepthinker_answer'] = r1_results[qid].get('deepthinker_answer')
            merged[qid]['deepthinker_thinking'] = r1_results[qid].get('deepthinker_thinking')
            merged[qid]['deepthinker_time'] = r1_results[qid].get('deepthinker_time')
    
    # Calculate accuracy
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    # LoVRR accuracy
    vm_acc = calculate_accuracy(merged, 'lovrr_answer')
    print(f"\nLoVRR Accuracy: {vm_acc['correct']}/{vm_acc['total']} = {vm_acc['accuracy']*100:.2f}%")
    
    # DeepThinker accuracy
    r1_acc = calculate_accuracy(merged, 'deepthinker_answer')
    print(f"DeepThinker Accuracy:  {r1_acc['correct']}/{r1_acc['total']} = {r1_acc['accuracy']*100:.2f}%")
    
    # Best-of-both accuracy
    best_correct = 0
    best_total = 0
    for qid, r in merged.items():
        gt = r.get('ground_truth', '')
        vm_pred = extract_answer_letter(r.get('lovrr_answer', ''))
        r1_pred = extract_answer_letter(r.get('deepthinker_answer', ''))
        
        if not gt:
            continue
        
        best_total += 1
        if (vm_pred and vm_pred == gt.upper()) or (r1_pred and r1_pred == gt.upper()):
            best_correct += 1
    
    if best_total > 0:
        print(f"Best-of-Both:       {best_correct}/{best_total} = {best_correct/best_total*100:.2f}%")
    
    # By category
    print("\n--- LoVRR by Category ---")
    for cat, stats in sorted(vm_acc['by_category'].items()):
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {cat}: {stats['correct']}/{stats['total']} = {acc*100:.1f}%")
    
    # Save aggregated results
    output_file = os.path.join(args.output_dir, 'aggregated_results.json')
    with open(output_file, 'w') as f:
        json.dump({
            'lovrr_accuracy': vm_acc,
            'deepthinker_accuracy': r1_acc,
            'best_of_both': {'correct': best_correct, 'total': best_total, 'accuracy': best_correct/best_total if best_total > 0 else 0},
            'results': list(merged.values())
        }, f, indent=2)
    
    print(f"\nAggregated results saved to {output_file}")


if __name__ == '__main__':
    main()
