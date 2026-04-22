"""
Task evaluator for ARC-Easy and HellaSwag.
Uses llama-cpp-python directly — no server, no lm_eval.

Usage:
    python evaluate_tasks.py \
        --model /path/to/model.gguf \
        --tasks arc_easy hellaswag \
        --output /path/to/results.json \
        --label qwen_q4km
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from datasets import load_dataset
from llama_cpp import Llama
from tqdm import tqdm


# ─── Model Loading ────────────────────────────────────────────────────────────

def load_model(model_path: str, n_gpu_layers: int = 99) -> Llama:
    """Load a GGUF model with llama-cpp-python."""
    print(f"\nLoading model: {Path(model_path).name}")
    model = Llama(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=512,
        logits_all=True,   # required for scoring all token positions
        verbose=False,
    )
    print("Model loaded.")
    return model


# ─── Scoring ──────────────────────────────────────────────────────────────────

def score_option(model: Llama, context: str, option: str) -> float:
    """
    Score how likely 'option' is as a continuation of 'context'.
    Returns the sum of log probabilities of the option tokens.
    A higher (less negative) score means the model finds this option more likely.
    """
    full_text = context + option

    # Tokenize context and full text
    context_tokens = model.tokenize(context.encode("utf-8"))
    full_tokens    = model.tokenize(full_text.encode("utf-8"))
    option_tokens  = full_tokens[len(context_tokens):]

    if len(option_tokens) == 0:
        return float("-inf")

    # Run forward pass to get logits
    model.reset()
    model.eval(full_tokens)

    # Extract log probabilities for the option tokens only
    log_probs = []
    for i, token_id in enumerate(option_tokens):
        # Position in full sequence where this option token was predicted
        pos = len(context_tokens) + i - 1
        if pos < 0:
            continue
        logits = np.array(model.scores[pos])
        # Softmax → log probabilities
        logits -= logits.max()
        log_prob_all = logits - np.log(np.sum(np.exp(logits)))
        log_probs.append(log_prob_all[token_id])

    return float(np.sum(log_probs)) if log_probs else float("-inf")


# ─── ARC-Easy ─────────────────────────────────────────────────────────────────

def evaluate_arc_easy(model: Llama, num_samples: int = None) -> dict:
    """
    Evaluate on ARC-Easy.
    Each example has a question + 3-5 multiple choice options.
    We score each option and pick the highest.
    """
    print("\nLoading ARC-Easy dataset...")
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")

    if num_samples:
        ds = ds.select(range(min(num_samples, len(ds))))

    correct = 0
    total   = 0
    results = []

    for example in tqdm(ds, desc="ARC-Easy"):
        question  = example["question"]
        choices   = example["choices"]
        answer_key = example["answerKey"]   # "A", "B", "C", or "D"

        # Build context
        context = f"Question: {question}\nAnswer:"

        # Score each option
        option_labels = choices["label"]    # ["A", "B", "C", "D"]
        option_texts  = choices["text"]     # ["the sky", "the ground", ...]

        scores = []
        for label, text in zip(option_labels, option_texts):
            score = score_option(model, context, f" {text}")
            scores.append((label, text, score))

        # Pick highest scoring option
        predicted_label = max(scores, key=lambda x: x[2])[0]
        is_correct      = (predicted_label == answer_key)

        correct += int(is_correct)
        total   += 1

        results.append({
            "question":        question,
            "correct_label":   answer_key,
            "predicted_label": predicted_label,
            "correct":         is_correct,
            "scores":          {s[0]: s[2] for s in scores},
        })

    accuracy = correct / total if total > 0 else 0.0
    print(f"ARC-Easy accuracy: {accuracy:.4f} ({correct}/{total})")

    return {
        "task":     "arc_easy",
        "accuracy": accuracy,
        "correct":  correct,
        "total":    total,
        "results":  results,
    }


# ─── HellaSwag ────────────────────────────────────────────────────────────────

def evaluate_hellaswag(model: Llama, num_samples: int = None) -> dict:
    """
    Evaluate on HellaSwag.
    Each example has an activity context + 4 possible sentence endings.
    We score each ending and pick the highest.
    """
    print("\nLoading HellaSwag dataset...")
    ds = load_dataset("Rowan/hellaswag", split="validation")

    if num_samples:
        ds = ds.select(range(min(num_samples, len(ds))))

    correct = 0
    total   = 0
    results = []

    for example in tqdm(ds, desc="HellaSwag"):
        context  = example["ctx"]           # the activity description
        endings  = example["endings"]       # list of 4 possible endings
        label    = int(example["label"])    # index of correct ending (0-3)

        scores = []
        for i, ending in enumerate(endings):
            score = score_option(model, context, " " + ending)
            scores.append(score)

        predicted = int(np.argmax(scores))
        is_correct = (predicted == label)

        correct += int(is_correct)
        total   += 1

        results.append({
            "context":   context,
            "correct":   label,
            "predicted": predicted,
            "is_correct": is_correct,
            "scores":    scores,
        })

    accuracy = correct / total if total > 0 else 0.0
    print(f"HellaSwag accuracy: {accuracy:.4f} ({correct}/{total})")

    return {
        "task":     "hellaswag",
        "accuracy": accuracy,
        "correct":  correct,
        "total":    total,
        "results":  results,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate GGUF model on ARC-Easy and HellaSwag")
    parser.add_argument("--model",   required=True,  help="Path to GGUF model file")
    parser.add_argument("--tasks",   nargs="+",      default=["arc_easy", "hellaswag"],
                        choices=["arc_easy", "hellaswag"], help="Tasks to evaluate")
    parser.add_argument("--output",  required=True,  help="Path to save results JSON")
    parser.add_argument("--label",   required=True,  help="Label for this run e.g. qwen_q4km")
    parser.add_argument("--n-gpu-layers", type=int,  default=99, help="Layers to offload to GPU")
    parser.add_argument("--num-samples", type=int,   default=None,
                        help="Limit samples per task (for quick testing, omit for full eval)")
    args = parser.parse_args()

    # Load model once, reuse across tasks
    model = load_model(args.model, n_gpu_layers=args.n_gpu_layers)

    all_results = {
        "label":      args.label,
        "model_path": args.model,
        "tasks":      {},
    }

    if "arc_easy" in args.tasks:
        all_results["tasks"]["arc_easy"] = evaluate_arc_easy(model, args.num_samples)

    if "hellaswag" in args.tasks:
        all_results["tasks"]["hellaswag"] = evaluate_hellaswag(model, args.num_samples)

    # Print summary
    print("\n" + "="*50)
    print(f"SUMMARY: {args.label}")
    print("="*50)
    for task, res in all_results["tasks"].items():
        print(f"  {task:20s}: {res['accuracy']:.4f} ({res['correct']}/{res['total']})")

    # Save results
    os.makedirs(Path(args.output).parent, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
