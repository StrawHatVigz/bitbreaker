"""
parse_reports.py — Unified baseline results parser for BitBreaker

Reads perplexity logs and task JSON files and outputs one complete
baseline table with all metrics: PPL, GPU memory, tokens/sec, ARC-Easy, HellaSwag.

Usage:
    python scripts/parse_reports.py

    # Specify platform explicitly (auto-detected by default)
    python scripts/parse_reports.py --platform mac
    python scripts/parse_reports.py --platform grendel

    # Save to file
    python scripts/parse_reports.py --output experiments/results/baseline/baseline_summary.txt
"""

import argparse
import json
import re
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent.parent
PERP_DIR   = BASE_DIR / "experiments/results/baseline/perplexity"
TASKS_DIR  = BASE_DIR / "experiments/results/baseline/tasks"

MODEL_ORDER = [
    "qwen_fp16",
    "qwen_q8",
    "qwen_q4km",
    "qwen_q4",
    "llama_fp16",
    "llama_q8",
    "llama_q4km",
    "llama_q4",
]

MODEL_LABELS = {
    "qwen_fp16":  "Qwen 0.5B  FP16",
    "qwen_q8":    "Qwen 0.5B  Q8  ",
    "qwen_q4km":  "Qwen 0.5B  Q4KM",
    "qwen_q4":    "Qwen 0.5B  Q4  ",
    "llama_fp16": "Llama 1B   FP16",
    "llama_q8":   "Llama 1B   Q8  ",
    "llama_q4km": "Llama 1B   Q4KM",
    "llama_q4":   "Llama 1B   Q4  ",
}

# ── Perplexity Log Parser ─────────────────────────────────────────────────────

def parse_perplexity_log(log_path: Path, platform: str) -> dict:
    """Extract PPL, GPU memory, and tokens/sec from a llama-perplexity log."""
    if not log_path.exists():
        return {}

    text = log_path.read_text()

    # PPL
    ppl_match = re.search(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)", text)
    ppl   = ppl_match.group(1) if ppl_match else "N/A"
    ppl_err = ppl_match.group(2) if ppl_match else "N/A"

    # GPU memory — different label on Mac (MTL0) vs Grendel (CUDA)
    if platform == "mac":
        mem_match = re.search(r"MTL0_Mapped model buffer size =\s+([\d.]+) MiB", text)
    else:
        mem_match = re.search(r"CUDA0\s+model buffer size =\s+([\d.]+) MiB", text)
        if not mem_match:
            # fallback pattern used by some llama.cpp versions
            mem_match = re.search(r"VRAM used:\s+([\d.]+) MiB", text)

    mem = mem_match.group(1) if mem_match else "N/A"

    # Tokens/sec
    toks_match = re.search(r"([\d.]+) tokens per second", text)
    toks = toks_match.group(1) if toks_match else "N/A"

    return {"ppl": ppl, "ppl_err": ppl_err, "mem_mib": mem, "tok_per_sec": toks}


# ── Task JSON Parser ───────────────────────────────────────────────────────────

def parse_task_json(json_path: Path) -> dict:
    """Extract ARC-Easy and HellaSwag accuracy from a task results JSON."""
    if not json_path.exists():
        return {}

    with open(json_path) as f:
        data = json.load(f)

    arc   = data["tasks"].get("arc_easy", {})
    hella = data["tasks"].get("hellaswag", {})

    return {
        "arc_acc":   arc.get("accuracy", None),
        "arc_n":     arc.get("total", None),
        "hella_acc": hella.get("accuracy", None),
        "hella_n":   hella.get("total", None),
    }


# ── Formatting Helpers ────────────────────────────────────────────────────────

def fmt_acc(acc, n):
    if acc is None:
        return "N/A"
    return f"{acc*100:.2f}% ({n})"

def fmt_mem(mem):
    if mem == "N/A":
        return "N/A"
    return f"{float(mem):.0f} MiB"

def fmt_toks(toks):
    if toks == "N/A":
        return "N/A"
    return f"{float(toks):.0f} t/s"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Parse BitBreaker baseline results")
    parser.add_argument("--platform", choices=["mac", "grendel"], default=None,
                        help="Platform (auto-detected from log content if not specified)")
    parser.add_argument("--output", default=None,
                        help="Save summary to this file path")
    args = parser.parse_args()

    # Auto-detect platform from first available log
    platform = args.platform
    if platform is None:
        for label in MODEL_ORDER:
            log = PERP_DIR / f"{label}_perplexity.log"
            if log.exists():
                text = log.read_text()
                platform = "mac" if "MTL0" in text else "grendel"
                break
        if platform is None:
            platform = "grendel"  # default fallback

    lines = []

    header = (
        f"\n{'='*90}\n"
        f"  BitBreaker Baseline Results  |  Platform: {platform.upper()}\n"
        f"{'='*90}\n"
        f"{'Model':<18} {'PPL':>8} {'±':>6} {'GPU Mem':>10} {'Tok/s':>8} "
        f"{'ARC-Easy':>16} {'HellaSwag':>16}\n"
        f"{'-'*90}"
    )
    lines.append(header)

    prev_family = None
    for label in MODEL_ORDER:
        family = "qwen" if "qwen" in label else "llama"
        if prev_family and family != prev_family:
            lines.append("")  # blank line between model families
        prev_family = family

        perp = parse_perplexity_log(PERP_DIR / f"{label}_perplexity.log", platform)
        task = parse_task_json(TASKS_DIR / f"{label}_tasks.json")

        name      = MODEL_LABELS.get(label, label)
        ppl       = perp.get("ppl", "N/A")
        ppl_err   = perp.get("ppl_err", "N/A")
        mem       = fmt_mem(perp.get("mem_mib", "N/A"))
        toks      = fmt_toks(perp.get("tok_per_sec", "N/A"))
        arc_str   = fmt_acc(task.get("arc_acc"), task.get("arc_n"))
        hella_str = fmt_acc(task.get("hella_acc"), task.get("hella_n"))

        line = (
            f"{name:<18} {ppl:>8} {ppl_err:>6} {mem:>10} {toks:>8} "
            f"{arc_str:>16} {hella_str:>16}"
        )
        lines.append(line)

    lines.append("=" * 90)
    lines.append(f"Perplexity logs : {PERP_DIR}")
    lines.append(f"Task results    : {TASKS_DIR}")

    output = "\n".join(lines)
    print(output)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output)
        print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
