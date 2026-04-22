#!/usr/bin/env python3
import os
import sys
import time
from pathlib import Path

# Add the project directories to sys.path so we can import our modules
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src/fault_injection"))

from model_map import ModelMap
import sanity_check_map as checker

# ── Configuration ─────────────────────────────────────────────────────────────

MODELS_DIR = project_root / "models"
MAPS_DIR = project_root / "experiments/maps"

MODELS = [
    "qwen2.5-0.5b-instruct-fp16.gguf",
    "qwen2.5-0.5b-instruct-q8_0.gguf",
    "qwen2.5-0.5b-instruct-q4_k_m.gguf",
    "qwen2.5-0.5b-instruct-q4_0.gguf",
    "Llama-3.2-1B-Instruct-f16.gguf",
    "Llama-3.2-1B-Instruct-Q8_0.gguf",
    "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    "Llama-3.2-1B-Instruct-Q4_0.gguf",
]

def run_model_pipeline(model_filename: str):
    """Orchestrates the build and check for a single model."""
    model_path = MODELS_DIR / model_filename
    map_path = MAPS_DIR / f"{model_path.stem}_map.json"
    
    if not model_path.exists():
        return "SKIPPED (File not found)"

    print(f"\n{'═'*60}\n  Model: {model_filename}\n{'═'*60}")

    try:
        # 1. Build & Save Map
        print(f"  [1/2] Building map...")
        # Infer quant from filename (simulating your build_model_map.py logic)
        quant = "FP16" if "f16" in model_filename.lower() else model_filename.split('-')[-1].replace('.gguf', '').upper()
        
        mm = ModelMap.build(str(model_path), quant)
        mm.save(str(map_path))

        # 2. Sanity Checks
        print(f"\n  [2/2] Running sanity checks...")
        checks = [
            ("File Bounds", lambda: checker.check_file_bounds(mm)),
            ("No Overlap", lambda: checker.check_no_overlap(mm)),
            ("Block Math", lambda: checker.check_block_math(mm)),
            ("Scope",      lambda: checker.check_scope_breakdown(mm)),
            ("Regions",    lambda: checker.check_region_types(mm)),
            ("Flip-Restore", lambda: checker.check_flip_restore(mm)),
        ]

        failed_checks = []
        for name, func in checks:
            if not func():
                failed_checks.append(name)
        
        if failed_checks:
            return f"FAILED ({', '.join(failed_checks)})"
        
        return "PASSED"

    except Exception as e:
        return f"ERROR: {str(e)}"

def main():
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    start_time = time.time()

    for model in MODELS:
        status = run_model_pipeline(model)
        results[model] = status

    # ── Final Summary ──────────────────────────────────────────────────────────
    print(f"\n\n{'═'*60}\n  FINAL SUMMARY\n{'═'*60}")
    for model, status in results.items():
        icon = "✓" if status == "PASSED" else "✗" if "FAILED" in status or "ERROR" in status else "-"
        print(f"  {icon} {model:<40} : {status}")
    
    elapsed = time.time() - start_time
    print(f"\nFinished in {elapsed:.2f}s")

if __name__ == "__main__":
    main()