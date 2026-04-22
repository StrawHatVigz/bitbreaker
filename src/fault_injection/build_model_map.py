#!/usr/bin/env python3
"""
build_model_map.py

Entry point: parse one GGUF model file and save its byte-range map as JSON.

Usage:
    python build_model_map.py --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
                               --output experiments/maps/qwen2.5-0.5b-q4_k_m_map.json

The shell script scripts/run_build_maps.sh calls this for all 8 models.
"""

import argparse
import os
import sys
import time

# Allow running from any directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_map import ModelMap


QUANT_LABEL_PATTERNS = [
    ('q4_k_m',  'Q4_K_M'),
    ('q4_k_s',  'Q4_K_S'),
    ('q4_k',    'Q4_K'),
    ('q4_0',    'Q4_0'),
    ('q8_0',    'Q8_0'),
    ('fp16',    'FP16'),
    ('f16',     'FP16'),
    ('fp32',    'FP32'),
    ('f32',     'FP32'),
]


def infer_quant_label(model_path: str) -> str:
    name = os.path.basename(model_path).lower()
    for pattern, label in QUANT_LABEL_PATTERNS:
        if pattern in name:
            return label
    return 'UNKNOWN'


def main():
    parser = argparse.ArgumentParser(
        description='Build GGUF tensor byte-range map for fault injection experiments'
    )
    parser.add_argument('--model',  required=True, help='Path to .gguf model file')
    parser.add_argument('--output', required=True, help='Output path for .json map file')
    parser.add_argument('--quant',  default=None,
                        help='Quant label override (e.g. Q4_K_M). Auto-inferred from filename if omitted.')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: Model file not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    quant_label = args.quant or infer_quant_label(args.model)
    file_size_mb = os.path.getsize(args.model) / 1024 / 1024

    print(f"Model    : {args.model}")
    print(f"Size     : {file_size_mb:.1f} MB")
    print(f"Quant    : {quant_label}")
    print(f"Output   : {args.output}")
    print()

    t0 = time.time()
    mm = ModelMap.build(args.model, quant_label)
    elapsed = time.time() - t0

    print(mm.summary())
    print(f"\nParse time: {elapsed:.2f}s")
    print()

    mm.save(args.output)


if __name__ == '__main__':
    main()
