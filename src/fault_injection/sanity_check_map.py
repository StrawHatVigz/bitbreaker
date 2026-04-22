#!/usr/bin/env python3
"""
sanity_check_map.py

Verifies a model map JSON is structurally correct and safe to use for fault injection.
Does NOT run model inference — purely validates the GGUF byte-range map.

Checks:
  1. All tensor byte ranges are within the actual file size
  2. No two tensors have overlapping byte ranges
  3. Block layout math is internally consistent
     (sum of region bytes == bytes_per_block, n_blocks * bpb == file range)
  4. Tensor scope breakdown is visually sensible
  5. Flip-and-restore: actually writes a flipped byte to the model file and restores it
     (verifies file I/O and offset math work end-to-end)

Usage:
    python sanity_check_map.py --map experiments/maps/qwen2.5-0.5b-q4_k_m_map.json
"""

import argparse
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_map import ModelMap


# ─────────────────────────────────────────────
# Individual checks
# ─────────────────────────────────────────────

def check_file_bounds(mm: ModelMap) -> bool:
    """No tensor should extend past the end of the file."""
    file_size = os.path.getsize(mm.model_path)
    errors = 0
    for t in mm.tensors:
        if t['file_byte_end'] is None:
            continue
        if t['file_byte_end'] > file_size:
            print(f"  FAIL  {t['name']}: ends at byte {t['file_byte_end']:,} "
                  f"but file is only {file_size:,} bytes")
            errors += 1
    if errors == 0:
        print(f"  PASS  All tensor ranges within file bounds  "
              f"(file = {file_size / 1024 / 1024:.1f} MB)")
    return errors == 0


def check_no_overlap(mm: ModelMap) -> bool:
    """
    Tensor byte ranges must not overlap.
    (They can be non-contiguous — GGUF packs them back-to-back with per-tensor alignment,
     but there should be no overlap.)
    """
    ranges = [
        (t['file_byte_start'], t['file_byte_end'], t['name'])
        for t in mm.tensors
        if t['file_byte_start'] is not None and t['file_byte_end'] is not None
    ]
    ranges.sort(key=lambda x: x[0])

    errors = 0
    for i in range(len(ranges) - 1):
        s1, e1, n1 = ranges[i]
        s2, e2, n2 = ranges[i + 1]
        if e1 > s2:
            print(f"  FAIL  Overlap: '{n1}' ends at {e1:,}, '{n2}' starts at {s2:,}")
            errors += 1

    if errors == 0:
        print(f"  PASS  No overlapping tensor byte ranges  ({len(ranges)} tensors checked)")
    return errors == 0


def check_block_math(mm: ModelMap) -> bool:
    """
    For every in-scope tensor:
      (a) sum of region bytes == bytes_per_block
      (b) n_blocks * bytes_per_block == file_byte_end - file_byte_start
    """
    errors = 0
    for t in mm.in_scope_tensors:
        if not t['block_layout']:
            continue

        # (a) region sum
        region_sum = sum(r['num_bytes'] for r in t['block_layout'])
        if region_sum != t['bytes_per_block']:
            print(f"  FAIL  {t['name']}: region bytes sum = {region_sum} "
                  f"but bytes_per_block = {t['bytes_per_block']}")
            errors += 1

        # (b) file range
        expected = t['n_blocks'] * t['bytes_per_block']
        actual   = t['file_byte_end'] - t['file_byte_start']
        if expected != actual:
            print(f"  FAIL  {t['name']}: n_blocks×bpb = {expected:,} "
                  f"but file range = {actual:,}")
            errors += 1

    if errors == 0:
        print(f"  PASS  Block layout math consistent  ({len(mm.in_scope_tensors)} in-scope tensors)")
    return errors == 0


def check_scope_breakdown(mm: ModelMap) -> bool:
    """Print scope counts by role — visual inspection."""
    by_role: dict = {}
    for t in mm.tensors:
        by_role.setdefault(t['role'], {'in': 0, 'out': 0})
        if t['in_scope']:
            by_role[t['role']]['in'] += 1
        else:
            by_role[t['role']]['out'] += 1

    print(f"  {'Role':<12}  {'In-scope':>8}  {'Excluded':>8}")
    print("  " + "-" * 32)
    for role, counts in sorted(by_role.items()):
        marker = ""
        if role == 'embedding' and counts['in'] > 0:
            marker = "  ← WARN: embeddings should be excluded"
        print(f"  {role:<12}  {counts['in']:>8}  {counts['out']:>8}{marker}")

    # Warn if no attn or ffn tensors are in scope
    issues = 0
    if not any(t['role'] == 'attn' and t['in_scope'] for t in mm.tensors):
        print("  WARN  No in-scope attention tensors found")
        issues += 1
    if not any(t['role'] == 'ffn' and t['in_scope'] for t in mm.tensors):
        print("  WARN  No in-scope FFN tensors found")
        issues += 1
    if issues == 0:
        print("  PASS  attn and ffn tensors present in scope")

    return issues == 0


def check_region_types(mm: ModelMap) -> bool:
    """
    Verify that the expected region types exist for each quant format.
    Q4_K_M must have super_scale_d; Q8_0/Q4_0 must have block_scale.
    """
    region_types = set()
    for t in mm.in_scope_tensors:
        if t['block_layout']:
            for r in t['block_layout']:
                region_types.add(r['name'])

    expected = {
        # Q4_K_M files mix Q4_K (large tensors) and Q6_K (small tensors).
        # We check that at minimum the Q4_K regions are present.
        'Q4_K_M': {'super_scale_d', 'super_scale_dmin', 'sub_scales', 'weights'},
        'Q4_K_S': {'super_scale_d', 'super_scale_dmin', 'sub_scales', 'weights'},
        'Q8_0':   {'block_scale', 'weights'},
        'Q4_0':   {'block_scale', 'weights'},
        'FP16':   {'weights'},
    }

    expected_for_quant = expected.get(mm.quant_label, set())
    missing = expected_for_quant - region_types
    if missing:
        print(f"  FAIL  Expected region types missing for {mm.quant_label}: {missing}")
        return False

    print(f"  PASS  All expected region types present: {sorted(region_types)}")
    return True


def check_flip_restore(mm: ModelMap) -> bool:
    """
    Actually flip a byte in the model file at an offset from the map, verify it changed,
    then restore it. Confirms file I/O and offset math end-to-end.
    Does not run inference.
    """
    try:
        target = mm.get_random_weight_byte(role_filter=['attn', 'ffn'])
    except ValueError as e:
        print(f"  SKIP  Could not sample a target byte: {e}")
        return True

    offset     = target['file_offset']
    bit_to_flip = random.randint(0, 7)

    try:
        with open(mm.model_path, 'r+b') as f:
            f.seek(offset)
            original_byte = f.read(1)[0]

            flipped_byte = original_byte ^ (1 << bit_to_flip)

            # Write flip
            f.seek(offset)
            f.write(bytes([flipped_byte]))
            f.flush()

            # Read back and verify
            f.seek(offset)
            readback = f.read(1)[0]
            if readback != flipped_byte:
                print(f"  FAIL  Write did not stick at offset 0x{offset:010X}")
                return False

            # Restore
            f.seek(offset)
            f.write(bytes([original_byte]))
            f.flush()

            # Verify restore
            f.seek(offset)
            final = f.read(1)[0]
            if final != original_byte:
                print(f"  FAIL  Restore failed at offset 0x{offset:010X}  "
                      f"(got 0x{final:02X}, expected 0x{original_byte:02X})")
                return False

    except PermissionError:
        print(f"  SKIP  File is read-only — cannot test flip-restore  ({mm.model_path})")
        return True

    print(
        f"  PASS  Flip-restore at 0x{offset:010X}  "
        f"tensor={target['tensor_name']}  bit={bit_to_flip}\n"
        f"        0x{original_byte:02X} → 0x{flipped_byte:02X} → 0x{original_byte:02X}"
    )
    return True


# ─────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Sanity-check a BitBreaker model map JSON'
    )
    parser.add_argument('--map', required=True, help='Path to model map .json file')
    args = parser.parse_args()

    if not os.path.exists(args.map):
        print(f"ERROR: Map file not found: {args.map}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading  : {args.map}")
    mm = ModelMap.load(args.map)
    print(f"Model    : {mm.model_path}")
    print(f"Quant    : {mm.quant_label}")
    print(f"Layers   : {mm.n_layers}")
    print()

    checks = [
        ("File bounds",          lambda: check_file_bounds(mm)),
        ("No overlap",           lambda: check_no_overlap(mm)),
        ("Block layout math",    lambda: check_block_math(mm)),
        ("Scope breakdown",      lambda: check_scope_breakdown(mm)),
        ("Region types",         lambda: check_region_types(mm)),
        ("Flip-and-restore",     lambda: check_flip_restore(mm)),
    ]

    results = []
    for name, fn in checks:
        print(f"[{name}]")
        try:
            passed = fn()
        except Exception as e:
            print(f"  ERROR  {e}")
            passed = False
        results.append((name, passed))
        print()

    # Summary line
    n_passed = sum(1 for _, p in results if p)
    n_total  = len(results)
    print("─" * 48)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}]  {name}")
    print("─" * 48)
    print(f"  {n_passed}/{n_total} checks passed")

    if n_passed < n_total:
        sys.exit(1)


if __name__ == '__main__':
    main()
