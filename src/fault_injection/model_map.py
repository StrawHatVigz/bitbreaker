"""
model_map.py

Orchestrates GGUFTensorMapper output into a queryable ModelMap:
  - Filters to in-scope tensors (no embeddings)
  - Tags role (attn/ffn/output) and layer index from tensor name
  - Saves to / loads from JSON (one file per model)
  - Exposes a clean query interface for the fault injection tool

No model weights are loaded — this operates on the GGUF binary layout only.
"""

import json
import os
import random
import math
from typing import Dict, List, Optional, Tuple

from gguf_tensor_mapper import GGUFTensorMapper, BLOCK_LAYOUTS, TYPE_TO_LAYOUT_KEY


# ─────────────────────────────────────────────
# FP16 bit-layout metadata (stored in JSON so flip tool can read it)
# ─────────────────────────────────────────────

FP16_BIT_LAYOUT = {
    'sign':         [15],
    'exponent_msb': [14],
    'exponent':     list(range(10, 15)),   # bits 14-10
    'mantissa_msb': [9],
    'mantissa':     list(range(0, 10)),    # bits 9-0
    'mantissa_lsb': [0],
}


class ModelMap:
    """
    Queryable byte-range map for a single GGUF model file.

    Build once:
        mm = ModelMap.build("models/qwen.gguf", quant_label="Q4_K_M")
        mm.save("experiments/maps/qwen_q4_k_m_map.json")

    Load later:
        mm = ModelMap.load("experiments/maps/qwen_q4_k_m_map.json")

    Query (used by fault injection tool):
        sample = mm.get_random_weight_byte(role_filter=['attn', 'ffn'])
        # → { file_offset, tensor_name, role, layer, region_type, ggml_type, block_idx }
    """

    def __init__(self):
        self.model_path:               Optional[str]  = None
        self.quant_label:              Optional[str]  = None   # e.g. "Q4_K_M"
        self.tensor_data_base_offset:  Optional[int]  = None
        self.n_layers:                 Optional[int]  = None
        self.tensors:                  List[Dict]     = []
        self.in_scope_tensors:         List[Dict]     = []

    # ── construction ─────────────────────────────────────────────────────

    @classmethod
    def build(cls, model_path: str, quant_label: str) -> 'ModelMap':
        """Parse a GGUF file and build the ModelMap. This is the slow path — run once per model."""
        mm             = cls()
        mm.model_path  = os.path.abspath(model_path)
        mm.quant_label = quant_label

        mapper        = GGUFTensorMapper(model_path)
        mm.tensors    = mapper.parse()
        mm.tensor_data_base_offset = mapper.tensor_data_base_offset

        mm.in_scope_tensors = [t for t in mm.tensors if t['in_scope']]

        layers       = [t['layer'] for t in mm.tensors if t['layer'] is not None]
        mm.n_layers  = (max(layers) + 1) if layers else 0

        return mm

    @classmethod
    def load(cls, map_path: str) -> 'ModelMap':
        """Load a pre-built ModelMap from JSON. Fast — no file parsing."""
        mm = cls()
        with open(map_path, 'r') as f:
            data = json.load(f)

        mm.model_path              = data['model_path']
        mm.quant_label             = data['quant_label']
        mm.tensor_data_base_offset = data['tensor_data_base_offset']
        mm.n_layers                = data['n_layers']
        mm.tensors                 = data['tensors']
        mm.in_scope_tensors        = [t for t in mm.tensors if t['in_scope']]
        return mm

    def save(self, output_path: str):
        """Serialize to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Embed block layout reference so the JSON is self-contained
        serializable_layouts = {}
        for k, v in BLOCK_LAYOUTS.items():
            serializable_layouts[k] = {
                'weights_per_block': v['weights_per_block'],
                'bytes_per_block':   v['bytes_per_block'],
                'regions':           v['regions'],
            }

        data = {
            'model_path':              self.model_path,
            'quant_label':             self.quant_label,
            'tensor_data_base_offset': self.tensor_data_base_offset,
            'n_layers':                self.n_layers,
            'fp16_bit_layout':         FP16_BIT_LAYOUT,
            'block_layouts_reference': serializable_layouts,
            'tensors':                 self.tensors,
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        size_kb = os.path.getsize(output_path) / 1024
        print(f"  Saved: {output_path}  ({size_kb:.1f} KB, {len(self.in_scope_tensors)} in-scope tensors)")

    # ── layer range helpers ───────────────────────────────────────────────

    def get_early_layer_range(self) -> Tuple[int, int]:
        """First 25% of transformer blocks."""
        return (0, max(0, self.n_layers // 4 - 1))

    def get_late_layer_range(self) -> Tuple[int, int]:
        """Last 25% of transformer blocks."""
        return (max(0, 3 * self.n_layers // 4), self.n_layers - 1)

    # ── query interface (used by fault_injector.py) ───────────────────────

    # Weight region names across all supported formats.
    # Q6_K splits weights into two regions (lower/upper bits of 6-bit values).
    # All other formats use a single 'weights' region.
    WEIGHT_REGION_NAMES = {'weights', 'weights_low', 'weights_high'}

    def get_random_weight_byte(
        self,
        role_filter:  Optional[List[str]]  = None,
        layer_range:  Optional[Tuple[int, int]] = None,
    ) -> Dict:
        """
        Sample a uniformly random byte from weight regions of in-scope tensors.
        Weighted by weight byte count so larger tensors are sampled proportionally.

        Handles both single-region formats (F16, Q8_0, Q4_0, Q4_K) and
        Q6_K's split weight regions (weights_low + weights_high).

        Args:
            role_filter:  e.g. ['attn', 'ffn'] — None means all in-scope roles
            layer_range:  (lo, hi) inclusive — None means all layers

        Returns dict:
            file_offset, tensor_name, role, layer, region_type, ggml_type, block_idx
        """
        candidates = self._filter_tensors(role_filter, layer_range)
        if not candidates:
            raise ValueError(
                f"No in-scope tensors for role_filter={role_filter}, layer_range={layer_range}"
            )

        # Build pool: (tensor, region_name) pairs, weighted by byte count
        pool        = []
        pool_weights = []
        for t in candidates:
            if not t['block_layout']:
                continue
            for region in t['block_layout']:
                if region['name'] in self.WEIGHT_REGION_NAMES:
                    byte_count = t['n_blocks'] * region['num_bytes']
                    pool.append((t, region['name']))
                    pool_weights.append(byte_count)

        if not pool:
            raise ValueError(
                f"No weight regions found for role_filter={role_filter}, layer_range={layer_range}"
            )

        # Weighted random selection of (tensor, region) pair
        total = sum(pool_weights)
        r     = random.uniform(0, total)
        cumsum = 0
        chosen_tensor, chosen_region = pool[-1]
        for (t, rname), w in zip(pool, pool_weights):
            cumsum += w
            if r <= cumsum:
                chosen_tensor, chosen_region = t, rname
                break

        return self._sample_byte_in_region(chosen_tensor, chosen_region)

    def get_random_scale_byte(
        self,
        scale_type:  str,
        role_filter: Optional[List[str]]       = None,
        layer_range: Optional[Tuple[int, int]] = None,
    ) -> Dict:
        """
        Sample a uniformly random byte from a specific scale region type.

        Args:
            scale_type:  one of 'block_scale', 'sub_scales', 'super_scale_d', 'super_scale_dmin'
            role_filter: e.g. ['attn'] or ['ffn'] — None means all in-scope roles
            layer_range: (lo, hi) inclusive — None means all layers

        Returns same dict schema as get_random_weight_byte.
        """
        candidates = self._filter_tensors(role_filter, layer_range)
        candidates = [
            t for t in candidates
            if t['block_layout'] and any(r['name'] == scale_type for r in t['block_layout'])
        ]
        if not candidates:
            raise ValueError(
                f"No in-scope tensors have region type '{scale_type}' "
                f"for role_filter={role_filter}, layer_range={layer_range}"
            )

        tensor = self._sample_tensor_by_region_bytes(candidates, region_name=scale_type)
        return self._sample_byte_in_region(tensor, region_name=scale_type)

    def get_all_byte_ranges_for_region(
        self,
        region_name:  str,
        role_filter:  Optional[List[str]]       = None,
        layer_range:  Optional[Tuple[int, int]] = None,
    ) -> List[Dict]:
        """
        Return ALL byte ranges of a given region type across filtered tensors.
        Used for systematic experiments (Exp 3: scale vs weight, Exp 5: Q4_K_M cascade).

        Returns list of:
            { file_byte_start, file_byte_end, tensor_name, layer, role, region_type, block_idx }
        """
        candidates = self._filter_tensors(role_filter, layer_range)
        results    = []

        for t in candidates:
            if not t['block_layout']:
                continue
            region = next((r for r in t['block_layout'] if r['name'] == region_name), None)
            if region is None:
                continue

            bpb          = t['bytes_per_block']
            block_offset = region['block_offset']
            region_bytes = region['num_bytes']

            for block_idx in range(t['n_blocks']):
                block_start = t['file_byte_start'] + block_idx * bpb
                results.append({
                    'file_byte_start': block_start + block_offset,
                    'file_byte_end':   block_start + block_offset + region_bytes,
                    'tensor_name':     t['name'],
                    'layer':           t['layer'],
                    'role':            t['role'],
                    'region_type':     region_name,
                    'block_idx':       block_idx,
                })

        return results

    # ── summary ───────────────────────────────────────────────────────────

    def summary(self) -> str:
        by_role: Dict[str, Dict] = {}
        for t in self.tensors:
            by_role.setdefault(t['role'], {'count': 0, 'in_scope': 0, 'weights': 0, 'bytes': 0})
            entry = by_role[t['role']]
            entry['count']   += 1
            if t['in_scope']:
                entry['in_scope'] += 1
                entry['weights']  += t['n_weights']
                if t['file_byte_end'] and t['file_byte_start']:
                    entry['bytes'] += t['file_byte_end'] - t['file_byte_start']

        lines = [
            f"Model    : {os.path.basename(self.model_path)}",
            f"Quant    : {self.quant_label}",
            f"Layers   : {self.n_layers}",
            f"Data base: 0x{self.tensor_data_base_offset:010X}",
            f"Tensors  : {len(self.tensors)} total, {len(self.in_scope_tensors)} in scope",
            "",
            f"  {'Role':<12}  {'Total':>5}  {'InScope':>7}  {'Weights':>14}  {'MB':>8}",
            "  " + "-" * 54,
        ]
        for role, d in sorted(by_role.items()):
            mb = d['bytes'] / 1024 / 1024
            lines.append(
                f"  {role:<12}  {d['count']:>5}  {d['in_scope']:>7}  {d['weights']:>14,}  {mb:>8.2f}"
            )

        # List which region types are present (useful sanity check for Q4_K_M vs Q4_0)
        region_types = set()
        for t in self.in_scope_tensors:
            if t['block_layout']:
                for r in t['block_layout']:
                    region_types.add(r['name'])
        lines += ["", f"  Region types in scope: {sorted(region_types)}"]

        return "\n".join(lines)

    # ── internal helpers ──────────────────────────────────────────────────

    def _filter_tensors(
        self,
        role_filter: Optional[List[str]],
        layer_range: Optional[Tuple[int, int]],
    ) -> List[Dict]:
        candidates = self.in_scope_tensors
        if role_filter:
            candidates = [t for t in candidates if t['role'] in role_filter]
        if layer_range:
            lo, hi = layer_range
            candidates = [
                t for t in candidates
                if t['layer'] is not None and lo <= t['layer'] <= hi
            ]
        return candidates

    def _sample_tensor_by_region_bytes(
        self, tensors: List[Dict], region_name: str
    ) -> Dict:
        """
        Sample a tensor with probability proportional to its byte count
        in the given region (so each byte is equally likely to be chosen).
        """
        weights = []
        for t in tensors:
            if not t['block_layout']:
                weights.append(0)
                continue
            region = next((r for r in t['block_layout'] if r['name'] == region_name), None)
            weights.append(t['n_blocks'] * region['num_bytes'] if region else 0)

        total = sum(weights)
        if total == 0:
            return random.choice(tensors)

        r = random.uniform(0, total)
        cumsum = 0
        for t, w in zip(tensors, weights):
            cumsum += w
            if r <= cumsum:
                return t
        return tensors[-1]

    def _sample_byte_in_region(self, tensor: Dict, region_name: str) -> Dict:
        """Pick a uniformly random byte within a given region across all blocks of a tensor."""
        region = next(
            (r for r in tensor['block_layout'] if r['name'] == region_name), None
        )
        if region is None:
            raise ValueError(f"Tensor '{tensor['name']}' has no region '{region_name}'")

        block_idx      = random.randrange(tensor['n_blocks'])
        byte_in_region = random.randrange(region['num_bytes'])

        file_offset = (
            tensor['file_byte_start']
            + block_idx * tensor['bytes_per_block']
            + region['block_offset']
            + byte_in_region
        )

        return {
            'file_offset':   file_offset,
            'tensor_name':   tensor['name'],
            'role':          tensor['role'],
            'layer':         tensor['layer'],
            'region_type':   region_name,
            'ggml_type':     tensor['ggml_type'],
            'block_idx':     block_idx,
        }
