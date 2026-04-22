"""
gguf_tensor_mapper.py

Parses a GGUF binary file and produces a coarse byte-range map for every tensor,
labeling regions as: weight bytes, block_scale, sub_scales, super_scale_d, super_scale_dmin.

Supports: FP16, Q8_0, Q4_0, Q4_K (Q4_K_M / Q4_K_S), Q6_K
"""

import struct
import os
import math
from typing import List, Dict, Optional, Tuple

# ─────────────────────────────────────────────
# GGUF constants
# ─────────────────────────────────────────────

GGUF_MAGIC     = b'GGUF'
GGUF_ALIGNMENT = 32  # tensor data section is aligned to 32 bytes

# GGML type IDs → string names (only types we care about are fully mapped)
GGML_TYPE_NAMES: Dict[int, str] = {
    0:  'F32',
    1:  'F16',
    2:  'Q4_0',
    3:  'Q4_1',
    6:  'Q5_0',
    7:  'Q5_1',
    8:  'Q8_0',
    9:  'Q8_1',
    10: 'Q2_K',
    11: 'Q3_K',
    12: 'Q4_K',   # Q4_K_M and Q4_K_S are both stored as type 12
    13: 'Q5_K',
    14: 'Q6_K',
    15: 'Q8_K',
}

# GGUF metadata value types: id → (name, struct_fmt, byte_size)
# fmt=None means special handling (string / array)
GGUF_METADATA_VALUE_TYPE: Dict[int, Tuple] = {
    0:  ('uint8',   'B', 1),
    1:  ('int8',    'b', 1),
    2:  ('uint16',  'H', 2),
    3:  ('int16',   'h', 2),
    4:  ('uint32',  'I', 4),
    5:  ('int32',   'i', 4),
    6:  ('float32', 'f', 4),
    7:  ('bool',    'B', 1),
    8:  ('string',  None, None),
    9:  ('array',   None, None),
    10: ('uint64',  'Q', 8),
    11: ('int64',   'q', 8),
    12: ('float64', 'd', 8),
}

# ─────────────────────────────────────────────
# Block layout definitions
#
# Each entry describes the internal structure of ONE block:
#   weights_per_block : how many model weights are in one block
#   bytes_per_block   : total bytes for one block
#   regions           : ordered list of sub-regions inside the block
#     - name         : semantic label (used by flip tool to target specific regions)
#     - block_offset : byte offset from the start of the block
#     - num_bytes    : byte count for this region
#     - description  : human note
#
# FP16 is treated as 1-weight blocks for uniformity.
# ─────────────────────────────────────────────

BLOCK_LAYOUTS: Dict[str, Dict] = {
    'F16': {
        'weights_per_block': 1,
        'bytes_per_block': 2,
        'regions': [
            {
                'name': 'weights',
                'block_offset': 0,
                'num_bytes': 2,
                'description': 'FP16 weight: bit15=sign, bits14-10=exponent, bits9-0=mantissa',
                'bit_layout': {
                    'sign':         [15],
                    'exponent_msb': [14],
                    'exponent':     list(range(10, 15)),
                    'mantissa_msb': [9],
                    'mantissa':     list(range(0, 10)),
                    'mantissa_lsb': [0],
                },
            },
        ],
    },

    'Q8_0': {
        'weights_per_block': 32,
        'bytes_per_block': 34,    # 2 (scale) + 32 (int8 weights)
        'regions': [
            {
                'name': 'block_scale',
                'block_offset': 0,
                'num_bytes': 2,
                'description': 'FP16 scale shared across 32 INT8 weights',
            },
            {
                'name': 'weights',
                'block_offset': 2,
                'num_bytes': 32,
                'description': '32 × INT8 weights (bit7=sign, max swing ±127×scale)',
            },
        ],
    },

    'Q4_0': {
        'weights_per_block': 32,
        'bytes_per_block': 18,    # 2 (scale) + 16 (packed nibbles)
        'regions': [
            {
                'name': 'block_scale',
                'block_offset': 0,
                'num_bytes': 2,
                'description': 'FP16 scale shared across 32 4-bit weights',
            },
            {
                'name': 'weights',
                'block_offset': 2,
                'num_bytes': 16,
                'description': '32 × 4-bit weights packed as 16 bytes (2 nibbles/byte, MSB=high nibble)',
            },
        ],
    },

    'Q4_K': {
        # block_q4_K in llama.cpp:
        #   ggml_half d      ( 2B) — super-scale multiplying all 8 sub-block scales
        #   ggml_half dmin   ( 2B) — super-scale multiplying all 8 sub-block mins
        #   uint8 scales[12] (12B) — 8×6-bit sub-scales + 8×6-bit sub-mins packed
        #   uint8 qs[128]   (128B) — 256 × 4-bit weights
        # Total: 144 bytes / 256 weights
        # Note: sub_scales region starts at block_offset=4; weight region at block_offset=16
        #       (offsets 12–15 are the tail of the sub-scales packing, already included in num_bytes=12)
        'weights_per_block': 256,
        'bytes_per_block': 144,
        'regions': [
            {
                'name': 'super_scale_d',
                'block_offset': 0,
                'num_bytes': 2,
                'description': 'FP16 super-scale d — multiplies all 8 sub-block scales; 1 bit flip corrupts all 256 weights',
            },
            {
                'name': 'super_scale_dmin',
                'block_offset': 2,
                'num_bytes': 2,
                'description': 'FP16 super-scale dmin — multiplies all 8 sub-block mins; 1 bit flip corrupts all 256 weights',
            },
            {
                'name': 'sub_scales',
                'block_offset': 4,
                'num_bytes': 12,
                'description': '8×6-bit sub-scales + 8×6-bit sub-mins packed into 12 bytes; each sub-scale affects 32 weights',
            },
            {
                'name': 'weights',
                'block_offset': 16,
                'num_bytes': 128,
                'description': '256 × 4-bit weights packed as 128 bytes (2 nibbles per byte)',
            },
        ],
    },

    'Q6_K': {
        # block_q6_K in llama.cpp (source: ggml-common.h):
        #   uint8_t ql[128]   — lower 4 bits of each 6-bit weight (256 weights × 4 bits / 8)
        #   uint8_t qh[ 64]   — upper 2 bits of each 6-bit weight (256 weights × 2 bits / 8)
        #   int8_t  scales[16]— 16 × INT8 sub-block scales (one per 16 weights)
        #   ggml_half d        — FP16 super-scale multiplying all 16 sub-block scales
        # Total: 128 + 64 + 16 + 2 = 210 bytes / 256 weights
        #
        # Blast radius of a single bit flip:
        #   d (super-scale)     → corrupts all 256 weights by a factor of 2
        #   scales[i] (INT8)    → corrupts 16 weights in that sub-block
        #   ql/qh (weight bits) → corrupts 1 weight
        'weights_per_block': 256,
        'bytes_per_block': 210,
        'regions': [
            {
                'name': 'weights_low',
                'block_offset': 0,
                'num_bytes': 128,
                'description': 'Lower 4 bits of 256 × 6-bit weights, packed as 128 bytes',
            },
            {
                'name': 'weights_high',
                'block_offset': 128,
                'num_bytes': 64,
                'description': 'Upper 2 bits of 256 × 6-bit weights, packed as 64 bytes',
            },
            {
                'name': 'sub_scales',
                'block_offset': 192,
                'num_bytes': 16,
                'description': '16 × INT8 sub-block scales; each affects 16 weights',
            },
            {
                'name': 'super_scale_d',
                'block_offset': 208,
                'num_bytes': 2,
                'description': 'FP16 super-scale d — multiplies all 16 sub-block scales; 1 bit flip corrupts all 256 weights',
            },
        ],
    },
}

# Map GGML type string → BLOCK_LAYOUTS key
# Types not listed here are not in scope for our experiments
TYPE_TO_LAYOUT_KEY: Dict[str, Optional[str]] = {
    'F16':  'F16',
    'Q8_0': 'Q8_0',
    'Q4_0': 'Q4_0',
    'Q4_K': 'Q4_K',
    'Q6_K': 'Q6_K',   # added: mixed into Q4_K_M files for smaller tensors
    # Everything else → None (out of scope)
    'F32':  None,
    'Q4_1': None,
    'Q5_0': None,
    'Q5_1': None,
    'Q2_K': None,
    'Q3_K': None,
    'Q5_K': None,
    'Q8_K': None,
    'Q8_1': None,
}


# ─────────────────────────────────────────────
# Tensor role classification
# ─────────────────────────────────────────────

def classify_tensor(name: str) -> Tuple[str, Optional[int]]:
    """
    Returns (role, layer_index).
    role ∈ {'attn', 'ffn', 'output', 'embedding', 'norm', 'other'}
    layer_index is None for non-block tensors.
    """
    # Embedding table
    if name in ('token_embd.weight', 'token_embd_norm.weight'):
        return 'embedding', None

    # Output projection and output norm
    if name == 'output.weight':
        return 'output', None
    if name in ('output_norm.weight', 'output_norm.bias'):
        return 'norm', None

    # Transformer blocks: blk.<layer>.<rest>
    if name.startswith('blk.'):
        parts = name.split('.')
        try:
            layer = int(parts[1])
        except (IndexError, ValueError):
            return 'other', None

        rest = '.'.join(parts[2:])

        if any(x in rest for x in ('attn_q', 'attn_k', 'attn_v', 'attn_output', 'attn_out')):
            return 'attn', layer
        if any(x in rest for x in ('ffn_up', 'ffn_down', 'ffn_gate')):
            return 'ffn', layer
        if 'attn_norm' in rest or 'ffn_norm' in rest:
            return 'norm', layer
        if 'attn' in rest:
            return 'attn', layer
        if 'ffn' in rest:
            return 'ffn', layer

        return 'other', layer

    return 'other', None


# ─────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────

class GGUFTensorMapper:
    """
    Parses a GGUF binary file and builds a coarse byte-range map for every tensor.

    Usage:
        mapper = GGUFTensorMapper("models/qwen2.5-0.5b-q4_k_m.gguf")
        tensors = mapper.parse()
        # tensors is a list of dicts — see _build_tensor_maps() for schema
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.file_size   = os.path.getsize(model_path)

        # Populated during parse()
        self.version:                  Optional[int]  = None
        self.n_tensors:                Optional[int]  = None
        self.n_kv:                     Optional[int]  = None
        self.tensor_data_base_offset:  Optional[int]  = None

        self._f                        = None
        self._tensor_descriptors_end:  Optional[int]  = None

    # ── public ──────────────────────────────────────────────────────────

    def parse(self) -> List[Dict]:
        """
        Main entry point.
        Returns a list of tensor info dicts (see _build_tensor_maps for schema).
        Also sets self.tensor_data_base_offset.
        """
        with open(self.model_path, 'rb') as f:
            self._f = f
            self._parse_header()
            self._skip_kv_metadata()
            raw_tensors = self._parse_tensor_descriptors()
            self._compute_tensor_data_base_offset()

        return self._build_tensor_maps(raw_tensors)

    # ── header / metadata parsing ────────────────────────────────────────

    def _read_fmt(self, fmt: str) -> tuple:
        """Read and unpack a little-endian struct from the current file position."""
        if not fmt:
            return ()
        size = struct.calcsize('<' + fmt)
        data = self._f.read(size)
        if len(data) < size:
            raise ValueError(
                f"Unexpected EOF at offset 0x{self._f.tell():08X} "
                f"(wanted {size} bytes, got {len(data)})"
            )
        return struct.unpack('<' + fmt, data)

    def _read_string(self) -> str:
        (length,) = self._read_fmt('Q')
        raw = self._f.read(length)
        if len(raw) < length:
            raise ValueError(f"Truncated string at offset 0x{self._f.tell():08X}")
        return raw.decode('utf-8', errors='replace')

    def _parse_header(self):
        magic = self._f.read(4)
        if magic != GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file — bad magic: {magic!r}")
        (self.version,)  = self._read_fmt('I')
        (self.n_tensors,) = self._read_fmt('Q')
        (self.n_kv,)     = self._read_fmt('Q')

    def _skip_kv_value(self, value_type: int):
        """Skip a single metadata value (recursive for arrays)."""
        if value_type not in GGUF_METADATA_VALUE_TYPE:
            raise ValueError(f"Unknown metadata value type: {value_type}")

        type_name, fmt, size = GGUF_METADATA_VALUE_TYPE[value_type]

        if type_name == 'string':
            (length,) = self._read_fmt('Q')
            self._f.read(length)

        elif type_name == 'array':
            (elem_type,) = self._read_fmt('I')
            (count,)     = self._read_fmt('Q')
            for _ in range(count):
                self._skip_kv_value(elem_type)

        else:
            self._f.read(size)

    def _skip_kv_metadata(self):
        for _ in range(self.n_kv):
            _key = self._read_string()
            (value_type,) = self._read_fmt('I')
            self._skip_kv_value(value_type)

    # ── tensor descriptor table ──────────────────────────────────────────

    def _parse_tensor_descriptors(self) -> List[Dict]:
        raw = []
        for _ in range(self.n_tensors):
            name         = self._read_string()
            (n_dims,)    = self._read_fmt('I')
            dims         = list(self._read_fmt('Q' * n_dims)) if n_dims > 0 else []
            (type_id,)   = self._read_fmt('I')
            (rel_offset,) = self._read_fmt('Q')   # relative to tensor data base

            ggml_type = GGML_TYPE_NAMES.get(type_id, f'UNKNOWN_{type_id}')
            n_weights = 1
            for d in dims:
                n_weights *= d

            raw.append({
                'name':            name,
                'ggml_type':       ggml_type,
                'dims':            dims,
                'n_weights':       n_weights,
                'offset_relative': rel_offset,
            })

        self._tensor_descriptors_end = self._f.tell()
        return raw

    def _compute_tensor_data_base_offset(self):
        pos       = self._tensor_descriptors_end
        remainder = pos % GGUF_ALIGNMENT
        if remainder != 0:
            pos += GGUF_ALIGNMENT - remainder
        self.tensor_data_base_offset = pos

    # ── byte-range map construction ──────────────────────────────────────

    def _byte_size_for_tensor(self, ggml_type: str, n_weights: int) -> int:
        """Compute the raw byte size of a tensor's data region."""
        layout_key = TYPE_TO_LAYOUT_KEY.get(ggml_type)
        if layout_key is None:
            if ggml_type == 'F32':
                return n_weights * 4
            return 0  # unknown — skip

        layout = BLOCK_LAYOUTS[layout_key]
        wpb    = layout['weights_per_block']
        bpb    = layout['bytes_per_block']
        n_blocks = n_weights if wpb == 1 else math.ceil(n_weights / wpb)
        return n_blocks * bpb

    def _build_tensor_maps(self, raw_tensors: List[Dict]) -> List[Dict]:
        """
        For each tensor, produce:
        {
          name, ggml_type, role, layer, dims, n_weights,
          file_byte_start, file_byte_end,
          n_blocks, bytes_per_block,
          in_scope, reason_out_of_scope,
          block_layout: [{ name, block_offset, num_bytes, description, ... }]
        }
        """
        result = []

        for t in raw_tensors:
            name      = t['name']
            ggml_type = t['ggml_type']
            n_weights = t['n_weights']

            file_byte_start = self.tensor_data_base_offset + t['offset_relative']
            role, layer     = classify_tensor(name)
            layout_key      = TYPE_TO_LAYOUT_KEY.get(ggml_type)

            # ── unsupported quant type (F32 norms, Q5, etc.) ──
            if layout_key is None:
                byte_size = self._byte_size_for_tensor(ggml_type, n_weights)
                result.append({
                    'name':               name,
                    'ggml_type':          ggml_type,
                    'role':               role,
                    'layer':              layer,
                    'dims':               t['dims'],
                    'n_weights':          n_weights,
                    'n_blocks':           None,
                    'bytes_per_block':    None,
                    'file_byte_start':    file_byte_start,
                    'file_byte_end':      file_byte_start + byte_size,
                    'in_scope':           False,
                    'reason_out_of_scope': f'Unsupported quantization type: {ggml_type}',
                    'block_layout':       None,
                })
                continue

            layout  = BLOCK_LAYOUTS[layout_key]
            wpb     = layout['weights_per_block']
            bpb     = layout['bytes_per_block']
            n_blocks = n_weights if wpb == 1 else math.ceil(n_weights / wpb)

            file_byte_end = file_byte_start + n_blocks * bpb

            # ── scope decision ──
            if role == 'embedding':
                in_scope           = False
                reason_out_of_scope = 'Embedding tensor excluded from flip pool'
            else:
                in_scope           = True
                reason_out_of_scope = None

            result.append({
                'name':               name,
                'ggml_type':          ggml_type,
                'role':               role,
                'layer':              layer,
                'dims':               t['dims'],
                'n_weights':          n_weights,
                'n_blocks':           n_blocks,
                'bytes_per_block':    bpb,
                'file_byte_start':    file_byte_start,
                'file_byte_end':      file_byte_end,
                'in_scope':           in_scope,
                'reason_out_of_scope': reason_out_of_scope,
                'block_layout':       layout['regions'],
            })

        return result
