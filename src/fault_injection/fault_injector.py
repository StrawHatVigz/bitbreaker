"""
fault_injector.py

Core fault injection engine for BitBreaker.

Handles the full flip -> benchmark -> restore cycle for one experiment run:
  1. Sample N bit-flip targets from a ModelMap
  2. Apply all flips to the GGUF file on disk
  3. Run benchmarks (perplexity and/or tasks) against the corrupted file
  4. Restore the file -- guaranteed via try/finally, survives eval crashes
  5. Return and save a structured results dict
"""

import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))
from model_map import ModelMap

# Match baseline scripts exactly
PERPLEXITY_CTX   = 512
TASK_NUM_SAMPLES = 2500


class FaultInjector:
    """
    Runs one fault injection experiment: inject -> benchmark -> restore -> save results.

    Args:
        model_map       : loaded ModelMap for the target model
        project_root    : Path to bitbreaker/ root
        n_gpu_layers    : passed to llama-perplexity and evaluate_tasks.py
        platform        : 'mac' or 'grendel'
        run_perplexity  : if True, run perplexity benchmark
        run_tasks       : if True, run ARC-Easy + HellaSwag
    """

    def __init__(
        self,
        model_map:      ModelMap,
        project_root:   Path,
        n_gpu_layers:   int  = 99,
        platform:       str  = 'mac',
        run_perplexity: bool = True,
        run_tasks:      bool = True,
    ):
        self.mm             = model_map
        self.project_root   = Path(project_root)
        self.n_gpu_layers   = n_gpu_layers
        self.platform       = platform
        self.run_perplexity = run_perplexity
        self.run_tasks      = run_tasks

        self.llama_perplexity_bin = (
            self.project_root / "llama.cpp" / "build" / "bin" / "llama-perplexity"
        )
        self.wikitext_path = self.project_root / "configs" / "wikitext2_test.txt"
        self.eval_script   = self.project_root / "src" / "evaluation" / "evaluate_tasks.py"

        self._validate_paths()

    def _validate_paths(self):
        missing = []
        if self.run_perplexity and not self.llama_perplexity_bin.exists():
            missing.append(str(self.llama_perplexity_bin))
        if self.run_perplexity and not self.wikitext_path.exists():
            missing.append(str(self.wikitext_path))
        if self.run_tasks and not self.eval_script.exists():
            missing.append(str(self.eval_script))
        if missing:
            raise FileNotFoundError(
                "Required files not found:\n" + "\n".join(f"  {p}" for p in missing)
            )

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(
        self,
        flip_count:    int,
        seed:          int,
        region_filter: Optional[str]            = None,
        role_filter:   Optional[List[str]]      = None,
        layer_range:   Optional[Tuple[int,int]] = None,
        output_path:   Optional[Path]           = None,
        bit_position:  Optional[int]            = None,
        high_impact:   bool                     = False, # Exp 2 Targeted Logic
    ) -> Dict:
        """
        Run one complete inject -> benchmark -> restore cycle.
        """
        if role_filter is None:
            role_filter = ['attn', 'ffn', 'output']

        random.seed(seed)

        bit_label = str(bit_position) if bit_position is not None else ('high_impact' if high_impact else 'random')
        print(f"\n  flip_count={flip_count}  seed={seed}  "
              f"region={region_filter or 'weights'}  "
              f"mode={bit_label}  roles={role_filter}")

        # Use updated sampling logic that handles targeted flips
        flip_targets = self._sample_flips(
            flip_count, region_filter, role_filter, layer_range, bit_position, high_impact
        )

        flip_records = []
        ppl_result   = None
        task_result  = None
        error        = None

        try:
            flip_records = self._apply_flips(flip_targets)
            print(f"    Applied {len(flip_records)} flips")

            if self.run_perplexity:
                t0 = time.time()
                ppl_result = self._run_perplexity()
                print(f"    PPL: {ppl_result.get('ppl', 'ERR')}  ({time.time()-t0:.0f}s)")

            if self.run_tasks:
                t0 = time.time()
                task_result = self._run_tasks()
                arc   = task_result.get('arc_easy', {}).get('accuracy', 'ERR')
                hella = task_result.get('hellaswag', {}).get('accuracy', 'ERR')
                if isinstance(arc, float):
                    arc = f"{arc*100:.2f}%"
                if isinstance(hella, float):
                    hella = f"{hella*100:.2f}%"
                print(f"    ARC-Easy: {arc}  HellaSwag: {hella}  ({time.time()-t0:.0f}s)")

        except Exception as e:
            error = str(e)
            print(f"    ERROR during eval: {e}")

        finally:
            restored = self._restore_flips(flip_records)
            if restored:
                print(f"    Restored {restored} bytes")
            else:
                print(f"    WARNING: restore returned 0 -- check file manually")

        results = {
            'model_path':    str(self.mm.model_path),
            'quant_label':   self.mm.quant_label,
            'platform':      self.platform,
            'flip_count':    flip_count,
            'seed':          seed,
            'region_filter': region_filter or 'weights',
            'bit_mode':      bit_label,
            'role_filter':   role_filter,
            'layer_range':   list(layer_range) if layer_range else None,
            'flips':         flip_records,
            'perplexity':    ppl_result,
            'tasks':         task_result,
            'error':         error,
        }

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)

        return results

    # ── Flip sampling ─────────────────────────────────────────────────────────

    def _sample_flips(
        self,
        n:             int,
        region_filter: Optional[str],
        role_filter:   List[str],
        layer_range:   Optional[Tuple[int, int]],
        bit_position:  Optional[int],
        high_impact:   bool = False,
    ) -> List[Dict]:
        """
        Sample N bit-flip targets from the model map.
        
        Logic Flow:
          1. If bit_position is set, first flip is hardcoded (Exp 1 legacy).
          2. If high_impact is True, ALL flips hit targeted "Nuclear Zones".
          3. Otherwise, flips are fully random.
        """
        targets = []
        for i in range(n):
            # Select the target byte region
            if region_filter is None or region_filter == 'weights':
                sample = self.mm.get_random_weight_byte(
                    role_filter=role_filter,
                    layer_range=layer_range,
                )
            else:
                sample = self.mm.get_random_scale_byte(
                    scale_type  = region_filter,
                    role_filter = role_filter,
                    layer_range = layer_range,
                )

            # --- Apply Bit Selection Logic ---
            
            # CASE 1: Manual bit_position override (First flip only)
            if i == 0 and bit_position is not None:
                byte_within_word = bit_position // 8
                bit_in_byte      = bit_position % 8
                sample['file_offset']             = sample['file_offset'] + byte_within_word
                sample['bit']                     = bit_in_byte
                sample['bit_position_word_level'] = bit_position
            
            # CASE 2: Targeted High-Impact Zone (Exp 2/3)
            elif high_impact:
                # Check if we are hitting a scale or an FP16 weight
                is_scale = region_filter is not None and region_filter != 'weights'
                
                if is_scale or self.mm.quant_label == 'FP16':
                    # Exponent zone: word-level bits 10 to 14
                    target_bit_word = random.randint(10, 14)
                    sample['file_offset'] += (target_bit_word // 8)
                    sample['bit'] = target_bit_word % 8
                    sample['targeted_zone'] = "FP16_Exponent"
                
                elif self.mm.quant_label == 'Q8_0':
                    # INT8 Sign Bit (Bit 7)
                    sample['bit'] = 7
                    sample['targeted_zone'] = "INT8_Sign"
                
                else: # Q4_0 or Q4_K_M
                    # Nibble MSB (Bit 3)
                    sample['bit'] = 3
                    sample['targeted_zone'] = "Nibble_MSB"

            # CASE 3: Standard Random Flip (Exp 1)
            else:
                sample['bit'] = random.randint(0, 7)

            targets.append(sample)

        return targets

    # ── File I/O: apply + restore ─────────────────────────────────────────────

    def _apply_flips(self, targets: List[Dict]) -> List[Dict]:
        """
        Write all bit flips to the GGUF file on disk.
        """
        records = []
        with open(self.mm.model_path, 'r+b') as f:
            for t in targets:
                offset = t['file_offset']
                bit    = t['bit']

                f.seek(offset)
                original = f.read(1)[0]
                flipped  = original ^ (1 << bit)

                f.seek(offset)
                f.write(bytes([flipped]))

                records.append({
                    **t,
                    'original_byte': original,
                    'flipped_byte':  flipped,
                })

            f.flush()
            os.fsync(f.fileno())

        return records

    def _restore_flips(self, records: List[Dict]) -> int:
        """
        Restore flipped bytes. Survives crashes via try/finally in run().
        """
        if not records:
            return 0

        restored = 0
        try:
            with open(self.mm.model_path, 'r+b') as f:
                for r in records:
                    try:
                        f.seek(r['file_offset'])
                        f.write(bytes([r['original_byte']]))
                        restored += 1
                    except Exception as e:
                        print(f"    RESTORE WARN: offset 0x{r['file_offset']:010X} -- {e}")
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            print(f"    RESTORE ERROR: could not open file for restore -- {e}")

        return restored

    # ── Benchmark Runners (Perplexity + Eval Tasks) ───────────────────────────

    def _run_perplexity(self) -> Dict:
        cmd = [
            str(self.llama_perplexity_bin),
            '-m',         str(self.mm.model_path),
            '-f',         str(self.wikitext_path),
            '-ngl',       str(self.n_gpu_layers),
            '--ctx-size', str(PERPLEXITY_CTX),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stdout + result.stderr

        # Match digits/dots AND nan/inf for corrupted model outputs
        ppl_match = re.search(
            r'Final estimate: PPL = ([\d.]+|nan|inf) \+/- ([\d.]+|nan|inf)',
            output, re.IGNORECASE
        )
        if ppl_match:
            try:
                ppl = float(ppl_match.group(1))
            except ValueError:
                ppl = None  # nan or inf -- total corruption
            try:
                ppl_err = float(ppl_match.group(2))
            except ValueError:
                ppl_err = None
        else:
            ppl     = None
            ppl_err = None

        if self.platform == 'mac':
            mem_match = re.search(r'MTL0_Mapped model buffer size =\s+([\d.]+) MiB', output)
        else:
            mem_match = re.search(r'CUDA0\s+model buffer size =\s+([\d.]+) MiB', output)

        return {
            'ppl':        ppl,
            'ppl_err':    ppl_err,
            'mem_mib':    float(mem_match.group(1)) if mem_match else None,
            'returncode': result.returncode,
            'raw_tail':   output[-2000:],  # always save for debugging None PPL
        }

    def _run_tasks(self) -> Dict:
        tmp_output = Path(self.mm.model_path).parent / '_tmp_task_result.json'
        cmd = [
            sys.executable,
            str(self.eval_script),
            '--model',        str(self.mm.model_path),
            '--tasks',        'arc_easy', 'hellaswag',
            '--output',       str(tmp_output),
            '--label',        'fault_injection_run',
            '--n-gpu-layers', str(self.n_gpu_layers),
            '--num-samples',  str(TASK_NUM_SAMPLES),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        task_data = {}
        if tmp_output.exists():
            try:
                with open(tmp_output) as f:
                    raw = json.load(f)
                task_data = raw.get('tasks', {})
            except Exception as e:
                task_data = {'parse_error': str(e)}
            finally:
                tmp_output.unlink(missing_ok=True)

        if result.returncode != 0 and not task_data:
            task_data = {'error': 'task_eval_failed', 'tail': (result.stdout + result.stderr)[-500:]}

        return task_data