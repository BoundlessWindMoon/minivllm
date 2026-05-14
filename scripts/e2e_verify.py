"""End-to-end correctness verification.

Runs the same short prompts through multiple backends / precisions and
compares greedy outputs.  A mismatch or garbled text indicates a regression.
"""

import argparse
import os
import sys
import tempfile
import warnings

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import GlobalConfig
from engine.model_runner import ModelRunner
from engine.loader import load_model


PROMPTS = [
    "1+1=",
    "The capital of France is",
    "import torch",
]


def run_once(cfg_path: str, prompt: str, max_new_tokens: int = 16):
    """Load model and run one inference. Returns generated text."""
    cfg = GlobalConfig.from_yaml(cfg_path)
    cfg.inference.prompt = prompt
    cfg.inference.max_new_tokens = max_new_tokens
    cfg.inference.sampling.sample_method = "greedy"
    cfg.inference.sampling.temperature = 1.0
    cfg.inference.sampling.topp = 1.0
    cfg.inference.stop_on_eos = True
    cfg.inference.use_cuda_graph = False

    torch.set_default_dtype(cfg.env.get_torch_dtype())
    torch.set_default_device(cfg.env.device)

    model, tokenizer = load_model(cfg)
    runner = ModelRunner(model=model, tokenizer=tokenizer, cfg=cfg)
    text = runner.inference()
    return text


def is_garbled(text: str) -> bool:
    """Heuristic: text is garbled if it is empty or dominated by special tokens."""
    if not text or not text.strip():
        return True
    # If more than 60 % of non-space chars are <|...|> style tokens, flag it
    non_space = [c for c in text if not c.isspace()]
    if not non_space:
        return True
    special_count = sum(1 for c in non_space if c in "<>|")
    if special_count / len(non_space) > 0.5:
        return True
    return False


def verify_consistency(results: dict[str, str]) -> list[str]:
    """Compare outputs across configurations. Returns list of mismatch messages."""
    issues = []
    base_key = "fp16_default"
    if base_key not in results:
        return issues
    base = results[base_key]
    for key, text in results.items():
        if key == base_key:
            continue
        if is_garbled(text):
            issues.append(f"[{key}] GARBLED: {text!r}")
            continue
        # Exact match for greedy short prompts is expected across backends
        if text != base:
            issues.append(
                f"[{key}] MISMATCH vs {base_key}\n"
                f"  base:  {base!r}\n"
                f"  this:  {text!r}"
            )
    return issues


def write_temp_config(base_cfg: str, overrides: dict) -> str:
    """Create a temporary YAML config derived from base_cfg with overrides applied."""
    import yaml

    with open(base_cfg) as f:
        d = yaml.safe_load(f)
    for key_path, value in overrides.items():
        parts = key_path.split(".")
        node = d
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = value

    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        yaml.dump(d, f, default_flow_style=False)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/default.yaml")
    parser.add_argument("--quant-config", default="configs/default.yaml")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--skip-megakernel", action="store_true")
    parser.add_argument("--skip-quant", action="store_true")
    args = parser.parse_args()

    # Single-process init for e2e test
    with tempfile.NamedTemporaryFile(delete=False) as f:
        init_file = f.name
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", world_size=1, rank=0
    )

    all_results = {}

    for prompt in PROMPTS:
        print(f"\n=== Prompt: {prompt!r} ===")
        results = {}

        # 1. FP16 / BF16 default backend
        overrides = {
            "inference.backend": "default",
            "inference.use_quantized_model": False,
            "path.model_path": "~/huggingface/Qwen3-0.6B/",
        }
        cfg_path = write_temp_config(args.base_config, overrides)
        try:
            text = run_once(cfg_path, prompt, args.max_new_tokens)
            results["fp16_default"] = text
            print(f"  fp16_default : {text!r}")
        except Exception as e:
            results["fp16_default"] = f"ERROR: {e}"
            print(f"  fp16_default : ERROR {e}")
        finally:
            os.unlink(cfg_path)

        # 2. Megakernel backend
        if not args.skip_megakernel:
            overrides = {
                "inference.backend": "megakernel_cuda",
                "inference.use_quantized_model": False,
                "path.model_path": "~/huggingface/Qwen3-0.6B/",
            }
            cfg_path = write_temp_config(args.base_config, overrides)
            try:
                text = run_once(cfg_path, prompt, args.max_new_tokens)
                results["fp16_megakernel"] = text
                print(f"  fp16_megaknl : {text!r}")
            except Exception as e:
                results["fp16_megakernel"] = f"ERROR: {e}"
                print(f"  fp16_megaknl : ERROR {e}")
            finally:
                os.unlink(cfg_path)

        # 3. Quantized model
        if not args.skip_quant:
            overrides = {
                "inference.backend": "default",
                "inference.use_quantized_model": True,
                "path.quantized_model_path": "~/huggingface/Qwen3-0.6B-AWQ_Cached",
                "inference.cpu_offload_modules": [],
            }
            cfg_path = write_temp_config(args.quant_config, overrides)
            try:
                text = run_once(cfg_path, prompt, args.max_new_tokens)
                results["quantized"] = text
                print(f"  quantized    : {text!r}")
            except Exception as e:
                results["quantized"] = f"ERROR: {e}"
                print(f"  quantized    : ERROR {e}")
            finally:
                os.unlink(cfg_path)

        issues = verify_consistency(results)
        if issues:
            print("  ISSUES:")
            for msg in issues:
                print(f"    - {msg}")
        else:
            print("  OK — no mismatches or garbled text.")

        all_results[prompt] = results

    dist.destroy_process_group()
    try:
        os.unlink(init_file)
    except FileNotFoundError:
        pass

    # Final summary
    print("\n" + "=" * 60)
    total_issues = sum(
        len(verify_consistency(r)) for r in all_results.values()
    )
    if total_issues == 0:
        print("E2E verification PASSED — all outputs consistent and clean.")
    else:
        print(f"E2E verification FAILED — {total_issues} issue(s) found.")
    print("=" * 60)


if __name__ == "__main__":
    main()
