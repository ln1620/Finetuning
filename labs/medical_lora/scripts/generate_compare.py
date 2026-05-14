#!/usr/bin/env python3
"""
Generate answers for the same prompts: base model vs base + LoRA adapter.
Writes markdown to results/lora_before_after.md (default).

Use --interactive to type one question in the terminal and see base vs LoRA answers
(without using eval/prompts.yaml).
"""
from __future__ import annotations

import warnings

# macOS Apple CLT Python uses LibreSSL; urllib3 v2 emits a noisy (harmless) warning.
warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")

import argparse
import inspect
import os
import time
from pathlib import Path


def _patch_inspect_for_torch_import() -> None:
    """Apple CLT Python + some torch wheels: inspect.getsource fails on torch internals at import time."""
    import inspect as _ins

    if getattr(_ins, "_medical_lora_inspect_patch", False):
        return
    _gs = _ins.getsource
    _gsl = _ins.getsourcelines

    def getsource(obj):  # noqa: ANN001
        try:
            return _gs(obj)
        except OSError:
            return "# __source_unavailable__\n"

    def getsourcelines(obj):  # noqa: ANN001
        try:
            return _gsl(obj)
        except OSError:
            return (["# __source_unavailable__\n"], 1)

    _ins.getsource = getsource  # type: ignore[assignment]
    _ins.getsourcelines = getsourcelines  # type: ignore[assignment]
    setattr(_ins, "_medical_lora_inspect_patch", True)


_patch_inspect_for_torch_import()

import torch
import yaml
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

LAB_ROOT = Path(__file__).resolve().parent.parent


def load_hf_token() -> str | None:
    load_dotenv(LAB_ROOT / ".env")
    t = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if t:
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", t)
    return t


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_prompts(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    prompts = data.get("prompts") or []
    out = []
    for item in prompts:
        if isinstance(item, dict) and item.get("prompt"):
            out.append({"id": item.get("id", "q"), "prompt": str(item["prompt"]).strip()})
    return out


@torch.inference_mode()
def generate_answer(
    model,
    tokenizer,
    user_prompt: str,
    max_new_tokens: int,
    *,
    do_sample: bool = True,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    enc = tokenizer(prompt, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}
    gen_kw: dict = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kw.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        gen_kw["do_sample"] = False
    out = model.generate(**enc, **gen_kw)
    cut = enc["input_ids"].shape[1]
    return tokenizer.decode(out[0, cut:], skip_special_tokens=True).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument(
        "--adapter-dir",
        type=Path,
        default=LAB_ROOT / "outputs" / "lora_adapter",
    )
    ap.add_argument(
        "--prompts",
        type=Path,
        default=LAB_ROOT / "eval" / "prompts.yaml",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=LAB_ROOT / "results" / "lora_before_after.md",
    )
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        metavar="N",
        help="Only the first N prompts from the YAML (default: all).",
    )
    ap.add_argument(
        "--greedy",
        action="store_true",
        help="Greedy decoding (faster than sampling; a bit less diverse).",
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Fast smoke run: 1 prompt, 96 new tokens, greedy (~minutes less wall time on MPS).",
    )
    ap.add_argument(
        "--print",
        dest="print_report",
        action="store_true",
        help="Also print the full markdown report to stdout after generation.",
    )
    ap.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Ask for one question in the terminal after the model loads; print base vs LoRA (saves results/ask_compare_last.md).",
    )
    args = ap.parse_args()

    if args.quick:
        max_prompts = 1
        max_new_tokens = 96
        do_sample = False
    else:
        max_prompts = args.max_prompts
        max_new_tokens = args.max_new_tokens
        do_sample = not args.greedy

    hf_token = load_hf_token()
    hub_kw = {"token": hf_token} if hf_token else {}

    device = pick_device()

    if args.interactive:
        prompts = []
    else:
        prompts = load_prompts(args.prompts)
        if not prompts:
            raise SystemExit(f"No prompts found in {args.prompts}")
        if max_prompts is not None:
            prompts = prompts[: max(0, max_prompts)]

    if not args.adapter_dir.is_dir():
        raise SystemExit(f"Adapter folder not found: {args.adapter_dir} (train LoRA first).")

    t0 = time.perf_counter()
    if args.interactive:
        print(
            "Interactive mode: loading the **full** Llama 3B model first (often several minutes on M2). "
            "You will be asked to type **one** question after loading finishes.\n",
            flush=True,
        )
    else:
        print(
            "Note: This script still loads the **full** base model (Llama 3B ≈6–7 GB from disk/cache). "
            "That step alone often takes **several minutes** on M2 with little new output; "
            "`--quick` only shortens **generation**, not this load.\n",
            flush=True,
        )

    print("Loading tokenizer…", flush=True)
    dtype = torch.float16 if device.type != "cpu" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, **hub_kw)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Tokenizer ready ({time.perf_counter() - t0:.0f}s).\n", flush=True)

    load_kw: dict = {"low_cpu_mem_usage": True, "trust_remote_code": True, **hub_kw}
    sig_lm = inspect.signature(AutoModelForCausalLM.from_pretrained)
    if "dtype" in sig_lm.parameters:
        load_kw["dtype"] = dtype
    else:
        load_kw["torch_dtype"] = dtype

    print("Loading base model (several GB from disk/cache; can take a few minutes)…", flush=True)
    base = AutoModelForCausalLM.from_pretrained(args.model_id, **load_kw).to(device)
    base.eval()
    print(f"  Base weights on {device} ({time.perf_counter() - t0:.0f}s).\n", flush=True)

    print("Loading LoRA adapter…", flush=True)
    model_lora = PeftModel.from_pretrained(base, str(args.adapter_dir))
    model_lora.eval()
    print(f"  Adapter attached ({time.perf_counter() - t0:.0f}s). Starting generation…\n", flush=True)

    if args.interactive:
        print("Type your question below, then press Enter.", flush=True)
        try:
            user_q = input("\nYour question: ").strip()
        except EOFError:
            user_q = ""
        if not user_q:
            raise SystemExit("No question entered.")
        prompts = [{"id": "interactive", "prompt": user_q}]

    lines: list[str] = [
        "# LoRA: interactive — base vs fine-tuned"
        if args.interactive
        else "# LoRA: same prompts, base vs fine-tuned",
        "",
        f"- Model: `{args.model_id}`",
        f"- Adapter: `{args.adapter_dir}`",
        f"- Device: `{device}`",
        f"- Generating up to **{max_new_tokens}** new tokens, **{'greedy' if not do_sample else 'sampling'}**",
        "",
    ]

    base_outputs: dict[str, str] = {}
    lora_outputs: dict[str, str] = {}
    for item in prompts:
        pid, ptxt = item["id"], item["prompt"]
        print(f"  Prompt `{pid}`: base model…", flush=True)
        with model_lora.disable_adapter():
            base_outputs[pid] = generate_answer(
                model_lora,
                tokenizer,
                ptxt,
                max_new_tokens,
                do_sample=do_sample,
            )
        print(f"  Prompt `{pid}`: +LoRA…", flush=True)
        lora_outputs[pid] = generate_answer(
            model_lora,
            tokenizer,
            ptxt,
            max_new_tokens,
            do_sample=do_sample,
        )
        print(f"  Prompt `{pid}`: done ({time.perf_counter() - t0:.0f}s since start)", flush=True)

    for item in prompts:
        pid, ptxt = item["id"], item["prompt"]
        lines.extend(
            [
                f"## Prompt `{pid}`",
                "",
                f"**Question:** {ptxt}",
                "",
                "### Base model",
                "",
                base_outputs[pid],
                "",
                "### After LoRA",
                "",
                lora_outputs[pid],
                "",
                "---",
                "",
            ]
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    if args.interactive:
        ask_path = LAB_ROOT / "results" / "ask_compare_last.md"
        ask_path.parent.mkdir(parents=True, exist_ok=True)
        ask_path.write_text(text, encoding="utf-8")
        print("\n" + "=" * 72 + "\n", flush=True)
        print(text, flush=True)
        print("=" * 72, flush=True)
        print("\nSaved to", ask_path, flush=True)
    else:
        args.out.write_text(text, encoding="utf-8")
        print("Wrote", args.out, flush=True)
        if args.print_report:
            print("\n" + "=" * 72 + "\n", flush=True)
            print(text, flush=True)
            print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
