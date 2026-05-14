#!/usr/bin/env python3
"""
LoRA fine-tune Meta Llama 3.2 Instruct on local train.jsonl (messages -> text via chat template).
Tuned for Mac M2 + 16 GB: small batch, gradient accumulation, modest seq length.
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")

import argparse
import inspect
import os
from pathlib import Path


def _patch_inspect_for_torch_import() -> None:
    """Apple CLT Python + some torch wheels: inspect.getsource fails on torch internals at import time."""
    if getattr(inspect, "_medical_lora_inspect_patch", False):
        return
    _gs = inspect.getsource
    _gsl = inspect.getsourcelines

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

    inspect.getsource = getsource  # type: ignore[assignment]
    inspect.getsourcelines = getsourcelines  # type: ignore[assignment]
    setattr(inspect, "_medical_lora_inspect_patch", True)


_patch_inspect_for_torch_import()

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

try:
    from trl import SFTConfig
except ImportError:  # very old trl
    SFTConfig = None  # type: ignore[misc, assignment]

LAB_ROOT = Path(__file__).resolve().parent.parent


def load_hf_token() -> str | None:
    """Load .env then return token string for gated models (Meta Llama)."""
    load_dotenv(LAB_ROOT / ".env")
    t = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if t:
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", t)
    return t


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def patch_transformers_trainer_save_for_torch_inspect_bug() -> None:
    """Trainer._save ends with torch.save(self.args, ...). On some macOS/Python stacks that
    triggers torch.serialization code paths that call inspect.getsource and raise
    OSError('could not get source code') even though model/tokenizer were already written.
    Swallow only that specific failure so long runs can finish.
    """
    import transformers.trainer as trn_mod

    if getattr(trn_mod.Trainer, "_medical_lora_trainer_save_patch", False):
        return
    _orig_save = trn_mod.Trainer._save

    def _save(self, output_dir=None, state_dict=None):  # type: ignore[no-untyped-def]
        try:
            return _orig_save(self, output_dir, state_dict)
        except OSError as exc:
            if "could not get source code" not in str(exc).lower():
                raise
            print(
                "Warning: skipped pickling training_args.bin (PyTorch/inspect issue on this Python build). "
                "Adapter weights from save_pretrained in this checkpoint should still be on disk.",
                flush=True,
            )

    trn_mod.Trainer._save = _save  # type: ignore[method-assign]
    setattr(trn_mod.Trainer, "_medical_lora_trainer_save_patch", True)


def main() -> None:
    patch_transformers_trainer_save_for_torch_inspect_bug()

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument(
        "--train-file",
        type=Path,
        default=LAB_ROOT / "data" / "train.jsonl",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=LAB_ROOT / "outputs" / "lora_adapter",
    )
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    args = ap.parse_args()

    hf_token = load_hf_token()
    hub_kw = {"token": hf_token} if hf_token else {}
    if not hf_token and str(args.model_id).startswith("meta-llama/"):
        print(
            "Note: HF_TOKEN not found in labs/medical_lora/.env — "
            "gated Llama downloads usually need HF_TOKEN=hf_... or `huggingface-cli login`.",
            flush=True,
        )

    device = pick_device()
    print("Using device:", device)

    if not args.train_file.is_file():
        raise SystemExit(f"Missing {args.train_file} — run scripts/prepare_data.py first.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, **hub_kw)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    load_kw = {"low_cpu_mem_usage": True, "trust_remote_code": True, **hub_kw}
    sig_lm = inspect.signature(AutoModelForCausalLM.from_pretrained)
    if "dtype" in sig_lm.parameters:
        load_kw["dtype"] = torch_dtype
    else:
        load_kw["torch_dtype"] = torch_dtype
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **load_kw)
    model = model.to(device)

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files={"train": str(args.train_file)}, split="train")

    def to_text(batch: dict) -> dict:
        texts = []
        for messages in batch["messages"]:
            texts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
        return {"text": texts}

    ds = ds.map(to_text, batched=True)
    drop_cols = [c for c in ds.column_names if c != "text"]
    if drop_cols:
        ds = ds.remove_columns(drop_cols)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    use_fp16 = device == "cuda"

    # TRL: max_seq_length / dataset_text_field / packing live on SFTConfig in newer releases;
    # older SFTTrainer accepted max_seq_length directly.
    train_kw = dict(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        gradient_checkpointing=True,
        fp16=use_fp16,
        bf16=False,
        optim="adamw_torch",
        max_grad_norm=0.3,
        lr_scheduler_type="cosine",
        report_to="none",
    )

    sig_trainer = inspect.signature(SFTTrainer.__init__)

    if SFTConfig is not None:
        sig_cfg = inspect.signature(SFTConfig.__init__)
        cfg_kw = {
            **train_kw,
            "packing": False,
            "dataset_text_field": "text",
        }
        if "max_seq_length" in sig_cfg.parameters:
            cfg_kw["max_seq_length"] = args.max_seq_length
        elif "max_length" in sig_cfg.parameters:
            cfg_kw["max_length"] = args.max_seq_length
        if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_cfg.parameters.values()):
            cfg_kw = {k: v for k, v in cfg_kw.items() if k in sig_cfg.parameters}
        training_args = SFTConfig(**cfg_kw)
        trainer_kw: dict = {
            "model": model,
            "args": training_args,
            "train_dataset": ds,
        }
    else:
        training_args = TrainingArguments(**train_kw)
        trainer_kw = {
            "model": model,
            "args": training_args,
            "train_dataset": ds,
            "packing": False,
            "dataset_text_field": "text",
        }
        if "max_seq_length" in sig_trainer.parameters:
            trainer_kw["max_seq_length"] = args.max_seq_length

    if "processing_class" in sig_trainer.parameters:
        trainer_kw["processing_class"] = tokenizer
    else:
        trainer_kw["tokenizer"] = tokenizer

    trainer = SFTTrainer(**trainer_kw)
    trainer.train()
    trainer.model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print("Saved LoRA adapter to", args.output_dir)


if __name__ == "__main__":
    main()
