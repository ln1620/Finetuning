#!/usr/bin/env python3
"""
Build train.jsonl from Hugging Face: mukulb/clustered_FUNPANG_dataset_with_groups
Each line: JSON with `messages` = [{role, content}, ...] for chat training.
"""
from typing import Any, Dict, Optional, Tuple

import warnings

warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")

import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset


def split_funpang_text(text: str) -> Tuple[str, str]:
    """Parse '<HUMAN>: ... <ASSISTANT>: ...' into (user, assistant)."""
    if not text or not isinstance(text, str):
        return "", ""
    t = text.strip()
    m = re.search(r"<ASSISTANT>\s*:", t, flags=re.IGNORECASE)
    if not m:
        return t, ""
    head, tail = t[: m.start()], t[m.end() :]
    head = re.sub(r"^\s*<HUMAN>\s*:\s*", "", head, flags=re.IGNORECASE).strip()
    return head, tail.strip()


def row_to_record(row: dict) -> Optional[dict]:
    user, assistant = split_funpang_text(row.get("text") or "")
    if not user:
        user = (row.get("query") or "").strip()
    if not assistant:
        return None
    messages = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    out: Dict[str, Any] = {"messages": messages}
    if row.get("group_name") is not None:
        out["group_name"] = row["group_name"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        default="mukulb/clustered_FUNPANG_dataset_with_groups",
        help="HF dataset id",
    )
    ap.add_argument("--split", default="train")
    ap.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "train.jsonl",
    )
    ap.add_argument(
        "--max-samples",
        type=int,
        default=4000,
        help="Cap rows for M2/16GB (raise after a successful dry run)",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset, split=args.split)
    if args.max_samples and len(ds) > args.max_samples:
        ds = ds.shuffle(seed=args.seed).select(range(args.max_samples))

    n_ok, n_skip = 0, 0
    with args.output.open("w", encoding="utf-8") as f:
        for row in ds:
            rec = row_to_record(row)
            if rec is None:
                n_skip += 1
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_ok += 1

    print(f"Wrote {args.output} with {n_ok} rows (skipped {n_skip}).")


if __name__ == "__main__":
    main()
