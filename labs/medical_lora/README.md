# Medical / mental-health LoRA lab (Mac M2, 16 GB)

Fine-tune **`meta-llama/Llama-3.2-3B-Instruct`** with **LoRA** on **[mukulb/clustered_FUNPANG_dataset_with_groups](https://huggingface.co/datasets/mukulb/clustered_FUNPANG_dataset_with_groups)** (FUNPANG-style Q&A), then compare **base vs adapter** on the same prompts.

**Disclaimer:** educational only. Outputs are **not** medical advice. Do not use for diagnosis or treatment decisions.

---

## 0. Prereqs

- Apple Silicon Mac with **~20 GB+ free disk** (model cache + outputs).
- **Hugging Face** account with access to **Meta Llama 3.2** and a **read token**.
- Python **3.10+** recommended.

---

## 1. Token (local only)

```bash
cd labs/medical_lora
cp .env.example .env
# Edit .env: HF_TOKEN=hf_...
```

Never commit `.env` (it is gitignored).

**401 / GatedRepoError:** Hugging Face is rejecting the download because **no valid token** was sent with the request.

1. Create `labs/medical_lora/.env` with exactly: `HF_TOKEN=hf_...` (no quotes, no spaces around `=`).
2. Confirm your HF account has **accepted the Llama license** and can open the model’s **Files** tab in the browser.
3. Re-run `python scripts/train_lora.py`. The script now passes `token=` into `from_pretrained` when `HF_TOKEN` is set.

Alternatively run **`huggingface-cli login`** once in the same venv (stores a token outside `.env`).

---

## 2. Virtual environment + deps

```bash
cd labs/medical_lora
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

Install **PyTorch** for your OS from [pytorch.org](https://pytorch.org/get-started/locally/) if `pip install torch` is not enough for MPS.

**`NotOpenSSLWarning` (urllib3 / LibreSSL):** On many Macs this is **harmless noise** (Apple’s Python is linked against LibreSSL, while urllib3 v2 prefers OpenSSL). `requirements.txt` pins **`urllib3<2`** to avoid it; if you still see the line after upgrading deps, you can ignore it or switch to **Python 3.10+** from [python.org](https://www.python.org/downloads/).

**`ImportError: cannot import name 'LNTuningConfig' from 'peft.tuners.ln_tuning'`:** Your **`peft`** install is broken (often an interrupted `pip` or a bad reinstall). Fix:

```bash
pip install --force-reinstall --no-deps 'peft>=0.11.0'
pip install 'urllib3>=1.26.14,<2'
```

(`--no-deps` avoids `pip` re-resolving the whole stack and re-bumping urllib3; the second line restores the LibreSSL-friendly pin.)

**`No module named torch._C`:** PyTorch’s native extension did not install correctly. Reinstall **torch** (same venv):

```bash
pip uninstall -y torch torchvision torchaudio
pip install --no-cache-dir 'torch>=2.2.0'
```

If `pip` warns about **`Ignoring invalid distribution -orch`**, delete broken folders under **`site-packages/`** whose names look like **`~orch-...dist-info`** or **`~unctorch`**, then run **`pip install torch`** again.

**`No module named torch._strobelight`:** Your **`torch`** tree is **incomplete** (that package ships inside the normal macOS **arm64** wheel). Fix by **removing** the broken install and reinstalling:

```bash
pip uninstall -y torch torchvision torchaudio
rm -rf .venv/lib/python3.9/site-packages/torch .venv/lib/python3.9/site-packages/functorch .venv/lib/python3.9/site-packages/torchgen
pip install --no-cache-dir 'torch>=2.2.0'
```

(Adjust the **`site-packages`** path if your venv uses another Python version.)

**`cannot import name 'S' from 'sympy' (unknown location)`:** **`sympy`** is broken (often missing **`sympy/__init__.py`** after a bad uninstall). Reinstall:

```bash
pip uninstall -y sympy
pip install --no-cache-dir 'sympy>=1.13.3'
```

---

## 3. Prepare training JSONL

Streams the dataset from the Hub and writes `data/train.jsonl` (default **4000** rows; raise `--max-samples` after a successful run):

```bash
source .venv/bin/activate
python scripts/prepare_data.py --max-samples 4000
```

---

## 4. Train LoRA

Defaults target **MPS** on M2, **fp16** weights on GPU, conservative **seq length 768**, **batch 1**, **grad accum 8**.

```bash
python scripts/train_lora.py \
  --model-id meta-llama/Llama-3.2-3B-Instruct \
  --train-file data/train.jsonl \
  --output-dir outputs/lora_adapter \
  --max-seq-length 768 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 8
```

If you **OOM**: lower `--max-seq-length` (e.g. 512), lower `--max-samples` in step 3, or switch to **`meta-llama/Llama-3.2-1B-Instruct`** with the same scripts.

Optional env tweak if MPS is flaky:

```bash
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

---

## 5. Before / after generations

Edit **`eval/prompts.yaml`** if you want different questions. Then:

```bash
# Full compare (4 prompts × 2 passes × up to 320 tokens — can take a long time on M2 MPS)
python scripts/generate_compare.py \
  --model-id meta-llama/Llama-3.2-3B-Instruct \
  --adapter-dir outputs/lora_adapter \
  --prompts eval/prompts.yaml \
  --out results/lora_before_after.md
```

**Faster runs:** **`--quick`** shortens **decoding** (one prompt, 96 tokens, greedy). It does **not** shrink the **3B checkpoint load**, which is usually **several minutes** on M2 the first time (disk → RAM → MPS). For more prompts without waiting forever, try e.g. `--max-prompts 2 --greedy --max-new-tokens 120`.

```bash
python scripts/generate_compare.py \
  --adapter-dir outputs/lora_adapter/checkpoint-500 \
  --quick
```

**Ask your own question in the terminal** (after the model loads, you get a prompt; answers print to stdout and to **`results/ask_compare_last.md`**):

```bash
python scripts/generate_compare.py \
  --adapter-dir outputs/lora_adapter/checkpoint-500 \
  --interactive
```

Add **`--quick`** with **`--interactive`** for shorter, greedy answers while you iterate.

Open **`results/lora_before_after.md`** to read **Base model** vs **After LoRA** side by side.

**In the terminal:** either `cat results/lora_before_after.md` (after a run), or pass **`--print`** so the same markdown is echoed when the script finishes:

```bash
python scripts/generate_compare.py --adapter-dir outputs/lora_adapter/checkpoint-500 --quick --print
```

---

## 6. Next (RLHF / preferences — not wired here yet)

Classic **RLHF** (reward model + PPO) is heavy on 16 GB. A practical follow-up is **DPO** with a preference JSONL (`prompt`, `chosen`, `rejected`). That can be added as `scripts/train_dpo.py` once you have or build preference pairs.

---

## Layout

| Path | Role |
|------|------|
| `data/train.jsonl` | Built by `prepare_data.py` (gitignored if large) |
| `outputs/lora_adapter/` | LoRA weights + tokenizer copy (gitignored) |
| `results/lora_before_after.md` | Compare report (commit or gitignore as you prefer) |
| `results/ask_compare_last.md` | Last **`--interactive`** Q&A (base vs LoRA) |
| `eval/prompts.yaml` | Fixed eval prompts |
