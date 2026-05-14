# Data for this lab

Training data comes from the Hugging Face dataset:

**[mukulb/clustered_FUNPANG_dataset_with_groups](https://huggingface.co/datasets/mukulb/clustered_FUNPANG_dataset_with_groups)**

Run `scripts/prepare_data.py` to build a local JSONL file (`train.jsonl`) with chat-style `messages` plus a `text` column for `SFTTrainer`.

You do not need to manually download the CSV unless you want offline mode; the prepare script streams from the Hub by default.
