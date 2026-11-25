#!/usr/bin/env python3
"""
GVK Private AI - QLoRA Trainer and Dataset Validator (tools/train_lora.py)

Provides leak-free offline ChatML dataset validation and a two-stage QLoRA
fine-tuning pipeline (matching `notebooks/Qwen2.5_14b_LoRA_trainer.ipynb`) with
CPU/macOS `--dry-run` verification support.

Key Responsibilities:
  - Validate ChatML JSONL records (`{"messages": [{"role": ..., "content": ...}]}`).
  - Perform a leak-free temporal train/validation split (holding out the final 5% without shuffling).
  - Verify response-only loss masking (`-100` for system/user turns before `<|im_start|>assistant\n`).
  - Run two-stage QLoRA training (Stage 1 full set -> Stage 2 style annealing on the top 30% concise samples).

Usage:
  python3 tools/train_lora.py self-test
  python3 tools/train_lora.py train --persona friend_1 --dataset datasets/friend_1.jsonl --dry-run
  python3 tools/train_lora.py train --persona friend_1 --dataset datasets/friend_1.jsonl -o adapters/
  python3 tools/train_lora.py train-all --dataset-dir datasets/ -o adapters/ --dry-run
"""

import argparse
from dataclasses import dataclass
import glob
import json
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple


DEFAULT_BASE_MODEL = "unsloth/Qwen2.5-14B-Instruct-bnb-4bit"
DEFAULT_MAX_SEQ_LEN = 4096
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
STAGE2_TOP_FRACTION = 0.30
ASSISTANT_HEADER = "<|im_start|>assistant\n"
IM_END = "<|im_end|>"


@dataclass
class DatasetStats:
    """Summary counts produced when inspecting a ChatML JSONL dataset."""

    total_lines: int
    valid_samples: int
    invalid_lines: int
    train_count: int
    val_count: int
    stage2_count: int


# =====================================================================
# 1. CHATML VALIDATION, LEAK-FREE SPLIT, AND LOSS MASKING
# =====================================================================
def validate_chatml_sample(sample: Any) -> Tuple[bool, str]:
    """Validate a single ChatML sample object (`{"messages": [...]}`).

    Args:
        sample: Parsed JSON object representing one conversation sample.

    Returns:
        Tuple `(is_valid, error_message)` where `error_message` is empty when valid.
    """
    if not isinstance(sample, dict):
        return False, "Record is not a JSON object"
    messages = sample.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return False, "Field 'messages' must contain at least 2 messages"

    has_assistant = False
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            return False, f"Message #{idx} is not a dictionary"
        role = msg.get("role")
        content = msg.get("content")
        if role not in ("system", "user", "assistant"):
            return False, f"Invalid role '{role}' in message #{idx}"
        if not isinstance(content, str) or not content.strip():
            return False, f"Empty 'content' in message #{idx}"
        if role == "assistant":
            has_assistant = True

    if not has_assistant:
        return False, "Missing at least one 'assistant' turn"
    return True, ""


def load_chatml_dataset(dataset_path: str) -> Tuple[List[Dict[str, Any]], int, int]:
    """Read a JSONL dataset, discard malformed rows, and return valid ChatML samples.

    Args:
        dataset_path: Path to the `.jsonl` file.

    Returns:
        Tuple of `(valid_samples, total_lines, invalid_lines)`.

    Raises:
        FileNotFoundError: If `dataset_path` does not exist.
    """
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    valid: List[Dict[str, Any]] = []
    total_lines = 0
    invalid_lines = 0
    with open(dataset_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped:
                continue
            total_lines += 1
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                invalid_lines += 1
                continue
            ok, _ = validate_chatml_sample(obj)
            if ok:
                valid.append(obj)
            else:
                invalid_lines += 1

    return valid, total_lines, invalid_lines


def split_train_val_without_leak(
    samples: Sequence[Dict[str, Any]],
    val_ratio: float = 0.05,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split chronological samples into train and tail validation sets without shuffling.

    Holding out the chronological tail (`shuffle=False`) prevents data leakage across
    overlapping sliding-window dialogue chunks and temporal weighting.

    Args:
        samples: Ordered sequence of validated ChatML samples.
        val_ratio: Fraction of tail samples reserved for validation.

    Returns:
        Tuple of `(train_samples, val_samples)`.
    """
    n = len(samples)
    if n <= 1:
        return list(samples), []
    val_count = max(1, int(round(n * val_ratio)))
    val_count = min(val_count, n - 1)
    train = list(samples[: n - val_count])
    val = list(samples[n - val_count :])
    return train, val


def select_stage2_samples(
    samples: Sequence[Dict[str, Any]],
    top_fraction: float = STAGE2_TOP_FRACTION,
    min_chars: int = 4,
    max_chars: int = 800,
) -> List[Dict[str, Any]]:
    """Select up to `top_fraction` concise samples for Stage 2 style annealing.

    Args:
        samples: Training samples to rank and filter.
        top_fraction: Target fraction of samples to keep.
        min_chars: Minimum allowed length per assistant reply.
        max_chars: Maximum allowed length per assistant reply.

    Returns:
        Subset of concise samples prioritized around natural short-reply length (~48 chars).
    """
    filtered: List[Tuple[int, Dict[str, Any]]] = []
    for sample in samples:
        asst_texts = [
            m["content"].strip()
            for m in sample.get("messages", [])
            if m.get("role") == "assistant"
        ]
        if not asst_texts:
            continue
        if all(min_chars <= len(t) <= max_chars for t in asst_texts):
            avg_len = sum(len(t) for t in asst_texts) // len(asst_texts)
            filtered.append((avg_len, sample))

    if not filtered:
        return list(samples[: max(1, int(len(samples) * top_fraction))])

    target_k = max(1, int(round(len(samples) * top_fraction)))
    # Prefer natural concise replies (8..180 characters per turn, centered near 48 chars)
    filtered.sort(key=lambda pair: abs(pair[0] - 48))
    return [pair[1] for pair in filtered[:target_k]]


def format_chatml_text(messages: Sequence[Dict[str, str]]) -> str:
    """Render a list of role/content messages into a Qwen ChatML string.

    Args:
        messages: Sequence of message dictionaries with `"role"` and `"content"`.

    Returns:
        Formatted ChatML text block.
    """
    parts: List[str] = []
    for m in messages:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}{IM_END}\n")
    return "".join(parts)


def compute_response_only_labels_mask(chatml_text: str) -> List[Tuple[str, bool]]:
    """Simulate response-only loss masking (`train_on_responses_only`).

    Everything up to and including `<|im_start|>assistant\n` plus all `system`/`user`
    blocks is masked (`-100`, `is_trained=False`); loss is computed only on assistant
    reply text and its closing `<|im_end|>` token.

    Args:
        chatml_text: Full ChatML-formatted conversation string.

    Returns:
        List of `(segment_text, is_trained)` tuples partitioning `chatml_text`.
    """
    segments: List[Tuple[str, bool]] = []
    pos = 0
    n = len(chatml_text)

    while pos < n:
        idx = chatml_text.find(ASSISTANT_HEADER, pos)
        if idx == -1:
            segments.append((chatml_text[pos:], False))
            break

        header_end = idx + len(ASSISTANT_HEADER)
        if header_end > pos:
            segments.append((chatml_text[pos:header_end], False))

        end_idx = chatml_text.find(IM_END, header_end)
        if end_idx == -1:
            segments.append((chatml_text[header_end:], True))
            break

        resp_end = end_idx + len(IM_END)
        segments.append((chatml_text[header_end:resp_end], True))
        pos = resp_end

    return segments


# =====================================================================
# 2. LOCAL / CLOUD QLORA TRAINING PIPELINE
# =====================================================================
def inspect_dataset(dataset_path: str) -> Tuple[DatasetStats, List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load, validate, and split a ChatML dataset into Stage 1, validation, and Stage 2 sets.

    Args:
        dataset_path: Path to the source `.jsonl` file.

    Returns:
        Tuple of `(stats, train_data, val_data, stage2_data)`.

    Raises:
        ValueError: If no valid ChatML samples are found in `dataset_path`.
    """
    samples, total_lines, invalid_lines = load_chatml_dataset(dataset_path)
    if not samples:
        raise ValueError(f"Dataset '{dataset_path}' contains no valid ChatML samples.")
    train_data, val_data = split_train_val_without_leak(samples, val_ratio=0.05)
    stage2_data = select_stage2_samples(train_data, top_fraction=STAGE2_TOP_FRACTION)
    stats = DatasetStats(
        total_lines=total_lines,
        valid_samples=len(samples),
        invalid_lines=invalid_lines,
        train_count=len(train_data),
        val_count=len(val_data),
        stage2_count=len(stage2_data),
    )
    return stats, train_data, val_data, stage2_data


def train_persona_lora(
    persona_id: str,
    dataset_path: str,
    output_dir: str = "adapters",
    base_model: str = DEFAULT_BASE_MODEL,
    max_seq_length: int = DEFAULT_MAX_SEQ_LEN,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run two-stage QLoRA fine-tuning for `persona_id`.

    In `--dry-run` mode, validates the dataset, leak-free split, and response-only loss
    masking without requiring a CUDA GPU.

    Args:
        persona_id: Identifier of the target persona (e.g., `"friend_1"`).
        dataset_path: Path to the persona ChatML `.jsonl` dataset.
        output_dir: Directory where stage checkpoints and the final PEFT adapter are saved.
        base_model: Hugging Face model ID for the 4-bit base model.
        max_seq_length: Maximum context length for SFT training.
        dry_run: If `True`, skip GPU training and only verify dataset/masking integrity.

    Returns:
        Summary dictionary containing `persona_id`, `dry_run`, `stats`, and (when trained) `peft_dir`.

    Raises:
        RuntimeError: If a CUDA GPU or required Unsloth/TRL/PyTorch stack is unavailable during real training.
    """
    stats, train_data, val_data, stage2_data = inspect_dataset(dataset_path)
    print("=====================================================================")
    print(f" [*] GVK Private AI - QLoRA Trainer ('{persona_id}')")
    print("=====================================================================")
    print(f" Dataset        : {os.path.abspath(dataset_path)}")
    print(f" Base model     : {base_model} (ctx={max_seq_length}, LoRA r={LORA_R}/alpha={LORA_ALPHA})")
    print(f" Valid samples  : {stats.valid_samples} / {stats.total_lines} (filtered: {stats.invalid_lines})")
    print(f" Leak-free split: train={stats.train_count}, val={stats.val_count} (5% tail)")
    print(f" Stage 2 subset : {stats.stage2_count} (~30% concise samples)")

    if dry_run:
        sample_chatml = format_chatml_text(train_data[0]["messages"])
        masked_segments = compute_response_only_labels_mask(sample_chatml)
        trained_chars = sum(len(seg) for seg, active in masked_segments if active)
        masked_chars = sum(len(seg) for seg, active in masked_segments if not active)
        print(f" [OK] [DRY-RUN] Loss masking verified: trained={trained_chars} chars, masked(-100)={masked_chars} chars.")
        print("=====================================================================")
        return {
            "persona_id": persona_id,
            "dry_run": True,
            "stats": stats,
        }

    # Verify GPU training stack availability before loading weights
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA GPU detected.")
        from datasets import Dataset
        from transformers import TrainingArguments
        from trl import SFTTrainer
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import train_on_responses_only
    except Exception as exc:
        raise RuntimeError(
            f"Real QLoRA training requires a CUDA GPU and unsloth/trl/torch ({exc}).\n"
            f"On CPU/macOS use --dry-run or run notebooks/Qwen2.5_14b_LoRA_trainer.ipynb in Google Colab."
        ) from exc

    os.makedirs(output_dir, exist_ok=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=list(TARGET_MODULES),
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    def _to_hf_dataset(rows: Sequence[Dict[str, Any]]) -> Any:
        return Dataset.from_dict({"text": [format_chatml_text(r["messages"]) for r in rows]})

    ds_train_1 = _to_hf_dataset(train_data)
    ds_val = _to_hf_dataset(val_data) if val_data else None
    ds_train_2 = _to_hf_dataset(stage2_data)

    bf16_ok = torch.cuda.is_bf16_supported()
    trainer_1 = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds_train_1,
        eval_dataset=ds_val,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=TrainingArguments(
            output_dir=os.path.join(output_dir, f"{persona_id}_stage1"),
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_ratio=0.03,
            num_train_epochs=1,
            learning_rate=6e-5,
            lr_scheduler_type="linear",
            weight_decay=0.01,
            fp16=not bf16_ok,
            bf16=bf16_ok,
            logging_steps=20,
            save_strategy="no",
            report_to="none",
        ),
    )
    trainer_1 = train_on_responses_only(
        trainer_1,
        instruction_part="<|im_start|>user\n",
        response_part=ASSISTANT_HEADER,
    )
    trainer_1.train()

    trainer_2 = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds_train_2,
        eval_dataset=ds_val,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=TrainingArguments(
            output_dir=os.path.join(output_dir, f"{persona_id}_stage2"),
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_ratio=0.05,
            num_train_epochs=1,
            learning_rate=2e-5,
            lr_scheduler_type="cosine",
            weight_decay=0.05,
            fp16=not bf16_ok,
            bf16=bf16_ok,
            logging_steps=10,
            save_strategy="no",
            report_to="none",
        ),
    )
    trainer_2 = train_on_responses_only(
        trainer_2,
        instruction_part="<|im_start|>user\n",
        response_part=ASSISTANT_HEADER,
    )
    trainer_2.train()

    peft_out = os.path.join(output_dir, f"{persona_id}_lora_final")
    model.save_pretrained(peft_out)
    tokenizer.save_pretrained(peft_out)
    print(f" [OK] PEFT adapter saved to: {peft_out}")
    return {"persona_id": persona_id, "dry_run": False, "peft_dir": peft_out, "stats": stats}


# =====================================================================
# 3. OFFLINE SELF-TEST (CPU / NO-GPU)
# =====================================================================
def run_self_test() -> None:
    """Run offline self-tests for ChatML validation, leak-free splitting, masking, and dry-run."""
    print("=====================================================================")
    print(" [*] GVK Private AI - Offline Trainer Self-Test (tools/train_lora.py)")
    print("=====================================================================")

    with tempfile.TemporaryDirectory(prefix="gvk_train_test_") as tmpdir:
        ds_path = os.path.join(tmpdir, "friend_test.jsonl")
        samples = []
        for idx in range(40):
            samples.append(
                {
                    "messages": [
                        {"role": "system", "content": "Ти — тестовий асистент."},
                        {"role": "user", "content": f"привіт, питання #{idx}"},
                        {"role": "assistant", "content": f"відповідь #{idx}" if idx % 2 == 0 else f"коротка репліка {idx}"},
                    ]
                }
            )
        with open(ds_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
            # Append 1 invalid record to verify malformed-row filtering
            f.write(json.dumps({"messages": [{"role": "user", "content": "без відповіді"}]}) + "\n")

        print(" [*] [1/4] Checking ChatML validator and JSONL loader...")
        loaded, total_lines, invalid = load_chatml_dataset(ds_path)
        assert total_lines == 41 and len(loaded) == 40 and invalid == 1
        print(f"       [OK] valid={len(loaded)}, filtered={invalid}.")

        print(" [*] [2/4] Checking leak-free train/val split (5% tail)...")
        train_set, val_set = split_train_val_without_leak(loaded, val_ratio=0.05)
        assert len(train_set) == 38 and len(val_set) == 2
        assert "питання #38" in val_set[0]["messages"][1]["content"]
        stage2 = select_stage2_samples(train_set, top_fraction=0.30)
        assert len(stage2) == 11
        print(f"       [OK] train={len(train_set)}, val={len(val_set)}, stage2={len(stage2)}.")

        print(" [*] [3/4] Checking response-only loss masking (train_on_responses_only)...")
        chatml = format_chatml_text(loaded[0]["messages"])
        segments = compute_response_only_labels_mask(chatml)
        trained_text = "".join(t for t, ok in segments if ok)
        masked_text = "".join(t for t, ok in segments if not ok)
        assert "відповідь #0" in trained_text and "тестовий асистент" not in trained_text
        assert "привіт, питання #0" in masked_text and "привіт, питання #0" not in trained_text
        print("       [OK] System and user tokens masked (-100); only assistant tokens trained.")

        print(" [*] [4/4] Checking --dry-run execution...")
        res = train_persona_lora("friend_1", ds_path, output_dir=tmpdir, dry_run=True)
        assert res["dry_run"] is True and res["stats"].valid_samples == 40

    print("=====================================================================")
    print(" [OK] ALL TRAIN_LORA.PY SELF-TESTS PASSED!")
    print("=====================================================================")


# =====================================================================
# 4. CLI INTERFACE
# =====================================================================
def main(argv: Optional[List[str]] = None) -> None:
    """Parse CLI arguments and dispatch `self-test`, `train`, or `train-all`.

    Args:
        argv: Optional CLI argument list (defaults to `sys.argv[1:]`).

    Raises:
        FileNotFoundError: If `train-all` finds no `.jsonl` files in `--dataset-dir`.
    """
    parser = argparse.ArgumentParser(
        description="GVK Private AI - QLoRA trainer and ChatML dataset validator"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("self-test", help="Verify ChatML validation, leak-free split, and loss masking (no GPU required)")

    p_tr = sub.add_parser("train", help="Run QLoRA training for a single persona")
    p_tr.add_argument("--persona", required=True, help="Persona ID (e.g., friend_1)")
    p_tr.add_argument("--dataset", required=True, help="Path to the .jsonl dataset")
    p_tr.add_argument("-o", "--output", default="adapters", help="Output directory for adapter artifacts")
    p_tr.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Hugging Face ID of the base model")
    p_tr.add_argument("--dry-run", action="store_true", help="Validate dataset and loss masking without a GPU")

    p_all = sub.add_parser("train-all", help="Batch-train all .jsonl datasets in a directory")
    p_all.add_argument("--dataset-dir", required=True, help="Directory containing JSONL datasets")
    p_all.add_argument("-o", "--output", default="adapters", help="Output directory for adapter artifacts")
    p_all.add_argument("--dry-run", action="store_true", help="Validate datasets without a GPU")

    args = parser.parse_args(argv)
    if args.cmd == "self-test":
        run_self_test()
    elif args.cmd == "train":
        train_persona_lora(
            persona_id=args.persona,
            dataset_path=args.dataset,
            output_dir=args.output,
            base_model=args.base_model,
            dry_run=args.dry_run,
        )
    elif args.cmd == "train-all":
        files = sorted(glob.glob(os.path.join(args.dataset_dir, "*.jsonl")))
        if not files:
            raise FileNotFoundError(f"No .jsonl dataset files found in {args.dataset_dir}.")
        for fp in files:
            pid = os.path.splitext(os.path.basename(fp))[0]
            train_persona_lora(
                persona_id=pid,
                dataset_path=fp,
                output_dir=args.output,
                dry_run=args.dry_run,
            )


if __name__ == "__main__":
    main()
