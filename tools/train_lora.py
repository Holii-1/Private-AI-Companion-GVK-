#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
GVK Private AI — QLoRA Fine-Tuning & GGUF Adapter Exporter
Track A: Stage 1 (Milestone 2) — tools/train_lora.py
=====================================================================
Скрипт для навчання 3-х індивідуальних QLoRA-адаптерів (friend_1, friend_2, friend_3)
на базі моделі Qwen2.5-7B-Instruct із подальшою конвертацією у формат GGUF
для миттєвого гарячого перемикання (Hot-Swapping) у llama-server.

Архітектурні особливості:
1. Базова модель: Qwen/Qwen2.5-7B-Instruct (або unsloth/Qwen2.5-7B-Instruct-bnb-4bit).
2. Конфігурація LoRA:
   - Rank (r): 32
   - Alpha: 64
   - Цільові шари: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
3. Маскування функції втрат (train_on_responses_only):
   - Модель навчається ВИКЛЮЧНО на відповідях цільового друга (<|im_start|>assistant),
     ігноруючи градієнти на системному промпті та репліках інших учасників (<|im_start|>user).
4. Експорт у GGUF:
   - Автоматичне збереження ваг у Safetensors + конвертація LoRA-адаптера у .gguf (F16)
     для безпосереднього шифрування через tools/disc_packer.py.
5. Режим локальної перевірки (--dry-run / self-test):
   - Дозволяє на macOS / CPU без GPU перевірити коректність датасетів ChatML,
     розмітку токенів та симулювати маскування лоссу перед запуском на GPU.

Приклади використання:
  # 1. Самоперевірка пайплайну та валідація шаблону ChatML (працює без GPU на Mac):
  python3 tools/train_lora.py self-test

  # 2. Перевірка реального датасету перед навчанням (Dry-Run):
  python3 tools/train_lora.py train --persona friend_1 --dataset datasets/friend_1_chatml.jsonl --dry-run

  # 3. Навчання одного адаптера на GPU (Colab / Linux CUDA) з експортом у GGUF:
  python3 tools/train_lora.py train --persona friend_1 --dataset datasets/friend_1_chatml.jsonl -o adapters/

  # 4. Пакетне навчання всіх 3-х друзів по черзі:
  python3 tools/train_lora.py train-all --dataset-dir datasets/ -o adapters/
=====================================================================
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# =====================================================================
# КОНСТАНТИ ТА ГІПЕРПАРАМЕТРИ ЗА ЗАМОВЧУВАННЯМ (ЗГІДНО З ROADMAP GVK)
# =====================================================================
DEFAULT_BASE_MODEL_UNSLOTH = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
DEFAULT_BASE_MODEL_HF = "Qwen/Qwen2.5-7B-Instruct"

LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.0
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

DEFAULT_EPOCHS = 1.8
DEFAULT_LR = 1.5e-4
DEFAULT_WARMUP_RATIO = 0.05
DEFAULT_BATCH_SIZE = 2
DEFAULT_GRAD_ACCUM = 4
DEFAULT_MAX_SEQ_LEN = 2048
DEFAULT_SEED = 3407

# Маркери шаблону ChatML для Qwen2.5
CHATML_INSTRUCTION_PART = "<|im_start|>user\n"
CHATML_RESPONSE_PART = "<|im_start|>assistant\n"
CHATML_END_TOKEN = "<|im_end|>\n"

PERSONA_IDS = ["friend_1", "friend_2", "friend_3"]


# =====================================================================
# ДОПОМІЖНІ ФУНКЦІЇ ФОРМАТУВАННЯ ТА ВАЛІДАЦІЇ CHATML
# =====================================================================
def format_chatml_conversation(messages: List[Dict[str, str]]) -> str:
    """
    Перетворює список повідомлень [{"role": "...", "content": "..."}]
    у канонічний текстовий рядок формату Qwen2.5 ChatML.
    """
    rendered = []
    for msg in messages:
        role = msg.get("role", "").strip()
        content = msg.get("content", "").strip()
        if role not in ("system", "user", "assistant"):
            raise ValueError(f"Невідома роль у повідомленні ChatML: '{role}'")
        rendered.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    return "".join(rendered)


def validate_jsonl_dataset(dataset_path: str) -> Tuple[int, List[Dict]]:
    """
    Перевіряє файл .jsonl на відповідність вимогам GVK ChatML:
    - Кожен рядок є валідним JSON з ключем 'messages'.
    - Кожен діалог містить щонайменше 1 репліку 'user' та 1 репліку 'assistant'.
    - Остання репліка в діалозі належить 'assistant' (цільовому другу).
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Файл датасету не знайдено: {dataset_path}")

    samples = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Помилка JSON у рядку {line_num} ({dataset_path}): {e}")

            messages = obj.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(
                    f"Рядок {line_num}: поле 'messages' має бути списком щонайменше з 2 повідомлень."
                )

            roles = [m.get("role") for m in messages]
            if "assistant" not in roles or "user" not in roles:
                raise ValueError(
                    f"Рядок {line_num}: діалог повинен містити і 'user', і 'assistant' ролі."
                )

            if messages[-1].get("role") != "assistant":
                raise ValueError(
                    f"Рядок {line_num}: остання репліка діалогу має належати 'assistant', "
                    f"знайдено '{messages[-1].get('role')}'."
                )

            samples.append(obj)

    if not samples:
        raise ValueError(f"Датасет порожній: {dataset_path}")

    return len(samples), samples


def simulate_loss_masking_preview(messages: List[Dict[str, str]]) -> str:
    """
    Візуалізує, як саме працює train_on_responses_only:
    показує, які частини тексту мають MASK (loss = -100, не тренуються),
    а які — ACTIVE GRADIENT (відповіді цільового друга).
    """
    preview_lines = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "assistant":
            preview_lines.append(f"  [🟢 LOSS ACTIVE  | {role.upper():9s}]: {content}")
        else:
            preview_lines.append(f"  [⚪ LOSS IGNORED | {role.upper():9s}]: {content}")
    return "\n".join(preview_lines)


# =====================================================================
# КОНВЕРТАЦІЯ LORA АДАПТЕРА У ФОРМАТ GGUF
# =====================================================================
def export_lora_to_gguf(
    model,
    tokenizer,
    safetensors_dir: str,
    output_gguf_path: str,
    base_model_hf: str = DEFAULT_BASE_MODEL_HF,
    llama_cpp_dir: Optional[str] = None,
) -> bool:
    """
    Експортує навчений LoRA-адаптер у бінарний файл .gguf для llama-server.
    Підтримує 2 методи:
    1. Вбудований експорт Unsloth (save_pretrained_gguf / convert_lora_to_gguf).
    2. Прямий виклик офіційного скрипта convert_lora_to_gguf.py із репозиторію llama.cpp.
    """
    print(f"\n[*] Початок конвертації LoRA-адаптера у формат GGUF -> {output_gguf_path}")
    os.makedirs(os.path.dirname(os.path.abspath(output_gguf_path)), exist_ok=True)

    # Метод 1: Перевірка наявності convert_lora_to_gguf.py (llama.cpp)
    search_paths = []
    if llama_cpp_dir:
        search_paths.append(Path(llama_cpp_dir) / "convert_lora_to_gguf.py")
    search_paths.extend([
        Path("llama.cpp/convert_lora_to_gguf.py"),
        Path("../llama.cpp/convert_lora_to_gguf.py"),
        Path("/content/llama.cpp/convert_lora_to_gguf.py"),
    ])

    converter_script = next((p for p in search_paths if p.exists()), None)

    if converter_script is not None:
        print(f"[*] Знайдено офіційний конвертер llama.cpp: {converter_script}")
        cmd = [
            sys.executable,
            str(converter_script),
            "--base",
            base_model_hf,
            "--outtype",
            "f16",
            "--outfile",
            output_gguf_path,
            safetensors_dir,
        ]
        res = subprocess.run(cmd, check=False)
        if res.returncode == 0 and os.path.exists(output_gguf_path):
            size_mb = os.path.getsize(output_gguf_path) / (1024 * 1024)
            print(f"[+] УСПІХ! GGUF LoRA-адаптер створено: {output_gguf_path} ({size_mb:.2f} МБ)")
            return True
        else:
            print("[!] Конвертер llama.cpp повернув помилку, пробуємо метод Unsloth...")

    # Метод 2: Спроба через вбудований метод Unsloth (якщо підтримується поточною версією)
    if model is not None and hasattr(model, "save_pretrained_gguf"):
        try:
            temp_export_dir = output_gguf_path + "_tmp_export"
            model.save_pretrained_gguf(
                temp_export_dir,
                tokenizer,
                quantization_method="f16",
            )
            # Знаходимо згенерований .gguf файл
            for root, _, files in os.walk(temp_export_dir):
                for file in files:
                    if file.endswith(".gguf"):
                        src_gguf = os.path.join(root, file)
                        shutil.move(src_gguf, output_gguf_path)
                        shutil.rmtree(temp_export_dir, ignore_errors=True)
                        size_mb = os.path.getsize(output_gguf_path) / (1024 * 1024)
                        print(f"[+] УСПІХ! GGUF створено через Unsloth: {output_gguf_path} ({size_mb:.2f} МБ)")
                        return True
        except Exception as e:
            print(f"[!] Попередження під час save_pretrained_gguf: {e}")

    print(
        "\n[i] Примітка: Ваги LoRA успішно збережено у форматі HuggingFace Safetensors:\n"
        f"    -> {safetensors_dir}\n"
        "    Для конвертації у GGUF виконайте команду:\n"
        f"    python3 llama.cpp/convert_lora_to_gguf.py --base {base_model_hf} "
        f"--outtype f16 --outfile {output_gguf_path} {safetensors_dir}"
    )
    return False


# =====================================================================
# ОСНОВНИЙ ПАЙПЛАЙН НАВЧАННЯ QLORA (UNSLOTH + TRL SFTTRAINER)
# =====================================================================
def train_single_persona(
    persona_id: str,
    dataset_path: str,
    output_dir: str,
    base_model: str = DEFAULT_BASE_MODEL_UNSLOTH,
    epochs: float = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    grad_accum: int = DEFAULT_GRAD_ACCUM,
    max_seq_length: int = DEFAULT_MAX_SEQ_LEN,
    lora_r: int = LORA_RANK,
    lora_alpha: int = LORA_ALPHA,
    dry_run: bool = False,
    llama_cpp_dir: Optional[str] = None,
) -> Dict[str, str]:
    """
    Повний цикл навчання одного QLoRA-адаптера для обраного друга.
    """
    print("=" * 70)
    print(f" GVK QLoRA TRAINING PIPELINE // ЦІЛЬОВИЙ АГЕНТ: [{persona_id.upper()}]")
    print("=" * 70)

    # 1. Валідація датасету
    num_samples, raw_samples = validate_jsonl_dataset(dataset_path)
    print(f"[+] Датасет валідний: {dataset_path} ({num_samples} мультитурнових діалогів)")

    # Перевірка форматування ChatML на першому прикладі
    sample_text = format_chatml_conversation(raw_samples[0]["messages"])
    print(f"[+] Перевірка шаблону ChatML пройдена (довжина першого зразка: {len(sample_text)} символів).")

    if dry_run:
        print("\n--- [DRY-RUN] ПЕРЕВІРКА МАСКУВАННЯ ФУНКЦІЇ ВТРАТ (train_on_responses_only) ---")
        print(simulate_loss_masking_preview(raw_samples[0]["messages"]))
        print("-" * 70)
        print(
            f"[+] DRY-RUN ЗАВЕРШЕНО УСПІШНО для '{persona_id}'.\n"
            f"    Гіперпараметри: Base={base_model}, r={lora_r}, alpha={lora_alpha}, "
            f"epochs={epochs}, lr={lr}, batch={batch_size}x{grad_accum}"
        )
        return {
            "persona_id": persona_id,
            "status": "dry_run_ok",
            "samples": str(num_samples),
        }

    # 2. Імпорт GPU-бібліотек (Unsloth, PyTorch, TRL, Datasets)
    try:
        import torch
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer
        from unsloth import FastLanguageModel, is_bfloat16_supported
        from unsloth.chat_templates import train_on_responses_only
    except ImportError as e:
        print(
            f"\n[!] ПОМИЛКА ІМПОРТУ GPU-БІБЛІОТЕК: {e}\n"
            "    Навчання QLoRA потребує середовища з NVIDIA GPU (CUDA) та встановлених пакетів:\n"
            "    pip install unsloth trl peft accelerate bitsandbytes datasets\n"
            "    Порада: Використовуйте прапорець --dry-run для перевірки датасету на macOS/CPU."
        )
        sys.exit(1)

    if not torch.cuda.is_available():
        print(
            "\n[!] УВАГА: CUDA GPU не виявлено! Unsloth QLoRA вимагає відеокарти NVIDIA (наприклад, Google Colab T4).\n"
            "    Запустіть із прапорцем --dry-run для тестування на CPU."
        )
        sys.exit(1)

    # 3. Завантаження базової моделі Qwen2.5-7B-Instruct у 4-бітному режимі (QLoRA)
    print(f"\n[*] Завантаження базової моделі: {base_model} (max_seq_length={max_seq_length})...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=None,  # Автовизначення (Float16 для Tesla T4, Bfloat16 для Ampere+)
        load_in_4bit=True,
    )

    # 4. Додавання матриць LoRA до всіх 7 лінійних модулів трансформера
    print(f"[*] Ініціалізація адаптера LoRA (r={lora_r}, alpha={lora_alpha}, modules={TARGET_MODULES})...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=TARGET_MODULES,
        lora_alpha=lora_alpha,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Економить 30% VRAM і дозволяє більший контекст
        random_state=DEFAULT_SEED,
        use_rslora=False,
        loftq_config=None,
    )

    # 5. Підготовка датасету у формат Hugging Face Dataset
    formatted_texts = [format_chatml_conversation(item["messages"]) for item in raw_samples]
    hf_dataset = Dataset.from_dict({"text": formatted_texts})

    persona_out_dir = os.path.join(output_dir, persona_id)
    checkpoints_dir = os.path.join(persona_out_dir, "checkpoints")
    safetensors_dir = os.path.join(persona_out_dir, "hf_lora")
    gguf_out_path = os.path.join(output_dir, f"{persona_id}_lora.gguf")

    os.makedirs(persona_out_dir, exist_ok=True)

    # 6. Конфігурація SFTTrainer (Cosine LR, adamw_8bit, 1.5-2.0 епохи)
    training_args = SFTConfig(
        output_dir=checkpoints_dir,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,  # Вимкнено для коректної роботи train_on_responses_only
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        warmup_ratio=DEFAULT_WARMUP_RATIO,
        num_train_epochs=epochs,
        learning_rate=lr,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=DEFAULT_SEED,
        save_strategy="no",
        report_to="none",  # Жодних зовнішніх трекерів (Zero-Cloud / Zero-Telemetry)
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=hf_dataset,
         args=training_args,
    )

    # 7. Маскування лоссу: тренуємося ТІЛЬКИ на репліках цільового друга (<|im_start|>assistant\n)
    print("[*] Активація маскування лоссу (train_on_responses_only)...")
    trainer = train_on_responses_only(
        trainer,
        instruction_part=CHATML_INSTRUCTION_PART,
        response_part=CHATML_RESPONSE_PART,
    )

    # 8. Запуск навчання
    print(f"\n[*] Старт навчання QLoRA для [{persona_id}] ({epochs} епох)...")
    trainer_stats = trainer.train()
    train_runtime = trainer_stats.metrics.get("train_runtime", 0.0)
    final_loss = trainer_stats.metrics.get("train_loss", 0.0)
    print(f"[+] Навчання [{persona_id}] завершено за {train_runtime:.1f} с | Фінальний Loss: {final_loss:.4f}")

    # 9. Збереження Safetensors адаптера
    print(f"[*] Збереження LoRA Safetensors у: {safetensors_dir}")
    model.save_pretrained(safetensors_dir)
    tokenizer.save_pretrained(safetensors_dir)

    # 10. Експорт у формат GGUF для llama-server
    export_lora_to_gguf(
        model=model,
        tokenizer=tokenizer,
        safetensors_dir=safetensors_dir,
        output_gguf_path=gguf_out_path,
        base_model_hf=DEFAULT_BASE_MODEL_HF,
        llama_cpp_dir=llama_cpp_dir,
    )

    # 11. Очищення GPU пам'яті перед наступним агентом
    del trainer
    del model
    torch.cuda.empty_cache()

    return {
        "persona_id": persona_id,
        "status": "trained",
        "safetensors_dir": safetensors_dir,
        "gguf_path": gguf_out_path,
        "loss": f"{final_loss:.4f}",
    }


# =====================================================================
# ВБУДОВАНИЙ SELF-TEST (ПЕРЕВІРКА БЕЗ GPU НА MAC / LINUX / WINDOWS)
# =====================================================================
def run_self_test() -> int:
    """
    Виконує повну автономну перевірку логіки валідації датасетів,
    генерації шаблонів Qwen2.5 ChatML та симуляції train_on_responses_only
    в оперативній пам'яті без потреби у відеокарті.
    """
    print("=" * 70)
    print(" GVK QLoRA TRAINER // АВТОНОМНИЙ SELF-TEST (CPU / RAM)")
    print("=" * 70)

    synthetic_dialogue = {
        "messages": [
            {
                "role": "system",
                "content": "Ти — цифровий двійник friend_1. Відповідай у своєму стилі.",
            },
            {
                "role": "user",
                "content": "[friend_2]: Хто пам'ятає, на якому порту в нас піднятий локальний сервер?\n"
                           "[friend_3]: Здається, на 8080, але я не впевнений.",
            },
            {
                "role": "assistant",
                "content": "На 8080 крутиться llama-server, я вчора особисто конфіг правив.",
            },
        ]
    }

    with tempfile.TemporaryDirectory(prefix="gvk_lora_test_") as tmp_dir:
        test_jsonl = os.path.join(tmp_dir, "friend_1_chatml.jsonl")
        with open(test_jsonl, "w", encoding="utf-8") as f:
            f.write(json.dumps(synthetic_dialogue, ensure_ascii=False) + "\n")

        # Тест 1: Валідація датасету
        count, samples = validate_jsonl_dataset(test_jsonl)
        assert count == 1, "Помилка підрахунку зразків у датасеті"
        print("[1/3] Валідація структури ChatML .jsonl ..................... [OK]")

        # Тест 2: Перевірка генерації маркера Qwen2.5 ChatML
        rendered = format_chatml_conversation(samples[0]["messages"])
        assert CHATML_INSTRUCTION_PART in rendered, "Відсутній тег <|im_start|>user"
        assert CHATML_RESPONSE_PART in rendered, "Відсутній тег <|im_start|>assistant"
        assert rendered.endswith(CHATML_END_TOKEN), "Відсутній завершальний <|im_end|>"
        print("[2/3] Форматування промпту Qwen2.5 ChatML .................. [OK]")

        # Тест 3: Dry-run пайплайну для одного персонажа
        res = train_single_persona(
            persona_id="friend_1",
            dataset_path=test_jsonl,
            output_dir=tmp_dir,
            dry_run=True,
        )
        assert res["status"] == "dry_run_ok"
        print("[3/3] Симуляція маскування лоссу (train_on_responses_only) .. [OK]")

    print("=" * 70)
    print(" [+] УСІ ПЕРЕВІРКИ ПРОЙДЕНО УСПІШНО! Скрипт готовий до роботи.")
    print("=" * 70)
    return 0


# =====================================================================
# CLI ІНТЕРФЕЙС
# =====================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GVK Private AI — QLoRA Fine-Tuning & GGUF Exporter (Qwen2.5-7B-Instruct)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Доступні команди")

    # 1. Команда self-test
    subparsers.add_parser(
        "self-test",
        help="Швидка перевірка валідатора датасетів та шаблонів ChatML без GPU",
    )

    # 2. Команда train (навчання одного адаптера)
    train_p = subparsers.add_parser(
        "train",
        help="Навчити QLoRA-адаптер для одного учасника (наприклад, friend_1)",
    )
    train_p.add_argument(
        "--persona",
        "-p",
        type=str,
        required=True,
        help="Ідентифікатор учасника (наприклад: friend_1, friend_2, friend_3)",
    )
    train_p.add_argument(
        "--dataset",
        "-d",
        type=str,
        required=True,
        help="Шлях до файлу датасету у форматі ChatML (.jsonl)",
    )
    train_p.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="adapters",
        help="Директорія для збереження навчених адаптерів та .gguf (за замовчуванням: adapters/)",
    )
    train_p.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_BASE_MODEL_UNSLOTH,
        help=f"Назва або шлях до базової моделі (за замовчуванням: {DEFAULT_BASE_MODEL_UNSLOTH})",
    )
    train_p.add_argument(
        "--epochs",
        type=float,
        default=DEFAULT_EPOCHS,
        help=f"Кількість епох навчання (рекомендовано 1.5 - 2.0, за замовчуванням: {DEFAULT_EPOCHS})",
    )
    train_p.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help=f"Learning Rate (за замовчуванням: {DEFAULT_LR})",
    )
    train_p.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size на GPU (за замовчуванням: {DEFAULT_BATCH_SIZE})",
    )
    train_p.add_argument(
        "--grad-accum",
        type=int,
        default=DEFAULT_GRAD_ACCUM,
        help=f"Кроки накопичення градієнта (за замовчуванням: {DEFAULT_GRAD_ACCUM})",
    )
    train_p.add_argument(
        "--max-seq-len",
        type=int,
        default=DEFAULT_MAX_SEQ_LEN,
        help=f"Максимальна довжина контексту в токенах (за замовчуванням: {DEFAULT_MAX_SEQ_LEN})",
    )
    train_p.add_argument(
        "--lora-r",
        type=int,
        default=LORA_RANK,
        help=f"Ранг матриць LoRA r (за замовчуванням: {LORA_RANK})",
    )
    train_p.add_argument(
        "--lora-alpha",
        type=int,
        default=LORA_ALPHA,
        help=f"Коефіцієнт масштабування LoRA alpha (за замовчуванням: {LORA_ALPHA})",
    )
    train_p.add_argument(
        "--llama-cpp-dir",
        type=str,
        default=None,
        help="Шлях до папки llama.cpp для конвертації у GGUF (опціонально)",
    )
    train_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Тільки перевірити датасет та маскування лоссу без запуску GPU навчання",
    )

    # 3. Команда train-all (пакетне навчання всіх 3-х друзів)
    all_p = subparsers.add_parser(
        "train-all",
        help="Пакетне навчання всіх 3-х адаптерів (friend_1, friend_2, friend_3) по черзі",
    )
    all_p.add_argument(
        "--dataset-dir",
        "-d",
        type=str,
        default="datasets",
        help="Папка з файлами friend_1_chatml.jsonl, friend_2_chatml.jsonl, friend_3_chatml.jsonl",
    )
    all_p.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="adapters",
        help="Директорія для збереження готових .gguf адаптерів",
    )
    all_p.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_BASE_MODEL_UNSLOTH,
        help=f"Базова модель (за замовчуванням: {DEFAULT_BASE_MODEL_UNSLOTH})",
    )
    all_p.add_argument(
        "--epochs",
        type=float,
        default=DEFAULT_EPOCHS,
        help=f"Кількість епох (за замовчуванням: {DEFAULT_EPOCHS})",
    )
    all_p.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help=f"Learning Rate (за замовчуванням: {DEFAULT_LR})",
    )
    all_p.add_argument(
        "--llama-cpp-dir",
        type=str,
        default=None,
        help="Шлях до папки llama.cpp для конвертації у GGUF",
    )
    all_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Перевірити всі 3 датасети без запуску GPU",
    )

    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.command or args.command == "self-test":
        return run_self_test()

    if args.command == "train":
        train_single_persona(
            persona_id=args.persona,
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            base_model=args.base_model,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            max_seq_length=args.max_seq_len,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            dry_run=args.dry_run,
            llama_cpp_dir=args.llama_cpp_dir,
        )
        return 0

    if args.command == "train-all":
        results = []
        for pid in PERSONA_IDS:
            ds_file = os.path.join(args.dataset_dir, f"{pid}_chatml.jsonl")
            if not os.path.exists(ds_file):
                print(f"[!] Пропуск [{pid}]: файл датасету не знайдено ({ds_file})")
                continue
            res = train_single_persona(
                persona_id=pid,
                dataset_path=ds_file,
                output_dir=args.output_dir,
                base_model=args.base_model,
                epochs=args.epochs,
                lr=args.lr,
                dry_run=args.dry_run,
                llama_cpp_dir=args.llama_cpp_dir,
            )
            results.append(res)

        print("\n" + "=" * 70)
        print(f" ПІДСУМОК ПАКЕТНОГО НАВЧАННЯ: Опрацьовано {len(results)}/3 агентів")
        print("=" * 70)
        for r in results:
            print(f"  • {r['persona_id']}: статус={r['status']} | GGUF={r.get('gguf_path', 'N/A')}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
