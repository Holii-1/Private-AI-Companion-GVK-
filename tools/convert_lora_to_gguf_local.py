import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
from typing import List, Optional

# ==============================================================================
# 1. Global Configuration Variables (Defaults; override via CLI flags)
# ==============================================================================
# Шлях до розпакованої папки lora_final (завантаженої з Google Drive)
LORA_INPUT_DIR: str = r"./adapters/lora_final"

# Шлях, куди зберегти готовий файл LoRA у форматі .gguf
GGUF_OUTPUT_PATH: str = r"./adapters/friend_1.gguf"

# Оригінальна базова модель на HuggingFace (скрипт скачає лише ~10 МБ конфігів/заголовків, без 28 ГБ ваг)
BASE_MODEL_HF_ID: str = "Qwen/Qwen2.5-14B-Instruct"

# Тип вихідних ваг адаптера: "f16" (рекомендовано), "bf16", "q8_0" або "f32"
OUTTYPE: str = "f16"

# Робоча папка для кешування скриптів конвертації llama.cpp (вже в .gitignore під build/)
LLAMA_CPP_DIR: str = r"./build/llama_cpp_converter"


# ==============================================================================
# 2. Helper Functions
# ==============================================================================
def resolve_lora_dir(input_dir: str) -> str:
    """Знаходить папку з adapter_config.json (навіть якщо архів розпакувався як lora_final/lora_final)."""
    if not os.path.exists(input_dir):
        raise FileNotFoundError(
            f"❌ Папку з адаптером не знайдено за шляхом:\n   {input_dir}\n"
            "Завантажте папку 'lora_final' із Google Drive і перевірте змінну LORA_INPUT_DIR."
        )

    if os.path.isfile(os.path.join(input_dir, "adapter_config.json")):
        return input_dir

    for root, _, files in os.walk(input_dir):
        if "adapter_config.json" in files:
            print(f"ℹ️ Знайдено adapter_config.json у вкладеній папці: {root}")
            return root

    raise FileNotFoundError(
        f"❌ У папці {input_dir} не знайдено 'adapter_config.json'.\n"
        "Переконайтеся, що там лежать файли adapter_config.json та adapter_model.safetensors."
    )


def _is_installed(mod_name: str) -> bool:
    try:
        return importlib.util.find_spec(mod_name) is not None
    except ModuleNotFoundError:
        return False


def ensure_dependencies() -> None:
    """Перевіряє та за потреби встановлює мінімальні Python-пакети для конвертації."""
    if not _is_installed("torch"):
        print("📦 Встановлення легкого CPU-варіанту PyTorch (потрібен для читання тензорів)...")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "torch",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
            ],
            check=True,
        )

    required = {
        "gguf": "gguf",
        "safetensors": "safetensors",
        "transformers": "transformers",
        "huggingface_hub": "huggingface_hub",
        "sentencepiece": "sentencepiece",
        "google.protobuf": "protobuf",
    }
    missing = [pkg for mod, pkg in required.items() if not _is_installed(mod)]
    if missing:
        print(f"📦 Встановлення залежностей: {', '.join(missing)}...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)


def ensure_llama_cpp(repo_dir: str) -> str:
    """Клонує репозиторій llama.cpp (тільки останній коміт), якщо скрипт конвертації ще не завантажено."""
    script_path = os.path.join(repo_dir, "convert_lora_to_gguf.py")
    if os.path.isfile(script_path):
        return script_path

    print(f"⬇️ Клонування скриптів конвертації llama.cpp у {repo_dir}...")
    os.makedirs(os.path.dirname(os.path.abspath(repo_dir)), exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp.git", repo_dir],
        check=True,
    )

    gguf_py_dir = os.path.join(repo_dir, "gguf-py")
    if os.path.isdir(gguf_py_dir):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", gguf_py_dir], check=True)

    return script_path


def resolve_default_paths(
    persona: Optional[str] = None,
    input_dir: Optional[str] = None,
    output_path: Optional[str] = None,
) -> tuple[str, str]:
    """Визначає вхідну папку PEFT-адаптера та вихідний шлях .gguf."""
    if input_dir:
        resolved_in = input_dir
    elif persona:
        cand = f"./adapters/{persona}_lora_final"
        resolved_in = cand if os.path.exists(cand) else LORA_INPUT_DIR
    else:
        resolved_in = LORA_INPUT_DIR

    if output_path:
        resolved_out = output_path
    elif persona:
        resolved_out = f"./adapters/{persona}.gguf"
    else:
        resolved_out = GGUF_OUTPUT_PATH

    return resolved_in, resolved_out


def build_convert_command(
    converter_script: str,
    lora_dir: str,
    output_path: str,
    base_model_id: str = BASE_MODEL_HF_ID,
    outtype: str = OUTTYPE,
    python_exe: str = sys.executable,
) -> List[str]:
    """Формує CLI-команду для convert_lora_to_gguf.py."""
    return [
        python_exe,
        converter_script,
        lora_dir,
        "--base-model-id",
        base_model_id,
        "--outfile",
        output_path,
        "--outtype",
        outtype,
    ]


# ==============================================================================
# 3. Main Execution & CLI
# ==============================================================================
def run_convert(
    input_dir: str,
    output_path: str,
    base_model_id: str = BASE_MODEL_HF_ID,
    outtype: str = OUTTYPE,
    llama_cpp_dir: str = LLAMA_CPP_DIR,
    dry_run: bool = False,
) -> List[str]:
    print("=" * 65)
    print("GVK Local LoRA -> GGUF Converter")
    print("=" * 65)

    lora_dir = resolve_lora_dir(input_dir)
    print(f"📂 Папка вхідного адаптера : {lora_dir}")
    print(f"🎯 Файл виходу GGUF        : {output_path}")
    print(f"🧬 Базова модель (HF ID)   : {base_model_id}")
    print(f"🗜️ Тип квантування        : {outtype}")
    print("-" * 65)

    if dry_run:
        converter_script = os.path.join(llama_cpp_dir, "convert_lora_to_gguf.py")
        cmd = build_convert_command(converter_script, lora_dir, output_path, base_model_id, outtype)
        print(" [DRY-RUN] Команда конвертації:")
        print(" ▶", " ".join(f'"{c}"' if " " in c else c for c in cmd))
        return cmd

    ensure_dependencies()
    converter_script = ensure_llama_cpp(llama_cpp_dir)

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    cmd = build_convert_command(converter_script, lora_dir, output_path, base_model_id, outtype)

    print("⚙️ Запускаю конвертацію LoRA -> GGUF...")
    print("▶", " ".join(f'"{c}"' if " " in c else c for c in cmd))
    subprocess.run(cmd, check=True)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print("=" * 65)
    print("✅ УСПІШНО! LoRA GGUF збережено:")
    print(f"   {output_path} ({size_mb:.1f} MB)")
    print("=" * 65)
    return cmd


def run_self_test() -> None:
    print("====================================================================")
    print(" [*] GVK Private AI — Self-Test (tools/convert_lora_to_gguf_local.py)")
    print("====================================================================")

    with tempfile.TemporaryDirectory(prefix="gvk_lora_conv_test_") as tmp_dir:
        nested = os.path.join(tmp_dir, "lora_final", "lora_final")
        os.makedirs(nested, exist_ok=True)
        with open(os.path.join(nested, "adapter_config.json"), "w", encoding="utf-8") as f:
            f.write('{"peft_type": "LORA"}\n')
        with open(os.path.join(nested, "adapter_model.safetensors"), "wb") as f:
            f.write(b"\x00" * 16)

        resolved = resolve_lora_dir(os.path.join(tmp_dir, "lora_final"))
        assert os.path.abspath(resolved) == os.path.abspath(nested), f"Очікувався вкладений шлях: {resolved}"

        out_gguf = os.path.join(tmp_dir, "friend_2.gguf")
        cmd = run_convert(
            input_dir=os.path.join(tmp_dir, "lora_final"),
            output_path=out_gguf,
            outtype="q8_0",
            llama_cpp_dir=os.path.join(tmp_dir, "llama_cpp"),
            dry_run=True,
        )
        assert "--outtype" in cmd and cmd[cmd.index("--outtype") + 1] == "q8_0"
        assert "--outfile" in cmd and cmd[cmd.index("--outfile") + 1] == out_gguf
        print(" [✓] resolve_lora_dir, q8_0 --dry-run та побудова CLI-команди OK!")

    print("====================================================================")
    print(" [✓] SELF-TEST CONVERT_LORA_TO_GGUF ПРОЙДЕНО УСПІШНО!")
    print("====================================================================")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GVK Private AI — Локальний конвертер PEFT LoRA у GGUF"
    )
    parser.add_argument(
        "cmd",
        nargs="?",
        choices=["convert", "self-test"],
        default="convert",
        help="Режим: 'convert' (за замовчуванням) або 'self-test'",
    )
    parser.add_argument("-p", "--persona", default=None, help="ID персони (friend_1, friend_2, friend_3)")
    parser.add_argument("-i", "--input", default=None, help="Папка з adapter_config.json (lora_final)")
    parser.add_argument("-o", "--output", default=None, help="Вихідний шлях до файлу .gguf")
    parser.add_argument(
        "-t",
        "--outtype",
        choices=("f16", "bf16", "q8_0", "f32"),
        default=OUTTYPE,
        help="Тип вихідних ваг (f16, bf16, q8_0, f32)",
    )
    parser.add_argument("--base-model", default=BASE_MODEL_HF_ID, help="Hugging Face ID базової моделі")
    parser.add_argument("--llama-cpp-dir", default=LLAMA_CPP_DIR, help="Папка кешу llama.cpp")
    parser.add_argument("--dry-run", action="store_true", help="Лише перевірити шляхи та команду без запуску")
    args = parser.parse_args()

    if args.cmd == "self-test":
        run_self_test()
        return

    in_dir, out_path = resolve_default_paths(persona=args.persona, input_dir=args.input, output_path=args.output)
    run_convert(
        input_dir=in_dir,
        output_path=out_path,
        base_model_id=args.base_model,
        outtype=args.outtype,
        llama_cpp_dir=args.llama_cpp_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
