#!/usr/bin/env python3
"""
GVK Private AI - Local PEFT LoRA to GGUF Converter (tools/convert_lora_to_gguf_local.py)

Converts a downloaded PEFT LoRA adapter directory (`adapter_config.json` and
`adapter_model.safetensors`) into a `llama.cpp`-compatible `.gguf` LoRA file locally
without downloading the full base model weights.

Key Responsibilities:
  - Locate `adapter_config.json` even inside nested archive subdirectories.
  - Bootstrap minimal CPU-only Python dependencies and shallow-clone `llama.cpp` conversion scripts.
  - Assemble and invoke `convert_lora_to_gguf.py` with configurable quantization (`f16`, `bf16`, `q8_0`, `f32`).
  - Support `--dry-run` inspection and an offline `self-test` command.

Usage:
  python3 tools/convert_lora_to_gguf_local.py --persona friend_1
  python3 tools/convert_lora_to_gguf_local.py -i ./adapters/lora_final -o ./adapters/friend_1.gguf -t f16
  python3 tools/convert_lora_to_gguf_local.py --dry-run
  python3 tools/convert_lora_to_gguf_local.py self-test
"""

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
from typing import List, Optional

# =====================================================================
# 1. GLOBAL CONFIGURATION
# =====================================================================
# Path to the extracted lora_final directory (e.g., downloaded from Google Drive)
LORA_INPUT_DIR: str = r"./adapters/lora_final"

# Destination path for the converted .gguf LoRA adapter file
GGUF_OUTPUT_PATH: str = r"./adapters/friend_1.gguf"

# Upstream base model on Hugging Face (only ~10 MB of config/header metadata is fetched, not 28 GB of weights)
BASE_MODEL_HF_ID: str = "Qwen/Qwen2.5-14B-Instruct"

# Output adapter weight type: "f16" (recommended), "bf16", "q8_0", or "f32"
OUTTYPE: str = "f16"

# Local cache directory for llama.cpp conversion scripts (ignored by git under build/)
LLAMA_CPP_DIR: str = r"./build/llama_cpp_converter"


# =====================================================================
# 2. HELPER FUNCTIONS
# =====================================================================
def resolve_lora_dir(input_dir: str) -> str:
    """Locate the directory containing `adapter_config.json`.

    Handles archives that extract into an extra nested folder (such as
    `lora_final/lora_final`).

    Args:
        input_dir: Root path to search for the PEFT adapter files.

    Returns:
        Path to the directory directly containing `adapter_config.json`.

    Raises:
        FileNotFoundError: If `input_dir` does not exist or contains no `adapter_config.json`.
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(
            f"[!] Adapter directory not found at:\n   {input_dir}\n"
            "Download the 'lora_final' directory from Google Drive and verify LORA_INPUT_DIR or --input."
        )

    if os.path.isfile(os.path.join(input_dir, "adapter_config.json")):
        return input_dir

    for root, _, files in os.walk(input_dir):
        if "adapter_config.json" in files:
            print(f"[*] Found adapter_config.json in nested directory: {root}")
            return root

    raise FileNotFoundError(
        f"[!] 'adapter_config.json' not found in {input_dir}.\n"
        "Ensure both adapter_config.json and adapter_model.safetensors are present."
    )


def _is_installed(mod_name: str) -> bool:
    """Check whether a Python module is importable in the current environment.

    Args:
        mod_name: Dotted module name to probe with `importlib.util.find_spec`.

    Returns:
        `True` if the module spec exists, `False` otherwise.
    """
    try:
        return importlib.util.find_spec(mod_name) is not None
    except ModuleNotFoundError:
        return False


def ensure_dependencies() -> None:
    """Install minimal CPU-only Python packages required for LoRA-to-GGUF conversion."""
    if not _is_installed("torch"):
        print("[*] Installing lightweight CPU build of PyTorch (required for reading tensors)...")
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
        print(f"[*] Installing missing dependencies: {', '.join(missing)}...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)


def ensure_llama_cpp(repo_dir: str) -> str:
    """Shallow-clone `llama.cpp` when `convert_lora_to_gguf.py` is not cached locally.

    Args:
        repo_dir: Local cache directory for the `llama.cpp` repository.

    Returns:
        Path to `convert_lora_to_gguf.py` inside `repo_dir`.
    """
    script_path = os.path.join(repo_dir, "convert_lora_to_gguf.py")
    if os.path.isfile(script_path):
        return script_path

    print(f"[*] Cloning llama.cpp conversion scripts into {repo_dir}...")
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
    """Resolve default input PEFT directory and output `.gguf` file path.

    Args:
        persona: Optional persona ID (e.g., `"friend_1"`).
        input_dir: Explicit input directory override.
        output_path: Explicit output `.gguf` file path override.

    Returns:
        Tuple of `(resolved_input_dir, resolved_output_path)`.
    """
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
    """Construct the subprocess command list for `convert_lora_to_gguf.py`.

    Args:
        converter_script: Path to `llama.cpp/convert_lora_to_gguf.py`.
        lora_dir: Resolved PEFT adapter directory.
        output_path: Destination `.gguf` file path.
        base_model_id: Hugging Face model ID of the base model.
        outtype: Target weight quantization/precision (`f16`, `bf16`, `q8_0`, `f32`).
        python_exe: Python interpreter executable path.

    Returns:
        Argument list suitable for `subprocess.run`.
    """
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


# =====================================================================
# 3. MAIN EXECUTION & CLI
# =====================================================================
def run_convert(
    input_dir: str,
    output_path: str,
    base_model_id: str = BASE_MODEL_HF_ID,
    outtype: str = OUTTYPE,
    llama_cpp_dir: str = LLAMA_CPP_DIR,
    dry_run: bool = False,
) -> List[str]:
    """Run (or preview via `dry_run`) the local PEFT LoRA to GGUF conversion.

    Args:
        input_dir: Path to the PEFT adapter folder.
        output_path: Destination `.gguf` file path.
        base_model_id: Hugging Face ID of the base model.
        outtype: Target adapter precision (`f16`, `bf16`, `q8_0`, or `f32`).
        llama_cpp_dir: Cache directory for the `llama.cpp` repository.
        dry_run: If `True`, validate paths and print the command without executing it.

    Returns:
        The assembled conversion command list.
    """
    print("=" * 69)
    print(" [*] GVK Private AI - Local LoRA to GGUF Converter")
    print("=" * 69)

    lora_dir = resolve_lora_dir(input_dir)
    print(f" Input adapter dir : {lora_dir}")
    print(f" Output GGUF file  : {output_path}")
    print(f" Base model (HF)   : {base_model_id}")
    print(f" Output weight type: {outtype}")
    print("-" * 69)

    if dry_run:
        converter_script = os.path.join(llama_cpp_dir, "convert_lora_to_gguf.py")
        cmd = build_convert_command(converter_script, lora_dir, output_path, base_model_id, outtype)
        print(" [*] [DRY-RUN] Conversion command:")
        print("     ", " ".join(f'"{c}"' if " " in c else c for c in cmd))
        return cmd

    ensure_dependencies()
    converter_script = ensure_llama_cpp(llama_cpp_dir)

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    cmd = build_convert_command(converter_script, lora_dir, output_path, base_model_id, outtype)

    print("[*] Starting LoRA -> GGUF conversion...")
    print("    ", " ".join(f'"{c}"' if " " in c else c for c in cmd))
    subprocess.run(cmd, check=True)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print("=" * 69)
    print("[OK] Conversion complete! LoRA GGUF saved to:")
    print(f"     {output_path} ({size_mb:.1f} MB)")
    print("=" * 69)
    return cmd


def run_self_test() -> None:
    """Run an offline self-test covering nested path resolution and dry-run CLI generation."""
    print("=====================================================================")
    print(" [*] GVK Private AI - Self-Test (tools/convert_lora_to_gguf_local.py)")
    print("=====================================================================")

    with tempfile.TemporaryDirectory(prefix="gvk_lora_conv_test_") as tmp_dir:
        nested = os.path.join(tmp_dir, "lora_final", "lora_final")
        os.makedirs(nested, exist_ok=True)
        with open(os.path.join(nested, "adapter_config.json"), "w", encoding="utf-8") as f:
            f.write('{"peft_type": "LORA"}\n')
        with open(os.path.join(nested, "adapter_model.safetensors"), "wb") as f:
            f.write(b"\x00" * 16)

        resolved = resolve_lora_dir(os.path.join(tmp_dir, "lora_final"))
        assert os.path.abspath(resolved) == os.path.abspath(nested), f"Expected nested path, got: {resolved}"

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
        print(" [OK] resolve_lora_dir, q8_0 --dry-run, and CLI command assembly passed.")

    print("=====================================================================")
    print(" [OK] CONVERT_LORA_TO_GGUF SELF-TEST PASSED!")
    print("=====================================================================")


def main() -> None:
    """Parse CLI arguments and run either LoRA conversion or offline self-test."""
    parser = argparse.ArgumentParser(
        description="GVK Private AI - Local PEFT LoRA to GGUF converter"
    )
    parser.add_argument(
        "cmd",
        nargs="?",
        choices=["convert", "self-test"],
        default="convert",
        help="Execution mode: 'convert' (default) or 'self-test'",
    )
    parser.add_argument("-p", "--persona", default=None, help="Persona ID (e.g., friend_1, friend_2, friend_3)")
    parser.add_argument("-i", "--input", default=None, help="Directory containing adapter_config.json (lora_final)")
    parser.add_argument("-o", "--output", default=None, help="Output path for the .gguf adapter file")
    parser.add_argument(
        "-t",
        "--outtype",
        choices=("f16", "bf16", "q8_0", "f32"),
        default=OUTTYPE,
        help="Output weight precision/quantization (f16, bf16, q8_0, f32)",
    )
    parser.add_argument("--base-model", default=BASE_MODEL_HF_ID, help="Hugging Face ID of the base model")
    parser.add_argument("--llama-cpp-dir", default=LLAMA_CPP_DIR, help="Local cache directory for llama.cpp")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths and print command without executing")
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
