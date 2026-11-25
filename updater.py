#!/usr/bin/env python3
"""
GVK Private AI - Media Verification, Signature Registry & Update Client (updater.py)

Manages local environment setup, cryptographic verification of GVK_VAULT media
(DVD or offline USB), incremental persona updates, and runtime asset caching.
Uses only the Python standard library so it can run directly from GVK_VAULT media
on a clean machine before `.venv` is created or `requirements.txt` is installed.

Key Responsibilities:
  - Initialize the local cache structure under `~/.gvk_ai/{base,bin,embeddings,personas,registry,logs}`.
  - Provision `.venv` (Python >= 3.10 with Tk >= 8.6) and keep `requirements.txt`
    dependencies synchronized via a SHA-256 marker file.
  - Verify and import `GVK_VAULT` media: validate the 32-byte `vault/master.key`,
    HMAC-SHA256 `persona/origin.sig`, manifest SHA-256 hashes, and `GVK1` container
    headers before copying encrypted containers to `~/.gvk_ai/`.
  - Perform Incremental Unlock from `personas_encrypted/`, updating only personas
    whose `origin.sig` is already registered locally.
  - Resolve, copy (from USB), symlink (from local repo), or resumable-download the
    public base model `Qwen2.5-14B-Instruct-Q4_K_M.gguf` (~9.0 GB) into `~/.gvk_ai/base/`.
  - Locate, unpack (from USB), or download the `llama-server` binary from GitHub
    Releases (`ggml-org/llama.cpp`) and pre-cache the RAG embedding model.

Usage:
  python3 updater.py --disc-path /Volumes/GVK_VAULT
  python3 updater.py sync --no-download
  python3 updater.py status
  python3 updater.py self-test
"""

import argparse
import glob
import hashlib
import hmac
import json
import os
import platform
import shutil
import ssl
import stat
import struct
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from typing import Dict, List, Optional, Tuple


# =====================================================================
# PROTOCOL CONSTANTS & SOURCES
# =====================================================================
DEFAULT_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GVK_HOME = os.environ.get("GVK_HOME") or os.path.expanduser("~/.gvk_ai")

DEFAULT_BASE_MODEL_FILENAME = "Qwen2.5-14B-Instruct-Q4_K_M.gguf"
DEFAULT_BASE_MODEL_URL = (
    "https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF/resolve/main/"
    "Qwen2.5-14B-Instruct-Q4_K_M.gguf"
)
LLAMA_CPP_RELEASES_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
LLAMA_CPP_RELEASES_LIST_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_EMBEDDING_SHORT_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

MIN_PYTHON_VERSION = (3, 10)
MIN_TK_VERSION = 8.6

MAGIC_BYTES = b"GVK1"
HEADER_SIZE = 36
KEY_SIZE = 32
SIG_SIZE = 32
GCM_TAG_SIZE = 16
GGUF_MAGIC = b"GGUF"


# =====================================================================
# HELPER UTILITIES (STDLIB ONLY)
# =====================================================================
def compute_sha256(file_path: str) -> str:
    """Compute the SHA-256 hex digest of a file using 4 MB streaming chunks.

    Args:
        file_path: Path to the file to hash.

    Returns:
        Lowercase hexadecimal SHA-256 digest string.
    """
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(4 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def verify_origin_sig_bytes(persona_id: str, master_key: bytes, sig_bytes: bytes) -> bool:
    """Verify a 32-byte HMAC-SHA256 `origin.sig` signature.

    Args:
        persona_id: Canonical persona identifier signed by the media key.
        master_key: 32-byte master key read from `vault/master.key`.
        sig_bytes: 32-byte signature read from `origin.sig`.

    Returns:
        True if the signature matches `HMAC-SHA256(master_key, persona_id)`, False otherwise.
    """
    if len(master_key) != KEY_SIZE or len(sig_bytes) != SIG_SIZE:
        return False
    expected = hmac.new(master_key, persona_id.strip().encode("utf-8"), hashlib.sha256).digest()
    return hmac.compare_digest(expected, sig_bytes)


def verify_gvk1_header(enc_file_path: str, master_key: Optional[bytes] = None) -> None:
    """Validate the 36-byte binary header of a `.enc` container.

    Header layout:
      `[Magic "GVK1" (4B) | Base Nonce (12B) | Header Auth Tag (16B) | Chunk Size (4B)]`

    Args:
        enc_file_path: Path to the encrypted `.enc` container file.
        master_key: Optional 32-byte master key; when provided, also verifies the
            cryptographic HMAC header authentication tag.

    Raises:
        ValueError: If the header is truncated, has invalid magic or chunk size,
            or fails `master_key` authentication.
    """
    with open(enc_file_path, "rb") as f:
        header = f.read(HEADER_SIZE)

    if len(header) != HEADER_SIZE:
        raise ValueError(
            f"Corrupted container {os.path.basename(enc_file_path)}: header is smaller than 36 bytes."
        )

    magic = header[:4]
    if magic != MAGIC_BYTES:
        raise ValueError(
            f"Invalid magic signature in {os.path.basename(enc_file_path)}: "
            f"expected {MAGIC_BYTES!r}, got {magic!r}."
        )

    base_nonce = header[4:16]
    header_tag = header[16:32]
    chunk_size_bytes = header[32:36]
    chunk_size = struct.unpack("<I", chunk_size_bytes)[0]
    if chunk_size == 0:
        raise ValueError(f"Invalid chunk size (0) in {os.path.basename(enc_file_path)}.")

    if master_key is not None:
        msg = MAGIC_BYTES + base_nonce + chunk_size_bytes
        expected_tag = hmac.new(master_key, msg, hashlib.sha256).digest()[:GCM_TAG_SIZE]
        if not hmac.compare_digest(header_tag, expected_tag):
            raise ValueError(
                f"Container {os.path.basename(enc_file_path)} failed master.key authentication!"
            )


def _normalize_disc_path(raw_path: str) -> str:
    """Normalize a media path (including Windows drive letters like `'D:'` -> `'D:\\'`)."""
    p = os.path.expanduser(raw_path.strip())
    if len(p) == 2 and p[1] == ":" and p[0].isalpha():
        p = p + "\\"
    return os.path.abspath(p)


def _copy_writable(src: str, dst: str) -> None:
    """Copy a file without preserving DVD read-only attributes.

    Clears read-only permissions on an existing destination before overwriting
    (critical on Windows when copying from optical media).

    Args:
        src: Source file path.
        dst: Destination file path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    if os.path.exists(dst):
        try:
            os.chmod(dst, 0o644)
        except OSError:
            pass
    shutil.copyfile(src, dst)
    try:
        os.chmod(dst, 0o644)
    except OSError:
        pass


def _build_ssl_context() -> ssl.SSLContext:
    """Build an SSL context with fallback to system `cert.pem` or `certifi` (for macOS python.org)."""
    ctx = ssl.create_default_context()
    try:
        stats = ctx.cert_store_stats()
        if stats.get("x509_ca", 0) > 0:
            return ctx
    except Exception:
        return ctx

    for ca_file in ("/etc/ssl/cert.pem", "/private/etc/ssl/cert.pem"):
        if os.path.isfile(ca_file):
            try:
                ctx.load_verify_locations(cafile=ca_file)
                return ctx
            except Exception:
                pass
    try:
        import certifi  # type: ignore

        ctx.load_verify_locations(cafile=certifi.where())
    except Exception:
        pass
    return ctx


def find_gvk_disc_path(explicit_path: Optional[str] = None) -> Optional[str]:
    """Locate a mounted `GVK_VAULT` medium containing a valid 32-byte `vault/master.key`.

    Args:
        explicit_path: Optional user-supplied path to check first.

    Returns:
        Absolute path to the detected `GVK_VAULT` root directory, or None if not found.
    """
    candidates: List[str] = []
    if explicit_path:
        candidates.append(_normalize_disc_path(explicit_path))

    env_disc = os.environ.get("GVK_DISC_PATH")
    if env_disc:
        candidates.append(_normalize_disc_path(env_disc))

    sys_name = platform.system().lower()
    if sys_name == "darwin":
        candidates.extend(sorted(glob.glob("/Volumes/GVK_VAULT*")))
        candidates.extend(sorted(glob.glob("/Volumes/*")))
    elif sys_name == "linux":
        user = os.environ.get("USER", "")
        candidates.extend(sorted(glob.glob(f"/media/{user}/GVK_VAULT*")))
        candidates.extend(sorted(glob.glob(f"/run/media/{user}/GVK_VAULT*")))
        candidates.extend(sorted(glob.glob("/media/*/GVK_VAULT*")))
        candidates.extend(sorted(glob.glob("/mnt/GVK_VAULT*")))
    elif sys_name == "windows":
        for letter in "DEFGHIJKLMNOPQRSTUVWXYZ":
            candidates.append(f"{letter}:\\")

    seen = set()
    for path in candidates:
        if not path or path in seen:
            continue
        seen.add(path)
        key_file = os.path.join(path, "vault", "master.key")
        try:
            if os.path.isfile(key_file) and os.path.getsize(key_file) == KEY_SIZE:
                return path
        except OSError:
            continue
    return None


def detect_media_mode(disc_path: Optional[str]) -> Dict[str, object]:
    """Detect the `GVK_VAULT` media mode and bundled offline assets.

    Determines one of three modes:
      - `'USB'`: `vault/media_mode.json` sets `mode=USB`, or offline assets
        (`base/*.gguf`, `bin/`, `embeddings/`, or `python_portable/`) are present.
      - `'DVD'`: Lightweight DVD-R (~150 MB) relying on cloud downloads for base assets.
      - `'NONE'`: No physical medium is mounted.

    Args:
        disc_path: Root path to the mounted `GVK_VAULT` directory, or None.

    Returns:
        Dictionary describing `mode`, `disc_path`, `has_base_model`, `has_llama_bins`,
        `has_embeddings`, and `has_portable_python`.
    """
    if not disc_path or not os.path.isdir(disc_path):
        return {
            "mode": "NONE",
            "disc_path": None,
            "has_base_model": False,
            "has_llama_bins": False,
            "has_embeddings": False,
            "has_portable_python": False,
        }

    explicit_mode = None
    mode_json_path = os.path.join(disc_path, "vault", "media_mode.json")
    if os.path.isfile(mode_json_path):
        try:
            with open(mode_json_path, "r", encoding="utf-8") as f:
                mdata = json.load(f)
            raw_m = str(mdata.get("mode") or "").strip().upper()
            if raw_m in ("USB", "DVD"):
                explicit_mode = raw_m
        except (OSError, json.JSONDecodeError):
            pass

    disc_base_dir = os.path.join(disc_path, "base")
    has_base = False
    if os.path.isdir(disc_base_dir):
        for gpath in glob.glob(os.path.join(disc_base_dir, "*.gguf")):
            if _is_valid_gguf_file(gpath):
                has_base = True
                break

    disc_bin_dir = os.path.join(disc_path, "bin")
    has_bins = False
    if os.path.isdir(disc_bin_dir):
        for entry in os.listdir(disc_bin_dir):
            el = entry.lower()
            if (
                el.endswith((".zip", ".tar.gz", ".tgz", ".exe"))
                or el == "llama-server"
                or os.path.isdir(os.path.join(disc_bin_dir, entry))
            ):
                has_bins = True
                break

    disc_emb_dir = os.path.join(disc_path, "embeddings")
    has_emb = os.path.isdir(disc_emb_dir) and any(True for _ in os.listdir(disc_emb_dir))

    disc_py_dir = os.path.join(disc_path, "python_portable")
    has_py = os.path.isdir(disc_py_dir) and any(
        f.endswith((".tar.gz", ".tgz", ".zip")) for f in os.listdir(disc_py_dir)
    )

    mode = explicit_mode or ("USB" if (has_base or has_bins or has_emb or has_py) else "DVD")
    return {
        "mode": mode,
        "disc_path": disc_path,
        "has_base_model": has_base,
        "has_llama_bins": has_bins,
        "has_embeddings": has_emb,
        "has_portable_python": has_py,
    }


def _copy_file_with_progress(src: str, dst: str, label: str = "Copying") -> str:
    """Stream-copy a large file in 8 MB chunks with terminal progress and atomic rename.

    Writes to `<dst>.part` first and atomically replaces `dst` on completion.

    Args:
        src: Source file path (e.g., a 9 GB GGUF model on USB media).
        dst: Destination file path (e.g., inside `~/.gvk_ai/base/`).
        label: Short label shown in the progress line.

    Returns:
        Destination file path `dst`.
    """
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    part_path = dst + ".part"
    total_size = os.path.getsize(src)
    copied = 0
    chunk_size = 8 * 1024 * 1024  # 8 MB
    t0 = time.time()

    if os.path.exists(dst):
        try:
            os.chmod(dst, 0o644)
        except OSError:
            pass

    with open(src, "rb") as f_in, open(part_path, "wb") as f_out:
        while True:
            buf = f_in.read(chunk_size)
            if not buf:
                break
            f_out.write(buf)
            copied += len(buf)
            elapsed = max(time.time() - t0, 1e-3)
            speed_mb = (copied / (1024 * 1024)) / elapsed
            if total_size > 0:
                pct = min(100.0, (copied / total_size) * 100.0)
                print(
                    f"\r [*] {label}: {pct:5.1f}% "
                    f"({_format_size(copied)} / {_format_size(total_size)}) [{speed_mb:.1f} MB/s]",
                    end="",
                    flush=True,
                )
            else:
                print(f"\r [*] {label}: {_format_size(copied)} [{speed_mb:.1f} MB/s]", end="", flush=True)

    print()
    try:
        os.chmod(part_path, 0o644)
    except OSError:
        pass
    os.replace(part_path, dst)
    return dst


def _is_valid_gguf_file(file_path: str) -> bool:
    """Return True if `file_path` exists and begins with the 4-byte `GGUF` magic header."""
    try:
        if not os.path.isfile(file_path) or os.path.getsize(file_path) < 4:
            return False
        with open(file_path, "rb") as f:
            return f.read(4) == GGUF_MAGIC
    except OSError:
        return False


def _format_size(num_bytes: int) -> str:
    """Format a byte count into a human-readable string (B, KB, MB, or GB)."""
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.1f} GB"


def _download_with_resume(
    url: str,
    dst_path: str,
    label: str = "Downloading",
    max_retries: int = 5,
) -> str:
    """Stream-download a file to `<dst_path>.part` with HTTP Range resume support.

    Automatically retries interrupted transfers up to `max_retries` times and
    verifies `Content-Length` completeness before atomically renaming to `dst_path`.

    Args:
        url: Source HTTP/HTTPS URL.
        dst_path: Destination file path.
        label: Short label shown in the progress line.
        max_retries: Maximum number of download attempts.

    Returns:
        Destination file path `dst_path`.

    Raises:
        IOError: If the downloaded byte count is smaller than `Content-Length`.
        Exception: Re-raises the last transport error if all retries fail.
    """
    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
    part_path = dst_path + ".part"
    ssl_ctx = _build_ssl_context()
    last_exc: Optional[Exception] = None

    for attempt in range(1, max(1, max_retries) + 1):
        existing_size = os.path.getsize(part_path) if os.path.isfile(part_path) else 0
        headers = {"User-Agent": "GVK-Private-AI-Updater/1.0"}
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"

        req = urllib.request.Request(url, headers=headers)
        try:
            try:
                resp = urllib.request.urlopen(req, timeout=30, context=ssl_ctx)
            except urllib.error.HTTPError as exc:
                # 416 Range Not Satisfiable: file may already be complete or server rejects Range
                if exc.code == 416 and existing_size > 0:
                    req = urllib.request.Request(url, headers={"User-Agent": "GVK-Private-AI-Updater/1.0"})
                    existing_size = 0
                    resp = urllib.request.urlopen(req, timeout=30, context=ssl_ctx)
                else:
                    raise

            status_code = getattr(resp, "status", 200)
            if existing_size > 0 and status_code == 206:
                mode = "ab"
                downloaded = existing_size
            else:
                mode = "wb"
                downloaded = 0

            content_len = resp.headers.get("Content-Length")
            total_size = (downloaded + int(content_len)) if content_len and content_len.isdigit() else None

            t0 = time.time()
            session_bytes = 0
            chunk_size = 4 * 1024 * 1024  # 4 MB

            with resp, open(part_path, mode) as f_out:
                while True:
                    buf = resp.read(chunk_size)
                    if not buf:
                        break
                    f_out.write(buf)
                    downloaded += len(buf)
                    session_bytes += len(buf)

                    elapsed = max(time.time() - t0, 1e-3)
                    speed_mb = (session_bytes / (1024 * 1024)) / elapsed
                    if total_size and total_size > 0:
                        pct = min(100.0, (downloaded / total_size) * 100.0)
                        print(
                            f"\r [*] {label}: {pct:5.1f}% "
                            f"({_format_size(downloaded)} / {_format_size(total_size)}) [{speed_mb:.1f} MB/s]",
                            end="",
                            flush=True,
                        )
                    else:
                        print(
                            f"\r [*] {label}: {_format_size(downloaded)} [{speed_mb:.1f} MB/s]",
                            end="",
                            flush=True,
                        )

            print()
            if total_size is not None and downloaded < total_size:
                raise IOError(
                    f"Incomplete download for {label}: received {downloaded} of {total_size} bytes."
                )

            os.replace(part_path, dst_path)
            return dst_path

        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                print(
                    f"\n [!] Download interrupted for {label} "
                    f"(attempt {attempt}/{max_retries}: {exc}). Resuming in 2s..."
                )
                time.sleep(2.0)
            else:
                raise

    if last_exc is not None:
        raise last_exc
    return dst_path


# =====================================================================
# 1. INITIALIZE ~/.gvk_ai CACHE
# =====================================================================
def get_gvk_paths(gvk_home: str = DEFAULT_GVK_HOME) -> Dict[str, str]:
    """Return a dictionary of standard `~/.gvk_ai` cache subdirectories without writing to disk.

    Args:
        gvk_home: Base path for the GVK local cache directory.

    Returns:
        Mapping of logical keys (`home`, `base`, `bin`, `embeddings`, `personas`,
        `registry`, `logs`) to absolute directory paths.
    """
    gvk_home = os.path.abspath(os.path.expanduser(gvk_home))
    return {
        "home": gvk_home,
        "base": os.path.join(gvk_home, "base"),
        "bin": os.path.join(gvk_home, "bin"),
        "embeddings": os.path.join(gvk_home, "embeddings"),
        "personas": os.path.join(gvk_home, "personas"),
        "registry": os.path.join(gvk_home, "registry"),
        "logs": os.path.join(gvk_home, "logs"),
    }


def init_gvk_home(gvk_home: str = DEFAULT_GVK_HOME) -> Dict[str, str]:
    """Create the `~/.gvk_ai` directory structure if it does not exist.

    Args:
        gvk_home: Base path for the GVK local cache directory.

    Returns:
        Mapping of standard GVK cache directory paths.
    """
    paths = get_gvk_paths(gvk_home)
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


# =====================================================================
# 2. PYTHON .VENV & REQUIREMENTS.TXT
# =====================================================================
def _venv_python_path(venv_dir: str) -> str:
    """Return the path to the Python interpreter inside `venv_dir`."""
    if platform.system().lower() == "windows":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    for name in ("python3", "python"):
        candidate = os.path.join(venv_dir, "bin", name)
        if os.path.isfile(candidate):
            return candidate
    return os.path.join(venv_dir, "bin", "python3")


def _portable_python_path(gvk_home: str = DEFAULT_GVK_HOME) -> str:
    """Return the expected path to the bundled portable Python binary under `gvk_home`."""
    home_abs = os.path.abspath(os.path.expanduser(gvk_home))
    if platform.system().lower() == "windows":
        return os.path.join(home_abs, "python", "python.exe")
    return os.path.join(home_abs, "python", "bin", "python3")


def _tcl_tk_env_for_python(py_bin: str) -> Dict[str, str]:
    """Build an environment dict with `TCL_LIBRARY`/`TK_LIBRARY` for portable Python builds."""
    env = dict(os.environ)
    if "TCL_LIBRARY" in env and "TK_LIBRARY" in env:
        return env
    real_bin = os.path.realpath(py_bin)
    py_dir = os.path.dirname(real_bin)
    portable_root = os.path.join(os.path.abspath(os.path.expanduser(DEFAULT_GVK_HOME)), "python")
    roots = [py_dir, os.path.dirname(py_dir), portable_root]
    for root in roots:
        for base_sub in ("tcl", "lib"):
            tcl_dir = os.path.join(root, base_sub, "tcl8.6")
            tk_dir = os.path.join(root, base_sub, "tk8.6")
            if os.path.isdir(tcl_dir) and "TCL_LIBRARY" not in env:
                env["TCL_LIBRARY"] = tcl_dir
            if os.path.isdir(tk_dir) and "TK_LIBRARY" not in env:
                env["TK_LIBRARY"] = tk_dir
    return env


def _probe_python_tk(py_bin: str) -> Tuple[bool, float, Tuple[int, int]]:
    """Probe a Python executable for its version and modern Tkinter (`Tk >= 8.6`) support.

    On macOS, the legacy system Tk 8.5 cannot render Cocoa widgets properly, so
    Tk >= 8.6 is required.

    Args:
        py_bin: Path to the Python interpreter to inspect.

    Returns:
        Tuple `(has_tk, tk_version, (major, minor))`.
    """
    if not py_bin or not os.path.isfile(py_bin):
        return False, 0.0, (0, 0)
    code = (
        "import sys; "
        "print(f'PY:{sys.version_info[0]}.{sys.version_info[1]}'); "
        "import tkinter as tk; "
        "print(f'TK:{float(tk.TkVersion)}')"
    )
    try:
        res = subprocess.run(
            [py_bin, "-c", code],
            capture_output=True,
            text=True,
            timeout=5,
            env=_tcl_tk_env_for_python(py_bin),
        )
    except Exception:
        return False, 0.0, (0, 0)

    py_ver = (0, 0)
    tk_ver = 0.0
    has_tk = False
    for line in (res.stdout or "").splitlines():
        line = line.strip()
        if line.startswith("PY:"):
            parts = line[3:].split(".")
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                py_ver = (int(parts[0]), int(parts[1]))
        elif line.startswith("TK:"):
            try:
                tk_ver = float(line[3:])
                has_tk = True
            except ValueError:
                pass
    return has_tk, tk_ver, py_ver


def _select_best_python(gvk_home: str = DEFAULT_GVK_HOME) -> str:
    """Select the best available Python interpreter for creating `.venv`.

    Prefers portable `~/.gvk_ai/python` or a system Python >= 3.10 with `TkVersion >= 8.6`.
    On macOS, attempts to install `python-tk@<major.minor>` via Homebrew if needed.

    Args:
        gvk_home: Base path for the GVK local cache directory.

    Returns:
        Path to the selected Python executable.
    """
    candidates: List[str] = []
    portable_py = _portable_python_path(gvk_home)
    if os.path.isfile(portable_py):
        candidates.append(portable_py)
    if platform.system().lower() == "darwin":
        for prefix in ("/opt/homebrew/bin", "/usr/local/bin"):
            for ver in ("3.12", "3.11", "3.13", "3.10", "3"):
                candidates.append(os.path.join(prefix, f"python{ver}"))
    for cmd_name in ("python3.12", "python3.11", "python3.13", "python3.10", "python3", "python"):
        found = shutil.which(cmd_name)
        if found:
            candidates.append(found)
    candidates.append(sys.executable)

    seen = set()
    probed: List[Tuple[str, bool, float, Tuple[int, int]]] = []
    for cand in candidates:
        if not cand or cand in seen or not os.path.isfile(cand):
            continue
        seen.add(cand)
        has_tk, tk_ver, py_ver = _probe_python_tk(cand)
        if has_tk and tk_ver >= MIN_TK_VERSION and py_ver >= MIN_PYTHON_VERSION:
            return cand
        probed.append((cand, has_tk, tk_ver, py_ver))

    # On macOS, if Homebrew Python >= 3.10 lacks Tkinter, attempt to install python-tk@X.Y
    if platform.system().lower() == "darwin":
        brew_bin = shutil.which("brew") or (
            "/opt/homebrew/bin/brew" if os.path.isfile("/opt/homebrew/bin/brew") else None
        )
        if brew_bin:
            for cand, has_tk, _tk_ver, py_ver in probed:
                if not has_tk and py_ver >= MIN_PYTHON_VERSION and (
                    "/opt/homebrew/" in os.path.realpath(cand) or "/usr/local/" in os.path.realpath(cand)
                ):
                    pkg = f"python-tk@{py_ver[0]}.{py_ver[1]}"
                    print(f" [*] Detected Python {py_ver[0]}.{py_ver[1]} without Tkinter; installing {pkg}...")
                    try:
                        env = dict(os.environ, HOMEBREW_NO_AUTO_UPDATE="1")
                        subprocess.run([brew_bin, "install", pkg], check=True, env=env, input="y\n", text=True)
                        ok2, tk2, _ = _probe_python_tk(cand)
                        if ok2 and tk2 >= MIN_TK_VERSION:
                            return cand
                    except Exception as exc:
                        print(f" [!] Failed to install {pkg} automatically: {exc}")

    for cand, has_tk, tk_ver, _py_ver in probed:
        if has_tk and tk_ver >= MIN_TK_VERSION:
            return cand
    return sys.executable


def resolve_venv_dir(gvk_home: str = DEFAULT_GVK_HOME, repo_root: str = DEFAULT_REPO_ROOT) -> str:
    """Return the path to the active or target `.venv` directory."""
    repo_venv = os.path.join(repo_root, ".venv")
    home_venv = os.path.join(gvk_home, ".venv")
    if os.path.isfile(_venv_python_path(repo_venv)):
        return repo_venv
    return home_venv


def check_requirements_status(venv_dir: str, req_path: str) -> Tuple[str, Optional[str], Optional[str]]:
    """Check virtual environment dependency status via `<venv>/.gvk_requirements.sha256`.

    Args:
        venv_dir: Path to the target `.venv` directory.
        req_path: Path to `requirements.txt`.

    Returns:
        Tuple `(status, current_sha256, saved_sha256)`, where `status` is one of:
          - `'missing_req'`: `requirements.txt` does not exist.
          - `'missing_venv'`: `.venv` or its Python binary does not exist.
          - `'needs_install'`: SHA-256 marker file is missing or unreadable.
          - `'outdated'`: `requirements.txt` hash differs from the saved marker.
          - `'up_to_date'`: Saved marker matches `requirements.txt`.
    """
    if not os.path.isfile(req_path):
        return "missing_req", None, None

    current_hash = compute_sha256(req_path)
    py_path = _venv_python_path(venv_dir)
    if not os.path.isfile(py_path):
        return "missing_venv", current_hash, None

    marker_path = os.path.join(venv_dir, ".gvk_requirements.sha256")
    if not os.path.isfile(marker_path):
        return "needs_install", current_hash, None

    try:
        with open(marker_path, "r", encoding="utf-8") as f:
            saved_hash = f.read().strip()
    except OSError:
        return "needs_install", current_hash, None

    if saved_hash != current_hash:
        return "outdated", current_hash, saved_hash
    return "up_to_date", current_hash, saved_hash


def ensure_venv_and_requirements(
    gvk_home: str = DEFAULT_GVK_HOME,
    repo_root: str = DEFAULT_REPO_ROOT,
    skip_pip: bool = False,
) -> Dict[str, Optional[str]]:
    """Ensure `.venv` (Python >= 3.10, Tk >= 8.6) exists and matches `requirements.txt`.

    Creates or recreates `.venv` when required and installs missing or updated
    Python packages unless `skip_pip` is enabled.

    Args:
        gvk_home: Base path for the GVK local cache directory.
        repo_root: Repository root directory containing `requirements.txt`.
        skip_pip: If True, skip venv creation/recreation and pip installation.

    Returns:
        Dictionary with keys `venv_dir`, `python`, and `status`.

    Raises:
        RuntimeError: If no Python >= 3.10 interpreter is available.
    """
    venv_dir = resolve_venv_dir(gvk_home, repo_root)
    req_path = os.path.join(repo_root, "requirements.txt")
    py_path = _venv_python_path(venv_dir)
    marker_path = os.path.join(venv_dir, ".gvk_requirements.sha256")

    if os.path.isfile(py_path):
        has_tk, tk_ver, py_ver = _probe_python_tk(py_path)
        if not skip_pip and (py_ver < MIN_PYTHON_VERSION or not has_tk or tk_ver < MIN_TK_VERSION):
            print(
                f" [!] Outdated Python or Tk detected in .venv (Python={py_ver[0]}.{py_ver[1]}, "
                f"Tk={tk_ver})."
            )
            best_py = _select_best_python(gvk_home)
            best_has_tk, best_tk, best_py_ver = _probe_python_tk(best_py)
            if best_py_ver >= MIN_PYTHON_VERSION and best_has_tk and best_tk >= MIN_TK_VERSION:
                print(
                    f" [*] Recreating {venv_dir} with {best_py} "
                    f"(Python {best_py_ver[0]}.{best_py_ver[1]}, Tk {best_tk})..."
                )
                subprocess.run([best_py, "-m", "venv", "--clear", venv_dir], check=True)
                if os.path.isfile(marker_path):
                    os.remove(marker_path)
                py_path = _venv_python_path(venv_dir)
            elif py_ver < MIN_PYTHON_VERSION:
                raise RuntimeError(
                    "Python >= 3.10 is required (current < 3.10). "
                    "macOS: brew install python@3.12 python-tk@3.12 | "
                    "Windows: python.org (Python 3.12) | Linux: sudo apt install python3.12 python3.12-venv python3-tk"
                )
            else:
                print(
                    " [!] WARNING: No Python interpreter with Tk >= 8.6 found. "
                    "On macOS run: brew install python@3.12 python-tk@3.12"
                )

    if not os.path.isfile(py_path):
        if skip_pip:
            return {
                "venv_dir": venv_dir,
                "python": None,
                "status": "missing_venv_skipped",
            }
        best_py = _select_best_python(gvk_home)
        _best_tk_ok, _best_tk_ver, best_py_ver = _probe_python_tk(best_py)
        if best_py_ver < MIN_PYTHON_VERSION:
            raise RuntimeError(
                f"Found only Python {best_py_ver[0]}.{best_py_ver[1]} ({best_py}), but Python >= 3.10 is required. "
                "macOS: brew install python@3.12 python-tk@3.12 | "
                "Windows: install Python 3.12 from python.org (Add to PATH) | "
                "Linux: sudo apt install python3.12 python3.12-venv python3-tk"
            )
        print(f" [*] Creating Python virtual environment at {venv_dir} ({best_py})...")
        subprocess.run([best_py, "-m", "venv", venv_dir], check=True)
        py_path = _venv_python_path(venv_dir)

    status, current_hash, _ = check_requirements_status(venv_dir, req_path)
    if status == "missing_req":
        return {"venv_dir": venv_dir, "python": py_path, "status": "missing_req"}
    if status == "up_to_date" and not skip_pip:
        imp_res = subprocess.run(
            [py_path, "-c", "import cryptography, argon2, certifi, numpy, faiss, sentence_transformers"],
            capture_output=True,
            text=True,
        )
        if imp_res.returncode != 0:
            status = "needs_install"
    if status == "up_to_date":
        print(f" [OK] requirements.txt dependencies are up to date ({venv_dir}).")
        return {"venv_dir": venv_dir, "python": py_path, "status": "up_to_date"}

    if skip_pip:
        print(f" [!] Skipping pip check (--skip-pip), status: {status}.")
        return {"venv_dir": venv_dir, "python": py_path, "status": f"{status}_skipped"}

    pip_check = subprocess.run([py_path, "-m", "pip", "--version"], capture_output=True, text=True)
    if pip_check.returncode != 0:
        subprocess.run([py_path, "-m", "ensurepip", "--upgrade"], check=False)

    print(f" [*] Updating Python dependencies ({status}) from {req_path}...")
    subprocess.run([py_path, "-m", "pip", "install", "--disable-pip-version-check", "-r", req_path], check=True)
    if current_hash:
        with open(marker_path, "w", encoding="utf-8") as f:
            f.write(current_hash + "\n")
    print(" [OK] requirements.txt dependencies installed successfully.")
    return {"venv_dir": venv_dir, "python": py_path, "status": "installed"}


# =====================================================================
# 3. GVK_VAULT MEDIA VERIFICATION & IMPORT
# =====================================================================
def _load_activated_registry(registry_dir: str) -> Dict[str, dict]:
    """Load the local activated-personas registry JSON (`activated.json`)."""
    reg_file = os.path.join(registry_dir, "activated.json")
    if not os.path.isfile(reg_file):
        return {}
    try:
        with open(reg_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_activated_registry(registry_dir: str, registry_data: Dict[str, dict]) -> None:
    """Persist the local activated-personas registry JSON (`activated.json`)."""
    reg_file = os.path.join(registry_dir, "activated.json")
    with open(reg_file, "w", encoding="utf-8") as f:
        json.dump(registry_data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _validate_manifest_and_files(
    source_dir: str,
    master_key: Optional[bytes] = None,
) -> dict:
    """Read `manifest.json`, verify SHA-256 digests of `.enc` containers, and check `GVK1` headers.

    Args:
        source_dir: Directory containing `manifest.json` and encrypted container files.
        master_key: Optional 32-byte master key for `GVK1` header authentication.

    Returns:
        Parsed manifest dictionary.

    Raises:
        FileNotFoundError: If `manifest.json` or a referenced container is missing.
        ValueError: If manifest fields are incomplete or integrity checks fail.
    """
    manifest_path = os.path.join(source_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Missing manifest.json in {source_dir}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    persona_id = manifest.get("persona_id")
    files_map = manifest.get("files") or {}
    sha_map = manifest.get("sha256") or {}
    if not persona_id or "adapter" not in files_map or "payload" not in files_map:
        raise ValueError(f"Incomplete manifest.json in {source_dir}")

    for key_name in ("adapter", "payload"):
        rel_name = files_map[key_name]
        fpath = os.path.join(source_dir, rel_name)
        if not os.path.isfile(fpath):
            raise FileNotFoundError(f"Missing container {rel_name} in {source_dir}")

        expected_sha = sha_map.get(key_name)
        if not expected_sha:
            raise ValueError(f"Missing SHA-256 entry for '{key_name}' in manifest.json")
        actual_sha = compute_sha256(fpath)
        if not hmac.compare_digest(actual_sha.lower(), str(expected_sha).lower()):
            raise ValueError(
                f"SHA-256 integrity mismatch for {rel_name}: "
                f"expected {expected_sha[:16]}..., got {actual_sha[:16]}..."
            )
        verify_gvk1_header(fpath, master_key=master_key)

    return manifest


def verify_and_import_disc(
    disc_path: Optional[str] = None,
    gvk_home: str = DEFAULT_GVK_HOME,
) -> Dict[str, Optional[object]]:
    """Detect `GVK_VAULT` media (DVD or USB), verify signatures/containers, and import into cache.

    Validates `vault/master.key`, `persona/origin.sig`, manifest SHA-256 hashes,
    and `GVK1` headers, registers the signature in `~/.gvk_ai/registry/`, and
    copies `.enc` containers onto local storage if not already cached.

    Args:
        disc_path: Optional explicit path to mounted `GVK_VAULT` media.
        gvk_home: Base path for the GVK local cache directory.

    Returns:
        Dictionary containing `found`, `imported`, `persona_id`, `disc_path`,
        `master_key`, `media_mode`, and `media_info`.

    Raises:
        FileNotFoundError: If required persona files or `origin.sig` are missing.
        ValueError: If `master.key`, `origin.sig`, SHA-256, or `GVK1` verification fails.
    """
    paths = init_gvk_home(gvk_home)
    detected_disc = find_gvk_disc_path(disc_path)
    if not detected_disc:
        return {
            "found": False,
            "imported": False,
            "persona_id": None,
            "disc_path": None,
            "master_key": None,
            "media_mode": "NONE",
            "media_info": detect_media_mode(None),
        }

    media_info = detect_media_mode(detected_disc)
    media_mode = str(media_info.get("mode") or "DVD")
    print(
        f" [OK] Detected GVK_VAULT media: {detected_disc} "
        f"[Mode: {media_mode} | Base model on media: {'YES' if media_info.get('has_base_model') else 'NO'} | "
        f"Binaries: {'YES' if media_info.get('has_llama_bins') else 'NO'}]"
    )

    key_path = os.path.join(detected_disc, "vault", "master.key")
    with open(key_path, "rb") as f:
        master_key = f.read()
    if len(master_key) != KEY_SIZE:
        raise ValueError(f"Invalid vault/master.key size: {len(master_key)} bytes (expected {KEY_SIZE} bytes).")

    disc_persona_dir = os.path.join(detected_disc, "persona")
    if not os.path.isdir(disc_persona_dir):
        print(f" [OK] GVK_VAULT ({media_mode}) detected ({detected_disc}), no persona/ directory - key-only mode.")
        return {
            "found": True,
            "imported": False,
            "persona_id": None,
            "disc_path": detected_disc,
            "master_key": master_key,
            "media_mode": media_mode,
            "media_info": media_info,
        }

    manifest = _validate_manifest_and_files(disc_persona_dir, master_key=master_key)
    persona_id = str(manifest["persona_id"]).strip()

    sig_path = os.path.join(disc_persona_dir, "origin.sig")
    if not os.path.isfile(sig_path):
        raise FileNotFoundError(f"Missing persona/origin.sig signature on media {detected_disc}!")
    with open(sig_path, "rb") as f:
        sig_bytes = f.read()

    if not verify_origin_sig_bytes(persona_id, master_key, sig_bytes):
        raise ValueError(
            f"Cryptographic error: origin.sig on media {detected_disc} "
            f"does not match persona_id='{persona_id}' or master.key!"
        )

    # Save origin.sig to the local signature registry
    reg_sig_path = os.path.join(paths["registry"], f"{persona_id}.sig")
    if os.path.exists(reg_sig_path):
        try:
            os.chmod(reg_sig_path, 0o644)
        except OSError:
            pass
    with open(reg_sig_path, "wb") as f:
        f.write(sig_bytes)

    # Skip re-copying 100+ MB from media if this persona is already valid in local SSD cache
    # (also prevents downgrading newer containers previously synced from GitHub).
    target_persona_dir = os.path.join(paths["personas"], persona_id)
    local_manifest_path = os.path.join(target_persona_dir, "manifest.json")
    already_cached = False
    cached_ver = str(manifest.get("version", "1.0.0"))
    if os.path.isfile(local_manifest_path):
        try:
            with open(local_manifest_path, "r", encoding="utf-8") as lm_f:
                local_manifest = json.load(lm_f)
            local_files = local_manifest.get("files") or {}
            if (
                str(local_manifest.get("persona_id") or "").strip() == persona_id
                and all(
                    os.path.isfile(os.path.join(target_persona_dir, local_files[k]))
                    for k in ("adapter", "payload")
                    if k in local_files
                )
            ):
                already_cached = True
                cached_ver = str(local_manifest.get("version") or cached_ver)
        except (OSError, json.JSONDecodeError):
            already_cached = False

    if already_cached:
        print(
            f" [OK] Persona '{persona_id}' is already activated in cache (v{cached_ver}) - "
            f"skipping re-copy from {media_mode}."
        )
        return {
            "found": True,
            "imported": False,
            "persona_id": persona_id,
            "disc_path": detected_disc,
            "master_key": master_key,
            "media_mode": media_mode,
            "media_info": media_info,
        }

    # Copy .enc containers and manifest.json to SSD cache (without origin.sig, master.key, or DVD read-only flags)
    os.makedirs(target_persona_dir, exist_ok=True)
    files_map = manifest["files"]
    for fname in ("manifest.json", files_map["adapter"], files_map["payload"]):
        _copy_writable(os.path.join(disc_persona_dir, fname), os.path.join(target_persona_dir, fname))

    registry_data = _load_activated_registry(paths["registry"])
    registry_data[persona_id] = {
        "persona_id": persona_id,
        "version": manifest.get("version", "1.0.0"),
        "origin_sig_sha256": hashlib.sha256(sig_bytes).hexdigest(),
        "adapter_sha256": manifest["sha256"]["adapter"],
        "payload_sha256": manifest["sha256"]["payload"],
        "updated_at": int(time.time()),
        "source": f"{media_mode.lower()}:{detected_disc}",
    }
    _save_activated_registry(paths["registry"], registry_data)

    print(
        f" [OK] Activated and imported '{persona_id}' (v{manifest.get('version', '1.0.0')}) "
        f"from {media_mode} -> {target_persona_dir}"
    )
    return {
        "found": True,
        "imported": True,
        "persona_id": persona_id,
        "disc_path": detected_disc,
        "master_key": master_key,
        "media_mode": media_mode,
        "media_info": media_info,
    }


# =====================================================================
# 4. INCREMENTAL UNLOCK & GITHUB PERSONA SYNC
# =====================================================================
def sync_activated_personas(
    gvk_home: str = DEFAULT_GVK_HOME,
    repo_root: str = DEFAULT_REPO_ROOT,
    master_key: Optional[bytes] = None,
) -> Dict[str, List[str]]:
    """Sync updated containers from `personas_encrypted/` only for already-activated personas.

    A persona is eligible for synchronization only when a valid `origin.sig` exists in
    `~/.gvk_ai/registry/<persona_id>.sig`.

    Args:
        gvk_home: Base path for the GVK local cache directory.
        repo_root: Repository root containing `personas_encrypted/`.
        master_key: Optional 32-byte master key used to authenticate signatures and `GVK1` headers.

    Returns:
        Dictionary with lists of persona IDs under `updated`, `up_to_date`,
        and `skipped_unactivated`.
    """
    paths = init_gvk_home(gvk_home)
    enc_root = os.path.join(repo_root, "personas_encrypted")
    result: Dict[str, List[str]] = {
        "updated": [],
        "up_to_date": [],
        "skipped_unactivated": [],
    }
    if not os.path.isdir(enc_root):
        return result

    registry_data = _load_activated_registry(paths["registry"])
    registry_changed = False

    for entry in sorted(os.listdir(enc_root)):
        src_dir = os.path.join(enc_root, entry)
        manifest_src = os.path.join(src_dir, "manifest.json")
        if not os.path.isdir(src_dir) or not os.path.isfile(manifest_src):
            continue

        with open(manifest_src, "r", encoding="utf-8") as f:
            remote_manifest = json.load(f)
        persona_id = str(remote_manifest.get("persona_id") or entry).strip()

        reg_sig_path = os.path.join(paths["registry"], f"{persona_id}.sig")
        if not os.path.isfile(reg_sig_path) or os.path.getsize(reg_sig_path) != SIG_SIZE:
            print(f" [-] Skipped '{persona_id}': physical GVK_VAULT media with origin.sig has not been activated.")
            result["skipped_unactivated"].append(persona_id)
            continue

        if master_key is not None:
            with open(reg_sig_path, "rb") as f:
                sig_bytes = f.read()
            if not verify_origin_sig_bytes(persona_id, master_key, sig_bytes):
                print(
                    f" [!] Warning: local origin.sig for '{persona_id}' failed verification "
                    f"against current master.key - update blocked."
                )
                result["skipped_unactivated"].append(persona_id)
                continue

        dst_dir = os.path.join(paths["personas"], persona_id)
        local_manifest_path = os.path.join(dst_dir, "manifest.json")
        needs_update = True

        if os.path.isfile(local_manifest_path):
            try:
                with open(local_manifest_path, "r", encoding="utf-8") as f:
                    local_manifest = json.load(f)
                same_ver = local_manifest.get("version") == remote_manifest.get("version")
                same_sha = local_manifest.get("sha256") == remote_manifest.get("sha256")
                files_map = remote_manifest.get("files") or {}
                files_present = all(
                    os.path.isfile(os.path.join(dst_dir, files_map[k]))
                    for k in ("adapter", "payload")
                    if k in files_map
                )
                if same_ver and same_sha and files_present:
                    needs_update = False
            except (OSError, json.JSONDecodeError):
                needs_update = True

        if not needs_update:
            result["up_to_date"].append(persona_id)
            continue

        verified_manifest = _validate_manifest_and_files(src_dir, master_key=master_key)
        os.makedirs(dst_dir, exist_ok=True)
        files_map = verified_manifest["files"]
        for fname in ("manifest.json", files_map["adapter"], files_map["payload"]):
            _copy_writable(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))

        entry_meta = registry_data.get(persona_id, {"persona_id": persona_id})
        entry_meta.update(
            {
                "version": verified_manifest.get("version", "1.0.0"),
                "adapter_sha256": verified_manifest["sha256"]["adapter"],
                "payload_sha256": verified_manifest["sha256"]["payload"],
                "updated_at": int(time.time()),
                "source": "github_sync",
            }
        )
        registry_data[persona_id] = entry_meta
        registry_changed = True
        result["updated"].append(persona_id)
        print(f" [OK] Updated '{persona_id}' -> v{verified_manifest.get('version', '1.0.0')}")

    if registry_changed:
        _save_activated_registry(paths["registry"], registry_data)
    return result


# =====================================================================
# 5. BASE 14B MODEL (LOCAL CACHE, USB MEDIA, OR HUGGING FACE)
# =====================================================================
def _collect_base_model_candidates(
    gvk_home: str,
    repo_root: str,
    disc_path: Optional[str] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """Return candidate paths for the base GGUF model across cache, USB media, and repo/env.

    Args:
        gvk_home: Base path for the GVK local cache directory.
        repo_root: Repository root directory.
        disc_path: Optional mounted `GVK_VAULT` media path.

    Returns:
        Three-tuple `(cache_candidates, disc_candidates, local_candidates)`.
    """
    paths = get_gvk_paths(gvk_home)
    cache_candidates = [os.path.join(paths["base"], DEFAULT_BASE_MODEL_FILENAME)]
    if os.path.isdir(paths["base"]):
        cache_candidates.extend(sorted(glob.glob(os.path.join(paths["base"], "*.gguf"))))

    disc_candidates: List[str] = []
    if disc_path and os.path.isdir(disc_path):
        disc_base = os.path.join(disc_path, "base")
        disc_candidates.append(os.path.join(disc_base, DEFAULT_BASE_MODEL_FILENAME))
        if os.path.isdir(disc_base):
            disc_candidates.extend(sorted(glob.glob(os.path.join(disc_base, "*.gguf"))))

    local_candidates: List[str] = []
    env_model = os.environ.get("GVK_BASE_MODEL")
    if env_model:
        local_candidates.append(os.path.abspath(os.path.expanduser(env_model)))
    local_candidates.append(os.path.join(repo_root, "base", DEFAULT_BASE_MODEL_FILENAME))
    repo_base_dir = os.path.join(repo_root, "base")
    if os.path.isdir(repo_base_dir):
        local_candidates.extend(sorted(glob.glob(os.path.join(repo_base_dir, "*.gguf"))))
    local_candidates.append(os.path.join(repo_root, "base_model.gguf"))
    return cache_candidates, disc_candidates, local_candidates


def find_base_model_path(
    gvk_home: str = DEFAULT_GVK_HOME,
    repo_root: str = DEFAULT_REPO_ROOT,
    disc_path: Optional[str] = None,
) -> Optional[str]:
    """Perform a read-only search for a valid base GGUF model without creating files or symlinks.

    Args:
        gvk_home: Base path for the GVK local cache directory.
        repo_root: Repository root directory.
        disc_path: Optional mounted `GVK_VAULT` media path.

    Returns:
        Resolved path to the first valid GGUF file found, or None.
    """
    cache_candidates, disc_candidates, local_candidates = _collect_base_model_candidates(
        gvk_home, repo_root, disc_path=disc_path
    )
    for cand in cache_candidates + disc_candidates + local_candidates:
        if cand and _is_valid_gguf_file(cand):
            return os.path.realpath(cand) if os.path.islink(cand) else cand
    return None


def ensure_base_model(
    gvk_home: str = DEFAULT_GVK_HOME,
    repo_root: str = DEFAULT_REPO_ROOT,
    download: bool = True,
    model_url: str = DEFAULT_BASE_MODEL_URL,
    disc_path: Optional[str] = None,
) -> Optional[str]:
    """Ensure the public base model (`Qwen2.5-14B-Instruct-Q4_K_M.gguf`) is available locally.

    Resolution order:
      1. Reuse an existing valid GGUF already cached in `~/.gvk_ai/base/`.
      2. Copy from USB media (`<disc_path>/base/*.gguf`) into `~/.gvk_ai/base/`
         with progress output so `llama-server` reads from fast local storage.
      3. Symlink (or return) a local copy found via `GVK_BASE_MODEL` or `<repo_root>/base/`.
      4. Download from Hugging Face with HTTP Range resume support (when `download=True`).

    Args:
        gvk_home: Base path for the GVK local cache directory.
        repo_root: Repository root directory.
        download: Allow downloading from `model_url` if not found locally or on media.
        model_url: Download URL for the base GGUF model.
        disc_path: Optional mounted `GVK_VAULT` media path.

    Returns:
        Path to the ready base GGUF model, or None if unavailable and `download=False`.

    Raises:
        ValueError: If a copied or downloaded file fails `GGUF` header validation.
    """
    paths = init_gvk_home(gvk_home)
    target_path = os.path.join(paths["base"], DEFAULT_BASE_MODEL_FILENAME)
    cache_candidates, disc_candidates, local_candidates = _collect_base_model_candidates(
        gvk_home, repo_root, disc_path=disc_path
    )

    for existing in cache_candidates:
        if existing and _is_valid_gguf_file(existing):
            print(f" [OK] Base model ready in cache: {existing} ({_format_size(os.path.getsize(existing))})")
            return existing

    # USB mode: copy the base model from removable media into ~/.gvk_ai/base/
    for usb_cand in disc_candidates:
        if usb_cand and _is_valid_gguf_file(usb_cand):
            src_size = os.path.getsize(usb_cand)
            print(
                f" [*] [USB-MODE] Found base model on media: {usb_cand} ({_format_size(src_size)}).\n"
                f"     Copying to local SSD cache ({target_path})..."
            )
            _copy_file_with_progress(
                usb_cand,
                target_path,
                label=f"Copying {os.path.basename(usb_cand)} from USB",
            )
            if not _is_valid_gguf_file(target_path):
                raise ValueError(f"Failed to copy valid GGUF from USB to {target_path}!")
            print(f" [OK] Base model copied from USB to cache: {target_path}")
            return target_path

    for cand in local_candidates:
        if cand and _is_valid_gguf_file(cand):
            real_src = os.path.realpath(cand)
            try:
                if os.path.lexists(target_path):
                    os.remove(target_path)
                os.symlink(real_src, target_path)
                print(f" [OK] Linked local base model: {target_path} -> {real_src}")
                return target_path
            except OSError:
                print(f" [OK] Found local base model: {real_src}")
                return real_src

    if not download:
        print(" [!] Base model not found locally or on media (download disabled via --no-download).")
        return None

    print(f" [*] [DVD/CLOUD-MODE] Downloading 14B base model (~9.0 GB) from Hugging Face...")
    print(f"     URL: {model_url}")
    _download_with_resume(model_url, target_path, label=DEFAULT_BASE_MODEL_FILENAME)
    if not _is_valid_gguf_file(target_path):
        os.remove(target_path)
        raise ValueError("Downloaded base model is not a valid GGUF file (missing GGUF magic header)!")
    print(f" [OK] Base model saved to: {target_path}")
    return target_path


# =====================================================================
# 6. LLAMA-SERVER BINARY & EMBEDDING MODEL
# =====================================================================
def _llama_server_names() -> Tuple[str, ...]:
    """Return platform-specific executable filenames for `llama-server`."""
    if platform.system().lower() == "windows":
        return ("llama-server.exe",)
    return ("llama-server",)


def find_llama_server(
    gvk_home: str = DEFAULT_GVK_HOME,
    repo_root: str = DEFAULT_REPO_ROOT,
    disc_path: Optional[str] = None,
) -> Optional[str]:
    """Search for an existing `llama-server` executable in cache, repo, media, or `PATH`.

    Args:
        gvk_home: Base path for the GVK local cache directory.
        repo_root: Repository root directory.
        disc_path: Optional mounted `GVK_VAULT` media path.

    Returns:
        Path to the `llama-server` binary if found, otherwise None.
    """
    names = _llama_server_names()
    bin_dir = os.path.join(os.path.abspath(os.path.expanduser(gvk_home)), "bin")
    for name in names:
        cand = os.path.join(bin_dir, name)
        if os.path.isfile(cand):
            return cand

    for base_root in ([repo_root] + ([disc_path] if disc_path else [])):
        runtime_dir = os.path.join(base_root, "base", "llama-pascal-runtime")
        if os.path.isdir(runtime_dir):
            for root, _, files in os.walk(runtime_dir):
                for name in names:
                    if name in files:
                        return os.path.join(root, name)
        direct_bin_dir = os.path.join(base_root, "bin")
        if os.path.isdir(direct_bin_dir):
            for name in names:
                cand = os.path.join(direct_bin_dir, name)
                if os.path.isfile(cand):
                    return cand

    for name in names:
        found = shutil.which(name)
        if found:
            return found
    return None


def _has_windows_vulkan() -> bool:
    """Check whether `vulkan-1.dll` is present on Windows before selecting a Vulkan build."""
    if os.name != "nt":
        return True
    sys_root = os.environ.get("SystemRoot", r"C:\Windows")
    return os.path.isfile(os.path.join(sys_root, "System32", "vulkan-1.dll"))


def _has_nvidia_gpu() -> bool:
    """Check for an NVIDIA CUDA driver (`nvcuda.dll` on Windows or `nvidia-smi` in `PATH`)."""
    if os.name == "nt":
        sys_root = os.environ.get("SystemRoot", r"C:\Windows")
        if os.path.isfile(os.path.join(sys_root, "System32", "nvcuda.dll")):
            return True
    return shutil.which("nvidia-smi") is not None


def _select_llama_release_asset(assets: List[dict], prefer_cuda: bool = False) -> Optional[dict]:
    """Select the highest-scoring `llama.cpp` release archive for the current OS and CPU/GPU.

    Args:
        assets: List of GitHub release asset dicts (each containing at least `name`).
        prefer_cuda: If True on Windows with an NVIDIA GPU, prefer CUDA 12 builds over Vulkan/CPU.

    Returns:
        Best matching asset dictionary, or None if no compatible archive was found.
    """
    sys_name = platform.system().lower()
    arch = platform.machine().lower()
    is_arm = arch in ("arm64", "aarch64")
    has_win_vulkan = _has_windows_vulkan() if sys_name == "windows" else False
    has_nv = (_has_nvidia_gpu() or prefer_cuda) if sys_name == "windows" else False

    def matches(name: str) -> int:
        n = name.lower()
        if not (n.endswith(".zip") or n.endswith(".tar.gz") or n.endswith(".tgz")):
            return -1
        if "llama-" not in n or n.startswith("cudart-llama-"):
            return -1
        if sys_name == "darwin":
            if "macos" not in n and "darwin" not in n:
                return -1
            if is_arm and "arm64" in n:
                return 100
            if not is_arm and ("x64" in n or "x86_64" in n):
                return 100
            return -1
        if sys_name == "linux":
            if "ubuntu" not in n and "linux" not in n:
                return -1
            if is_arm and ("arm64" in n or "aarch64" in n):
                return 90
            if not is_arm and ("x64" in n or "x86_64" in n):
                return 100 if "vulkan" not in n and "sycl" not in n and "rocm" not in n and "cuda" not in n else 80
            return -1
        if sys_name == "windows":
            if "win" not in n:
                return -1
            if is_arm:
                if "arm64" in n and "vulkan" in n:
                    return 100
                if "arm64" in n and "cpu" in n:
                    return 95
                return -1
            if "arm64" in n:
                return -1
            # Prefer CUDA 12 when explicitly requested for offline bundles on NVIDIA machines
            if prefer_cuda and has_nv and "cuda-12" in n and "x64" in n:
                return 105
            if "vulkan" in n and "x64" in n:
                return 100 if has_win_vulkan else 75
            if "cpu-x64" in n:
                return 90 if has_win_vulkan else 100
            if has_nv and "cuda-12" in n and "x64" in n:
                return 88
            if "x64" in n and "cuda" not in n and "sycl" not in n and "rocm" not in n and "openvino" not in n:
                return 85
            if "cuda" in n and "x64" in n:
                return 70
            return -1
        return -1

    best_score = -1
    best_asset = None
    for asset in assets:
        score = matches(asset.get("name", ""))
        if score > best_score:
            best_score = score
            best_asset = asset
    return best_asset


def _select_cudart_asset(assets: List[dict], chosen_llama_name: str = "") -> Optional[dict]:
    """Find the companion `cudart-llama-bin-win-cuda-12.*.zip` archive for Windows CUDA builds."""
    want_12 = "cuda-12" in chosen_llama_name.lower() or not chosen_llama_name
    for asset in assets:
        n = asset.get("name", "").lower()
        if n.startswith("cudart-llama-") and "win" in n and "x64" in n and n.endswith(".zip"):
            if want_12 and "cuda-12" in n:
                return asset
    for asset in assets:
        n = asset.get("name", "").lower()
        if n.startswith("cudart-llama-") and "win" in n and "x64" in n and n.endswith(".zip"):
            return asset
    return None


def _extract_llama_archive(archive_path: str, bin_dir: str) -> Optional[str]:
    """Extract `llama-server` and companion shared libraries/shaders into `bin_dir`.

    Args:
        archive_path: Path to a `.zip`, `.tar.gz`, or `.tgz` release archive.
        bin_dir: Target directory for extracted executables and runtime libraries.

    Returns:
        Path to the extracted `llama-server` binary, or None if not found in the archive.
    """
    os.makedirs(bin_dir, exist_ok=True)
    wanted_bins = set(_llama_server_names())
    lib_exts = (".dylib", ".so", ".dll", ".metal", ".metallib")
    found_server = None

    with tempfile.TemporaryDirectory(prefix="gvk_llama_extract_") as tmp_ext:
        if archive_path.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(tmp_ext)
        else:
            with tarfile.open(archive_path, "r:*") as tf:
                if "filter" in tf.extractall.__code__.co_varnames:
                    tf.extractall(tmp_ext, filter="data")
                else:
                    tf.extractall(tmp_ext)

        for root, _, files in os.walk(tmp_ext):
            for fname in files:
                src = os.path.join(root, fname)
                if not os.path.isfile(src):
                    continue
                if fname in wanted_bins or any(ext in fname for ext in lib_exts):
                    dst = os.path.join(bin_dir, fname)
                    _copy_writable(src, dst)
                    if fname in wanted_bins:
                        os.chmod(dst, os.stat(dst).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                        found_server = dst
    return found_server


def _install_llama_from_directory(src_dir: str, dst_bin_dir: str) -> Optional[str]:
    """Copy an unpacked `llama-server` runtime (e.g., `llama-pascal-runtime`) into `~/.gvk_ai/bin/`."""
    wanted_bins = set(_llama_server_names())
    lib_exts = (".dylib", ".so", ".dll", ".metal", ".metallib")
    found_server = None
    os.makedirs(dst_bin_dir, exist_ok=True)
    for root, _, files in os.walk(src_dir):
        for fname in files:
            if fname in wanted_bins or any(fname.lower().endswith(ext) for ext in lib_exts):
                src = os.path.join(root, fname)
                dst = os.path.join(dst_bin_dir, fname)
                _copy_writable(src, dst)
                if fname in wanted_bins:
                    os.chmod(dst, os.stat(dst).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                    found_server = dst
    return found_server


def _try_install_llama_from_usb(disc_path: Optional[str], repo_root: str, dst_bin_dir: str) -> Optional[str]:
    """Attempt to install `llama-server` from offline bundles on USB media or `<repo_root>/base/llama_bins/`.

    Checks archives first (picking the best OS/GPU match and unpacking companion CUDA
    DLLs if present), then falls back to unpacked Windows runtime directories.

    Args:
        disc_path: Optional mounted `GVK_VAULT` media path.
        repo_root: Repository root directory.
        dst_bin_dir: Destination `~/.gvk_ai/bin/` directory.

    Returns:
        Path to the installed `llama-server` binary, or None if no offline bundle matched.
    """
    search_dirs: List[str] = []
    if disc_path and os.path.isdir(disc_path):
        search_dirs.append(os.path.join(disc_path, "bin"))
        search_dirs.append(os.path.join(disc_path, "base", "llama_bins"))
    search_dirs.append(os.path.join(repo_root, "base", "llama_bins"))

    sys_name = platform.system().lower()
    has_nv = _has_nvidia_gpu() if sys_name == "windows" else False

    for bdir in search_dirs:
        if not os.path.isdir(bdir):
            continue

        # 1. Check archive bundles (.zip / .tar.gz / .tgz) inside the bin directory
        local_assets: List[dict] = []
        for fname in sorted(os.listdir(bdir)):
            fpath = os.path.join(bdir, fname)
            if os.path.isfile(fpath) and fname.lower().endswith((".zip", ".tar.gz", ".tgz")):
                local_assets.append({"name": fname, "local_path": fpath})

        if local_assets:
            picked = _select_llama_release_asset(local_assets, prefer_cuda=has_nv)
            if picked and picked.get("local_path"):
                archive_path = str(picked["local_path"])
                print(
                    f" [*] [USB-MODE] Installing llama-server from offline archive: "
                    f"{os.path.basename(archive_path)}..."
                )
                # If a CUDA build was chosen and a matching cudart-llama-*.zip is present, unpack CUDA DLLs too
                if "cuda" in picked["name"].lower():
                    cudart = _select_cudart_asset(local_assets, picked["name"])
                    if cudart and cudart.get("local_path"):
                        print(f" [*] [USB-MODE] Extracting CUDA runtime DLLs: {cudart['name']}...")
                        _extract_llama_archive(str(cudart["local_path"]), dst_bin_dir)
                srv = _extract_llama_archive(archive_path, dst_bin_dir)
                if srv:
                    print(f" [OK] Installed llama-server from USB to: {srv}")
                    return srv

        # 2. On Windows, check for pre-extracted runtime directories (e.g., win-cuda-pascal or llama-pascal-runtime)
        if sys_name == "windows":
            for sub_name in ("win-cuda-pascal", "llama-pascal-runtime", "win-x64"):
                sub_dir = os.path.join(bdir, sub_name)
                if os.path.isdir(sub_dir) and os.path.isfile(os.path.join(sub_dir, "llama-server.exe")):
                    print(f" [*] [USB-MODE] Copying unpacked llama-server runtime from {sub_dir}...")
                    srv = _install_llama_from_directory(sub_dir, dst_bin_dir)
                    if srv:
                        print(f" [OK] Installed llama-server from USB to: {srv}")
                        return srv

    if sys_name == "windows" and disc_path:
        pascal_usb = os.path.join(disc_path, "base", "llama-pascal-runtime")
        if os.path.isdir(pascal_usb) and os.path.isfile(os.path.join(pascal_usb, "llama-server.exe")):
            print(f" [*] [USB-MODE] Copying llama-pascal-runtime from {pascal_usb}...")
            srv = _install_llama_from_directory(pascal_usb, dst_bin_dir)
            if srv:
                print(f" [OK] Installed llama-server from USB to: {srv}")
                return srv

    return None


def _fetch_latest_llama_release_with_assets() -> dict:
    """Fetch the latest `ggml-org/llama.cpp` GitHub release containing compatible binary assets.

    If `/releases/latest` points to a tag without prebuilt binaries (e.g., a marker tag),
    scans `/releases?per_page=10` to find the newest release containing matching archives.

    Returns:
        GitHub release JSON dictionary.
    """
    headers = {"User-Agent": "GVK-Private-AI-Updater/1.0", "Accept": "application/vnd.github+json"}
    ssl_ctx = _build_ssl_context()

    try:
        req = urllib.request.Request(LLAMA_CPP_RELEASES_API, headers=headers)
        with urllib.request.urlopen(req, timeout=30, context=ssl_ctx) as resp:
            release = json.loads(resp.read().decode("utf-8"))
        if _select_llama_release_asset(release.get("assets") or []) is not None:
            return release
    except Exception:
        release = {}

    req_list = urllib.request.Request(LLAMA_CPP_RELEASES_LIST_API, headers=headers)
    with urllib.request.urlopen(req_list, timeout=30, context=ssl_ctx) as resp:
        releases_list = json.loads(resp.read().decode("utf-8"))

    if isinstance(releases_list, list):
        for rel in releases_list:
            if isinstance(rel, dict) and _select_llama_release_asset(rel.get("assets") or []) is not None:
                return rel
        if releases_list and isinstance(releases_list[0], dict):
            return releases_list[0]
    return release


def ensure_llama_server(
    gvk_home: str = DEFAULT_GVK_HOME,
    repo_root: str = DEFAULT_REPO_ROOT,
    download: bool = True,
    disc_path: Optional[str] = None,
) -> Optional[str]:
    """Ensure `llama-server` is available via cache, USB media (`bin/`), or GitHub Releases.

    Args:
        gvk_home: Base path for the GVK local cache directory.
        repo_root: Repository root directory.
        download: Allow downloading from `ggml-org/llama.cpp` Releases if not found locally.
        disc_path: Optional mounted `GVK_VAULT` media path.

    Returns:
        Path to the `llama-server` executable, or None if unavailable and `download=False`.

    Raises:
        RuntimeError: If no compatible release asset is found or the archive lacks `llama-server`.
    """
    paths = init_gvk_home(gvk_home)
    cached_bin = os.path.join(paths["bin"], _llama_server_names()[0])
    if os.path.isfile(cached_bin):
        print(f" [OK] llama-server ready in cache: {cached_bin}")
        return cached_bin

    # Attempt offline installation from USB media (or repo base/llama_bins) into ~/.gvk_ai/bin/
    usb_installed = _try_install_llama_from_usb(disc_path, repo_root, paths["bin"])
    if usb_installed:
        return usb_installed

    existing = find_llama_server(paths["home"], repo_root, disc_path=disc_path)
    if existing:
        print(f" [OK] llama-server ready: {existing}")
        return existing

    if not download:
        print(" [!] llama-server not found locally (download disabled via --no-download).")
        return None

    print(" [*] Querying latest ggml-org/llama.cpp release on GitHub...")
    release = _fetch_latest_llama_release_with_assets()

    asset = _select_llama_release_asset(release.get("assets") or [])
    if not asset:
        raise RuntimeError(
            f"No compatible llama.cpp release build found for {platform.system()} / {platform.machine()}."
        )

    asset_name = asset["name"]
    asset_url = asset["browser_download_url"]
    with tempfile.TemporaryDirectory(prefix="gvk_llama_dl_") as tmp_dl:
        archive_path = os.path.join(tmp_dl, asset_name)
        _download_with_resume(asset_url, archive_path, label=asset_name)
        bin_path = _extract_llama_archive(archive_path, paths["bin"])
        if not bin_path:
            raise RuntimeError(f"llama-server executable not found inside downloaded archive {asset_name}!")
        print(f" [OK] Installed llama-server to: {bin_path}")
        return bin_path


def ensure_embedding_model(
    venv_python: Optional[str],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    download: bool = True,
    gvk_home: str = DEFAULT_GVK_HOME,
    disc_path: Optional[str] = None,
) -> bool:
    """Ensure the RAG `sentence-transformers` embedding model is cached for offline use.

    Copies `<disc_path>/embeddings/paraphrase-multilingual-MiniLM-L12-v2` from USB
    media when available, or pre-caches the model via `venv_python` so the GUI does
    not stall on cold start.

    Args:
        venv_python: Path to the virtual environment Python interpreter, or None.
        model_name: Hugging Face model identifier for `SentenceTransformer`.
        download: Allow downloading the embedding model when not cached locally.
        gvk_home: Base path for the GVK local cache directory.
        disc_path: Optional mounted `GVK_VAULT` media path.

    Returns:
        True if the embedding model is ready in local cache, False otherwise.
    """
    paths = init_gvk_home(gvk_home)
    local_emb_dst = os.path.join(paths["embeddings"], DEFAULT_EMBEDDING_SHORT_NAME)

    if os.path.isdir(local_emb_dst) and os.path.isfile(os.path.join(local_emb_dst, "config.json")):
        print(f" [OK] RAG embedding model ready in local cache: {local_emb_dst}")
        return True

    if disc_path and os.path.isdir(disc_path):
        usb_emb_src = os.path.join(disc_path, "embeddings", DEFAULT_EMBEDDING_SHORT_NAME)
        if os.path.isdir(usb_emb_src) and os.path.isfile(os.path.join(usb_emb_src, "config.json")):
            print(f" [*] [USB-MODE] Copying RAG embedding model from media: {usb_emb_src} -> {local_emb_dst}...")
            for root, _, files in os.walk(usb_emb_src):
                rel_dir = os.path.relpath(root, usb_emb_src)
                target_sub = local_emb_dst if rel_dir == "." else os.path.join(local_emb_dst, rel_dir)
                os.makedirs(target_sub, exist_ok=True)
                for fname in files:
                    _copy_writable(os.path.join(root, fname), os.path.join(target_sub, fname))
            print(f" [OK] RAG embedding model copied from USB to: {local_emb_dst}")
            return True

    if not venv_python or not os.path.isfile(venv_python):
        return False

    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
    )
    check_code = (
        "import sys; "
        "from sentence_transformers import SentenceTransformer; "
        "SentenceTransformer(sys.argv[1], device='cpu', local_files_only=True)"
    )
    try:
        res = subprocess.run(
            [venv_python, "-c", check_code, model_name],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if res.returncode == 0:
            print(f" [OK] RAG embedding model cached: {model_name}")
            return True
    except Exception as exc:
        print(f" [!] Failed to check RAG embedding model cache: {exc}")
        return False

    if not download:
        print(" [!] RAG embedding model not found in cache (download disabled via --no-download).")
        return False

    print(f" [*] Pre-caching RAG embedding model {model_name} (~470 MB)...")
    dl_code = (
        "import sys; "
        "from sentence_transformers import SentenceTransformer; "
        "m = SentenceTransformer(sys.argv[1], device='cpu'); "
        "m.save(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else None"
    )
    try:
        res_dl = subprocess.run(
            [venv_python, "-c", dl_code, model_name, local_emb_dst],
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if res_dl.returncode == 0:
            print(f" [OK] RAG embedding model cached successfully: {local_emb_dst}")
            return True
        err_tail = (res_dl.stderr or res_dl.stdout or "").strip().splitlines()[-3:]
        print(f" [!] Failed to cache RAG embedding model (offline?): {' | '.join(err_tail)}")
    except Exception as exc:
        print(f" [!] Error caching RAG embedding model: {exc}")
    return False


# =====================================================================
# 7. HIGH-LEVEL COMMANDS: SYNC, STATUS, SELF-TEST
# =====================================================================
def run_sync(
    disc_path: Optional[str] = None,
    gvk_home: str = DEFAULT_GVK_HOME,
    repo_root: str = DEFAULT_REPO_ROOT,
    download: bool = True,
    skip_pip: bool = False,
    pull: bool = False,
) -> dict:
    """Run the full GVK Private AI sync pipeline (supports DVD and offline USB modes).

    Args:
        disc_path: Optional explicit path to mounted `GVK_VAULT` media.
        gvk_home: Base path for the GVK local cache directory.
        repo_root: Repository root directory.
        download: Allow network downloads for base model, `llama-server`, and embeddings.
        skip_pip: If True, skip `.venv` creation/recreation and `pip install`.
        pull: If True and `<repo_root>/.git` exists, run `git pull --ff-only` first.

    Returns:
        Summary dictionary with cache paths, `.venv` state, media import results,
        persona sync summary, and resolved runtime asset paths.
    """
    print("====================================================================")
    print(" [*] GVK Private AI - Environment Sync & Verification (updater)")
    print("====================================================================")
    paths = init_gvk_home(gvk_home)

    if pull and os.path.isdir(os.path.join(repo_root, ".git")):
        print(" [*] Updating repository code (git pull --ff-only)...")
        subprocess.run(["git", "-C", repo_root, "pull", "--ff-only"], check=False)

    venv_info = ensure_venv_and_requirements(paths["home"], repo_root, skip_pip=skip_pip)
    disc_info = verify_and_import_disc(disc_path=disc_path, gvk_home=paths["home"])
    detected_disc = str(disc_info.get("disc_path")) if disc_info.get("disc_path") else None
    media_mode = str(disc_info.get("media_mode") or "NONE")
    master_key = disc_info.get("master_key")
    sync_info = sync_activated_personas(
        gvk_home=paths["home"],
        repo_root=repo_root,
        master_key=master_key if isinstance(master_key, bytes) else None,
    )
    base_model_path = ensure_base_model(
        paths["home"], repo_root, download=download, disc_path=detected_disc
    )
    llama_path = ensure_llama_server(
        paths["home"], repo_root, download=download, disc_path=detected_disc
    )
    emb_ok = (
        ensure_embedding_model(
            venv_info.get("python"),
            download=download,
            gvk_home=paths["home"],
            disc_path=detected_disc,
        )
        if venv_info.get("python") or detected_disc
        else False
    )

    print("====================================================================")
    print(f" [OK] GVK cache: {paths['home']} [Media mode: {media_mode}]")
    print("====================================================================")
    return {
        "paths": paths,
        "venv": venv_info,
        "disc_found": bool(disc_info.get("found")),
        "disc_imported": bool(disc_info.get("imported")),
        "disc_persona": disc_info.get("persona_id"),
        "media_mode": media_mode,
        "personas_sync": sync_info,
        "base_model": base_model_path,
        "llama_server": llama_path,
        "embedding_model": emb_ok,
    }


def run_status(gvk_home: str = DEFAULT_GVK_HOME, repo_root: str = DEFAULT_REPO_ROOT) -> None:
    """Print read-only diagnostics for cache, `.venv`, activated personas, media mode, and models.

    Args:
        gvk_home: Base path for the GVK local cache directory.
        repo_root: Repository root directory.
    """
    paths = get_gvk_paths(gvk_home)
    home_exists = os.path.isdir(paths["home"])
    venv_dir = resolve_venv_dir(paths["home"], repo_root)
    req_path = os.path.join(repo_root, "requirements.txt")
    req_status, _, _ = check_requirements_status(venv_dir, req_path)
    py_path = _venv_python_path(venv_dir)
    has_tk, tk_ver, py_ver = _probe_python_tk(py_path)
    tk_desc = f"Python {py_ver[0]}.{py_ver[1]}, Tk {tk_ver}" if has_tk else "Tk MISSING"
    registry = _load_activated_registry(paths["registry"])
    disc = find_gvk_disc_path(None)
    media_info = detect_media_mode(disc)
    base_path = find_base_model_path(paths["home"], repo_root, disc_path=disc)
    llama_path = find_llama_server(paths["home"], repo_root, disc_path=disc)

    print("====================================================================")
    print(" [*] GVK Private AI - Local Node Status")
    print("====================================================================")
    print(f" GVK_HOME cache  : {paths['home']} ({'OK' if home_exists else 'not initialized yet'})")
    print(f" Repository      : {repo_root}")
    print(f" Python .venv    : {venv_dir} ({tk_desc}, requirements: {req_status})")
    print(f" Base model      : {base_path or 'NOT FOUND'}")
    print(f" llama-server    : {llama_path or 'NOT FOUND'}")
    if disc:
        print(
            f" GVK_VAULT media : {disc} [Mode: {media_info['mode']} | "
            f"base_gguf={media_info['has_base_model']} | bins={media_info['has_llama_bins']}]"
        )
    else:
        print(" GVK_VAULT media : Not mounted")
    print(f" Activated ({len(registry)}):")
    if not registry:
        print("   - (no activated origin.sig entries yet)")
    else:
        for pid, meta in sorted(registry.items()):
            sig_ok = os.path.isfile(os.path.join(paths["registry"], f"{pid}.sig"))
            print(
                f"   - {pid} [v{meta.get('version', '?')}, source={meta.get('source', '?')}, "
                f"origin.sig={'OK' if sig_ok else 'MISSING'}]"
            )
    print("====================================================================")


# =====================================================================
# 8. OFFLINE SELF-TEST SUITE
# =====================================================================
def _make_fake_gvk1_container(path: str, payload: bytes, master_key: bytes) -> str:
    """Create a valid 36-byte `GVK1` container with an authenticated HMAC header tag."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    base_nonce = os.urandom(12)
    chunk_size = 64 * 1024 * 1024
    msg = MAGIC_BYTES + base_nonce + struct.pack("<I", chunk_size)
    header_tag = hmac.new(master_key, msg, hashlib.sha256).digest()[:GCM_TAG_SIZE]
    header = MAGIC_BYTES + base_nonce + header_tag + struct.pack("<I", chunk_size)
    with open(path, "wb") as f:
        f.write(header)
        f.write(payload)
    return compute_sha256(path)


def _write_test_persona_dir(
    target_dir: str,
    persona_id: str,
    version: str,
    master_key: bytes,
    seed: bytes,
) -> None:
    """Populate a synthetic encrypted persona directory (`manifest.json` + `.enc` files) for tests."""
    adapter_sha = _make_fake_gvk1_container(
        os.path.join(target_dir, "adapter.gguf.enc"),
        b"ENC_ADAPTER_" + seed,
        master_key,
    )
    payload_sha = _make_fake_gvk1_container(
        os.path.join(target_dir, "payload.tar.enc"),
        b"ENC_PAYLOAD_" + seed,
        master_key,
    )
    manifest = {
        "persona_id": persona_id,
        "version": version,
        "min_core_version": "1.0.0",
        "files": {"adapter": "adapter.gguf.enc", "payload": "payload.tar.enc"},
        "sha256": {"adapter": adapter_sha, "payload": payload_sha},
    }
    with open(os.path.join(target_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def run_self_test() -> None:
    """Execute the standalone offline self-test suite covering all `updater.py` workflows."""
    print("====================================================================")
    print(" [*] GVK Private AI - Offline Self-Test (updater.py)")
    print("====================================================================")

    with tempfile.TemporaryDirectory(prefix="gvk_updater_test_") as tmpdir:
        gvk_home = os.path.join(tmpdir, "gvk_home")
        repo_root = os.path.join(tmpdir, "repo")
        disc_dir = os.path.join(tmpdir, "GVK_VAULT_DVD")
        os.makedirs(repo_root, exist_ok=True)

        # 1. Initialize ~/.gvk_ai cache layout
        print(" [1/6] Testing ~/.gvk_ai cache directory initialization...")
        paths = init_gvk_home(gvk_home)
        for key in ("base", "bin", "personas", "registry", "logs"):
            assert os.path.isdir(paths[key]), f"Directory not created: {key}"
        print("       [OK] All cache subdirectories created.")

        # 2. Verify DVD import and anti-tampering protection
        print(" [2/6] Testing GVK_VAULT DVD import and cryptographic verification...")
        master_key = hashlib.sha256(b"GVK_UPDATER_TEST_KEY_2026").digest()
        os.makedirs(os.path.join(disc_dir, "vault"), exist_ok=True)
        os.makedirs(os.path.join(disc_dir, "persona"), exist_ok=True)
        with open(os.path.join(disc_dir, "vault", "master.key"), "wb") as f:
            f.write(master_key)

        _write_test_persona_dir(
            os.path.join(disc_dir, "persona"),
            persona_id="friend_1",
            version="1.0.0",
            master_key=master_key,
            seed=b"v100",
        )
        valid_sig = hmac.new(master_key, b"friend_1", hashlib.sha256).digest()
        bad_sig = bytes([valid_sig[0] ^ 0xFF]) + valid_sig[1:]

        # First confirm a forged origin.sig is rejected
        with open(os.path.join(disc_dir, "persona", "origin.sig"), "wb") as f:
            f.write(bad_sig)
        try:
            verify_and_import_disc(disc_path=disc_dir, gvk_home=gvk_home)
            raise AssertionError("Forged origin.sig was not rejected!")
        except ValueError:
            pass

        # Write valid origin.sig and import
        with open(os.path.join(disc_dir, "persona", "origin.sig"), "wb") as f:
            f.write(valid_sig)
        res_disc = verify_and_import_disc(disc_path=disc_dir, gvk_home=gvk_home)
        assert res_disc["imported"] is True and res_disc["persona_id"] == "friend_1"
        assert os.path.isfile(os.path.join(paths["registry"], "friend_1.sig"))
        assert os.path.isfile(os.path.join(paths["personas"], "friend_1", "adapter.gguf.enc"))
        print("       [OK] Valid DVD imported; forged origin.sig rejected.")

        # 3. Incremental Unlock: update friend_1 to v1.1.0 and block unactivated friend_2
        print(" [3/6] Testing Incremental Unlock (update friend_1, block friend_2)...")
        _write_test_persona_dir(
            os.path.join(repo_root, "personas_encrypted", "friend_1"),
            persona_id="friend_1",
            version="1.1.0",
            master_key=master_key,
            seed=b"v110_new",
        )
        _write_test_persona_dir(
            os.path.join(repo_root, "personas_encrypted", "friend_2"),
            persona_id="friend_2",
            version="1.0.0",
            master_key=master_key,
            seed=b"friend2",
        )
        sync_res = sync_activated_personas(gvk_home=gvk_home, repo_root=repo_root, master_key=master_key)
        assert sync_res["updated"] == ["friend_1"], f"Expected friend_1 to update: {sync_res}"
        assert sync_res["skipped_unactivated"] == ["friend_2"], f"Expected friend_2 to be skipped: {sync_res}"
        with open(os.path.join(paths["personas"], "friend_1", "manifest.json"), "r", encoding="utf-8") as f:
            assert json.load(f)["version"] == "1.1.0"
        assert not os.path.exists(os.path.join(paths["personas"], "friend_2"))
        print("       [OK] friend_1 updated to v1.1.0; unactivated friend_2 ignored.")

        # 4. Base GGUF model detection and USB copy
        print(" [4/6] Testing local and USB base GGUF model discovery...")
        assert detect_media_mode(disc_dir)["mode"] == "DVD"
        usb_base_dir = os.path.join(disc_dir, "base")
        os.makedirs(usb_base_dir, exist_ok=True)
        usb_gguf = os.path.join(usb_base_dir, DEFAULT_BASE_MODEL_FILENAME)
        with open(usb_gguf, "wb") as f:
            f.write(b"GGUF" + b"\x01" * 2048)
        assert detect_media_mode(disc_dir)["mode"] == "USB"
        base_out = ensure_base_model(gvk_home=gvk_home, repo_root=repo_root, download=False, disc_path=disc_dir)
        assert base_out == os.path.join(paths["base"], DEFAULT_BASE_MODEL_FILENAME)
        assert _is_valid_gguf_file(base_out) and os.path.getsize(base_out) == 2052
        print(f"       [OK] Base model copied from USB to cache: {base_out}")

        # 5. llama-server binary discovery and release asset selection
        print(" [5/6] Testing local llama-server binary discovery...")
        srv_name = _llama_server_names()[0]
        fake_srv = os.path.join(repo_root, "base", "llama-pascal-runtime", srv_name)
        os.makedirs(os.path.dirname(fake_srv), exist_ok=True)
        with open(fake_srv, "wb") as f:
            f.write(b"#!/bin/sh\necho ok\n")
        os.chmod(fake_srv, 0o755)
        found_srv = ensure_llama_server(gvk_home=gvk_home, repo_root=repo_root, download=False)
        assert found_srv == fake_srv
        assert ensure_embedding_model(None, download=False) is False
        from unittest.mock import patch
        win_assets = [
            {"name": "llama-b1234-bin-win-cpu-x64.zip"},
            {"name": "llama-b1234-bin-win-vulkan-x64.zip"},
            {"name": "llama-b1234-bin-win-cuda-12.4-x64.zip"},
        ]
        with patch("updater.platform.system", return_value="Windows"), patch("updater.platform.machine", return_value="AMD64"):
            picked = _select_llama_release_asset(win_assets)
            assert picked and "vulkan" in picked["name"].lower(), f"Expected win-vulkan-x64: {picked}"
        print(f"       [OK] Found llama-server: {found_srv}")

        # 6. requirements.txt SHA-256 marker and .venv probe
        print(" [6/6] Testing requirements.txt change detector for .venv...")
        fake_venv = os.path.join(repo_root, ".venv")
        fake_py = _venv_python_path(fake_venv)
        os.makedirs(os.path.dirname(fake_py), exist_ok=True)
        with open(fake_py, "wb") as f:
            f.write(b"python")
        req_file = os.path.join(repo_root, "requirements.txt")
        with open(req_file, "w", encoding="utf-8") as f:
            f.write("cryptography>=42.0.0\n")

        st1, sha1, _ = check_requirements_status(fake_venv, req_file)
        assert st1 == "needs_install" and sha1 is not None
        with open(os.path.join(fake_venv, ".gvk_requirements.sha256"), "w", encoding="utf-8") as f:
            f.write(sha1 + "\n")
        st2, _, _ = check_requirements_status(fake_venv, req_file)
        assert st2 == "up_to_date"
        with open(req_file, "a", encoding="utf-8") as f:
            f.write("faiss-cpu>=1.8.0\n")
        st3, _, _ = check_requirements_status(fake_venv, req_file)
        assert st3 == "outdated"
        best_py = _select_best_python(gvk_home)
        assert os.path.isfile(best_py)
        ok_tk, tk_v, py_v = _probe_python_tk(best_py)
        print(
            f"       [OK] requirements.txt SHA-256 detector OK; selected Python={py_v[0]}.{py_v[1]}, "
            f"Tk={tk_v if ok_tk else 'MISSING'}."
        )

    print("====================================================================")
    print(" [OK] ALL UPDATER.PY SELF-TESTS PASSED!")
    print("====================================================================")


# =====================================================================
# CLI INTERFACE
# =====================================================================
def main(argv: Optional[List[str]] = None) -> None:
    """Parse CLI arguments and dispatch to `sync`, `status`, or `self-test`."""
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    known_cmds = {"sync", "status", "self-test"}
    if not raw_argv or (raw_argv[0] not in known_cmds and raw_argv[0] not in ("-h", "--help")):
        raw_argv = ["sync"] + raw_argv

    parser = argparse.ArgumentParser(
        description="GVK Private AI - Media Verification, Dependency Sync & Runtime Asset Updater"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sync = sub.add_parser("sync", help="Verify .venv, import GVK_VAULT media, and sync local cache")
    p_sync.add_argument("--disc-path", default=None, help="Explicit path to mounted GVK_VAULT media")
    p_sync.add_argument("--gvk-home", default=DEFAULT_GVK_HOME, help="Path to local ~/.gvk_ai cache directory")
    p_sync.add_argument("--repo-root", default=DEFAULT_REPO_ROOT, help="Path to the repository root")
    p_sync.add_argument("--no-download", action="store_true", help="Do not download base model or llama-server from the network")
    p_sync.add_argument("--skip-pip", action="store_true", help="Skip .venv creation/verification and pip install")
    p_sync.add_argument("--pull", action="store_true", help="Update the repository via git pull before syncing")

    p_status = sub.add_parser("status", help="Display ~/.gvk_ai cache status and activated personas")
    p_status.add_argument("--gvk-home", default=DEFAULT_GVK_HOME, help="Path to local ~/.gvk_ai cache directory")
    p_status.add_argument("--repo-root", default=DEFAULT_REPO_ROOT, help="Path to the repository root")

    sub.add_parser("self-test", help="Run the standalone offline self-test suite for updater.py")

    args = parser.parse_args(raw_argv)

    if args.cmd == "self-test":
        run_self_test()
    elif args.cmd == "status":
        run_status(gvk_home=args.gvk_home, repo_root=args.repo_root)
    elif args.cmd == "sync":
        run_sync(
            disc_path=args.disc_path,
            gvk_home=args.gvk_home,
            repo_root=args.repo_root,
            download=not args.no_download,
            skip_pip=args.skip_pip,
            pull=args.pull,
        )


if __name__ == "__main__":
    main()
