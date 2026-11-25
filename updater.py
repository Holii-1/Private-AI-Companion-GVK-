#!/usr/bin/env python3
"""
GVK Private AI — Клієнт перевірки носія, реєстрації origin.sig та оновлень (updater.py)

Працює виключно на стандартній бібліотеці Python, щоб запускатися з DVD-диска `GVK_VAULT`
навіть на «чистому» комп'ютері до створення `.venv` і встановлення `requirements.txt`.

Основні задачі модуля:
  1. Ініціалізація локального кешу `~/.gvk_ai/{base,bin,personas,registry,logs}`.
  2. Перевірка та автоматичне розгортання `.venv` + встановлення/оновлення пакетів
     із `requirements.txt` (через звірку SHA-256 маркера).
  3. Імпорт та криптографічна верифікація DVD-диска `GVK_VAULT`:
     - перевірка 32-байтного `vault/master.key`;
     - верифікація апаратного HMAC-підпису `persona/origin.sig`;
     - перевірка SHA-256 гешів у `manifest.json` та заголовків `GVK1` контейнерів `.enc`;
     - копіювання `adapter.gguf.enc`, `payload.tar.enc` і `origin.sig` у `~/.gvk_ai/`.
  4. Накопичувальна активація (Incremental Unlock) із `personas_encrypted/`:
     - оновлює контейнери лише тих учасників, чий `origin.sig` вже зареєстрований локально.
  5. Одноразове завантаження (або лінкування локальної копії) публічної базової моделі
     `Qwen2.5-14B-Instruct-Q4_K_M.gguf` (~9.0 ГБ) з Hugging Face у `~/.gvk_ai/base/`
     із підтримкою HTTP Range (докачування після обриву зв'язку).
  6. Пошук або автоматичне завантаження бінарника `llama-server` із GitHub Releases `llama.cpp`.

Приклади використання:
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
# КОНСТАНТИ ПРОТОКОЛУ ТА ДЖЕРЕЛ
# =====================================================================
DEFAULT_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GVK_HOME = os.environ.get("GVK_HOME") or os.path.expanduser("~/.gvk_ai")

DEFAULT_BASE_MODEL_FILENAME = "Qwen2.5-14B-Instruct-Q4_K_M.gguf"
DEFAULT_BASE_MODEL_URL = (
    "https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF/resolve/main/"
    "Qwen2.5-14B-Instruct-Q4_K_M.gguf"
)
LLAMA_CPP_RELEASES_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

MIN_PYTHON_VERSION = (3, 10)
MIN_TK_VERSION = 8.6

MAGIC_BYTES = b"GVK1"
HEADER_SIZE = 36
KEY_SIZE = 32
SIG_SIZE = 32
GCM_TAG_SIZE = 16
GGUF_MAGIC = b"GGUF"


# =====================================================================
# ДОПОМІЖНІ УТИЛІТИ (STDLIB ONLY)
# =====================================================================
def compute_sha256(file_path: str) -> str:
    """Обчислює SHA-256 геш файлу потоково (блоками по 4 МБ)."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(4 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def verify_origin_sig_bytes(persona_id: str, master_key: bytes, sig_bytes: bytes) -> bool:
    """Перевіряє 32-байтний HMAC-SHA256 підпис `origin.sig`."""
    if len(master_key) != KEY_SIZE or len(sig_bytes) != SIG_SIZE:
        return False
    expected = hmac.new(master_key, persona_id.strip().encode("utf-8"), hashlib.sha256).digest()
    return hmac.compare_digest(expected, sig_bytes)


def verify_gvk1_header(enc_file_path: str, master_key: Optional[bytes] = None) -> None:
    """
    Перевіряє 36-байтний бінарний заголовок `.enc` контейнера:
      [Magic "GVK1" (4B) | Base Nonce (12B) | Header Auth Tag (16B) | Chunk Size (4B)]
    Якщо передано `master_key`, додатково перевіряє криптографічний тег заголовка.
    """
    with open(enc_file_path, "rb") as f:
        header = f.read(HEADER_SIZE)

    if len(header) != HEADER_SIZE:
        raise ValueError(f"Пошкоджений контейнер {os.path.basename(enc_file_path)}: заголовок < 36 Б.")

    magic = header[:4]
    if magic != MAGIC_BYTES:
        raise ValueError(
            f"Невірна сигнатура {os.path.basename(enc_file_path)}: "
            f"очікується {MAGIC_BYTES!r}, отримано {magic!r}."
        )

    base_nonce = header[4:16]
    header_tag = header[16:32]
    chunk_size_bytes = header[32:36]
    chunk_size = struct.unpack("<I", chunk_size_bytes)[0]
    if chunk_size == 0:
        raise ValueError(f"Некоректний розмір чанка (0) у {os.path.basename(enc_file_path)}.")

    if master_key is not None:
        msg = MAGIC_BYTES + base_nonce + chunk_size_bytes
        expected_tag = hmac.new(master_key, msg, hashlib.sha256).digest()[:GCM_TAG_SIZE]
        if not hmac.compare_digest(header_tag, expected_tag):
            raise ValueError(
                f"Контейнер {os.path.basename(enc_file_path)} не пройшов автентифікацію master.key!"
            )


def _normalize_disc_path(raw_path: str) -> str:
    """Нормалізує шлях до диска (зокрема Windows 'D:' -> 'D:\\' перед abspath)."""
    p = os.path.expanduser(raw_path.strip())
    if len(p) == 2 and p[1] == ":" and p[0].isalpha():
        p = p + "\\"
    return os.path.abspath(p)


def _copy_writable(src: str, dst: str) -> None:
    """
    Копіює файл без перенесення атрибуту Read-Only із DVD-носія та знімає
    Read-Only з цільового файлу перед перезаписом (критично для Windows).
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
    """Формує SSL-контекст із фолбеком на системний cert.pem / certifi (для macOS python.org)."""
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
    """Шукає вставлений DVD `GVK_VAULT` із `vault/master.key`."""
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


def _is_valid_gguf_file(file_path: str) -> bool:
    try:
        if not os.path.isfile(file_path) or os.path.getsize(file_path) < 4:
            return False
        with open(file_path, "rb") as f:
            return f.read(4) == GGUF_MAGIC
    except OSError:
        return False


def _format_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if value < 1024.0 or unit == "ГБ":
            return f"{value:.1f} {unit}" if unit != "Б" else f"{int(value)} Б"
        value /= 1024.0
    return f"{value:.1f} ГБ"


def _download_with_resume(
    url: str,
    dst_path: str,
    label: str = "Завантаження",
    max_retries: int = 5,
) -> str:
    """
    Потоково завантажує великий файл у `<dst_path>.part` із підтримкою HTTP Range
    (автоматичне докачування з місця обриву до `max_retries` спроб) та перевіркою
    повноти `Content-Length` перед перейменуванням у `dst_path`.
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
                # 416 Range Not Satisfiable — файл міг бути вже докачаний або сервер не підтримує Range
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
            chunk_size = 4 * 1024 * 1024  # 4 МБ

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
                            f"({_format_size(downloaded)} / {_format_size(total_size)}) [{speed_mb:.1f} МБ/с]",
                            end="",
                            flush=True,
                        )
                    else:
                        print(
                            f"\r [*] {label}: {_format_size(downloaded)} [{speed_mb:.1f} МБ/с]",
                            end="",
                            flush=True,
                        )

            print()
            if total_size is not None and downloaded < total_size:
                raise IOError(
                    f"Неповне завантаження {label}: отримано {downloaded} із {total_size} Б."
                )

            os.replace(part_path, dst_path)
            return dst_path

        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                print(f"\n [!] Обрив завантаження {label} (спроба {attempt}/{max_retries}: {exc}). Докачування через 2 с...")
                time.sleep(2.0)
            else:
                raise

    if last_exc is not None:
        raise last_exc
    return dst_path


# =====================================================================
# 1. ІНІЦІАЛІЗАЦІЯ КЕШУ ~/.gvk_ai
# =====================================================================
def get_gvk_paths(gvk_home: str = DEFAULT_GVK_HOME) -> Dict[str, str]:
    """Повертає словник стандартних піддиректорій кешу `~/.gvk_ai` без запису на диск."""
    gvk_home = os.path.abspath(os.path.expanduser(gvk_home))
    return {
        "home": gvk_home,
        "base": os.path.join(gvk_home, "base"),
        "bin": os.path.join(gvk_home, "bin"),
        "personas": os.path.join(gvk_home, "personas"),
        "registry": os.path.join(gvk_home, "registry"),
        "logs": os.path.join(gvk_home, "logs"),
    }


def init_gvk_home(gvk_home: str = DEFAULT_GVK_HOME) -> Dict[str, str]:
    """Створює структуру робочої директорії `~/.gvk_ai`."""
    paths = get_gvk_paths(gvk_home)
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


# =====================================================================
# 2. ПЕРЕВІРКА .VENV ТА REQUIREMENTS.TXT
# =====================================================================
def _venv_python_path(venv_dir: str) -> str:
    if platform.system().lower() == "windows":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    for name in ("python3", "python"):
        candidate = os.path.join(venv_dir, "bin", name)
        if os.path.isfile(candidate):
            return candidate
    return os.path.join(venv_dir, "bin", "python3")


def _portable_python_path(gvk_home: str = DEFAULT_GVK_HOME) -> str:
    home_abs = os.path.abspath(os.path.expanduser(gvk_home))
    if platform.system().lower() == "windows":
        return os.path.join(home_abs, "python", "python.exe")
    return os.path.join(home_abs, "python", "bin", "python3")


def _tcl_tk_env_for_python(py_bin: str) -> Dict[str, str]:
    """Підтягує TCL_LIBRARY / TK_LIBRARY для портативного python-build-standalone, якщо вони поруч."""
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
    """
    Перевіряє версію Python та наявність сучасного Tkinter (`Tk >= 8.6`).
    Повертає `(has_tk, tk_version, (major, minor))`.
    На macOS системний Tk 8.5 не відмальовує Cocoa-віджети, тому вимагається Tk >= 8.6.
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
    """
    Обирає найкращий інтерпретатор Python для `.venv`:
      - Пріоритет: портативний `~/.gvk_ai/python` або системний Python >= 3.10 із `TkVersion >= 8.6`.
      - На macOS за потреби автоматично доінстальовує `python-tk@<major.minor>` через Homebrew.
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

    # На macOS: якщо знайшли Homebrew Python >= 3.10 без python-tk — спробуємо поставити python-tk@X.Y
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
                    print(f" [*] Виявлено Python {py_ver[0]}.{py_ver[1]} без Tkinter — встановлюємо {pkg}...")
                    try:
                        env = dict(os.environ, HOMEBREW_NO_AUTO_UPDATE="1")
                        subprocess.run([brew_bin, "install", pkg], check=True, env=env, input="y\n", text=True)
                        ok2, tk2, _ = _probe_python_tk(cand)
                        if ok2 and tk2 >= MIN_TK_VERSION:
                            return cand
                    except Exception as exc:
                        print(f" [!] Не вдалося встановити {pkg} автоматично: {exc}")

    for cand, has_tk, tk_ver, _py_ver in probed:
        if has_tk and tk_ver >= MIN_TK_VERSION:
            return cand
    return sys.executable


def resolve_venv_dir(gvk_home: str = DEFAULT_GVK_HOME, repo_root: str = DEFAULT_REPO_ROOT) -> str:
    """Повертає шлях до активної або цільової директорії `.venv`."""
    repo_venv = os.path.join(repo_root, ".venv")
    home_venv = os.path.join(gvk_home, ".venv")
    if os.path.isfile(_venv_python_path(repo_venv)):
        return repo_venv
    return home_venv


def check_requirements_status(venv_dir: str, req_path: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Перевіряє стан залежностей за SHA-256 маркером `<venv>/.gvk_requirements.sha256`.
    Повертає (status, current_sha256, saved_sha256), де status:
      - 'missing_req': файл requirements.txt відсутній
      - 'missing_venv': директорія .venv або інтерпретатор відсутні
      - 'needs_install': маркер відсутній
      - 'outdated': хеш requirements.txt змінився
      - 'up_to_date': маркер збігається
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
    """
    Перевіряє наявність `.venv` (із Python >= 3.10 та Tk >= 8.6) і відповідність
    встановлених пакетів до `requirements.txt`. За потреби створює або оновлює `.venv`.
    """
    venv_dir = resolve_venv_dir(gvk_home, repo_root)
    req_path = os.path.join(repo_root, "requirements.txt")
    py_path = _venv_python_path(venv_dir)
    marker_path = os.path.join(venv_dir, ".gvk_requirements.sha256")

    if os.path.isfile(py_path):
        has_tk, tk_ver, py_ver = _probe_python_tk(py_path)
        if not skip_pip and (py_ver < MIN_PYTHON_VERSION or not has_tk or tk_ver < MIN_TK_VERSION):
            print(
                f" [!] Виявлено застарілий Python або Tk у .venv (Python={py_ver[0]}.{py_ver[1]}, "
                f"Tk={tk_ver})."
            )
            best_py = _select_best_python(gvk_home)
            best_has_tk, best_tk, best_py_ver = _probe_python_tk(best_py)
            if best_py_ver >= MIN_PYTHON_VERSION and best_has_tk and best_tk >= MIN_TK_VERSION:
                print(f" [*] Перестворення {venv_dir} на базі {best_py} (Python {best_py_ver[0]}.{best_py_ver[1]}, Tk {best_tk})...")
                subprocess.run([best_py, "-m", "venv", "--clear", venv_dir], check=True)
                if os.path.isfile(marker_path):
                    os.remove(marker_path)
                py_path = _venv_python_path(venv_dir)
            elif py_ver < MIN_PYTHON_VERSION:
                raise RuntimeError(
                    "Потрібен Python >= 3.10 (поточний < 3.10). "
                    "macOS: brew install python@3.12 python-tk@3.12 | "
                    "Windows: python.org (Python 3.12) | Linux: sudo apt install python3.12 python3.12-venv python3-tk"
                )
            else:
                print(" [!] УВАГА: Не знайдено Python із Tk >= 8.6. Для macOS: brew install python@3.12 python-tk@3.12")

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
                f"Знайдено лише Python {best_py_ver[0]}.{best_py_ver[1]} ({best_py}), але потрібен Python >= 3.10. "
                "macOS: brew install python@3.12 python-tk@3.12 | "
                "Windows: встановіть Python 3.12 з python.org (Add to PATH) | "
                "Linux: sudo apt install python3.12 python3.12-venv python3-tk"
            )
        print(f" [*] Створення віртуального середовища Python у {venv_dir} ({best_py})...")
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
        print(f" [OK] Залежності requirements.txt актуальні ({venv_dir}).")
        return {"venv_dir": venv_dir, "python": py_path, "status": "up_to_date"}

    if skip_pip:
        print(f" [!] Перевірку pip пропущено (--skip-pip), статус: {status}.")
        return {"venv_dir": venv_dir, "python": py_path, "status": f"{status}_skipped"}

    pip_check = subprocess.run([py_path, "-m", "pip", "--version"], capture_output=True, text=True)
    if pip_check.returncode != 0:
        subprocess.run([py_path, "-m", "ensurepip", "--upgrade"], check=False)

    print(f" [*] Оновлення залежностей Python ({status}) із {req_path}...")
    subprocess.run([py_path, "-m", "pip", "install", "--disable-pip-version-check", "-r", req_path], check=True)
    if current_hash:
        with open(marker_path, "w", encoding="utf-8") as f:
            f.write(current_hash + "\n")
    print(" [✓] Залежності requirements.txt успішно встановлено.")
    return {"venv_dir": venv_dir, "python": py_path, "status": "installed"}


# =====================================================================
# 3. ВЕРИФІКАЦІЯ ТА ІМПОРТ DVD-ДИСКА GVK_VAULT
# =====================================================================
def _load_activated_registry(registry_dir: str) -> Dict[str, dict]:
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
    reg_file = os.path.join(registry_dir, "activated.json")
    with open(reg_file, "w", encoding="utf-8") as f:
        json.dump(registry_data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _validate_manifest_and_files(
    source_dir: str,
    master_key: Optional[bytes] = None,
) -> dict:
    """Читає `manifest.json`, перевіряє SHA-256 геші `.enc` файлів та заголовок `GVK1`."""
    manifest_path = os.path.join(source_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Відсутній manifest.json у {source_dir}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    persona_id = manifest.get("persona_id")
    files_map = manifest.get("files") or {}
    sha_map = manifest.get("sha256") or {}
    if not persona_id or "adapter" not in files_map or "payload" not in files_map:
        raise ValueError(f"Неповний manifest.json у {source_dir}")

    for key_name in ("adapter", "payload"):
        rel_name = files_map[key_name]
        fpath = os.path.join(source_dir, rel_name)
        if not os.path.isfile(fpath):
            raise FileNotFoundError(f"Відсутній контейнер {rel_name} у {source_dir}")

        expected_sha = sha_map.get(key_name)
        if not expected_sha:
            raise ValueError(f"У manifest.json відсутній SHA-256 для '{key_name}'")
        actual_sha = compute_sha256(fpath)
        if not hmac.compare_digest(actual_sha.lower(), str(expected_sha).lower()):
            raise ValueError(
                f"Порушено цілісність SHA-256 для {rel_name}: "
                f"очікується {expected_sha[:16]}..., отримано {actual_sha[:16]}..."
            )
        verify_gvk1_header(fpath, master_key=master_key)

    return manifest


def verify_and_import_disc(
    disc_path: Optional[str] = None,
    gvk_home: str = DEFAULT_GVK_HOME,
) -> Dict[str, Optional[object]]:
    """
    Виявляє DVD `GVK_VAULT`, перевіряє `vault/master.key`, `origin.sig`, SHA-256
    та заголовки `GVK1`, реєструє підпис у `~/.gvk_ai/registry/` і копіює `.enc` на SSD.
    """
    paths = init_gvk_home(gvk_home)
    detected_disc = find_gvk_disc_path(disc_path)
    if not detected_disc:
        return {"found": False, "imported": False, "persona_id": None, "disc_path": None, "master_key": None}

    key_path = os.path.join(detected_disc, "vault", "master.key")
    with open(key_path, "rb") as f:
        master_key = f.read()
    if len(master_key) != KEY_SIZE:
        raise ValueError(f"Невірний розмір vault/master.key: {len(master_key)} Б (очікується {KEY_SIZE} Б).")

    disc_persona_dir = os.path.join(detected_disc, "persona")
    if not os.path.isdir(disc_persona_dir):
        print(f" [OK] Диск GVK_VAULT виявлено ({detected_disc}), папка persona/ відсутня — лише ключ.")
        return {
            "found": True,
            "imported": False,
            "persona_id": None,
            "disc_path": detected_disc,
            "master_key": master_key,
        }

    manifest = _validate_manifest_and_files(disc_persona_dir, master_key=master_key)
    persona_id = str(manifest["persona_id"]).strip()

    sig_path = os.path.join(disc_persona_dir, "origin.sig")
    if not os.path.isfile(sig_path):
        raise FileNotFoundError(f"На DVD {detected_disc} відсутній підпис persona/origin.sig!")
    with open(sig_path, "rb") as f:
        sig_bytes = f.read()

    if not verify_origin_sig_bytes(persona_id, master_key, sig_bytes):
        raise ValueError(
            f"Криптографічна помилка: origin.sig на диску {detected_disc} "
            f"не відповідає persona_id='{persona_id}' або master.key!"
        )

    # Зберігаємо origin.sig у локальний реєстр
    reg_sig_path = os.path.join(paths["registry"], f"{persona_id}.sig")
    if os.path.exists(reg_sig_path):
        try:
            os.chmod(reg_sig_path, 0o644)
        except OSError:
            pass
    with open(reg_sig_path, "wb") as f:
        f.write(sig_bytes)

    # Перевіряємо, чи персонаж уже активований та валідний у кеші SSD (щоб не перекопійовувати 100+ МБ із DVD і не відкочувати оновлення з GitHub)
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
            f" [OK] Персонаж '{persona_id}' вже активований у кеші (v{cached_ver}) — "
            f"повторне копіювання з DVD пропущено."
        )
        return {
            "found": True,
            "imported": False,
            "persona_id": persona_id,
            "disc_path": detected_disc,
            "master_key": master_key,
        }

    # Копіюємо .enc контейнери та manifest.json у кеш SSD (без origin.sig, master.key і без Read-Only біта DVD)
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
        "source": f"dvd:{detected_disc}",
    }
    _save_activated_registry(paths["registry"], registry_data)

    print(
        f" [✓] Активовано та імпортовано з DVD '{persona_id}' (v{manifest.get('version', '1.0.0')}) "
        f"-> {target_persona_dir}"
    )
    return {
        "found": True,
        "imported": True,
        "persona_id": persona_id,
        "disc_path": detected_disc,
        "master_key": master_key,
    }


# =====================================================================
# 4. НАКОПИЧУВАЛЬНА АКТИВАЦІЯ ТА ОНОВЛЕННЯ ІЗ GITHUB (INCREMENTAL UNLOCK)
# =====================================================================
def sync_activated_personas(
    gvk_home: str = DEFAULT_GVK_HOME,
    repo_root: str = DEFAULT_REPO_ROOT,
    master_key: Optional[bytes] = None,
) -> Dict[str, List[str]]:
    """
    Перевіряє оновлення в `personas_encrypted/<persona_id>/` і синхронізує ТІЛЬКИ
    тих персонажів, для яких у `~/.gvk_ai/registry/<persona_id>.sig` є валідний `origin.sig`.
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
            print(f" [-] Пропущено '{persona_id}': фізичний DVD із origin.sig ще не активовано.")
            result["skipped_unactivated"].append(persona_id)
            continue

        if master_key is not None:
            with open(reg_sig_path, "rb") as f:
                sig_bytes = f.read()
            if not verify_origin_sig_bytes(persona_id, master_key, sig_bytes):
                print(
                    f" [!] Попередження: локальний origin.sig для '{persona_id}' "
                    f"не пройшов перевірку поточним master.key — оновлення заблоковано."
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
        print(f" [✓] Оновлено '{persona_id}' -> v{verified_manifest.get('version', '1.0.0')}")

    if registry_changed:
        _save_activated_registry(paths["registry"], registry_data)
    return result


# =====================================================================
# 5. БАЗОВА МОДЕЛЬ 14B (З ЛОКАЛЬНОГО КЕШУ АБО HUGGING FACE)
# =====================================================================
def _collect_base_model_candidates(gvk_home: str, repo_root: str) -> Tuple[List[str], List[str]]:
    """Повертає (кандидати в кеші gvk_home/base, локальні кандидати в env/репо)."""
    paths = get_gvk_paths(gvk_home)
    cache_candidates = [os.path.join(paths["base"], DEFAULT_BASE_MODEL_FILENAME)]
    if os.path.isdir(paths["base"]):
        cache_candidates.extend(sorted(glob.glob(os.path.join(paths["base"], "*.gguf"))))

    local_candidates: List[str] = []
    env_model = os.environ.get("GVK_BASE_MODEL")
    if env_model:
        local_candidates.append(os.path.abspath(os.path.expanduser(env_model)))
    local_candidates.append(os.path.join(repo_root, "base", DEFAULT_BASE_MODEL_FILENAME))
    repo_base_dir = os.path.join(repo_root, "base")
    if os.path.isdir(repo_base_dir):
        local_candidates.extend(sorted(glob.glob(os.path.join(repo_base_dir, "*.gguf"))))
    local_candidates.append(os.path.join(repo_root, "base_model.gguf"))
    return cache_candidates, local_candidates


def find_base_model_path(
    gvk_home: str = DEFAULT_GVK_HOME,
    repo_root: str = DEFAULT_REPO_ROOT,
) -> Optional[str]:
    """Read-only пошук валідної базової GGUF-моделі без створення файлів чи симлінків."""
    cache_candidates, local_candidates = _collect_base_model_candidates(gvk_home, repo_root)
    for cand in cache_candidates + local_candidates:
        if cand and _is_valid_gguf_file(cand):
            return os.path.realpath(cand) if os.path.islink(cand) else cand
    return None


def ensure_base_model(
    gvk_home: str = DEFAULT_GVK_HOME,
    repo_root: str = DEFAULT_REPO_ROOT,
    download: bool = True,
    model_url: str = DEFAULT_BASE_MODEL_URL,
) -> Optional[str]:
    """
    Забезпечує наявність публічної базової моделі `Qwen2.5-14B-Instruct-Q4_K_M.gguf`
    у `~/.gvk_ai/base/`. Спершу шукає валідний GGUF у локальних шляхах (і створює symlink),
    інакше завантажує з Hugging Face із підтримкою докачування (HTTP Range).
    """
    paths = init_gvk_home(gvk_home)
    target_path = os.path.join(paths["base"], DEFAULT_BASE_MODEL_FILENAME)
    cache_candidates, local_candidates = _collect_base_model_candidates(gvk_home, repo_root)

    for existing in cache_candidates:
        if existing and _is_valid_gguf_file(existing):
            print(f" [OK] Базова модель готова: {existing} ({_format_size(os.path.getsize(existing))})")
            return existing

    for cand in local_candidates:
        if cand and _is_valid_gguf_file(cand):
            real_src = os.path.realpath(cand)
            try:
                if os.path.lexists(target_path):
                    os.remove(target_path)
                os.symlink(real_src, target_path)
                print(f" [✓] Прив'язано локальну базову модель: {target_path} -> {real_src}")
                return target_path
            except OSError:
                print(f" [OK] Знайдено локальну базову модель: {real_src}")
                return real_src

    if not download:
        print(" [!] Базову модель не знайдено локально (завантаження вимкнено через --no-download).")
        return None

    print(f" [*] Завантаження базової моделі 14B (~9.0 ГБ) із Hugging Face...")
    print(f"     URL: {model_url}")
    _download_with_resume(model_url, target_path, label=DEFAULT_BASE_MODEL_FILENAME)
    if not _is_valid_gguf_file(target_path):
        os.remove(target_path)
        raise ValueError("Завантажений файл базової моделі не є валідним GGUF (відсутній заголовок GGUF)!")
    print(f" [✓] Базову модель збережено у: {target_path}")
    return target_path


# =====================================================================
# 6. БІНАРНИК LLAMA-SERVER (ПОШУК АБО АВТОЗАВАНТАЖЕННЯ З GITHUB RELEASES)
# =====================================================================
def _llama_server_names() -> Tuple[str, ...]:
    if platform.system().lower() == "windows":
        return ("llama-server.exe",)
    return ("llama-server",)


def find_llama_server(gvk_home: str = DEFAULT_GVK_HOME, repo_root: str = DEFAULT_REPO_ROOT) -> Optional[str]:
    """Шукає виконуваний файл `llama-server` у `~/.gvk_ai/bin/`, репозиторії та PATH."""
    names = _llama_server_names()
    bin_dir = os.path.join(os.path.abspath(os.path.expanduser(gvk_home)), "bin")
    for name in names:
        cand = os.path.join(bin_dir, name)
        if os.path.isfile(cand):
            return cand

    runtime_dir = os.path.join(repo_root, "base", "llama-pascal-runtime")
    if os.path.isdir(runtime_dir):
        for root, _, files in os.walk(runtime_dir):
            for name in names:
                if name in files:
                    return os.path.join(root, name)

    for name in names:
        found = shutil.which(name)
        if found:
            return found
    return None


def _has_windows_vulkan() -> bool:
    """На реальній Windows перевіряє наявність vulkan-1.dll перед вибором win-vulkan збірки."""
    if os.name != "nt":
        return True
    sys_root = os.environ.get("SystemRoot", r"C:\Windows")
    return os.path.isfile(os.path.join(sys_root, "System32", "vulkan-1.dll"))


def _select_llama_release_asset(assets: List[dict]) -> Optional[dict]:
    sys_name = platform.system().lower()
    arch = platform.machine().lower()
    is_arm = arch in ("arm64", "aarch64")
    has_win_vulkan = _has_windows_vulkan() if sys_name == "windows" else False

    def matches(name: str) -> int:
        n = name.lower()
        if not (n.endswith(".zip") or n.endswith(".tar.gz") or n.endswith(".tgz")):
            return -1
        if "llama-" not in n:
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
                return 100 if "vulkan" not in n and "sycl" not in n else 80
            return -1
        if sys_name == "windows":
            if "win" not in n:
                return -1
            if "vulkan" in n and "x64" in n:
                return 100 if has_win_vulkan else 75
            if "cpu-x64" in n:
                return 90 if has_win_vulkan else 100
            if "x64" in n and "cuda" not in n and "sycl" not in n:
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


def _extract_llama_archive(archive_path: str, bin_dir: str) -> Optional[str]:
    """Розпаковує `llama-server` та супутні динамічні бібліотеки / шейдери у `bin_dir`."""
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


def ensure_llama_server(
    gvk_home: str = DEFAULT_GVK_HOME,
    repo_root: str = DEFAULT_REPO_ROOT,
    download: bool = True,
) -> Optional[str]:
    """Знаходить або скачує `llama-server` із GitHub Releases `ggml-org/llama.cpp`."""
    paths = init_gvk_home(gvk_home)
    existing = find_llama_server(paths["home"], repo_root)
    if existing:
        print(f" [OK] llama-server готовий: {existing}")
        return existing

    if not download:
        print(" [!] llama-server не знайдено локально (завантаження вимкнено через --no-download).")
        return None

    print(" [*] Запит останнього релізу ggml-org/llama.cpp на GitHub...")
    req = urllib.request.Request(
        LLAMA_CPP_RELEASES_API,
        headers={"User-Agent": "GVK-Private-AI-Updater/1.0", "Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(req, timeout=30, context=_build_ssl_context()) as resp:
        release = json.loads(resp.read().decode("utf-8"))

    asset = _select_llama_release_asset(release.get("assets") or [])
    if not asset:
        raise RuntimeError(
            f"Не знайдено відповідної збірки llama.cpp для {platform.system()} / {platform.machine()}."
        )

    asset_name = asset["name"]
    asset_url = asset["browser_download_url"]
    with tempfile.TemporaryDirectory(prefix="gvk_llama_dl_") as tmp_dl:
        archive_path = os.path.join(tmp_dl, asset_name)
        _download_with_resume(asset_url, archive_path, label=asset_name)
        bin_path = _extract_llama_archive(archive_path, paths["bin"])
        if not bin_path:
            raise RuntimeError(f"У завантаженому архіві {asset_name} не знайдено llama-server!")
        print(f" [✓] llama-server встановлено у: {bin_path}")
        return bin_path


def ensure_embedding_model(
    venv_python: Optional[str],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    download: bool = True,
) -> bool:
    """
    Перевіряє та прекешує ембедінг-модель RAG (`sentence-transformers`) у `.venv`,
    щоб GUI не підвисав на холодному старті і працював в офлайні.
    """
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
            print(f" [OK] RAG-модель кешована: {model_name}")
            return True
    except Exception as exc:
        print(f" [!] Не вдалося перевірити кеш RAG-моделі: {exc}")
        return False

    if not download:
        print(" [!] RAG-модель не знайдено в кеші (завантаження вимкнено через --no-download).")
        return False

    print(f" [*] Прекешування RAG-моделі {model_name} (~470 МБ)...")
    dl_code = (
        "import sys; "
        "from sentence_transformers import SentenceTransformer; "
        "SentenceTransformer(sys.argv[1], device='cpu')"
    )
    try:
        res_dl = subprocess.run(
            [venv_python, "-c", dl_code, model_name],
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if res_dl.returncode == 0:
            print(f" [✓] RAG-модель успішно кешовано: {model_name}")
            return True
        err_tail = (res_dl.stderr or res_dl.stdout or "").strip().splitlines()[-3:]
        print(f" [!] Не вдалося кешувати RAG-модель (офлайн?): {' | '.join(err_tail)}")
    except Exception as exc:
        print(f" [!] Помилка кешування RAG-моделі: {exc}")
    return False


# =====================================================================
# 7. ЗВЕДЕНІ КОМАНДИ: SYNC, STATUS, SELF-TEST
# =====================================================================
def run_sync(
    disc_path: Optional[str] = None,
    gvk_home: str = DEFAULT_GVK_HOME,
    repo_root: str = DEFAULT_REPO_ROOT,
    download: bool = True,
    skip_pip: bool = False,
    pull: bool = False,
) -> dict:
    """Повний цикл синхронізації GVK Private AI."""
    print("====================================================================")
    print(" [*] GVK Private AI — Синхронізація та перевірка оточення (updater)")
    print("====================================================================")
    paths = init_gvk_home(gvk_home)

    if pull and os.path.isdir(os.path.join(repo_root, ".git")):
        print(" [*] Оновлення коду репозиторію (git pull --ff-only)...")
        subprocess.run(["git", "-C", repo_root, "pull", "--ff-only"], check=False)

    venv_info = ensure_venv_and_requirements(paths["home"], repo_root, skip_pip=skip_pip)
    disc_info = verify_and_import_disc(disc_path=disc_path, gvk_home=paths["home"])
    master_key = disc_info.get("master_key")
    sync_info = sync_activated_personas(
        gvk_home=paths["home"],
        repo_root=repo_root,
        master_key=master_key if isinstance(master_key, bytes) else None,
    )
    base_model_path = ensure_base_model(paths["home"], repo_root, download=download)
    llama_path = ensure_llama_server(paths["home"], repo_root, download=download)
    emb_ok = ensure_embedding_model(venv_info.get("python"), download=download) if venv_info.get("python") else False

    print("====================================================================")
    print(f" [✓] Кеш GVK: {paths['home']}")
    print("====================================================================")
    return {
        "paths": paths,
        "venv": venv_info,
        "disc_found": bool(disc_info.get("found")),
        "disc_imported": bool(disc_info.get("imported")),
        "disc_persona": disc_info.get("persona_id"),
        "personas_sync": sync_info,
        "base_model": base_model_path,
        "llama_server": llama_path,
        "embedding_model": emb_ok,
    }


def run_status(gvk_home: str = DEFAULT_GVK_HOME, repo_root: str = DEFAULT_REPO_ROOT) -> None:
    """Виводить поточний статус кешу, активованих персонажів, моделі та .venv (read-only)."""
    paths = get_gvk_paths(gvk_home)
    home_exists = os.path.isdir(paths["home"])
    venv_dir = resolve_venv_dir(paths["home"], repo_root)
    req_path = os.path.join(repo_root, "requirements.txt")
    req_status, _, _ = check_requirements_status(venv_dir, req_path)
    py_path = _venv_python_path(venv_dir)
    has_tk, tk_ver, py_ver = _probe_python_tk(py_path)
    tk_desc = f"Python {py_ver[0]}.{py_ver[1]}, Tk {tk_ver}" if has_tk else "Tk MISSING"
    registry = _load_activated_registry(paths["registry"])
    base_path = find_base_model_path(paths["home"], repo_root)
    llama_path = find_llama_server(paths["home"], repo_root)
    disc = find_gvk_disc_path(None)

    print("====================================================================")
    print(" [*] GVK Private AI — Статус локального вузла")
    print("====================================================================")
    print(f" Кеш GVK_HOME    : {paths['home']} ({'OK' if home_exists else 'ще не ініціалізовано'})")
    print(f" Репозиторій     : {repo_root}")
    print(f" Python .venv    : {venv_dir} ({tk_desc}, requirements: {req_status})")
    print(f" Базова модель   : {base_path or 'НЕ ЗНАЙДЕНО'}")
    print(f" llama-server    : {llama_path or 'НЕ ЗНАЙДЕНО'}")
    print(f" DVD GVK_VAULT   : {disc or 'Не вставлено'}")
    print(f" Активовані ({len(registry)}):")
    if not registry:
        print("   - (немає жодного активованого origin.sig)")
    else:
        for pid, meta in sorted(registry.items()):
            sig_ok = os.path.isfile(os.path.join(paths["registry"], f"{pid}.sig"))
            print(f"   - {pid} [v{meta.get('version', '?')}, origin.sig={'OK' if sig_ok else 'MISSING'}]")
    print("====================================================================")


# =====================================================================
# 8. АВТОМАТИЧНИЙ SELF-TEST (ПОВНІСТЮ ОФЛАЙН)
# =====================================================================
def _make_fake_gvk1_container(path: str, payload: bytes, master_key: bytes) -> str:
    """Формує валідний 36-байтний заголовок GVK1 із правильним HMAC Header Auth Tag."""
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
    print("====================================================================")
    print(" [*] GVK Private AI — Офлайн Self-Test (updater.py)")
    print("====================================================================")

    with tempfile.TemporaryDirectory(prefix="gvk_updater_test_") as tmpdir:
        gvk_home = os.path.join(tmpdir, "gvk_home")
        repo_root = os.path.join(tmpdir, "repo")
        disc_dir = os.path.join(tmpdir, "GVK_VAULT_DVD")
        os.makedirs(repo_root, exist_ok=True)

        # 1. Ініціалізація ~/.gvk_ai
        print(" [1/6] Перевірка ініціалізації структури кешу ~/.gvk_ai...")
        paths = init_gvk_home(gvk_home)
        for key in ("base", "bin", "personas", "registry", "logs"):
            assert os.path.isdir(paths[key]), f"Не створено папку {key}"
        print("       [OK] Усі піддиректорії кешу створено.")

        # 2. Верифікація та імпорт DVD + перевірка захисту від підробки
        print(" [2/6] Перевірка імпорту DVD GVK_VAULT та криптографічної верифікації...")
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

        # Спочатку перевіряємо блокування підробленого origin.sig
        with open(os.path.join(disc_dir, "persona", "origin.sig"), "wb") as f:
            f.write(bad_sig)
        try:
            verify_and_import_disc(disc_path=disc_dir, gvk_home=gvk_home)
            raise AssertionError("Підроблений origin.sig не заблоковано!")
        except ValueError:
            pass

        # Записуємо валідний origin.sig та імпортуємо
        with open(os.path.join(disc_dir, "persona", "origin.sig"), "wb") as f:
            f.write(valid_sig)
        res_disc = verify_and_import_disc(disc_path=disc_dir, gvk_home=gvk_home)
        assert res_disc["imported"] is True and res_disc["persona_id"] == "friend_1"
        assert os.path.isfile(os.path.join(paths["registry"], "friend_1.sig"))
        assert os.path.isfile(os.path.join(paths["personas"], "friend_1", "adapter.gguf.enc"))
        print("       [OK] Валідний DVD імпортовано; підроблений origin.sig відхилено.")

        # 3. Incremental Unlock: оновлення friend_1 до v1.1.0 та блокування friend_2 без DVD
        print(" [3/6] Перевірка Incremental Unlock (оновлення friend_1, блокування friend_2)...")
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
        assert sync_res["updated"] == ["friend_1"], f"Очікувалось оновлення friend_1: {sync_res}"
        assert sync_res["skipped_unactivated"] == ["friend_2"], f"friend_2 мав бути заблокований: {sync_res}"
        with open(os.path.join(paths["personas"], "friend_1", "manifest.json"), "r", encoding="utf-8") as f:
            assert json.load(f)["version"] == "1.1.0"
        assert not os.path.exists(os.path.join(paths["personas"], "friend_2"))
        print("       [OK] friend_1 оновлено до v1.1.0; friend_2 без origin.sig проігноровано.")

        # 4. Локальний лінк базової моделі GGUF
        print(" [4/6] Перевірка підхоплення локальної базової моделі GGUF...")
        local_gguf = os.path.join(repo_root, "base_model.gguf")
        with open(local_gguf, "wb") as f:
            f.write(b"GGUF" + b"\x00" * 1024)
        base_out = ensure_base_model(gvk_home=gvk_home, repo_root=repo_root, download=False)
        assert base_out and _is_valid_gguf_file(base_out)
        print(f"       [OK] Базову модель підключено у: {base_out}")

        # 5. Пошук бінарника llama-server
        print(" [5/6] Перевірка виявлення локального бінарника llama-server...")
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
            assert picked and "vulkan" in picked["name"].lower(), f"Очікувався win-vulkan-x64: {picked}"
        print(f"       [OK] Знайдено llama-server: {found_srv}")

        # 6. Перевірка маркера requirements.txt + .venv
        print(" [6/6] Перевірка детектора змін requirements.txt для .venv...")
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
            f"       [OK] Детектор SHA-256 requirements.txt OK; обраний Python={py_v[0]}.{py_v[1]}, "
            f"Tk={tk_v if ok_tk else 'MISSING'}."
        )

    print("====================================================================")
    print(" [✓] УСІ ТЕСТИ UPDATER.PY ПРОЙДЕНО УСПІШНО!")
    print("====================================================================")


# =====================================================================
# CLI ІНТЕРФЕЙС
# =====================================================================
def main(argv: Optional[List[str]] = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    known_cmds = {"sync", "status", "self-test"}
    if not raw_argv or (raw_argv[0] not in known_cmds and raw_argv[0] not in ("-h", "--help")):
        raw_argv = ["sync"] + raw_argv

    parser = argparse.ArgumentParser(
        description="GVK Private AI — Клієнт перевірки DVD, requirements.txt, базової моделі та оновлень"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sync = sub.add_parser("sync", help="Перевірити .venv, імпортувати DVD та синхронізувати кеш")
    p_sync.add_argument("--disc-path", default=None, help="Явний шлях до змонтованого DVD GVK_VAULT")
    p_sync.add_argument("--gvk-home", default=DEFAULT_GVK_HOME, help="Шлях до кешу ~/.gvk_ai")
    p_sync.add_argument("--repo-root", default=DEFAULT_REPO_ROOT, help="Шлях до кореня репозиторію")
    p_sync.add_argument("--no-download", action="store_true", help="Не докачувати базову модель і llama-server з мережі")
    p_sync.add_argument("--skip-pip", action="store_true", help="Пропустити перевірку/встановлення pip")
    p_sync.add_argument("--pull", action="store_true", help="Оновити репозиторій через git pull перед синхронізацією")

    p_status = sub.add_parser("status", help="Показати статус кешу ~/.gvk_ai та активованих персонажів")
    p_status.add_argument("--gvk-home", default=DEFAULT_GVK_HOME, help="Шлях до кешу ~/.gvk_ai")
    p_status.add_argument("--repo-root", default=DEFAULT_REPO_ROOT, help="Шлях до кореня репозиторію")

    sub.add_parser("self-test", help="Запустити автономний офлайн self-test усіх функцій updater.py")

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
