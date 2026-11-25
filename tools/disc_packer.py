#!/usr/bin/env python3
"""
GVK Private AI — Генератор ключів, Шифратор контейнерів та Збирач диска (tools/disc_packer.py)

Реалізує криптографічний протокол GVK Private AI (Zero-Plaintext-At-Rest & Zero-Leak Metadata):
  1. Деривація 256-бітного майстер-ключа через Argon2id:
     - Сіль домену: GVK_PRIVATE_AI_DOMAIN_SALT_2026
     - Параметри: time_cost=4, memory_cost=262144 (256 МБ ОЗП), parallelism=4, hash_len=32
  2. Потокове шифрування та дешифрування великих файлів (AES-256-GCM) чанками по 64 МБ:
     - 36-байтний бінарний заголовок:
       [Magic "GVK1" (4B) | Base Nonce (12B) | Header Auth Tag (16B) | Chunk Size (4B)]
     - Кожен чанк шифрується власним унікальним Nonce (base_nonce XOR chunk_index)
       із прив'язкою порядкового номера чанка в AAD (Associated Authenticated Data),
       що унеможливлює повторне використання Nonce або перестановку чанків.
  3. Генерація апаратного підпису активації диска `origin.sig`:
     - HMAC-SHA256(persona_id, master_key) -> 32 байти.
  4. Пакування приватного архіву особистості `payload.tar.enc` напряму в пам'яті (io.BytesIO)
     та автоматична генерація публічного паспорта `manifest.json` із SHA-256 гешами.
  5. Збірка повної структури DVD-диска `GVK_VAULT` (DVD-R 4.7 ГБ, ~150–160 МБ,
     включно з лаунчерами `run.sh` / `run.bat`).

Приклади використання:
  # 1. Швидка самоперевірка всіх криптографічних функцій в ОЗП:
  python3 tools/disc_packer.py self-test

  # 2. Генерація master.key із секретної фрази:
  python3 tools/disc_packer.py keygen -o vault/master.key

  # 3. Шифрування файлу (наприклад, адаптера):
  python3 tools/disc_packer.py encrypt -k vault/master.key -i adapter.gguf -o adapter.gguf.enc

  # 4. Повна збірка папки персонажа (adapter.gguf.enc + payload.tar.enc + manifest.json + origin.sig):
  python3 tools/disc_packer.py pack-persona -k vault/master.key --id friend_1 \
      --adapter raw/friend_1.gguf --payload-dir raw/friend_1_payload/ -o build/friend_1/

  # 5. Збірка готового образу DVD-диска GVK_VAULT (публічна база докачується updater.py з хмари):
  python3 tools/disc_packer.py build-disc -k vault/master.key --id friend_1 \
      --persona-dir build/friend_1/ -o dist/GVK_VAULT_DISC_1/
"""

import argparse
import datetime
import getpass
import hashlib
import hmac
import io
import json
import os
import shutil
import stat
import struct
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Any, BinaryIO, Dict, List, Optional, Tuple

# =====================================================================
# КОНСТАНТИ КРИПТОГРАФІЧНОГО ПРОТОКОЛУ GVK
# =====================================================================
MAGIC_BYTES = b"GVK1"
HEADER_SIZE = 36  # 4B Magic + 12B Nonce + 16B Tag + 4B ChunkSize
DEFAULT_CHUNK_SIZE = 64 * 1024 * 1024  # 64 МБ
GCM_TAG_SIZE = 16
NONCE_SIZE = 12
KEY_SIZE = 32  # 256 біт

DOMAIN_SALT = b"GVK_PRIVATE_AI_DOMAIN_SALT_2026"
ARGON2_TIME_COST = 4
ARGON2_MEMORY_COST = 262144  # 256 МБ у КіБ (262144 KiB)
ARGON2_PARALLELISM = 4
MIN_PASSPHRASE_LEN = 24


# =====================================================================
# ЗАВАНТАЖЕННЯ КРИПТОГРАФІЧНИХ ПРИМІТИВІВ
# =====================================================================
def _get_aesgcm_class():
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        return AESGCM
    except ImportError:
        print(
            "\n[!] Помилка: Бібліотеку 'cryptography' не встановлено.\n"
            "    Встановіть залежності командою: pip3 install -r requirements.txt\n",
            file=sys.stderr,
        )
        sys.exit(1)


# =====================================================================
# 1. ДЕРИВАЦІЯ КЛЮЧА (Argon2id -> 32-byte master.key)
# =====================================================================
def derive_master_key(
    passphrase: str,
    salt: bytes = DOMAIN_SALT,
    time_cost: int = ARGON2_TIME_COST,
    memory_cost: int = ARGON2_MEMORY_COST,
    parallelism: int = ARGON2_PARALLELISM,
    allow_short: bool = False,
) -> bytes:
    """
    Обчислює 32-байтний (256-бітний) master.key із секретної фрази через Argon2id.
    Підтримує як `argon2-cffi`, так і вбудований у `cryptography` Argon2id.
    """
    if not allow_short and len(passphrase.strip()) < MIN_PASSPHRASE_LEN:
        raise ValueError(
            f"Секретна фраза занадто коротка ({len(passphrase.strip())} симв.). "
            f"За специфікацією потрібно щонайменше {MIN_PASSPHRASE_LEN} символи "
            f"(або передайте --allow-short для тестового режиму)."
        )

    secret_bytes = passphrase.encode("utf-8")

    # Спроба 1: Через argon2-cffi (якщо встановлено)
    try:
        from argon2.low_level import Type, hash_secret_raw

        return hash_secret_raw(
            secret=secret_bytes,
            salt=salt,
            time_cost=time_cost,
            memory_cost=memory_cost,
            parallelism=parallelism,
            hash_len=KEY_SIZE,
            type=Type.ID,
        )
    except ImportError:
        pass

    # Спроба 2: Через cryptography (>= 42.0 має нативний OpenSSL Argon2id)
    try:
        from cryptography.hazmat.primitives.kdf.argon2 import Argon2id

        kdf = Argon2id(
            salt=salt,
            length=KEY_SIZE,
            iterations=time_cost,
            lanes=parallelism,
            memory_cost=memory_cost,
        )
        return kdf.derive(secret_bytes)
    except ImportError:
        print(
            "\n[!] Помилка: Для Argon2id потрібна бібліотека 'argon2-cffi' або 'cryptography>=42.0'.\n"
            "    Виконайте: pip3 install -r requirements.txt\n",
            file=sys.stderr,
        )
        sys.exit(1)


# =====================================================================
# 2. АПАРАТНИЙ ПІДПИС АКТИВАЦІЇ (origin.sig)
# =====================================================================
def generate_origin_sig(persona_id: str, master_key: bytes) -> bytes:
    """
    Генерує 32-байтний підпис фізичної активації учасника:
    origin.sig = HMAC-SHA256(persona_id, master_key)
    """
    if len(master_key) != KEY_SIZE:
        raise ValueError(f"Невірний розмір master_key: очікується {KEY_SIZE} байтів.")
    return hmac.new(master_key, persona_id.strip().encode("utf-8"), hashlib.sha256).digest()


def verify_origin_sig(persona_id: str, master_key: bytes, sig_bytes: bytes) -> bool:
    """Перевіряє валідність підпису origin.sig у захищеному від таймінг-атак режимі."""
    expected = generate_origin_sig(persona_id, master_key)
    return hmac.compare_digest(expected, sig_bytes)


# =====================================================================
# 3. ПОТОКОВЕ ШИФРУВАННЯ ТА ДЕШИФРУВАННЯ (AES-256-GCM, чанки 64 МБ)
# =====================================================================
def _derive_chunk_nonce(base_nonce: bytes, chunk_index: int) -> bytes:
    """
    Безпечно модифікує 12-байтний Nonce для чанка з номером chunk_index:
    виконує XOR останніх 8 байтів base_nonce із 64-бітним лічильником чанка (NIST SP 800-38D).
    Гарантує, що жодні два чанки у файлі ніколи не матимуть однаковий Nonce.
    """
    if len(base_nonce) != NONCE_SIZE:
        raise ValueError("Base nonce повинен мати довжину 12 байтів.")
    prefix = base_nonce[:4]
    counter_part = struct.unpack("<Q", base_nonce[4:])[0] ^ chunk_index
    return prefix + struct.pack("<Q", counter_part)


def _compute_header_tag(base_nonce: bytes, chunk_size: int, master_key: bytes) -> bytes:
    """
    Обчислює 16-байтний тег заголовка (Header Auth Tag).
    Дозволяє миттєво (за 0.001 с) перевірити правильність ключа master.key
    ще до читання першого 64-мегабайтного блоку моделі!
    """
    msg = MAGIC_BYTES + base_nonce + struct.pack("<I", chunk_size)
    return hmac.new(master_key, msg, hashlib.sha256).digest()[:GCM_TAG_SIZE]


def encrypt_stream(
    src_stream: BinaryIO,
    dst_stream: BinaryIO,
    master_key: bytes,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    total_size: Optional[int] = None,
    label: str = "Шифрування",
) -> int:
    """
    Потоково шифрує дані з src_stream у dst_stream за стандартом GVK1 (.enc).
    Споживає в ОЗП лише розмір одного чанка (64 МБ), навіть для файлів 5–50 ГБ.
    """
    if len(master_key) != KEY_SIZE:
        raise ValueError(f"master_key має містити рівно {KEY_SIZE} байти.")

    AESGCM = _get_aesgcm_class()
    aesgcm = AESGCM(master_key)

    base_nonce = os.urandom(NONCE_SIZE)
    header_tag = _compute_header_tag(base_nonce, chunk_size, master_key)

    # Записуємо 36-байтний заголовок:
    # [Magic (4B) | File Nonce (12B) | Header GCM/Auth Tag (16B) | Chunk Size (4B)]
    header = MAGIC_BYTES + base_nonce + header_tag + struct.pack("<I", chunk_size)
    assert len(header) == HEADER_SIZE
    dst_stream.write(header)

    chunk_index = 0
    bytes_processed = 0
    start_t = time.time()

    while True:
        plain_chunk = src_stream.read(chunk_size)
        if not plain_chunk:
            break

        c_len = len(plain_chunk)
        chunk_nonce = _derive_chunk_nonce(base_nonce, chunk_index)
        # Прив'язуємо індекс і розмір чанка до Associated Data (захист від підміни порядку чанків)
        aad = MAGIC_BYTES + struct.pack("<QI", chunk_index, c_len)
        encrypted_chunk = aesgcm.encrypt(chunk_nonce, plain_chunk, aad)

        # Записуємо: [довжина відкритого чанка (4B uint32)] + [шифротекст + 16B GCM Tag]
        dst_stream.write(struct.pack("<I", c_len))
        dst_stream.write(encrypted_chunk)

        bytes_processed += c_len
        chunk_index += 1

        if total_size and total_size > 0:
            pct = (bytes_processed / total_size) * 100.0
            elapsed = max(time.time() - start_t, 1e-3)
            speed_mb = (bytes_processed / (1024 * 1024)) / elapsed
            print(
                f"\r[*] {label}: {pct:5.1f}% ({bytes_processed // (1024 * 1024)} / "
                f"{total_size // (1024 * 1024)} МБ) [{speed_mb:.1f} МБ/с]",
                end="",
                flush=True,
            )

    if total_size and total_size > 0:
        print()
    return bytes_processed


def decrypt_stream(
    src_stream: BinaryIO,
    dst_stream: BinaryIO,
    master_key: bytes,
    label: str = "Дешифрування",
) -> int:
    """
    Потоково перевіряє та розшифровує контейнер .enc із src_stream у dst_stream.
    Якщо ключ невірний або пошкоджено хоча б 1 біт — викидає ValueError.
    """
    if len(master_key) != KEY_SIZE:
        raise ValueError(f"master_key має містити рівно {KEY_SIZE} байти.")

    AESGCM = _get_aesgcm_class()
    aesgcm = AESGCM(master_key)

    header = src_stream.read(HEADER_SIZE)
    if len(header) != HEADER_SIZE:
        raise ValueError("Пошкоджений файл .enc: заголовок менший за 36 байтів.")

    magic = header[:4]
    if magic != MAGIC_BYTES:
        raise ValueError(f"Невірна сигнатура файлу: очікується {MAGIC_BYTES!r}, отримано {magic!r}.")

    base_nonce = header[4:16]
    header_tag = header[16:32]
    chunk_size = struct.unpack("<I", header[32:36])[0]

    # Миттєва перевірка ключа за тегом заголовка!
    expected_hdr_tag = _compute_header_tag(base_nonce, chunk_size, master_key)
    if not hmac.compare_digest(header_tag, expected_hdr_tag):
        raise ValueError("Помилка автентифікації ключа: невірний master.key або пошкоджений заголовок!")

    chunk_index = 0
    bytes_out = 0

    while True:
        len_bytes = src_stream.read(4)
        if not len_bytes:
            break
        if len(len_bytes) != 4:
            raise ValueError(f"Несподіваний кінець файлу на чанку #{chunk_index}.")

        c_len = struct.unpack("<I", len_bytes)[0]
        if c_len == 0 or c_len > chunk_size:
            raise ValueError(f"Некоректний розмір чанка #{chunk_index}: {c_len} байтів.")

        expected_enc_len = c_len + GCM_TAG_SIZE
        encrypted_chunk = src_stream.read(expected_enc_len)
        if len(encrypted_chunk) != expected_enc_len:
            raise ValueError(f"Обрізаний шифротекст на чанку #{chunk_index}.")

        chunk_nonce = _derive_chunk_nonce(base_nonce, chunk_index)
        aad = MAGIC_BYTES + struct.pack("<QI", chunk_index, c_len)

        try:
            plain_chunk = aesgcm.decrypt(chunk_nonce, encrypted_chunk, aad)
        except Exception as exc:
            raise ValueError(
                f"Порушення цілісності AES-GCM у чанку #{chunk_index}! Дані пошкоджено або підмінено."
            ) from exc

        dst_stream.write(plain_chunk)
        bytes_out += len(plain_chunk)
        chunk_index += 1

    return bytes_out


def encrypt_file(
    input_path: str,
    output_path: str,
    master_key: bytes,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> str:
    """Шифрує файл на диску та повертає його SHA-256 геш."""
    total_size = os.path.getsize(input_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(input_path, "rb") as f_in, open(output_path, "wb") as f_out:
        encrypt_stream(
            f_in,
            f_out,
            master_key=master_key,
            chunk_size=chunk_size,
            total_size=total_size,
            label=f"Шифрування {os.path.basename(input_path)}",
        )
    return compute_sha256(output_path)


def decrypt_file(input_path: str, output_path: str, master_key: bytes) -> int:
    """Розшифровує .enc файл у вказаний шлях (для тестів або RAM-диска)."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(input_path, "rb") as f_in, open(output_path, "wb") as f_out:
        return decrypt_stream(f_in, f_out, master_key=master_key)


# =====================================================================
# 4. ПАКУВАННЯ АРХІВУ ОСОБИСТОСТІ (payload.tar.enc) ТА МАНІФЕСТУ
# =====================================================================
def compute_sha256(file_path: str) -> str:
    """Обчислює SHA-256 геш файлу потоково."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            block = f.read(4 * 1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def pack_payload_encrypted(
    source_dir: str,
    output_enc_path: str,
    master_key: bytes,
    roster_path: Optional[str] = None,
) -> str:
    """
    Збирає вміст папки особистості (persona_config.json, prompt.txt, portraits/,
    memory.faiss, tools.py) у стиснутий TAR.GZ архів прямо в оперативній пам'яті
    та одразу шифрує його у файл `payload.tar.enc` (Zero-Plaintext-At-Rest).
    Якщо всередині `source_dir` немає `roster.json`, автоматично підхоплює спільний
    `roster_path` або `../roster.json` (наприклад `personas/roster.json`).
    Повертає SHA-256 геш створеного `payload.tar.enc`.
    """
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"Папку корисного навантаження не знайдено: {source_dir}")

    def _tar_filter(tarinfo: tarfile.TarInfo) -> Optional[tarfile.TarInfo]:
        parts = [p for p in tarinfo.name.replace("\\", "/").split("/") if p and p != "."]
        if any(p.startswith(".") or p in ("__pycache__", "raw") for p in parts):
            return None
        base = parts[-1].lower() if parts else ""
        if base.endswith((".psd", ".xcf", ".pyc", ".pyo")):
            return None
        return tarinfo

    tar_buffer = io.BytesIO()
    has_roster = False
    with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
        for entry in sorted(os.listdir(source_dir)):
            if entry.startswith(".") or entry in ("__pycache__", "raw"):
                continue
            if entry == "roster.json":
                has_roster = True
            full_path = os.path.join(source_dir, entry)
            tar.add(full_path, arcname=entry, filter=_tar_filter)

        if not has_roster:
            shared_roster = (
                roster_path
                if roster_path
                else os.path.join(os.path.dirname(os.path.abspath(source_dir)), "roster.json")
            )
            if shared_roster and os.path.isfile(shared_roster):
                tar.add(shared_roster, arcname="roster.json", filter=_tar_filter)

    tar_bytes = tar_buffer.getvalue()
    tar_stream = io.BytesIO(tar_bytes)

    os.makedirs(os.path.dirname(os.path.abspath(output_enc_path)), exist_ok=True)
    with open(output_enc_path, "wb") as f_out:
        encrypt_stream(
            tar_stream,
            f_out,
            master_key=master_key,
            total_size=len(tar_bytes),
            label=f"Пакування {os.path.basename(output_enc_path)}",
        )

    return compute_sha256(output_enc_path)


def unpack_payload_decrypted(input_enc_path: str, target_ram_dir: str, master_key: bytes) -> None:
    """
    Розшифровує `payload.tar.enc` у пам'яті (io.BytesIO) і безпечно розпаковує
    всі файли персонажа в цільову папку (наприклад, на змонтований RAM-диск).
    """
    tar_buffer = io.BytesIO()
    with open(input_enc_path, "rb") as f_in:
        decrypt_stream(f_in, tar_buffer, master_key=master_key)

    tar_buffer.seek(0)
    os.makedirs(target_ram_dir, exist_ok=True)
    with tarfile.open(fileobj=tar_buffer, mode="r:gz") as tar:
        # Захист від Path Traversal у назвах файлів архіву
        for member in tar.getmembers():
            member_path = os.path.abspath(os.path.join(target_ram_dir, member.name))
            if not member_path.startswith(os.path.abspath(target_ram_dir)):
                raise ValueError(f"Небезпечний шлях в архіві: {member.name}")
        if "filter" in tar.extractall.__code__.co_varnames:
            tar.extractall(path=target_ram_dir, filter="data")
        else:
            tar.extractall(path=target_ram_dir)


def write_public_manifest(
    output_dir: str,
    persona_id: str,
    adapter_sha256: str,
    payload_sha256: str,
    version: str = "1.0.0",
    min_core_version: str = "1.0.0",
) -> str:
    """
    Створює публічний Zero-Leak файл `manifest.json` із SHA-256 гешами зашифрованих контейнерів.
    """
    manifest = {
        "persona_id": persona_id,
        "version": version,
        "min_core_version": min_core_version,
        "files": {
            "adapter": "adapter.gguf.enc",
            "payload": "payload.tar.enc",
        },
        "sha256": {
            "adapter": adapter_sha256,
            "payload": payload_sha256,
        },
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return manifest_path


# =====================================================================
# 5. ЗБІРКА СТРУКТУРИ DVD-ДИСКА (GVK_VAULT)
# =====================================================================
RUN_SH_TEMPLATE = r"""#!/usr/bin/env bash
# =====================================================================
# GVK Private AI — Універсальний Zero-Install лаунчер (macOS / Linux)
# Працює з локального репо або з DVD GVK_VAULT (навіть без Git та Python).
# =====================================================================
set -e

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GVK_HOME="${GVK_HOME:-${HOME}/.gvk_ai}"
REPO_URL="https://github.com/Holii-1/Private-AI-Companion-GVK-.git"
REPO_TARBALL_URL="https://github.com/Holii-1/Private-AI-Companion-GVK-/archive/refs/heads/main.tar.gz"
PORTABLE_PY_TAG="20250317"
PORTABLE_PY_VER="3.12.9"

CMD_NAME="${1:-}"
DISC_ARGS=()

has_real_git() {
    if [ "$(uname -s)" = "Darwin" ]; then
        # На чистій macOS /usr/bin/git — заглушка, що відкриває вікно Xcode CLT
        xcode-select -p >/dev/null 2>&1 || return 1
    fi
    command -v git >/dev/null 2>&1
}

export_portable_tk_env() {
    local py_root="${GVK_HOME}/python"
    if [ -d "${py_root}/lib/tcl8.6" ] && [ -z "${TCL_LIBRARY:-}" ]; then
        export TCL_LIBRARY="${py_root}/lib/tcl8.6"
    fi
    if [ -d "${py_root}/lib/tk8.6" ] && [ -z "${TK_LIBRARY:-}" ]; then
        export TK_LIBRARY="${py_root}/lib/tk8.6"
    fi
}

seed_repo_from_disc_bundle() {
    local target_dir="$1"
    if [ -f "${SCRIPT_DIR}/core_bundle/core/main.py" ] && [ -f "${SCRIPT_DIR}/core_bundle/updater.py" ]; then
        mkdir -p "${target_dir}"
        cp -R "${SCRIPT_DIR}/core_bundle/." "${target_dir}/"
        chmod -R u+w "${target_dir}" 2>/dev/null || true
        echo " [OK] Локальне ядро з DVD (core_bundle) розгорнуто у: ${target_dir}"
        return 0
    fi
    return 1
}

sync_repo_via_tarball() {
    local target_dir="$1"
    local tmp_dir archive extracted
    if ! command -v curl >/dev/null 2>&1 || ! command -v tar >/dev/null 2>&1; then
        if [ -f "${target_dir}/core/main.py" ]; then
            echo " [!] curl/tar недоступні — використовуємо локальну копію ядра."
            return 0
        fi
        echo " [X] Помилка: не знайдено git або curl+tar для завантаження ядра." >&2
        return 1
    fi

    tmp_dir="$(mktemp -d "${GVK_HOME}/repo_dl_XXXXXX")"
    archive="${tmp_dir}/repo.tar.gz"
    echo " [*] Завантаження ядра GVK Private AI (без git)..."
    if ! curl -fSL --retry 2 --connect-timeout 15 -o "${archive}" "${REPO_TARBALL_URL}"; then
        rm -rf "${tmp_dir}"
        if [ -f "${target_dir}/core/main.py" ]; then
            echo " [!] Немає мережі — використовуємо кешовану копію ядра."
            return 0
        fi
        echo " [X] Не вдалося завантажити ядро з GitHub. Перевірте інтернет на 1-й запуск." >&2
        return 1
    fi

    if ! tar -xzf "${archive}" -C "${tmp_dir}"; then
        rm -rf "${tmp_dir}"
        echo " [X] Помилка розпакування архіву репозиторію." >&2
        return 1
    fi

    extracted="$(find "${tmp_dir}" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
    if [ -z "${extracted}" ] || [ ! -f "${extracted}/updater.py" ]; then
        rm -rf "${tmp_dir}"
        echo " [X] Некоректний архів репозиторію (відсутній updater.py)." >&2
        return 1
    fi

    mkdir -p "${target_dir}"
    cp -R "${extracted}/." "${target_dir}/"
    rm -rf "${tmp_dir}"
    echo " [✓] Ядро GVK Private AI оновлено у: ${target_dir}"
    return 0
}

is_good_python() {
    local py_bin="$1"
    [ -n "${py_bin}" ] && [ -x "${py_bin}" ] || return 1
    if [[ "${py_bin}" == "${GVK_HOME}/python/"* ]]; then
        export_portable_tk_env
    fi
    "${py_bin}" -c "import sys, tkinter as tk; raise SystemExit(0 if sys.version_info >= (3, 10) and float(tk.TkVersion) >= 8.6 else 1)" >/dev/null 2>&1
}

bootstrap_portable_python() {
    local os_name arch_name triple py_url tmp_dir archive
    os_name="$(uname -s)"
    arch_name="$(uname -m)"
    triple=""

    if [ "${os_name}" = "Darwin" ]; then
        case "${arch_name}" in
            arm64|aarch64) triple="aarch64-apple-darwin" ;;
            x86_64)        triple="x86_64-apple-darwin" ;;
        esac
    elif [ "${os_name}" = "Linux" ]; then
        case "${arch_name}" in
            x86_64|amd64)  triple="x86_64-unknown-linux-gnu" ;;
            aarch64|arm64) triple="aarch64-unknown-linux-gnu" ;;
        esac
    fi

    if [ -z "${triple}" ]; then
        echo " [X] Непідтримувана архітектура (${os_name}/${arch_name}) для автозавантаження Python. Встановіть Python 3.12 вручну." >&2
        return 1
    fi
    if ! command -v curl >/dev/null 2>&1 || ! command -v tar >/dev/null 2>&1; then
        echo " [X] Потрібен curl + tar для автозавантаження портативного Python 3.12." >&2
        return 1
    fi

    mkdir -p "${GVK_HOME}"
    py_url="https://github.com/astral-sh/python-build-standalone/releases/download/${PORTABLE_PY_TAG}/cpython-${PORTABLE_PY_VER}%2B${PORTABLE_PY_TAG}-${triple}-install_only_stripped.tar.gz"
    tmp_dir="$(mktemp -d "${GVK_HOME}/py_dl_XXXXXX")"
    archive="${tmp_dir}/python.tar.gz"

    local usb_py_cand=""
    for cand_f in "${SCRIPT_DIR}/python_portable/"*"${triple}"*.tar.gz; do
        if [ -f "${cand_f}" ]; then
            usb_py_cand="${cand_f}"
            break
        fi
    done

    if [ -n "${usb_py_cand}" ]; then
        echo " [*] [USB-РЕЖИМ] Встановлення портативного Python ${PORTABLE_PY_VER} із носія: $(basename "${usb_py_cand}")..." >&2
        cp "${usb_py_cand}" "${archive}"
    else
        echo " [*] Python 3.10+ (з Tk >= 8.6) не знайдено — завантажуємо портативний Python ${PORTABLE_PY_VER} (~35 МБ)..." >&2
        if ! curl -fSL --retry 2 --connect-timeout 15 -o "${archive}" "${py_url}"; then
            rm -rf "${tmp_dir}"
            echo " [X] Не вдалося завантажити портативний Python. Перевірте інтернет або встановіть Python 3.12 вручну." >&2
            return 1
        fi
    fi

    if ! tar -xzf "${archive}" -C "${tmp_dir}" || [ ! -x "${tmp_dir}/python/bin/python3" ]; then
        rm -rf "${tmp_dir}"
        echo " [X] Помилка розпакування портативного Python." >&2
        return 1
    fi

    rm -rf "${GVK_HOME}/python"
    mv "${tmp_dir}/python" "${GVK_HOME}/python"
    rm -rf "${tmp_dir}"
    export_portable_tk_env
    echo " [✓] Портативний Python ${PORTABLE_PY_VER} встановлено у: ${GVK_HOME}/python" >&2
    printf '%s\n' "${GVK_HOME}/python/bin/python3"
    return 0
}

resolve_any_python() {
    local cand
    for cand in \
        "${GVK_HOME}/python/bin/python3" \
        "/opt/homebrew/bin/python3.12" \
        "/opt/homebrew/bin/python3.11" \
        "/opt/homebrew/bin/python3.13" \
        "/opt/homebrew/bin/python3" \
        "/usr/local/bin/python3.12" \
        "/usr/local/bin/python3" \
        "$(command -v python3 2>/dev/null || true)" \
        "${REPO_DIR}/.venv/bin/python3" \
        "${GVK_HOME}/.venv/bin/python3"; do
        if [ -n "${cand}" ] && [ -x "${cand}" ]; then
            printf '%s\n' "${cand}"
            return 0
        fi
    done
    return 1
}

resolve_system_python() {
    local cand
    for cand in \
        "${GVK_HOME}/python/bin/python3" \
        "/opt/homebrew/bin/python3.12" \
        "/opt/homebrew/bin/python3.11" \
        "/opt/homebrew/bin/python3.13" \
        "/opt/homebrew/bin/python3" \
        "/usr/local/bin/python3.12" \
        "/usr/local/bin/python3.11" \
        "/usr/local/bin/python3" \
        "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3" \
        "/Library/Frameworks/Python.framework/Versions/Current/bin/python3" \
        "$(command -v python3.12 2>/dev/null || true)" \
        "$(command -v python3.11 2>/dev/null || true)" \
        "$(command -v python3 2>/dev/null || true)" \
        "${REPO_DIR}/.venv/bin/python3" \
        "${GVK_HOME}/.venv/bin/python3"; do
        if is_good_python "${cand}"; then
            printf '%s\n' "${cand}"
            return 0
        fi
    done
    bootstrap_portable_python
}

resolve_venv_python() {
    local cand
    for cand in \
        "${REPO_DIR}/.venv/bin/python3" \
        "${GVK_HOME}/.venv/bin/python3"; do
        if [ -x "${cand}" ]; then
            printf '%s\n' "${cand}"
            return 0
        fi
    done
    resolve_system_python
}

if [ -f "${SCRIPT_DIR}/core/main.py" ] && [ -f "${SCRIPT_DIR}/updater.py" ]; then
    REPO_DIR="${SCRIPT_DIR}"
    echo "===================================================================="
    echo " [*] GVK Private AI — Локальний запуск: ${REPO_DIR}"
    echo "===================================================================="
    if [ -f "${SCRIPT_DIR}/persona/manifest.json" ]; then
        DISC_ARGS=(--disc-path "${SCRIPT_DIR}")
    fi
    if [ "${CMD_NAME}" != "self-test" ] && [ "${CMD_NAME}" != "--self-test" ] && [ "${CMD_NAME}" != "status" ]; then
        if [ -d "${REPO_DIR}/.git" ] && has_real_git; then
            if [ -z "$(git -C "${REPO_DIR}" status --porcelain 2>/dev/null || true)" ]; then
                echo " [*] Перевірка оновлень коду..."
                git -C "${REPO_DIR}" pull --ff-only || echo " [!] Не вдалося виконати git pull (офлайн), продовжуємо..."
            fi
        fi
    fi
else
    REPO_DIR="${GVK_HOME}/repo"
    DISC_ARGS=(--disc-path "${SCRIPT_DIR}")
    MEDIA_MODE="DVD"
    if [ -d "${SCRIPT_DIR}/base" ] || [ -d "${SCRIPT_DIR}/bin" ] || [ -f "${SCRIPT_DIR}/vault/media_mode.json" ]; then
        MEDIA_MODE="USB"
    fi
    echo "===================================================================="
    echo " [*] GVK_VAULT Виявлено: ${SCRIPT_DIR} [Режим: ${MEDIA_MODE}]"
    echo "===================================================================="
    mkdir -p "${GVK_HOME}"
    if [ ! -f "${REPO_DIR}/core/main.py" ]; then
        seed_repo_from_disc_bundle "${REPO_DIR}" || true
    fi
    if [ "${MEDIA_MODE}" = "USB" ] && [ -f "${REPO_DIR}/core/main.py" ]; then
        seed_repo_from_disc_bundle "${REPO_DIR}" || true
    elif has_real_git && { [ -d "${REPO_DIR}/.git" ] || [ ! -f "${REPO_DIR}/core/main.py" ]; }; then
        if [ ! -d "${REPO_DIR}/.git" ]; then
            echo " [*] Перший запуск: клонування ядра GVK Private AI..."
            git clone "${REPO_URL}" "${REPO_DIR}" || sync_repo_via_tarball "${REPO_DIR}"
        else
            echo " [*] Перевірка оновлень ядра та personas_encrypted/..."
            git -C "${REPO_DIR}" pull --ff-only || echo " [!] Не вдалося виконати git pull (офлайн або локальні зміни), продовжуємо..."
        fi
    else
        sync_repo_via_tarball "${REPO_DIR}"
    fi
    # Якщо на носії core_bundle свіжіший за завантажене з GitHub репо — поверх накладаємо файли з носія
    if [ -f "${SCRIPT_DIR}/core_bundle/core/main.py" ] && { [ ! -f "${REPO_DIR}/core/main.py" ] || [ "${SCRIPT_DIR}/core_bundle/core/main.py" -nt "${REPO_DIR}/core/main.py" ]; }; then
        seed_repo_from_disc_bundle "${REPO_DIR}" || true
    fi
fi

if [ "${CMD_NAME}" = "status" ]; then
    shift
    STAT_PY="$(resolve_any_python)" || {
        echo " [!] Python ще не встановлено (буде завантажено автоматично при першому запуску ./run.sh)." >&2
        exit 1
    }
    exec "${STAT_PY}" "${REPO_DIR}/updater.py" status --repo-root "${REPO_DIR}" "$@"
fi

SYS_PY="$(resolve_system_python)" || {
    echo " [X] Помилка: не вдалося знайти або завантажити Python 3.10+ (Tk >= 8.6)." >&2
    exit 1
}
export_portable_tk_env

if [ "${CMD_NAME}" = "self-test" ] || [ "${CMD_NAME}" = "--self-test" ]; then
    shift
    VENV_PY="$(resolve_venv_python)"
    exec "${VENV_PY}" "${REPO_DIR}/core/main.py" self-test "$@"
fi

if [ ${#DISC_ARGS[@]} -gt 0 ]; then
    "${SYS_PY}" "${REPO_DIR}/updater.py" sync --repo-root "${REPO_DIR}" "${DISC_ARGS[@]}"
else
    "${SYS_PY}" "${REPO_DIR}/updater.py" sync --repo-root "${REPO_DIR}"
fi

VENV_PY="$(resolve_venv_python)"

if [ -f "${REPO_DIR}/core/main.py" ]; then
    if [ ${#DISC_ARGS[@]} -gt 0 ]; then
        exec "${VENV_PY}" "${REPO_DIR}/core/main.py" "${DISC_ARGS[@]}" "$@"
    else
        exec "${VENV_PY}" "${REPO_DIR}/core/main.py" "$@"
    fi
else
    echo " [!] Модуль core/main.py ще не створено. Статус кешу:"
    exec "${SYS_PY}" "${REPO_DIR}/updater.py" status --repo-root "${REPO_DIR}"
fi
"""

RUN_COMMAND_TEMPLATE = r"""#!/usr/bin/env bash
cd "$(dirname "$0")"
exec bash ./run.sh "$@"
"""

RUN_BAT_TEMPLATE = r"""@echo off
REM =====================================================================
REM GVK Private AI — Універсальний Zero-Install лаунчер (Windows)
REM Працює з локального репо або з DVD GVK_VAULT (навіть без Git та Python).
REM =====================================================================
chcp 65001 >nul 2>&1
setlocal EnableExtensions

if not defined OMP_NUM_THREADS set "OMP_NUM_THREADS=1"
if not defined MKL_NUM_THREADS set "MKL_NUM_THREADS=1"
if not defined TOKENIZERS_PARALLELISM set "TOKENIZERS_PARALLELISM=false"
if not defined KMP_DUPLICATE_LIB_OK set "KMP_DUPLICATE_LIB_OK=TRUE"

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

if not defined GVK_HOME set "GVK_HOME=%USERPROFILE%\.gvk_ai"
set "REPO_URL=https://github.com/Holii-1/Private-AI-Companion-GVK-.git"
set "REPO_TARBALL_URL=https://github.com/Holii-1/Private-AI-Companion-GVK-/archive/refs/heads/main.tar.gz"
set "PORTABLE_PY_URL=https://github.com/astral-sh/python-build-standalone/releases/download/20250317/cpython-3.12.9%%2B20250317-x86_64-pc-windows-msvc-install_only_stripped.tar.gz"
set "DISC_PATH="
set "CMD_NAME=%~1"

if not exist "%SCRIPT_DIR%\core\main.py" goto :dvd_mode
if not exist "%SCRIPT_DIR%\updater.py" goto :dvd_mode

set "REPO_DIR=%SCRIPT_DIR%"
echo ====================================================================
echo  [*] GVK Private AI — Local Launch: %REPO_DIR%
echo ====================================================================
if exist "%SCRIPT_DIR%\persona\manifest.json" set "DISC_PATH=%SCRIPT_DIR%"
if /I "%CMD_NAME%"=="self-test" goto :resolve_py
if /I "%CMD_NAME%"=="--self-test" goto :resolve_py
if /I "%CMD_NAME%"=="status" goto :resolve_py
if not exist "%REPO_DIR%\.git" goto :resolve_py

where git >nul 2>&1
if errorlevel 1 goto :resolve_py

set "DIRTY_FILES="
for /f "delims=" %%i in ('git -C "%REPO_DIR%" status --porcelain 2^>nul') do set "DIRTY_FILES=1"
if not defined DIRTY_FILES (
    echo  [*] Checking repository updates via git pull...
    git -C "%REPO_DIR%" pull --ff-only || echo  [!] git pull skipped ^(offline^).
)
goto :resolve_py

:dvd_mode
set "REPO_DIR=%GVK_HOME%\repo"
set "DISC_PATH=%SCRIPT_DIR%"
set "MEDIA_MODE=DVD"
if exist "%SCRIPT_DIR%\base\*.gguf" set "MEDIA_MODE=USB"
if exist "%SCRIPT_DIR%\bin\*" set "MEDIA_MODE=USB"
if exist "%SCRIPT_DIR%\vault\media_mode.json" set "MEDIA_MODE=USB"
echo ====================================================================
echo  [*] GVK_VAULT Media Detected: %SCRIPT_DIR% [Mode: %MEDIA_MODE%]
echo ====================================================================
if not exist "%GVK_HOME%" mkdir "%GVK_HOME%"

if exist "%SCRIPT_DIR%\core_bundle\core\main.py" (
    if not exist "%REPO_DIR%" mkdir "%REPO_DIR%"
    xcopy "%SCRIPT_DIR%\core_bundle\*" "%REPO_DIR%\" /E /I /H /Y /Q >nul
    attrib -R "%REPO_DIR%\*.*" /S /D >nul 2>&1
    echo  [OK] Seeded local GVK core from %MEDIA_MODE% ^(core_bundle^).
    if /I "%MEDIA_MODE%"=="USB" goto :resolve_py
)

where git >nul 2>&1
if errorlevel 1 goto :download_repo_tarball

if exist "%REPO_DIR%\.git" (
    echo  [*] Checking repository updates via git pull...
    git -C "%REPO_DIR%" pull --ff-only || echo  [!] git pull skipped ^(offline or local changes^).
    goto :resolve_py
)
if exist "%REPO_DIR%\core\main.py" goto :download_repo_tarball

echo  [*] First launch: cloning GVK Private AI core...
git clone "%REPO_URL%" "%REPO_DIR%"
if not errorlevel 1 goto :resolve_py
echo  [!] git clone failed, trying direct archive download...

:download_repo_tarball
where curl.exe >nul 2>&1
if errorlevel 1 goto :repo_no_tools
where tar.exe >nul 2>&1
if errorlevel 1 goto :repo_no_tools

set "REPO_TMP=%GVK_HOME%\repo_dl_tmp"
set "REPO_ARCHIVE=%GVK_HOME%\repo_dl.tar.gz"
if exist "%REPO_TMP%" rmdir /S /Q "%REPO_TMP%"
if exist "%REPO_ARCHIVE%" del /F /Q "%REPO_ARCHIVE%"
mkdir "%REPO_TMP%"

echo  [*] Downloading GVK Private AI core archive ^(no Git required^)...
curl.exe -fSL --retry 2 --connect-timeout 15 -o "%REPO_ARCHIVE%" "%REPO_TARBALL_URL%"
if errorlevel 1 goto :repo_dl_failed

tar.exe -xzf "%REPO_ARCHIVE%" -C "%REPO_TMP%"
if errorlevel 1 goto :repo_dl_failed

set "EXTRACTED_DIR="
for /d %%D in ("%REPO_TMP%\*") do set "EXTRACTED_DIR=%%~fD"
if not defined EXTRACTED_DIR goto :repo_dl_failed
if not exist "%EXTRACTED_DIR%\updater.py" goto :repo_dl_failed

if not exist "%REPO_DIR%" mkdir "%REPO_DIR%"
xcopy "%EXTRACTED_DIR%\*" "%REPO_DIR%\" /E /I /H /Y /Q >nul
rmdir /S /Q "%REPO_TMP%" >nul 2>&1
del /F /Q "%REPO_ARCHIVE%" >nul 2>&1
echo  [OK] GVK Private AI core synced to: %REPO_DIR%
goto :resolve_py

:repo_dl_failed
if exist "%REPO_TMP%" rmdir /S /Q "%REPO_TMP%" >nul 2>&1
if exist "%REPO_ARCHIVE%" del /F /Q "%REPO_ARCHIVE%" >nul 2>&1
if exist "%REPO_DIR%\core\main.py" (
    echo  [!] Offline — using cached GVK core: %REPO_DIR%
    goto :resolve_py
)
echo  [X] Error: failed to download GVK core from GitHub. Internet is required on first launch.
call :fail 1
exit /b 1

:repo_no_tools
if exist "%REPO_DIR%\core\main.py" (
    echo  [!] Offline / no git or curl — using cached GVK core.
    goto :resolve_py
)
echo  [X] Error: neither git nor curl.exe/tar.exe found. Install Git for Windows and retry.
call :fail 1
exit /b 1

:resolve_py
set "SYS_PY="
set "SYS_FLAG="
if exist "%GVK_HOME%\python\tcl\tcl8.6" if not defined TCL_LIBRARY set "TCL_LIBRARY=%GVK_HOME%\python\tcl\tcl8.6"
if exist "%GVK_HOME%\python\tcl\tk8.6" if not defined TK_LIBRARY set "TK_LIBRARY=%GVK_HOME%\python\tcl\tk8.6"
if exist "%GVK_HOME%\python\lib\tcl8.6" if not defined TCL_LIBRARY set "TCL_LIBRARY=%GVK_HOME%\python\lib\tcl8.6"
if exist "%GVK_HOME%\python\lib\tk8.6" if not defined TK_LIBRARY set "TK_LIBRARY=%GVK_HOME%\python\lib\tk8.6"

if not exist "%GVK_HOME%\python\python.exe" goto :try_py_launcher
"%GVK_HOME%\python\python.exe" -c "import sys, tkinter as tk; raise SystemExit(0 if sys.version_info >= (3, 10) and float(tk.TkVersion) >= 8.6 else 1)" >nul 2>&1
if not errorlevel 1 (
    set "SYS_PY=%GVK_HOME%\python\python.exe"
    goto :sys_ready
)

:try_py_launcher
py -3 -c "import sys, tkinter as tk; raise SystemExit(0 if sys.version_info >= (3, 10) and float(tk.TkVersion) >= 8.6 else 1)" >nul 2>&1
if not errorlevel 1 (
    set "SYS_PY=py"
    set "SYS_FLAG=-3"
    goto :sys_ready
)

python -c "import sys, tkinter as tk; raise SystemExit(0 if sys.version_info >= (3, 10) and float(tk.TkVersion) >= 8.6 else 1)" >nul 2>&1
if not errorlevel 1 (
    set "SYS_PY=python"
    goto :sys_ready
)

if exist "%REPO_DIR%\.venv\Scripts\python.exe" (
    set "SYS_PY=%REPO_DIR%\.venv\Scripts\python.exe"
    goto :sys_ready
)
if exist "%GVK_HOME%\.venv\Scripts\python.exe" (
    set "SYS_PY=%GVK_HOME%\.venv\Scripts\python.exe"
    goto :sys_ready
)

:bootstrap_py
if /I "%CMD_NAME%"=="status" goto :py_missing
where tar.exe >nul 2>&1
if errorlevel 1 goto :py_missing

if not exist "%GVK_HOME%" mkdir "%GVK_HOME%"
set "PY_TMP=%GVK_HOME%\py_dl_tmp"
set "PY_ARCHIVE=%GVK_HOME%\py_portable.tar.gz"
if exist "%PY_TMP%" rmdir /S /Q "%PY_TMP%"
if exist "%PY_ARCHIVE%" del /F /Q "%PY_ARCHIVE%"
mkdir "%PY_TMP%"

set "USB_PY_ARCHIVE="
for %%F in ("%SCRIPT_DIR%\python_portable\*windows*.tar.gz") do set "USB_PY_ARCHIVE=%%~fF"
if defined USB_PY_ARCHIVE (
    echo  [*] [USB-MODE] Unpacking portable Python 3.12 from media: "%USB_PY_ARCHIVE%"...
    tar.exe -xzf "%USB_PY_ARCHIVE%" -C "%PY_TMP%"
    if not errorlevel 1 if exist "%PY_TMP%\python\python.exe" goto :py_extracted_ok
)

where curl.exe >nul 2>&1
if errorlevel 1 goto :py_missing

echo  [*] Python 3.10+ ^(with Tkinter^) not found — downloading portable Python 3.12 ^(~35 MB^)...
curl.exe -fSL --retry 2 --connect-timeout 15 -o "%PY_ARCHIVE%" "%PORTABLE_PY_URL%"
if errorlevel 1 goto :py_dl_failed

tar.exe -xzf "%PY_ARCHIVE%" -C "%PY_TMP%"
if errorlevel 1 goto :py_dl_failed
if not exist "%PY_TMP%\python\python.exe" goto :py_dl_failed

:py_extracted_ok

if exist "%GVK_HOME%\python" rmdir /S /Q "%GVK_HOME%\python"
move "%PY_TMP%\python" "%GVK_HOME%\python" >nul
rmdir /S /Q "%PY_TMP%" >nul 2>&1
del /F /Q "%PY_ARCHIVE%" >nul 2>&1

if exist "%GVK_HOME%\python\tcl\tcl8.6" set "TCL_LIBRARY=%GVK_HOME%\python\tcl\tcl8.6"
if exist "%GVK_HOME%\python\tcl\tk8.6" set "TK_LIBRARY=%GVK_HOME%\python\tcl\tk8.6"
if exist "%GVK_HOME%\python\lib\tcl8.6" if not defined TCL_LIBRARY set "TCL_LIBRARY=%GVK_HOME%\python\lib\tcl8.6"
if exist "%GVK_HOME%\python\lib\tk8.6" if not defined TK_LIBRARY set "TK_LIBRARY=%GVK_HOME%\python\lib\tk8.6"

set "SYS_PY=%GVK_HOME%\python\python.exe"
echo  [OK] Portable Python 3.12 installed to: %GVK_HOME%\python
goto :sys_ready

:py_dl_failed
if exist "%PY_TMP%" rmdir /S /Q "%PY_TMP%" >nul 2>&1
if exist "%PY_ARCHIVE%" del /F /Q "%PY_ARCHIVE%" >nul 2>&1

:py_missing
echo  [X] Error: Python 3.10+ not found and portable download failed. Check internet or install Python 3.12 from python.org.
call :fail 1
exit /b 1

:sys_ready
if /I "%CMD_NAME%"=="status" (
    "%SYS_PY%" %SYS_FLAG% "%REPO_DIR%\updater.py" status --repo-root "%REPO_DIR%"
    exit /b %errorlevel%
)

set "VENV_PY=%SYS_PY%"
set "VENV_FLAG=%SYS_FLAG%"
if exist "%REPO_DIR%\.venv\Scripts\python.exe" (
    set "VENV_PY=%REPO_DIR%\.venv\Scripts\python.exe"
    set "VENV_FLAG="
) else if exist "%GVK_HOME%\.venv\Scripts\python.exe" (
    set "VENV_PY=%GVK_HOME%\.venv\Scripts\python.exe"
    set "VENV_FLAG="
)

if /I "%CMD_NAME%"=="self-test" (
    "%VENV_PY%" %VENV_FLAG% "%REPO_DIR%\core\main.py" self-test
    exit /b %errorlevel%
)
if /I "%CMD_NAME%"=="--self-test" (
    "%VENV_PY%" %VENV_FLAG% "%REPO_DIR%\core\main.py" self-test
    exit /b %errorlevel%
)

if defined DISC_PATH (
    "%SYS_PY%" %SYS_FLAG% "%REPO_DIR%\updater.py" sync --repo-root "%REPO_DIR%" --disc-path "%DISC_PATH%"
) else (
    "%SYS_PY%" %SYS_FLAG% "%REPO_DIR%\updater.py" sync --repo-root "%REPO_DIR%"
)
if errorlevel 1 (
    call :fail %errorlevel%
    exit /b %errorlevel%
)

set "VENV_PY=%SYS_PY%"
set "VENV_FLAG=%SYS_FLAG%"
if exist "%REPO_DIR%\.venv\Scripts\python.exe" (
    set "VENV_PY=%REPO_DIR%\.venv\Scripts\python.exe"
    set "VENV_FLAG="
) else if exist "%GVK_HOME%\.venv\Scripts\python.exe" (
    set "VENV_PY=%GVK_HOME%\.venv\Scripts\python.exe"
    set "VENV_FLAG="
)

if exist "%REPO_DIR%\core\main.py" (
    if defined DISC_PATH (
        "%VENV_PY%" %VENV_FLAG% "%REPO_DIR%\core\main.py" --disc-path "%DISC_PATH%" %*
    ) else (
        "%VENV_PY%" %VENV_FLAG% "%REPO_DIR%\core\main.py" %*
    )
    if errorlevel 1 (
        call :fail %errorlevel%
        exit /b %errorlevel%
    )
    exit /b 0
)

echo  [!] core\main.py not found yet. Cache status:
"%SYS_PY%" %SYS_FLAG% "%REPO_DIR%\updater.py" status --repo-root "%REPO_DIR%"
exit /b %errorlevel%

:fail
echo.
if not defined GVK_NO_PAUSE pause
exit /b %~1
"""

def build_disc_tree(
    output_disc_dir: str,
    master_key: bytes,
    persona_id: str,
    persona_build_dir: str,
    base_model_enc_path: Optional[str] = None,
) -> None:
    """
    Формує повну структуру оптичного DVD-диска [DVD Disc: GVK_VAULT],
    готову до запису на DVD-R 4.7 ГБ (~150–160 МБ).
    За замовчуванням базова модель на диск не копіюється (вона докачується з хмари),
    але якщо передано `base_model_enc_path`, підтримується опційне додавання `base/`.
    """
    vault_dir = os.path.join(output_disc_dir, "vault")
    persona_dir = os.path.join(output_disc_dir, "persona")

    os.makedirs(vault_dir, exist_ok=True)
    os.makedirs(persona_dir, exist_ok=True)

    # 1. Запис 32-байтного ключа master.key
    key_path = os.path.join(vault_dir, "master.key")
    with open(key_path, "wb") as f:
        f.write(master_key)

    # 2. Запис origin.sig
    sig_bytes = generate_origin_sig(persona_id, master_key)
    with open(os.path.join(persona_dir, "origin.sig"), "wb") as f:
        f.write(sig_bytes)

    # 3. Копіювання manifest.json, adapter.gguf.enc, payload.tar.enc
    for fname in ("manifest.json", "adapter.gguf.enc", "payload.tar.enc"):
        src = os.path.join(persona_build_dir, fname)
        if not os.path.isfile(src):
            raise FileNotFoundError(f"Відсутній обов'язковий файл персонажа: {src}")
        shutil.copy2(src, os.path.join(persona_dir, fname))

    # 4. Опційне копіювання base_model.gguf.enc (лише якщо явно передано --base-enc)
    if base_model_enc_path:
        if not os.path.isfile(base_model_enc_path):
            raise FileNotFoundError(f"Зашифровану базову модель не знайдено: {base_model_enc_path}")
        base_dir = os.path.join(output_disc_dir, "base")
        os.makedirs(base_dir, exist_ok=True)
        shutil.copy2(base_model_enc_path, os.path.join(base_dir, "base_model.gguf.enc"))

    # 4b. Вбудовування автономної копії ядра (core_bundle/, ~350 КБ) на DVD,
    # щоб запуск із диска працював навіть без доступу до GitHub або до git push
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bundle_dir = os.path.join(output_disc_dir, "core_bundle")
    os.makedirs(os.path.join(bundle_dir, "core"), exist_ok=True)
    os.makedirs(os.path.join(bundle_dir, "tools"), exist_ok=True)
    for top_fname in ("updater.py", "requirements.txt"):
        top_src = os.path.join(repo_root, top_fname)
        if os.path.isfile(top_src):
            shutil.copy2(top_src, os.path.join(bundle_dir, top_fname))
    core_src_dir = os.path.join(repo_root, "core")
    if os.path.isdir(core_src_dir):
        for fname in sorted(os.listdir(core_src_dir)):
            if fname.endswith(".py"):
                shutil.copy2(os.path.join(core_src_dir, fname), os.path.join(bundle_dir, "core", fname))
    packer_src = os.path.join(repo_root, "tools", "disc_packer.py")
    if os.path.isfile(packer_src):
        shutil.copy2(packer_src, os.path.join(bundle_dir, "tools", "disc_packer.py"))

    # 5. Створення стартових скриптів run.sh та run.bat
    run_sh_path = os.path.join(output_disc_dir, "run.sh")
    with open(run_sh_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(RUN_SH_TEMPLATE)
    os.chmod(run_sh_path, os.stat(run_sh_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    run_cmd_path = os.path.join(output_disc_dir, "run.command")
    with open(run_cmd_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(RUN_COMMAND_TEMPLATE)
    os.chmod(run_cmd_path, os.stat(run_cmd_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    run_bat_path = os.path.join(output_disc_dir, "run.bat")
    with open(run_bat_path, "w", encoding="utf-8", newline="\r\n") as f:
        f.write(RUN_BAT_TEMPLATE)

    print(f"[✓] Образ DVD-диска GVK_VAULT для '{persona_id}' успішно зібрано у: {output_disc_dir}")


def fetch_usb_llama_bins(output_bins_dir: str, include_cuda_zip: bool = False) -> List[str]:
    """
    Автоматично завантажує кросплатформний набір релізних архівів `llama.cpp`
    для офлайн USB-дистрибутиву:
      - Windows Vulkan x64 (працює на AMD Radeon, Intel Arc/iGPU та NVIDIA)
      - Windows CPU x64 (універсальний фолбек)
      - macOS Apple Silicon ARM64 (Metal)
      - macOS Intel x64
      - Опційно (--include-cuda-zip): Windows CUDA 12.4 x64 + cudart DLLs
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from updater import _download_with_resume, _fetch_latest_llama_release_with_assets

    os.makedirs(output_bins_dir, exist_ok=True)
    release = _fetch_latest_llama_release_with_assets()
    tag = release.get("tag_name", "latest")
    assets = release.get("assets") or []
    print(f" [*] Отримано реліз llama.cpp: {tag} ({len(assets)} файлів)")

    wanted_predicates = [
        ("win-vulkan-x64", lambda n: n.startswith("llama-") and "win-vulkan-x64.zip" in n),
        ("win-cpu-x64", lambda n: n.startswith("llama-") and "win-cpu-x64.zip" in n),
        ("macos-arm64", lambda n: n.startswith("llama-") and "macos-arm64.tar.gz" in n),
        ("macos-x64", lambda n: n.startswith("llama-") and "macos-x64.tar.gz" in n),
    ]
    if include_cuda_zip:
        wanted_predicates.extend(
            [
                ("win-cuda-12.4-x64", lambda n: n.startswith("llama-") and "win-cuda-12.4-x64.zip" in n),
                ("cudart-win-cuda-12.4-x64", lambda n: n.startswith("cudart-llama-") and "win-cuda-12.4-x64.zip" in n),
            ]
        )

    downloaded_paths: List[str] = []
    for label, pred in wanted_predicates:
        matched = next((a for a in assets if pred(str(a.get("name", "")).lower())), None)
        if not matched:
            print(f" [!] Не знайдено асет для '{label}' у релізі {tag}")
            continue
        fname = matched["name"]
        url = matched["browser_download_url"]
        dst = os.path.join(output_bins_dir, fname)
        expected_size = int(matched.get("size") or 0)
        if os.path.isfile(dst) and (expected_size <= 0 or os.path.getsize(dst) == expected_size):
            print(f" [OK] Уже завантажено: {fname} ({os.path.getsize(dst) // (1024 * 1024)} МБ)")
            downloaded_paths.append(dst)
            continue
        _download_with_resume(url, dst, label=fname)
        downloaded_paths.append(dst)
    return downloaded_paths


def build_usb_tree(
    output_usb_dir: str,
    master_key: bytes,
    persona_id: str,
    persona_build_dir: str,
    base_model_path: Optional[str] = None,
    llama_bins_dir: Optional[str] = None,
    pascal_runtime_dir: Optional[str] = None,
    embeddings_dir: Optional[str] = None,
    fetch_bins: bool = False,
    include_cuda_zip: bool = False,
) -> None:
    """
    Формує повний офлайн USB-дистрибутив `GVK_VAULT [Mode: USB]` (під флешку 16–64 ГБ):
      1. Базова структура `vault/`, `persona/`, `core_bundle/`, `run.bat`, `run.sh`, `run.command`.
      2. Маркер `vault/media_mode.json` (`{"mode": "USB"}`).
      3. Публічна базова модель `base/Qwen2.5-14B-Instruct-Q4_K_M.gguf` (~9.0 ГБ), яка при першому
         запуску автоматично копіюється у `~/.gvk_ai/base/` без звернення до Hugging Face.
      4. Кросплатформні бінарники `bin/`:
         - Архівні релізи для Windows (Vulkan AMD/Intel/NVIDIA + CPU x64) та macOS (ARM64 + Intel x64).
         - Готовий CUDA 12 / Pascal рантайм `bin/win-cuda-pascal/` із `cublas64_12.dll`, `cudart64_12.dll`, `ggml-cuda.dll`.
      5. Опційно: офлайн RAG-модель `embeddings/paraphrase-multilingual-MiniLM-L12-v2/`.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from updater import DEFAULT_BASE_MODEL_FILENAME, DEFAULT_EMBEDDING_SHORT_NAME, _copy_file_with_progress

    build_disc_tree(
        output_disc_dir=output_usb_dir,
        master_key=master_key,
        persona_id=persona_id,
        persona_build_dir=persona_build_dir,
    )

    # 1. Записуємо явний маркер USB-режиму
    mode_meta = {
        "mode": "USB",
        "persona_id": persona_id,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(os.path.join(output_usb_dir, "vault", "media_mode.json"), "w", encoding="utf-8") as f:
        json.dump(mode_meta, f, indent=2, ensure_ascii=False)
        f.write("\n")

    # 2. Базова модель 14B -> <USB>/base/Qwen2.5-14B-Instruct-Q4_K_M.gguf
    resolved_base = base_model_path
    if not resolved_base:
        default_repo_base = os.path.join(repo_root, "base", DEFAULT_BASE_MODEL_FILENAME)
        if os.path.isfile(default_repo_base):
            resolved_base = default_repo_base

    if resolved_base and os.path.isfile(resolved_base):
        usb_base_dir = os.path.join(output_usb_dir, "base")
        os.makedirs(usb_base_dir, exist_ok=True)
        dst_gguf = os.path.join(usb_base_dir, DEFAULT_BASE_MODEL_FILENAME)
        if (
            os.path.isfile(dst_gguf)
            and os.path.getsize(dst_gguf) == os.path.getsize(resolved_base)
        ):
            print(f" [OK] Базова модель вже присутня на USB: {dst_gguf}")
        else:
            _copy_file_with_progress(
                resolved_base,
                dst_gguf,
                label=f"Запис {os.path.basename(resolved_base)} у USB-образ",
            )
    else:
        print(" [!] УВАГА: Базову модель не знайдено для копіювання в USB-образ (передайте --base-model).")

    # 3. Бінарники llama-server (Windows Vulkan/CPU + macOS ARM64/x64 + CUDA DLLs)
    usb_bin_dir = os.path.join(output_usb_dir, "bin")
    os.makedirs(usb_bin_dir, exist_ok=True)

    staging_bins = llama_bins_dir or os.path.join(repo_root, "base", "llama_bins")
    if fetch_bins:
        fetch_usb_llama_bins(staging_bins, include_cuda_zip=include_cuda_zip)

    if os.path.isdir(staging_bins):
        for fname in sorted(os.listdir(staging_bins)):
            src_f = os.path.join(staging_bins, fname)
            if os.path.isfile(src_f) and fname.lower().endswith((".zip", ".tar.gz", ".tgz")):
                dst_f = os.path.join(usb_bin_dir, fname)
                if not (os.path.isfile(dst_f) and os.path.getsize(dst_f) == os.path.getsize(src_f)):
                    shutil.copy2(src_f, dst_f)
                print(f" [✓] Додано архів llama-server у USB bin/: {fname}")

    # 3b. Копіювання готового CUDA 12 / Pascal рантайму (з cublas64_12.dll, cudart64_12.dll, ggml-cuda.dll)
    resolved_pascal = pascal_runtime_dir
    if not resolved_pascal:
        default_pascal = os.path.join(repo_root, "base", "llama-pascal-runtime")
        if os.path.isdir(default_pascal):
            resolved_pascal = default_pascal

    if resolved_pascal and os.path.isdir(resolved_pascal):
        usb_cuda_dir = os.path.join(usb_bin_dir, "win-cuda-pascal")
        os.makedirs(usb_cuda_dir, exist_ok=True)
        copied_cuda = 0
        for fname in sorted(os.listdir(resolved_pascal)):
            src_f = os.path.join(resolved_pascal, fname)
            if not os.path.isfile(src_f):
                continue
            fl = fname.lower()
            if fl == "llama-server.exe" or fl.endswith(".dll") or fl.startswith("license"):
                shutil.copy2(src_f, os.path.join(usb_cuda_dir, fname))
                copied_cuda += 1
        print(f" [✓] Додано Windows NVIDIA CUDA 12 рантайм у USB bin/win-cuda-pascal/ ({copied_cuda} файлів)")

    # 4. Опційна офлайн RAG-модель embeddings/paraphrase-multilingual-MiniLM-L12-v2
    resolved_emb = embeddings_dir
    if not resolved_emb:
        for cand_emb in (
            os.path.join(repo_root, "base", "embeddings", DEFAULT_EMBEDDING_SHORT_NAME),
            os.path.join(os.path.expanduser("~/.gvk_ai"), "embeddings", DEFAULT_EMBEDDING_SHORT_NAME),
        ):
            if os.path.isdir(cand_emb) and os.path.isfile(os.path.join(cand_emb, "config.json")):
                resolved_emb = cand_emb
                break

    if resolved_emb and os.path.isdir(resolved_emb):
        usb_emb_dst = os.path.join(output_usb_dir, "embeddings", DEFAULT_EMBEDDING_SHORT_NAME)
        shutil.copytree(resolved_emb, usb_emb_dst, dirs_exist_ok=True)
        print(f" [✓] Додано офлайн RAG-модель у USB embeddings/: {usb_emb_dst}")

    print(f"[✓] Повний USB-дистрибутив GVK_VAULT [Mode: USB] зібрано у: {output_usb_dir}")


# =====================================================================
# 5b. ГЕНЕРАЦІЯ ISO-9660 / JOLIET ОБРАЗУ (.ISO)
# =====================================================================
ISO_SECTOR_SIZE = 2048


def _both16(val: int) -> bytes:
    return struct.pack("<H", val & 0xFFFF) + struct.pack(">H", val & 0xFFFF)


def _both32(val: int) -> bytes:
    return struct.pack("<I", val & 0xFFFFFFFF) + struct.pack(">I", val & 0xFFFFFFFF)


def _sanitize_iso_name(name: str, is_dir: bool, used: set) -> bytes:
    """Формує валідне ECMA-119 ім'я файлу/каталогу без колізій."""
    if is_dir:
        base = "".join(ch if (ch.isascii() and (ch.isalnum() or ch == "_")) else "_" for ch in name.upper())
        stem = (base or "DIR")[:28]
        cand = stem
        n = 1
        while cand in used:
            suffix = f"_{n}"
            cand = stem[: max(1, 28 - len(suffix))] + suffix
            n += 1
        used.add(cand)
        return cand.encode("ascii")

    if "." in name:
        raw_base, raw_ext = name.rsplit(".", 1)
    else:
        raw_base, raw_ext = name, ""
    base = "".join(ch if (ch.isascii() and (ch.isalnum() or ch == "_")) else "_" for ch in raw_base.upper()) or "FILE"
    ext = "".join(ch if (ch.isascii() and (ch.isalnum() or ch == "_")) else "_" for ch in raw_ext.upper())
    ext = ext[:8]
    max_base = max(1, 28 - (len(ext) + 1 if ext else 0))
    base = base[:max_base]
    cand = f"{base}.{ext}" if ext else base
    n = 1
    while cand in used:
        suffix = f"_{n}"
        b = base[: max(1, max_base - len(suffix))] + suffix
        cand = f"{b}.{ext}" if ext else b
        n += 1
    used.add(cand)
    return f"{cand};1".encode("ascii")


def _encode_joliet_name(name: str, is_dir: bool) -> bytes:
    """Кодує ім'я в UCS-2 / UTF-16BE для Joliet SVD (макс. 64 символи)."""
    trimmed = name[:64]
    return trimmed.encode("utf-16-be")


def _iso_dir_record(
    file_id: bytes,
    extent_lba: int,
    data_len: int,
    is_dir: bool,
    rec_date_7: bytes,
) -> bytes:
    pad = b"\x00" if (len(file_id) % 2 == 0) else b""
    rec_len = 33 + len(file_id) + len(pad)
    flags = 0x02 if is_dir else 0x00
    return (
        bytes([rec_len, 0])
        + _both32(extent_lba)
        + _both32(data_len)
        + rec_date_7
        + bytes([flags, 0, 0])
        + _both16(1)
        + bytes([len(file_id)])
        + file_id
        + pad
    )


def _pack_dir_records_into_sectors(records: List[bytes]) -> bytes:
    out = bytearray()
    for rec in records:
        pos = len(out) % ISO_SECTOR_SIZE
        if pos != 0 and pos + len(rec) > ISO_SECTOR_SIZE:
            out.extend(b"\x00" * (ISO_SECTOR_SIZE - pos))
        out.extend(rec)
    if len(out) == 0:
        out.extend(b"\x00" * ISO_SECTOR_SIZE)
    else:
        rem = len(out) % ISO_SECTOR_SIZE
        if rem:
            out.extend(b"\x00" * (ISO_SECTOR_SIZE - rem))
    return bytes(out)


def _build_path_tables(
    dirs_info: List[Dict[str, Any]],
    lba_key: str,
    name_key: str,
) -> Tuple[bytes, bytes]:
    tbl_l = bytearray()
    tbl_m = bytearray()
    for idx, d in enumerate(dirs_info):
        name_bytes: bytes = d[name_key] if idx > 0 else b"\x00"
        pad = b"\x00" if (len(name_bytes) % 2 == 1) else b""
        parent_num = 1 if idx == 0 else (d["parent_idx"] + 1)
        lba = int(d[lba_key])
        tbl_l.extend(struct.pack("<BBIH", len(name_bytes), 0, lba, parent_num))
        tbl_l.extend(name_bytes + pad)
        tbl_m.extend(struct.pack(">BBIH", len(name_bytes), 0, lba, parent_num))
        tbl_m.extend(name_bytes + pad)
    return bytes(tbl_l), bytes(tbl_m)


def _pad_to_sector(data: bytes) -> bytes:
    rem = len(data) % ISO_SECTOR_SIZE
    if rem == 0:
        return data if data else (b"\x00" * ISO_SECTOR_SIZE)
    return data + b"\x00" * (ISO_SECTOR_SIZE - rem)


def _build_volume_descriptor(
    vd_type: int,
    volume_name: str,
    total_sectors: int,
    path_table_size: int,
    path_l_lba: int,
    path_m_lba: int,
    root_record: bytes,
    dec_time_17: bytes,
    is_joliet: bool = False,
) -> bytes:
    vd = bytearray(ISO_SECTOR_SIZE)
    vd[0] = vd_type
    vd[1:6] = b"CD001"
    vd[6] = 1
    if is_joliet:
        sys_id = "GVK PRIVATE AI".encode("utf-16-be")[:32].ljust(32, b"\x00")
        vol_id = volume_name.encode("utf-16-be")[:32].ljust(32, b"\x00")
        vd[88:91] = b"%/@"
    else:
        ascii_vol = "".join(ch if (ch.isascii() and (ch.isalnum() or ch == "_")) else "_" for ch in volume_name.upper())
        sys_id = b"GVK PRIVATE AI".ljust(32, b" ")
        vol_id = ascii_vol.encode("ascii")[:32].ljust(32, b" ")

    vd[8:40] = sys_id
    vd[40:72] = vol_id
    vd[80:88] = _both32(total_sectors)
    vd[120:124] = _both16(1)
    vd[124:128] = _both16(1)
    vd[128:132] = _both16(ISO_SECTOR_SIZE)
    vd[132:140] = _both32(path_table_size)
    struct.pack_into("<I", vd, 140, path_l_lba)
    struct.pack_into(">I", vd, 148, path_m_lba)
    vd[156 : 156 + len(root_record)] = root_record
    vd[813:830] = dec_time_17
    vd[830:847] = dec_time_17
    vd[881] = 1
    return bytes(vd)


def _write_iso9660_image(disc_dir: str, output_iso_path: str, volume_name: str = "GVK_VAULT") -> str:
    """Кросплатформний pure-Python збирач ISO-9660 + Joliet образу (.iso)."""
    now = datetime.datetime.now(datetime.timezone.utc)
    dec_time_17 = now.strftime("%Y%m%d%H%M%S00").encode("ascii") + b"\x00"
    rec_date_7 = bytes(
        [max(0, min(255, now.year - 1900)), now.month, now.day, now.hour, now.minute, now.second, 0]
    )

    dirs_info: List[Dict[str, Any]] = [
        {
            "abs_dir": os.path.abspath(disc_dir),
            "parent_idx": 0,
            "iso_name": b"\x00",
            "jol_name": b"\x00",
            "subdirs": [],
            "files": [],
        }
    ]

    idx = 0
    while idx < len(dirs_info):
        d = dirs_info[idx]
        abs_dir = d["abs_dir"]
        used_iso_names = set()
        entries = sorted(os.listdir(abs_dir))
        for name in entries:
            full = os.path.join(abs_dir, name)
            if os.path.isdir(full):
                iso_n = _sanitize_iso_name(name, is_dir=True, used=used_iso_names)
                jol_n = _encode_joliet_name(name, is_dir=True)
                child_idx = len(dirs_info)
                dirs_info.append(
                    {
                        "abs_dir": full,
                        "parent_idx": idx,
                        "iso_name": iso_n,
                        "jol_name": jol_n,
                        "subdirs": [],
                        "files": [],
                    }
                )
                d["subdirs"].append(child_idx)
            elif os.path.isfile(full):
                iso_n = _sanitize_iso_name(name, is_dir=False, used=used_iso_names)
                jol_n = _encode_joliet_name(name, is_dir=False)
                d["files"].append(
                    {
                        "abs_path": full,
                        "size": os.path.getsize(full),
                        "iso_name": iso_n,
                        "jol_name": jol_n,
                        "lba": 0,
                    }
                )
        idx += 1

    # Перший прохід: рахуємо точні розміри кожного каталогу у секторах (LBA у записах не впливає на розмір)
    for d in dirs_info:
        dummy_iso = [_iso_dir_record(b"\x00", 0, 0, True, rec_date_7), _iso_dir_record(b"\x01", 0, 0, True, rec_date_7)]
        dummy_jol = [_iso_dir_record(b"\x00", 0, 0, True, rec_date_7), _iso_dir_record(b"\x01", 0, 0, True, rec_date_7)]
        for c_idx in d["subdirs"]:
            child = dirs_info[c_idx]
            dummy_iso.append(_iso_dir_record(child["iso_name"], 0, 0, True, rec_date_7))
            dummy_jol.append(_iso_dir_record(child["jol_name"], 0, 0, True, rec_date_7))
        for f_info in d["files"]:
            dummy_iso.append(_iso_dir_record(f_info["iso_name"], 0, f_info["size"], False, rec_date_7))
            dummy_jol.append(_iso_dir_record(f_info["jol_name"], 0, f_info["size"], False, rec_date_7))
        d["iso_size"] = len(_pack_dir_records_into_sectors(dummy_iso))
        d["jol_size"] = len(_pack_dir_records_into_sectors(dummy_jol))
        d["iso_lba"] = 0
        d["jol_lba"] = 0

    pt_l_dummy, _ = _build_path_tables(dirs_info, "iso_lba", "iso_name")
    jpt_l_dummy, _ = _build_path_tables(dirs_info, "jol_lba", "jol_name")
    iso_pt_sectors = len(_pad_to_sector(pt_l_dummy)) // ISO_SECTOR_SIZE
    jol_pt_sectors = len(_pad_to_sector(jpt_l_dummy)) // ISO_SECTOR_SIZE

    cursor = 19  # 0..15 System, 16 PVD, 17 SVD, 18 Terminator
    iso_pt_l_lba = cursor
    cursor += iso_pt_sectors
    iso_pt_m_lba = cursor
    cursor += iso_pt_sectors
    jol_pt_l_lba = cursor
    cursor += jol_pt_sectors
    jol_pt_m_lba = cursor
    cursor += jol_pt_sectors

    for d in dirs_info:
        d["iso_lba"] = cursor
        cursor += d["iso_size"] // ISO_SECTOR_SIZE
    for d in dirs_info:
        d["jol_lba"] = cursor
        cursor += d["jol_size"] // ISO_SECTOR_SIZE

    ordered_files: List[Dict[str, Any]] = []
    for d in dirs_info:
        for f_info in d["files"]:
            f_info["lba"] = cursor
            sectors = max(1, (f_info["size"] + ISO_SECTOR_SIZE - 1) // ISO_SECTOR_SIZE) if f_info["size"] > 0 else 0
            cursor += sectors
            ordered_files.append(f_info)

    total_sectors = cursor

    # Другий прохід: формуємо фінальні таблиці шляхів та сектори каталогів із реальними LBA
    iso_pt_l_raw, iso_pt_m_raw = _build_path_tables(dirs_info, "iso_lba", "iso_name")
    jol_pt_l_raw, jol_pt_m_raw = _build_path_tables(dirs_info, "jol_lba", "jol_name")

    iso_dir_blocks: List[bytes] = []
    jol_dir_blocks: List[bytes] = []
    for d in dirs_info:
        parent = dirs_info[d["parent_idx"]]
        iso_recs = [
            _iso_dir_record(b"\x00", d["iso_lba"], d["iso_size"], True, rec_date_7),
            _iso_dir_record(b"\x01", parent["iso_lba"], parent["iso_size"], True, rec_date_7),
        ]
        jol_recs = [
            _iso_dir_record(b"\x00", d["jol_lba"], d["jol_size"], True, rec_date_7),
            _iso_dir_record(b"\x01", parent["jol_lba"], parent["jol_size"], True, rec_date_7),
        ]
        for c_idx in d["subdirs"]:
            child = dirs_info[c_idx]
            iso_recs.append(_iso_dir_record(child["iso_name"], child["iso_lba"], child["iso_size"], True, rec_date_7))
            jol_recs.append(_iso_dir_record(child["jol_name"], child["jol_lba"], child["jol_size"], True, rec_date_7))
        for f_info in d["files"]:
            iso_recs.append(_iso_dir_record(f_info["iso_name"], f_info["lba"], f_info["size"], False, rec_date_7))
            jol_recs.append(_iso_dir_record(f_info["jol_name"], f_info["lba"], f_info["size"], False, rec_date_7))
        iso_dir_blocks.append(_pack_dir_records_into_sectors(iso_recs))
        jol_dir_blocks.append(_pack_dir_records_into_sectors(jol_recs))

    root_iso_rec = _iso_dir_record(b"\x00", dirs_info[0]["iso_lba"], dirs_info[0]["iso_size"], True, rec_date_7)
    root_jol_rec = _iso_dir_record(b"\x00", dirs_info[0]["jol_lba"], dirs_info[0]["jol_size"], True, rec_date_7)

    pvd = _build_volume_descriptor(
        1, volume_name, total_sectors, len(iso_pt_l_raw), iso_pt_l_lba, iso_pt_m_lba, root_iso_rec, dec_time_17, False
    )
    svd = _build_volume_descriptor(
        2, volume_name, total_sectors, len(jol_pt_l_raw), jol_pt_l_lba, jol_pt_m_lba, root_jol_rec, dec_time_17, True
    )
    term = bytearray(ISO_SECTOR_SIZE)
    term[0] = 255
    term[1:6] = b"CD001"
    term[6] = 1

    out_abs = os.path.abspath(output_iso_path)
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    with open(out_abs, "wb") as iso_f:
        iso_f.write(b"\x00" * (16 * ISO_SECTOR_SIZE))
        iso_f.write(pvd)
        iso_f.write(svd)
        iso_f.write(term)
        iso_f.write(_pad_to_sector(iso_pt_l_raw))
        iso_f.write(_pad_to_sector(iso_pt_m_raw))
        iso_f.write(_pad_to_sector(jol_pt_l_raw))
        iso_f.write(_pad_to_sector(jol_pt_m_raw))
        for blk in iso_dir_blocks:
            iso_f.write(blk)
        for blk in jol_dir_blocks:
            iso_f.write(blk)
        for f_info in ordered_files:
            written = 0
            with open(f_info["abs_path"], "rb") as src_f:
                while True:
                    chunk = src_f.read(1024 * 1024)
                    if not chunk:
                        break
                    iso_f.write(chunk)
                    written += len(chunk)
            if written > 0:
                rem = written % ISO_SECTOR_SIZE
                if rem:
                    iso_f.write(b"\x00" * (ISO_SECTOR_SIZE - rem))

    return out_abs


def build_disc_iso(disc_dir: str, output_iso_path: str, volume_name: str = "GVK_VAULT") -> str:
    """Створює готовий ISO-образ DVD-диска з каталогу `disc_dir`."""
    disc_abs = os.path.abspath(disc_dir)
    if not os.path.isdir(disc_abs):
        raise FileNotFoundError(f"Папку диска не знайдено: {disc_dir}")

    out_abs = os.path.abspath(output_iso_path)
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)

    if sys.platform == "darwin" and shutil.which("hdiutil"):
        try:
            subprocess.run(
                [
                    "hdiutil",
                    "makehybrid",
                    "-iso",
                    "-joliet",
                    "-default-volume-name",
                    volume_name,
                    "-o",
                    out_abs,
                    disc_abs,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            if os.path.isfile(out_abs):
                print(f"[✓] ISO-образ (hdiutil) створено: {out_abs}")
                return out_abs
        except Exception:
            pass

    for tool in ("xorrisofs", "genisoimage", "mkisofs"):
        if shutil.which(tool):
            try:
                subprocess.run(
                    [tool, "-J", "-R", "-V", volume_name, "-o", out_abs, disc_abs],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                if os.path.isfile(out_abs):
                    print(f"[✓] ISO-образ ({tool}) створено: {out_abs}")
                    return out_abs
            except Exception:
                pass

    _write_iso9660_image(disc_abs, out_abs, volume_name=volume_name)
    print(f"[✓] ISO-образ (ISO-9660/Joliet) створено: {out_abs}")
    return out_abs


# =====================================================================
# 6. АВТОМАТИЧНИЙ SELF-TEST (ПЕРЕВІРКА MILESTONE 1)
# =====================================================================
def run_self_test() -> None:
    print("====================================================================")
    print(" [*] GVK Private AI — Криптографічний Self-Test (disc_packer.py)")
    print("====================================================================")

    test_phrase = "GVK_Secret_Lore_Passphrase_2026_Test!"
    print(f" [1/5] Деривація майстер-ключа Argon2id (256 МБ RAM, time_cost=4)...")
    t0 = time.time()
    master_key = derive_master_key(test_phrase)
    dt = time.time() - t0
    assert len(master_key) == 32
    print(f"       [OK] Отримано 32-байтний master.key за {dt:.2f} с: {master_key.hex()[:24]}...")

    print(" [2/5] Перевірка генерації та верифікації HMAC підпису origin.sig...")
    sig_f1 = generate_origin_sig("friend_1", master_key)
    sig_f2 = generate_origin_sig("friend_2", master_key)
    assert len(sig_f1) == 32
    assert sig_f1 != sig_f2
    assert verify_origin_sig("friend_1", master_key, sig_f1) is True
    assert verify_origin_sig("friend_2", master_key, sig_f1) is False
    print(f"       [OK] origin.sig (friend_1): {sig_f1.hex()[:24]}...")

    print(" [3/5] Перевірка потокового шифрування та дешифрування AES-256-GCM...")
    sample_data = os.urandom(3 * 1024 * 1024)  # 3 МБ випадкових даних
    src_buf = io.BytesIO(sample_data)
    enc_buf = io.BytesIO()
    # Використовуємо чанки по 1 МБ, щоб протестувати мульти-чанковий потік (3 чанки)
    encrypt_stream(src_buf, enc_buf, master_key=master_key, chunk_size=1024 * 1024)

    enc_bytes = enc_buf.getvalue()
    assert enc_bytes[:4] == b"GVK1"
    dec_buf = io.BytesIO()
    decrypt_stream(io.BytesIO(enc_bytes), dec_buf, master_key=master_key)
    assert dec_buf.getvalue() == sample_data
    print("       [OK] Мульти-чанковий потік зашифровано та розшифровано в ОЗП без втрат!")

    # Перевірка захисту від пошкодження (1 змінений біт повинен викликати помилку)
    corrupted = bytearray(enc_bytes)
    corrupted[-10] ^= 0x01
    try:
        decrypt_stream(io.BytesIO(bytes(corrupted)), io.BytesIO(), master_key=master_key)
        raise AssertionError("Пошкоджений шифротекст не викликав помилку!")
    except ValueError:
        print("       [OK] Захист цілісності GCM спрацював: зміну 1 біта миттєво виявлено.")

    print(" [4/5] Перевірка пакування та розпакування payload.tar.enc у пам'яті...")
    with tempfile.TemporaryDirectory() as tmpdir:
        shared_roster_src = os.path.join(tmpdir, "roster.json")
        with open(shared_roster_src, "w", encoding="utf-8") as rf:
            json.dump({"users_dict": {"2": {"name": "Учасник 02", "locked": False, "experimental": False}}}, rf)

        src_payload = os.path.join(tmpdir, "src_payload")
        os.makedirs(os.path.join(src_payload, "portraits", "raw"), exist_ok=True)
        with open(os.path.join(src_payload, "prompt.txt"), "w", encoding="utf-8") as f:
            f.write("Ти — friend_1. Починай репліку з [MOOD: СПОКІЙ].")
        with open(os.path.join(src_payload, "portraits", "мовчить.txt"), "w", encoding="utf-8") as f:
            f.write("(o_o)")
        with open(os.path.join(src_payload, "portraits", ".DS_Store"), "wb") as f:
            f.write(b"JUNK")
        with open(os.path.join(src_payload, "portraits", "raw", "photo.psd"), "wb") as f:
            f.write(b"8BPS" + b"\x00" * 64)

        enc_payload_path = os.path.join(tmpdir, "payload.tar.enc")
        sha = pack_payload_encrypted(src_payload, enc_payload_path, master_key)

        ram_extract_dir = os.path.join(tmpdir, "ram_extracted")
        unpack_payload_decrypted(enc_payload_path, ram_extract_dir, master_key)
        assert os.path.isfile(os.path.join(ram_extract_dir, "prompt.txt"))
        assert os.path.isfile(os.path.join(ram_extract_dir, "roster.json")), "Спільний ../roster.json має пакуватися автоматично!"
        assert os.path.isfile(os.path.join(ram_extract_dir, "portraits", "мовчить.txt"))
        assert not os.path.exists(os.path.join(ram_extract_dir, "portraits", ".DS_Store")), ".DS_Store не має пакуватися!"
        assert not os.path.exists(os.path.join(ram_extract_dir, "portraits", "raw")), "portraits/raw/ не має пакуватися!"
        print(f"       [OK] payload.tar.enc (SHA-256: {sha[:16]}...) розпаковано в ОЗП!")

        print(" [5/5] Перевірка збірки образу DVD-диска GVK_VAULT (без base_model.gguf.enc)...")
        persona_build = os.path.join(tmpdir, "persona_build")
        os.makedirs(persona_build, exist_ok=True)
        raw_adapter = os.path.join(tmpdir, "adapter.gguf")
        with open(raw_adapter, "wb") as f:
            f.write(b"GGUF" + os.urandom(64 * 1024))
        adapter_enc = os.path.join(persona_build, "adapter.gguf.enc")
        adapter_sha = encrypt_file(raw_adapter, adapter_enc, master_key)
        shutil.copy2(enc_payload_path, os.path.join(persona_build, "payload.tar.enc"))
        write_public_manifest(
            persona_build,
            persona_id="friend_1",
            adapter_sha256=adapter_sha,
            payload_sha256=sha,
        )

        disc_out = os.path.join(tmpdir, "GVK_VAULT_DVD")
        build_disc_tree(
            output_disc_dir=disc_out,
            master_key=master_key,
            persona_id="friend_1",
            persona_build_dir=persona_build,
        )
        for rel in (
            "vault/master.key",
            "persona/origin.sig",
            "persona/manifest.json",
            "persona/adapter.gguf.enc",
            "persona/payload.tar.enc",
            "run.sh",
            "run.bat",
        ):
            assert os.path.isfile(os.path.join(disc_out, rel)), f"Відсутній файл DVD: {rel}"
        assert not os.path.exists(os.path.join(disc_out, "base")), "Папка base/ не має створюватися на DVD за замовчуванням"
        iso_out = os.path.join(tmpdir, "GVK_VAULT_DISC_1.iso")
        built_iso = build_disc_iso(disc_out, iso_out, volume_name="GVK_VAULT")
        assert os.path.isfile(built_iso) and os.path.getsize(built_iso) >= 20 * ISO_SECTOR_SIZE
        with open(built_iso, "rb") as iso_f:
            iso_f.seek(0x8001)
            assert iso_f.read(5) == b"CD001", "Не знайдено сигнатуру CD001 у Primary Volume Descriptor!"
        print("       [OK] Образ DVD GVK_VAULT + ISO-9660 (.iso) зібрано коректно!")

    print("====================================================================")
    print(" [✓] УСІ ТЕСТИ ПРОЙДЕНО УСПІШНО!")
    print("====================================================================")


# =====================================================================
# CLI ІНТЕРФЕЙС
# =====================================================================
def _load_or_prompt_key(key_file: Optional[str], allow_short: bool = False) -> bytes:
    if key_file and os.path.isfile(key_file):
        with open(key_file, "rb") as f:
            key = f.read()
        if len(key) != KEY_SIZE:
            raise ValueError(f"Файл ключа {key_file} має розмір {len(key)} Б (очікується {KEY_SIZE} Б).")
        return key
    phrase = getpass.getpass(" [?] Введіть секретну майстер-фразу (від 24 символів): ")
    return derive_master_key(phrase, allow_short=allow_short)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GVK Private AI — Пакувальник дисків та криптографічний інструмент (disc_packer.py)"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # 1. self-test
    sub.add_parser("self-test", help="Запустити повну самоперевірку Argon2id + AES-256-GCM у пам'яті")

    # 2. keygen
    p_key = sub.add_parser("keygen", help="Згенерувати 32-байтний master.key із секретної фрази")
    p_key.add_argument("-o", "--output", required=True, help="Шлях збереження файлу master.key")
    p_key.add_argument("--passphrase", help="Секретна фраза (якщо не вказано — запитає приховано)")
    p_key.add_argument("--allow-short", action="store_true", help="Дозволити фразу коротшу за 24 символи (для тестів)")

    # 3. sign
    p_sig = sub.add_parser("sign", help="Згенерувати підпис фізичної активації origin.sig")
    p_sig.add_argument("-k", "--key-file", help="Шлях до файлу master.key")
    p_sig.add_argument("--id", required=True, help="ID персонажа (наприклад: friend_1)")
    p_sig.add_argument("-o", "--output", required=True, help="Шлях збереження origin.sig")

    # 4. encrypt
    p_enc = sub.add_parser("encrypt", help="Зашифрувати файл у формат .enc (потоково по 64 МБ)")
    p_enc.add_argument("-k", "--key-file", help="Шлях до файлу master.key")
    p_enc.add_argument("-i", "--input", required=True, help="Вхідний відкритий файл")
    p_enc.add_argument("-o", "--output", required=True, help="Вихідний зашифрований .enc файл")
    p_enc.add_argument("--allow-short", action="store_true", help="Дозволити коротку фразу для тестів")

    # 5. decrypt
    p_dec = sub.add_parser("decrypt", help="Розшифрувати .enc файл та перевірити цілісність GCM")
    p_dec.add_argument("-k", "--key-file", help="Шлях до файлу master.key")
    p_dec.add_argument("-i", "--input", required=True, help="Вхідний .enc файл")
    p_dec.add_argument("-o", "--output", required=True, help="Вихідний розшифрований файл")
    p_dec.add_argument("--allow-short", action="store_true", help="Дозволити коротку фразу для тестів")

    # 6. pack-persona
    p_pack = sub.add_parser(
        "pack-persona",
        help="Зашифрувати адаптер та папку скілів/портретів учасника + створити manifest.json і origin.sig",
    )
    p_pack.add_argument("-k", "--key-file", help="Шлях до файлу master.key")
    p_pack.add_argument("--id", required=True, help="ID персонажа (friend_1, friend_2, friend_3)")
    p_pack.add_argument("--adapter", required=True, help="Шлях до відкритого файлу adapter.gguf")
    p_pack.add_argument("--payload-dir", required=True, help="Шлях до папки з prompt.txt, portraits/, memory.faiss")
    p_pack.add_argument("--roster", default=None, help="Опційно: шлях до спільного roster.json (за замовч. ../roster.json)")
    p_pack.add_argument("-o", "--output-dir", required=True, help="Вихідна папка для зашифрованих файлів")
    p_pack.add_argument("--version", default="1.0.0", help="Версія пакета (за замовч. 1.0.0)")

    # 7. build-disc
    p_disc = sub.add_parser("build-disc", help="Зібрати повну структуру DVD-диска GVK_VAULT (DVD-R 4.7 ГБ)")
    p_disc.add_argument("-k", "--key-file", required=True, help="Шлях до файлу master.key")
    p_disc.add_argument("--id", required=True, help="ID персонажа на цьому диску (напр. friend_1)")
    p_disc.add_argument(
        "--base-enc",
        required=False,
        default=None,
        help="Опційно: шлях до base_model.gguf.enc (для DVD не потрібно — базова модель качається з хмари)",
    )
    p_disc.add_argument("--persona-dir", required=True, help="Папка зібраного персонажа (з pack-persona)")
    p_disc.add_argument("-o", "--output-dir", required=True, help="Вихідна папка образу диска GVK_VAULT")
    p_disc.add_argument("--iso", default=None, help="Опційно: шлях вихідного .iso файлу (напр. dist/GVK_DISC_1.iso)")

    # 8. build-iso
    p_iso = sub.add_parser("build-iso", help="Зібрати ISO-9660/Joliet файл (.iso) із готової папки диска")
    p_iso.add_argument("-i", "--input-dir", required=True, help="Папка зібраного диска GVK_VAULT")
    p_iso.add_argument("-o", "--output", required=True, help="Вихідний шлях до файлу .iso")
    p_iso.add_argument("--volume-name", default="GVK_VAULT", help="Мітка тому DVD (за замовч. GVK_VAULT)")

    # 9. fetch-usb-bins
    p_fbins = sub.add_parser(
        "fetch-usb-bins",
        help="Завантажити релізні архіви llama.cpp (Windows Vulkan/CPU + macOS ARM64/Intel) для офлайн USB",
    )
    p_fbins.add_argument("-o", "--output-dir", default="base/llama_bins", help="Папка збереження архівів")
    p_fbins.add_argument(
        "--include-cuda-zip",
        action="store_true",
        help="Додатково скачати офіційні архіви win-cuda-12.4-x64.zip + cudart-llama-*.zip (~720 МБ)",
    )

    # 10. build-usb
    p_usb = sub.add_parser(
        "build-usb",
        help="Зібрати повний офлайн USB-дистрибутив GVK_VAULT [Mode: USB] з моделлю 14B та бінарниками Win/macOS/CUDA",
    )
    p_usb.add_argument("-k", "--key-file", required=True, help="Шлях до файлу master.key")
    p_usb.add_argument("--id", default="friend_1", help="ID персонажа (за замовч. friend_1)")
    p_usb.add_argument(
        "--persona-dir",
        default="personas_encrypted/friend_1",
        help="Папка зібраного персонажа (за замовч. personas_encrypted/friend_1)",
    )
    p_usb.add_argument("-o", "--output-dir", required=True, help="Вихідна папка або буква USB-диска (напр. dist/GVK_VAULT_USB або E:\\)")
    p_usb.add_argument("--base-model", default=None, help="Шлях до Qwen2.5-14B-Instruct-Q4_K_M.gguf (за замовч. base/Qwen2.5-14B-Instruct-Q4_K_M.gguf)")
    p_usb.add_argument("--llama-bins-dir", default=None, help="Папка з архівами llama.cpp (за замовч. base/llama_bins)")
    p_usb.add_argument("--pascal-runtime-dir", default=None, help="Папка з CUDA DLL рантаймом (за замовч. base/llama-pascal-runtime)")
    p_usb.add_argument("--embeddings-dir", default=None, help="Опційно: папка з офлайн моделлю paraphrase-multilingual-MiniLM-L12-v2")
    p_usb.add_argument("--fetch-bins", action="store_true", help="Автоматично докачати відсутні архіви Win/macOS llama.cpp з GitHub")
    p_usb.add_argument("--include-cuda-zip", action="store_true", help="При --fetch-bins також скачати офіційні CUDA 12.4 zip-архіви")

    args = parser.parse_args()

    if args.cmd == "self-test":
        run_self_test()

    elif args.cmd == "keygen":
        phrase = args.passphrase or getpass.getpass(" [?] Введіть секретну майстер-фразу (від 24 символів): ")
        print(" [*] Обчислення ключа через Argon2id (256 МБ RAM)...")
        key = derive_master_key(phrase, allow_short=args.allow_short)
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "wb") as f:
            f.write(key)
        print(f" [✓] 32-байтний ключ збережено у: {args.output}")

    elif args.cmd == "sign":
        key = _load_or_prompt_key(args.key_file)
        sig = generate_origin_sig(args.id, key)
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "wb") as f:
            f.write(sig)
        print(f" [✓] Підпис origin.sig для '{args.id}' збережено у: {args.output}")

    elif args.cmd == "encrypt":
        key = _load_or_prompt_key(args.key_file, allow_short=args.allow_short)
        sha = encrypt_file(args.input, args.output, key)
        print(f" [✓] Зашифровано у: {args.output} (SHA-256: {sha})")

    elif args.cmd == "decrypt":
        key = _load_or_prompt_key(args.key_file, allow_short=args.allow_short)
        size = decrypt_file(args.input, args.output, key)
        print(f" [✓] Розшифровано {size} байтів у: {args.output}")

    elif args.cmd == "pack-persona":
        key = _load_or_prompt_key(args.key_file)
        os.makedirs(args.output_dir, exist_ok=True)

        adapter_enc = os.path.join(args.output_dir, "adapter.gguf.enc")
        payload_enc = os.path.join(args.output_dir, "payload.tar.enc")
        sig_path = os.path.join(args.output_dir, "origin.sig")

        adapter_sha = encrypt_file(args.adapter, adapter_enc, key)
        payload_sha = pack_payload_encrypted(args.payload_dir, payload_enc, key, roster_path=args.roster)
        manifest_path = write_public_manifest(
            args.output_dir,
            persona_id=args.id,
            adapter_sha256=adapter_sha,
            payload_sha256=payload_sha,
            version=args.version,
        )
        with open(sig_path, "wb") as f:
            f.write(generate_origin_sig(args.id, key))

        print(f" [✓] Пакет персонажа '{args.id}' зібрано в: {args.output_dir}")
        print(f"     - adapter.gguf.enc (SHA-256: {adapter_sha[:16]}...)")
        print(f"     - payload.tar.enc  (SHA-256: {payload_sha[:16]}...)")
        print(f"     - manifest.json    ({manifest_path})")
        print(f"     - origin.sig       (УВАГА: тільки для DVD-диска, не комітити в GitHub!)")

    elif args.cmd == "build-disc":
        key = _load_or_prompt_key(args.key_file)
        build_disc_tree(
            output_disc_dir=args.output_dir,
            master_key=key,
            persona_id=args.id,
            persona_build_dir=args.persona_dir,
            base_model_enc_path=args.base_enc,
        )
        if args.iso:
            build_disc_iso(args.output_dir, args.iso, volume_name="GVK_VAULT")

    elif args.cmd == "build-iso":
        build_disc_iso(args.input_dir, args.output, volume_name=args.volume_name)

    elif args.cmd == "fetch-usb-bins":
        fetch_usb_llama_bins(args.output_dir, include_cuda_zip=args.include_cuda_zip)

    elif args.cmd == "build-usb":
        key = _load_or_prompt_key(args.key_file)
        build_usb_tree(
            output_usb_dir=args.output_dir,
            master_key=key,
            persona_id=args.id,
            persona_build_dir=args.persona_dir,
            base_model_path=args.base_model,
            llama_bins_dir=args.llama_bins_dir,
            pascal_runtime_dir=args.pascal_runtime_dir,
            embeddings_dir=args.embeddings_dir,
            fetch_bins=args.fetch_bins,
            include_cuda_zip=args.include_cuda_zip,
        )


if __name__ == "__main__":
    main()
