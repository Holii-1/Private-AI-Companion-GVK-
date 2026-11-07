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
  5. Збірка повної структури Blu-ray диска `GVK_VAULT` (включно з лаунчерами `run.sh` / `run.bat`).

Приклади використання:
  # 1. Швидка самоперевірка всіх криптографічних функцій в ОЗП:
  python3 tools/disc_packer.py self-test

  # 2. Генерація master.key із секретної фрази:
  python3 tools/disc_packer.py keygen -o vault/master.key

  # 3. Шифрування великого файлу (наприклад, базової моделі):
  python3 tools/disc_packer.py encrypt -k vault/master.key -i base_model.gguf -o base_model.gguf.enc

  # 4. Повна збірка папки персонажа (adapter.gguf.enc + payload.tar.enc + manifest.json + origin.sig):
  python3 tools/disc_packer.py pack-persona -k vault/master.key --id friend_1 \
      --adapter raw/friend_1.gguf --payload-dir raw/friend_1_payload/ -o build/friend_1/

  # 5. Збірка готового образу Blu-ray диска GVK_VAULT:
  python3 tools/disc_packer.py build-disc -k vault/master.key --id friend_1 \
      --base-enc base_model.gguf.enc --persona-dir build/friend_1/ -o dist/GVK_VAULT_DISC_1/
"""

import argparse
import getpass
import hashlib
import hmac
import io
import json
import os
import shutil
import stat
import struct
import sys
import tarfile
import tempfile
import time
from typing import BinaryIO, Optional

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


def pack_payload_encrypted(source_dir: str, output_enc_path: str, master_key: bytes) -> str:
    """
    Збирає вміст папки особистості (persona_config.json, prompt.txt, portraits/,
    memory.faiss, tools.py) у стиснутий TAR.GZ архів прямо в оперативній пам'яті
    та одразу шифрує його у файл `payload.tar.enc` (Zero-Plaintext-At-Rest).
    Повертає SHA-256 геш створеного `payload.tar.enc`.
    """
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"Папку корисного навантаження не знайдено: {source_dir}")

    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
        for entry in sorted(os.listdir(source_dir)):
            if entry.startswith(".") or entry == "__pycache__":
                continue
            full_path = os.path.join(source_dir, entry)
            tar.add(full_path, arcname=entry)

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
# 5. ЗБІРКА СТРУКТУРИ BLU-RAY ДИСКА (GVK_VAULT)
# =====================================================================
RUN_SH_TEMPLATE = """#!/usr/bin/env bash
# =====================================================================
# GVK Private AI — Автономний лаунчер Blu-ray диска (macOS / Linux)
# =====================================================================
set -e

DISC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GVK_HOME="${HOME}/.gvk_ai"
REPO_DIR="${GVK_HOME}/repo"
REPO_URL="https://github.com/Holii-1/Private-AI-Companion-GVK-.git"

echo "===================================================================="
echo " [*] GVK_VAULT Виявлено: ${DISC_DIR}"
echo "===================================================================="

mkdir -p "${GVK_HOME}"

if [ ! -d "${REPO_DIR}/.git" ]; then
    echo " [*] Перший запуск: клонування ядра GVK Private AI..."
    git clone "${REPO_URL}" "${REPO_DIR}"
fi

exec python3 "${REPO_DIR}/core/main.py" --disc-path "${DISC_DIR}" "$@"
"""

RUN_BAT_TEMPLATE = """@echo off
REM =====================================================================
REM GVK Private AI — Автономний лаунчер Blu-ray диска (Windows)
REM =====================================================================
setlocal

set "DISC_DIR=%~dp0"
set "GVK_HOME=%USERPROFILE%\\.gvk_ai"
set "REPO_DIR=%GVK_HOME%\\repo"
set "REPO_URL=https://github.com/Holii-1/Private-AI-Companion-GVK-.git"

echo ====================================================================
echo  [*] GVK_VAULT Disc Detected: %DISC_DIR%
echo ====================================================================

if not exist "%GVK_HOME%" mkdir "%GVK_HOME%"

if not exist "%REPO_DIR%\\.git" (
    echo  [*] First launch: cloning GVK Private AI core...
    git clone "%REPO_URL%" "%REPO_DIR%"
)

python "%REPO_DIR%\\core\\main.py" --disc-path "%DISC_DIR%" %*
"""


def build_disc_tree(
    output_disc_dir: str,
    master_key: bytes,
    persona_id: str,
    base_model_enc_path: str,
    persona_build_dir: str,
) -> None:
    """
    Формує повну структуру оптичного диска [Blu-ray Disc: GVK_VAULT],
    готову до запису на BD-R 25GB.
    """
    vault_dir = os.path.join(output_disc_dir, "vault")
    base_dir = os.path.join(output_disc_dir, "base")
    persona_dir = os.path.join(output_disc_dir, "persona")

    os.makedirs(vault_dir, exist_ok=True)
    os.makedirs(base_dir, exist_ok=True)
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

    # 4. Копіювання base_model.gguf.enc
    if not os.path.isfile(base_model_enc_path):
        raise FileNotFoundError(f"Зашифровану базову модель не знайдено: {base_model_enc_path}")
    shutil.copy2(base_model_enc_path, os.path.join(base_dir, "base_model.gguf.enc"))

    # 5. Створення стартових скриптів run.sh та run.bat
    run_sh_path = os.path.join(output_disc_dir, "run.sh")
    with open(run_sh_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(RUN_SH_TEMPLATE)
    os.chmod(run_sh_path, os.stat(run_sh_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    run_bat_path = os.path.join(output_disc_dir, "run.bat")
    with open(run_bat_path, "w", encoding="utf-8", newline="\r\n") as f:
        f.write(RUN_BAT_TEMPLATE)

    print(f"[✓] Образ Blu-ray диска GVK_VAULT для '{persona_id}' успішно зібрано у: {output_disc_dir}")


# =====================================================================
# 6. АВТОМАТИЧНИЙ SELF-TEST (ПЕРЕВІРКА MILESTONE 1)
# =====================================================================
def run_self_test() -> None:
    print("====================================================================")
    print(" [*] GVK Private AI — Криптографічний Self-Test (disc_packer.py)")
    print("====================================================================")

    test_phrase = "GVK_Secret_Lore_Passphrase_2026_Test!"
    print(f" [1/4] Деривація майстер-ключа Argon2id (256 МБ RAM, time_cost=4)...")
    t0 = time.time()
    master_key = derive_master_key(test_phrase)
    dt = time.time() - t0
    assert len(master_key) == 32
    print(f"       [OK] Отримано 32-байтний master.key за {dt:.2f} с: {master_key.hex()[:24]}...")

    print(" [2/4] Перевірка генерації та верифікації HMAC підпису origin.sig...")
    sig_f1 = generate_origin_sig("friend_1", master_key)
    sig_f2 = generate_origin_sig("friend_2", master_key)
    assert len(sig_f1) == 32
    assert sig_f1 != sig_f2
    assert verify_origin_sig("friend_1", master_key, sig_f1) is True
    assert verify_origin_sig("friend_2", master_key, sig_f1) is False
    print(f"       [OK] origin.sig (friend_1): {sig_f1.hex()[:24]}...")

    print(" [3/4] Перевірка потокового шифрування та дешифрування AES-256-GCM...")
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

    print(" [4/4] Перевірка пакування та розпакування payload.tar.enc у пам'яті...")
    with tempfile.TemporaryDirectory() as tmpdir:
        src_payload = os.path.join(tmpdir, "src_payload")
        os.makedirs(os.path.join(src_payload, "portraits"), exist_ok=True)
        with open(os.path.join(src_payload, "prompt.txt"), "w", encoding="utf-8") as f:
            f.write("Ти — friend_1. Починай репліку з [MOOD: СПОКІЙ].")
        with open(os.path.join(src_payload, "portraits", "мовчить.txt"), "w", encoding="utf-8") as f:
            f.write("(o_o)")

        enc_payload_path = os.path.join(tmpdir, "payload.tar.enc")
        sha = pack_payload_encrypted(src_payload, enc_payload_path, master_key)

        ram_extract_dir = os.path.join(tmpdir, "ram_extracted")
        unpack_payload_decrypted(enc_payload_path, ram_extract_dir, master_key)
        assert os.path.isfile(os.path.join(ram_extract_dir, "prompt.txt"))
        assert os.path.isfile(os.path.join(ram_extract_dir, "portraits", "мовчить.txt"))
        print(f"       [OK] payload.tar.enc (SHA-256: {sha[:16]}...) розпаковано в ОЗП!")

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
    p_pack.add_argument("-o", "--output-dir", required=True, help="Вихідна папка для зашифрованих файлів")
    p_pack.add_argument("--version", default="1.0.0", help="Версія пакета (за замовч. 1.0.0)")

    # 7. build-disc
    p_disc = sub.add_parser("build-disc", help="Зібрати повну структуру Blu-ray диска GVK_VAULT")
    p_disc.add_argument("-k", "--key-file", required=True, help="Шлях до файлу master.key")
    p_disc.add_argument("--id", required=True, help="ID персонажа на цьому диску (напр. friend_1)")
    p_disc.add_argument("--base-enc", required=True, help="Шлях до зашифрованої бази base_model.gguf.enc")
    p_disc.add_argument("--persona-dir", required=True, help="Папка зібраного персонажа (з pack-persona)")
    p_disc.add_argument("-o", "--output-dir", required=True, help="Вихідна папка образу диска GVK_VAULT")

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
        payload_sha = pack_payload_encrypted(args.payload_dir, payload_enc, key)
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
        print(f"     - origin.sig       (УВАГА: тільки для Blu-ray диска, не комітити в GitHub!)")

    elif args.cmd == "build-disc":
        key = _load_or_prompt_key(args.key_file)
        build_disc_tree(
            output_disc_dir=args.output_dir,
            master_key=key,
            persona_id=args.id,
            base_model_enc_path=args.base_enc,
            persona_build_dir=args.persona_dir,
        )


if __name__ == "__main__":
    main()
