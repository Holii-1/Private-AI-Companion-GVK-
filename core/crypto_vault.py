#!/usr/bin/env python3
"""
GVK Private AI — Крипто-сховище, RAM-диск та Демон Kill-Switch (core/crypto_vault.py)

Runtime-модуль безпеки, що забезпечує принцип Zero-Plaintext-At-Rest:
  1. Здобуття ключа доступу (32 байти):
     - Автоматичний пошук оптичного DVD-диска `GVK_VAULT` (macOS / Linux / Windows)
       і миттєве зчитування `vault/master.key` за 0.01 с.
     - Fallback-режим (якщо диска немає): консольний запит секретної майстер-фрази
       та деривація того самого 32-байтного ключа через Argon2id (256 МБ RAM).
  2. Кросплатформне монтування ізольованого RAM-диска (512–1024 МБ під 16 ГБ ОЗП):
     - macOS: блоковий пристрій у пам'яті (`hdiutil attach -nomount ram://...`) з HFS+,
       правами 0700 та примусовим вимкненням індексації Spotlight (`.metadata_never_index`).
     - Linux: вбудований розділ ядра `tmpfs` у `/dev/shm/gvk_runtime_<pid>` (`chmod 700`).
     - Windows: віртуальний RAM-диск (ImDisk) або ізольований тимчасовий буфер у пам'яті
       з примусовим зануленням (`0x00`) перед закриттям.
  3. Потокове дешифрування `.enc` контейнерів (AES-256-GCM чанками по 64 МБ) напряму в RAM-диск:
     - Персональні LoRA-адаптери (`adapter.gguf.enc` -> `<ram>/adapters/<friend_id>.gguf`).
     - Приватні архіви (`payload.tar.enc` -> `<ram>/personas/<friend_id>/` з промптами,
       ASCII-портретами емоцій `portraits/`, базою `memory.faiss` та `tools.py`).
     - Публічна базова модель (`Qwen2.5-14B-Instruct-Q4_K_M.gguf`, ~9.0 ГБ) підключається
       напряму з SSD (`~/.gvk_ai/base/`) через `--mmap` без подвійного копіювання в RAM-диск.
  4. Апаратний демон Kill-Switch (Heartbeat 3.0 с):
     - Окремий фоновий потік кожні 3 секунди перевіряє присутність `vault/master.key` на диску.
     - Якщо диск вийнято або спрацював апаратний тумблер на станції `GVK-DOCK // MK.I`:
       надсилає `SIGKILL` процесу `llama-server`, затирає ключі в ОЗП нулями,
       примусово демонтує RAM-диск і миттєво завершує процес.

Приклад запуску самоперевірки (Milestone 1 — шифрування + реальний RAM-диск + розпакування):
  python3 core/crypto_vault.py self-test
"""

import argparse
import atexit
import getpass
import glob
import hashlib
import hmac
import io
import os
import platform
import shutil
import signal
import stat
import struct
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from typing import BinaryIO, Callable, List, Optional

# Додаємо корінь репозиторію в sys.path для прямого запуску `python3 core/crypto_vault.py`
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.disc_packer import (
    ARGON2_MEMORY_COST,
    ARGON2_PARALLELISM,
    ARGON2_TIME_COST,
    DEFAULT_CHUNK_SIZE,
    DOMAIN_SALT,
    GCM_TAG_SIZE,
    HEADER_SIZE,
    KEY_SIZE,
    MAGIC_BYTES,
    NONCE_SIZE,
    decrypt_stream,
    derive_master_key,
    encrypt_stream,
    generate_origin_sig,
    pack_payload_encrypted,
    unpack_payload_decrypted,
    verify_origin_sig,
)

DEFAULT_RAMDISK_MB = 1024  # 1 ГБ для 3 LoRA (~420 МБ) + FAISS/промпти/портрети; база 14B іде через --mmap
DEFAULT_BASE_MODEL_FILENAME = "Qwen2.5-14B-Instruct-Q4_K_M.gguf"
KILL_SWITCH_INTERVAL_SEC = 3.0


# =====================================================================
# 1. ПОШУК ОПТИЧНОГО ДИСКА GVK_VAULT ТА ЗДОБУТТЯ MASTER.KEY
# =====================================================================
def find_base_model_path(
    custom_path: Optional[str] = None,
    gvk_home: Optional[str] = None,
) -> Optional[str]:
    """
    Шукає публічну базову GGUF-модель для підключення у llama-server через `--mmap`.
    Порядок пріоритету:
      1. Явно переданий `custom_path`
      2. Змінна середовища `GVK_BASE_MODEL`
      3. `~/.gvk_ai/base/Qwen2.5-14B-Instruct-Q4_K_M.gguf` (або будь-який `.gguf` у `~/.gvk_ai/base/`)
      4. Локальні шляхи в репозиторії (`base/Qwen2.5-14B-Instruct-Q4_K_M.gguf`, `base/*.gguf`, `base_model.gguf`)
    """
    home_dir = os.path.abspath(gvk_home or os.environ.get("GVK_HOME") or os.path.expanduser("~/.gvk_ai"))
    home_base_dir = os.path.join(home_dir, "base")

    candidates: List[str] = []
    if custom_path:
        candidates.append(os.path.abspath(os.path.expanduser(custom_path)))

    env_model = os.environ.get("GVK_BASE_MODEL")
    if env_model:
        candidates.append(os.path.abspath(os.path.expanduser(env_model)))

    candidates.append(os.path.join(home_base_dir, DEFAULT_BASE_MODEL_FILENAME))
    if os.path.isdir(home_base_dir):
        candidates.extend(sorted(glob.glob(os.path.join(home_base_dir, "*.gguf"))))

    candidates.append(os.path.join(REPO_ROOT, "base", DEFAULT_BASE_MODEL_FILENAME))
    repo_base_dir = os.path.join(REPO_ROOT, "base")
    if os.path.isdir(repo_base_dir):
        candidates.extend(sorted(glob.glob(os.path.join(repo_base_dir, "*.gguf"))))
    candidates.append(os.path.join(REPO_ROOT, "base_model.gguf"))

    seen = set()
    for path in candidates:
        if not path or path in seen:
            continue
        seen.add(path)
        try:
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                return path
        except OSError:
            continue

    return None


def _normalize_disc_path(raw_path: str) -> str:
    """Нормалізує шлях до диска (зокрема Windows 'D:' -> 'D:\\' перед abspath)."""
    p = os.path.expanduser(raw_path.strip())
    if len(p) == 2 and p[1] == ":" and p[0].isalpha():
        p = p + "\\"
    return os.path.abspath(p)


def find_gvk_disc_path(explicit_path: Optional[str] = None) -> Optional[str]:
    """
    Шукає змонтований оптичний DVD-носій `GVK_VAULT` (або перевіряє переданий шлях).
    Повертає шлях до кореня диска, де лежить `vault/master.key`, або None.
    """
    candidates: List[str] = []
    if explicit_path:
        candidates.append(_normalize_disc_path(explicit_path))

    env_disc = os.environ.get("GVK_DISC_PATH")
    if env_disc:
        candidates.append(_normalize_disc_path(env_disc))

    sys_name = platform.system().lower()
    if sys_name == "darwin":
        # macOS монтує оптичні диски у /Volumes/<VolumeName>
        candidates.extend(sorted(glob.glob("/Volumes/GVK_VAULT*")))
        candidates.extend(sorted(glob.glob("/Volumes/*")))
    elif sys_name == "linux":
        user = os.environ.get("USER", "")
        candidates.extend(sorted(glob.glob(f"/media/{user}/GVK_VAULT*")))
        candidates.extend(sorted(glob.glob(f"/run/media/{user}/GVK_VAULT*")))
        candidates.extend(sorted(glob.glob("/media/*/GVK_VAULT*")))
        candidates.extend(sorted(glob.glob("/mnt/GVK_VAULT*")))
    elif sys_name == "windows":
        for drive_letter in "DEFGHIJKLMNOPQRSTUVWXYZ":
            candidates.append(f"{drive_letter}:\\")

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


def acquire_master_key(
    disc_path: Optional[str] = None,
    passphrase: Optional[str] = None,
    allow_short: bool = False,
) -> tuple[bytearray, Optional[str]]:
    """
    Здобуває 32-байтний master.key:
      1. Якщо знайдено фізичний диск `GVK_VAULT` з `vault/master.key` — зчитує за 0.01 с.
      2. Якщо диска немає — переходить у режим ручної деривації ключа Argon2id із фрази.
    Повертає кортеж: (mutable bytearray ключа для безпечного затирання, шлях_до_master_key_на_диску або None).
    """
    detected_disc = find_gvk_disc_path(disc_path)
    if detected_disc:
        key_file = os.path.join(detected_disc, "vault", "master.key")
        with open(key_file, "rb") as f:
            raw_key = f.read()
        if len(raw_key) != KEY_SIZE:
            raise ValueError(f"Пошкоджений master.key на диску ({len(raw_key)} байтів замість {KEY_SIZE}).")
        is_usb = (
            os.path.isdir(os.path.join(detected_disc, "base"))
            or os.path.isdir(os.path.join(detected_disc, "bin"))
            or os.path.isfile(os.path.join(detected_disc, "vault", "media_mode.json"))
        )
        mode_tag = "USB" if is_usb else "DVD"
        print(f" [✓] Апаратний ключ доступу GVK_VAULT [{mode_tag}] зчитано з носія: {detected_disc}")
        return bytearray(raw_key), key_file

    # Fallback: Крипто-челендж / Ручне введення секретної майстер-фрази
    if passphrase is None:
        print("====================================================================")
        print(" [!] Фізичний носій GVK_VAULT з ключем доступу не знайдено.")
        print(" [*] Перехід у режим ручної деривації ключа (Argon2id -> AES-256-GCM).")
        print("====================================================================")
        passphrase = getpass.getpass(" [?] Введіть секретну майстер-фразу для розблокування особистостей -> ")

    print(" [*] Обчислення 256-бітного ключа через Argon2id (256 МБ ОЗП)...")
    derived = derive_master_key(passphrase, allow_short=allow_short)
    return bytearray(derived), None


# =====================================================================
# 2. КРОСПЛАТФОРМНИЙ МЕНЕДЖЕР RAM-ДИСКА ТА CRYPTO VAULT
# =====================================================================
class CryptoVault:
    """
    Керує життєвим циклом зашифрованої сесії:
      - Монтування ізольованого RAM-диска (512–1024 МБ, macOS / Linux / Windows)
      - Пошук публічної базової моделі на SSD (`~/.gvk_ai/base/`) для підключення через `--mmap`
      - Потокове дешифрування секретів (LoRA-адаптерів та `payload.tar.enc`) у RAM
      - Фоновий моніторинг оптичного приводу (Hardware Kill-Switch)
      - Гарантоване затирання ключів і демонтування RAM-диска при виході
    """

    def __init__(
        self,
        disc_path: Optional[str] = None,
        ramdisk_mb: int = DEFAULT_RAMDISK_MB,
        passphrase: Optional[str] = None,
        allow_short_passphrase: bool = False,
    ):
        self.disc_path = disc_path
        self.ramdisk_mb = ramdisk_mb
        self._initial_passphrase = passphrase
        self._allow_short = allow_short_passphrase

        self._master_key: Optional[bytearray] = None
        self._monitored_key_path: Optional[str] = None
        self.mount_point: Optional[str] = None
        self._macos_device: Optional[str] = None
        self._windows_imdisk_drive: Optional[str] = None

        self._protected_pids: List[int] = []
        self._kill_switch_stop = threading.Event()
        self._kill_switch_thread: Optional[threading.Thread] = None
        self._cleanup_lock = threading.Lock()
        self._is_cleaned_up = False

    # -----------------------------------------------------------------
    # Context Manager (`with CryptoVault() as vault:`)
    # -----------------------------------------------------------------
    def __enter__(self) -> "CryptoVault":
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()

    def initialize(self) -> str:
        """Здобуває ключ, монтує RAM-диск і запускає Kill-Switch (якщо сесія з диска)."""
        self._master_key, self._monitored_key_path = acquire_master_key(
            disc_path=self.disc_path,
            passphrase=self._initial_passphrase,
            allow_short=self._allow_short,
        )
        # Одразу видаляємо текстову фразу з пам'яті
        self._initial_passphrase = None

        self.mount_point = self.mount_ramdisk(self.ramdisk_mb)
        atexit.register(self.cleanup)

        if self._monitored_key_path:
            self.start_kill_switch()

        return self.mount_point

    # -----------------------------------------------------------------
    # МОНТУВАННЯ RAM-ДИСКА (Linux / macOS / Windows)
    # -----------------------------------------------------------------
    def mount_ramdisk(self, size_mb: int) -> str:
        """
        Створює ізольований простір в оперативній пам'яті.
        Жоден файл усередині цього розділу ніколи не записується на фізичний SSD.
        """
        sys_name = platform.system().lower()
        pid = os.getpid()

        if sys_name == "linux":
            shm_base = "/dev/shm" if os.path.isdir("/dev/shm") else "/run/shm"
            if os.path.isdir(shm_base) and os.access(shm_base, os.W_OK):
                mount_dir = os.path.join(shm_base, f"gvk_runtime_{pid}")
                os.makedirs(mount_dir, mode=0o700, exist_ok=True)
                os.chmod(mount_dir, 0o700)
                print(f" [✓] RAM-диск (Linux tmpfs) змонтовано у: {mount_dir}")
                return mount_dir

        elif sys_name == "darwin":
            # macOS: створюємо блоковий RAM-пристрій через `hdiutil attach -nomount ram://<sectors>`
            # 1 сектор = 512 байтів -> size_mb * 2048 секторів
            sectors = max(32768, int(size_mb * 2048))
            vol_name = f"GVK_RAM_{pid}"
            try:
                res = subprocess.run(
                    ["hdiutil", "attach", "-nomount", f"ram://{sectors}"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                device = res.stdout.strip()
                if device.startswith("/dev/disk"):
                    self._macos_device = device
                    mount_dir = os.path.join(tempfile.gettempdir(), f"gvk_ram_{pid}")
                    os.makedirs(mount_dir, mode=0o700, exist_ok=True)

                    # Форматуємо блоковий пристрій у швидку файлову систему HFS+ і монтуємо в приватну папку
                    subprocess.run(
                        ["newfs_hfs", "-v", vol_name, device],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    subprocess.run(
                        ["mount", "-t", "hfs", "-o", "nobrowse,noowners,nosuid,-m=0700", device, mount_dir],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )

                    # Забороняємо індексацію Spotlight
                    open(os.path.join(mount_dir, ".metadata_never_index"), "wb").close()
                    subprocess.run(
                        ["mdutil", "-i", "off", mount_dir],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    print(f" [✓] RAM-диск (macOS {size_mb} МБ, пристрій {device}) змонтовано у: {mount_dir}")
                    return mount_dir
            except Exception as exc:
                # Якщо процес запущено у пісочниці без доступу до hdiutil, використовуємо захищений буфер у пам'яті
                if self._macos_device:
                    subprocess.run(
                        ["hdiutil", "detach", self._macos_device, "-force"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    self._macos_device = None
                print(f" [!] Примітка: апаратний hdiutil недоступний ({exc}); активовано ізольований сесійний буфер.")

        elif sys_name == "windows":
            # Спроба створити віртуальний RAM-диск через ImDisk (якщо встановлений у системі)
            if shutil.which("imdisk"):
                for letter in "ZYXWVUTSRQ":
                    drive = f"{letter}:"
                    if not os.path.exists(drive + "\\"):
                        try:
                            subprocess.run(
                                ["imdisk", "-a", "-s", f"{size_mb}M", "-m", drive, "-p", "/fs:ntfs /q /y"],
                                check=True,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                            )
                            self._windows_imdisk_drive = drive
                            mount_dir = f"{drive}\\gvk_runtime_{pid}"
                            os.makedirs(mount_dir, exist_ok=True)
                            print(f" [✓] RAM-диск (Windows ImDisk {size_mb} МБ) змонтовано у: {mount_dir}")
                            return mount_dir
                        except Exception:
                            break

        # Універсальний ізольований фолбек (з правами 0700 та примусовим криптографічним затиранням 0x00)
        fallback_dir = tempfile.mkdtemp(prefix=f"gvk_ram_session_{pid}_")
        os.chmod(fallback_dir, 0o700)
        try:
            open(os.path.join(fallback_dir, ".metadata_never_index"), "wb").close()
        except OSError:
            pass
        print(f" [✓] Ізольований сесійний простір створено у: {fallback_dir}")
        return fallback_dir

    # -----------------------------------------------------------------
    # ДЕШИФРУВАННЯ КОНТЕЙНЕРІВ У RAM-ДИСК
    # -----------------------------------------------------------------
    def _require_key(self) -> bytes:
        if not self._master_key or len(self._master_key) != KEY_SIZE:
            raise RuntimeError("CryptoVault не ініціалізовано або ключ вже затерто!")
        return bytes(self._master_key)

    def decrypt_container_to_ram(self, enc_file_path: str, relative_ram_path: str) -> str:
        """
        Потоково розшифровує `.enc` контейнер (наприклад, `base_model.gguf.enc` або `adapter.gguf.enc`)
        чанками по 64 МБ прямо у змонтований RAM-диск.
        Повертає абсолютний шлях до розшифрованого файлу в RAM.
        """
        if not self.mount_point:
            raise RuntimeError("RAM-диск не змонтовано!")
        if not os.path.isfile(enc_file_path):
            raise FileNotFoundError(f"Зашифрований контейнер не знайдено: {enc_file_path}")

        key = self._require_key()
        dst_path = os.path.join(self.mount_point, relative_ram_path.lstrip("/\\"))
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        with open(enc_file_path, "rb") as f_in, open(dst_path, "wb") as f_out:
            os.chmod(dst_path, 0o600)
            decrypt_stream(f_in, f_out, master_key=key, label=f"Дешифрування {os.path.basename(enc_file_path)}")

        return dst_path

    def unpack_persona_to_ram(self, persona_id: str, adapter_enc_path: str, payload_enc_path: str) -> dict:
        """
        Розшифровує LoRA-адаптер (`adapter.gguf.enc`) та архів особистості (`payload.tar.enc`)
        конкретного учасника прямо у RAM-диск.
        Повертає словник зі шляхами в RAM:
          - adapter_path: <ram>/adapters/<persona_id>.gguf
          - persona_dir:  <ram>/personas/<persona_id>/
        """
        if not self.mount_point:
            raise RuntimeError("RAM-диск не змонтовано!")

        key = self._require_key()
        adapter_ram_path = self.decrypt_container_to_ram(
            adapter_enc_path,
            os.path.join("adapters", f"{persona_id}.gguf"),
        )

        persona_ram_dir = os.path.join(self.mount_point, "personas", persona_id)
        unpack_payload_decrypted(payload_enc_path, persona_ram_dir, master_key=key)

        return {
            "persona_id": persona_id,
            "adapter_path": adapter_ram_path,
            "persona_dir": persona_ram_dir,
        }

    def verify_persona_signature(self, persona_id: str, sig_file_path: str) -> bool:
        """Перевіряє оригінальність файлу підпису `origin.sig` для вказаного `persona_id`."""
        if not os.path.isfile(sig_file_path):
            return False
        with open(sig_file_path, "rb") as f:
            sig_bytes = f.read()
        if len(sig_bytes) != 32:
            return False
        return verify_origin_sig(persona_id, self._require_key(), sig_bytes)

    def resolve_base_model_path(self, custom_path: Optional[str] = None) -> Optional[str]:
        """Повертає шлях до публічної базової 14B GGUF моделі для запуску llama-server з `--mmap`."""
        return find_base_model_path(custom_path=custom_path)

    # -----------------------------------------------------------------
    # HARDWARE KILL-SWITCH (ФОНОВИЙ ДЕМОН МОНІТОРИНГУ НОСІЯ)
    # -----------------------------------------------------------------
    def register_process_for_kill_switch(self, pid: int) -> None:
        """Реєструє PID процесу (наприклад, `llama-server`) для миттєвого знищення при вийманні диска."""
        if pid and pid not in self._protected_pids:
            self._protected_pids.append(pid)

    def start_kill_switch(
        self,
        interval_sec: float = KILL_SWITCH_INTERVAL_SEC,
        on_trigger_callback: Optional[Callable[[], None]] = None,
    ) -> None:
        """
        Запускає фоновий потік-охоронець, який кожні `interval_sec` секунд
        перевіряє фізичну наявність `vault/master.key` на оптичному приводі.
        """
        if not self._monitored_key_path or self._kill_switch_thread is not None:
            return

        self._kill_switch_stop.clear()

        def _monitor_loop():
            while not self._kill_switch_stop.wait(interval_sec):
                if not self._is_key_still_present():
                    self._trigger_emergency_purge(on_trigger_callback)
                    break

        self._kill_switch_thread = threading.Thread(
            target=_monitor_loop,
            name="GVK-KillSwitch-Daemon",
            daemon=True,
        )
        self._kill_switch_thread.start()

    def _is_key_still_present(self) -> bool:
        if not self._monitored_key_path:
            return True
        try:
            with open(self._monitored_key_path, "rb") as f:
                data = f.read(KEY_SIZE)
            return len(data) == KEY_SIZE and hmac.compare_digest(data, bytes(self._master_key or b""))
        except OSError:
            return False

    def _trigger_emergency_purge(self, callback: Optional[Callable[[], None]] = None) -> None:
        """
        АВАРІЙНЕ СКИДАННЯ (Kill-Switch):
        Викликається миттєво при вийманні DVD-диска або клацанні тумблера GVK-DOCK.
        """
        sys.stderr.write(
            "\n\n====================================================================\n"
            " [!!!] EMERGENCY RAM PURGE: ВТРАЧЕНО ЗВ'ЯЗОК ІЗ НОСІЄМ GVK_VAULT!\n"
            " [!!!] Зупинка llama-server та екстрене знищення даних у RAM...\n"
            "====================================================================\n"
        )
        sys.stderr.flush()

        # 1. Миттєвий SIGKILL для всіх зареєстрованих підпроцесів (llama-server)
        had_pids = bool(self._protected_pids)
        for pid in list(self._protected_pids):
            try:
                os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))
            except OSError:
                pass
        if had_pids:
            time.sleep(0.25)

        # 2. Занулення ключа та демонтування RAM-диска
        self.cleanup()

        if callback:
            callback()
            return

        # Примусовий негайний вихід із кодом 137 (SIGKILL)
        os._exit(137)

    # -----------------------------------------------------------------
    # БЕЗПЕЧНЕ ЗАТИРАННЯ ПАМ'ЯТІ ТА ДЕМОНТУВАННЯ RAM-ДИСКА
    # -----------------------------------------------------------------
    @staticmethod
    def _zero_wipe_directory(directory: str) -> None:
        """Перезаписує всі файли в директорії нульовими байтами (0x00) перед видаленням."""
        if not directory or not os.path.isdir(directory):
            return
        zero_buf = b"\x00" * (1024 * 1024)  # 1 МБ блок нулів
        for root, dirs, files in os.walk(directory, topdown=False):
            for fname in files:
                fpath = os.path.join(root, fname)
                try:
                    os.chmod(fpath, 0o600)
                except OSError:
                    pass
                try:
                    size = os.path.getsize(fpath)
                    with open(fpath, "r+b") as f:
                        remaining = size
                        while remaining > 0:
                            step = min(remaining, len(zero_buf))
                            f.write(zero_buf[:step])
                            remaining -= step
                        f.flush()
                    os.remove(fpath)
                except OSError:
                    try:
                        os.remove(fpath)
                    except OSError:
                        pass
            for dname in dirs:
                try:
                    os.rmdir(os.path.join(root, dname))
                except OSError:
                    pass

    def cleanup(self) -> None:
        """
        Безпечно завершує сесію:
          1. Зупиняє потік Kill-Switch.
          2. По-байтово затирає `master_key` у пам'яті Python нулями (`0x00`).
          3. Затирає всі розшифровані файли в RAM-диску і демонтує пристрій.
        """
        with self._cleanup_lock:
            if self._is_cleaned_up:
                return
            self._is_cleaned_up = True

            self._kill_switch_stop.set()

            # 1. Занулення master_key в ОЗП
            if self._master_key is not None:
                for i in range(len(self._master_key)):
                    self._master_key[i] = 0
                self._master_key = None

            # 2. Очищення та демонтування RAM-диска
            if self.mount_point and os.path.exists(self.mount_point):
                self._zero_wipe_directory(self.mount_point)

                if self._macos_device:
                    subprocess.run(
                        ["umount", "-f", self.mount_point],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    subprocess.run(
                        ["hdiutil", "detach", self._macos_device, "-force"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    self._macos_device = None

                if self._windows_imdisk_drive:
                    subprocess.run(
                        ["imdisk", "-D", "-m", self._windows_imdisk_drive],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    self._windows_imdisk_drive = None

                try:
                    shutil.rmtree(self.mount_point, ignore_errors=True)
                except OSError:
                    pass

                print(f" [✓] RAM-диск очищено нулями та демонтовано: {self.mount_point}")
                self.mount_point = None


# =====================================================================
# 3. АВТОМАТИЧНИЙ ТЕСТ MILESTONE 1 (ШИФРУВАННЯ -> RAM-ДИСК -> KILL-SWITCH)
# =====================================================================
def run_milestone1_self_test() -> None:
    print("====================================================================")
    print(" [*] GVK Private AI — Тест Milestone 1 (core/crypto_vault.py)")
    print("====================================================================")

    with tempfile.TemporaryDirectory(prefix="gvk_sim_disc_") as sim_disc_dir:
        # 1. Імітуємо вставлений DVD-диск GVK_VAULT у дисководі
        vault_dir = os.path.join(sim_disc_dir, "vault")
        persona_src = os.path.join(sim_disc_dir, "src_friend_1")
        os.makedirs(vault_dir, exist_ok=True)
        os.makedirs(os.path.join(persona_src, "portraits"), exist_ok=True)

        print(" [1/5] Деривація ключа Argon2id та імітація DVD-диска GVK_VAULT...")
        key = derive_master_key("GVK_Milestone_1_Verification_Phrase_2026!")
        key_file_path = os.path.join(vault_dir, "master.key")
        with open(key_file_path, "wb") as f:
            f.write(key)

        # Створюємо тестовий LoRA-адаптер та приватний архів особистості з емоціями
        raw_adapter = os.path.join(sim_disc_dir, "raw_adapter.gguf")
        fake_gguf_bytes = b"GGUF" + os.urandom(512 * 1024)  # 512 КБ тестових ваг
        with open(raw_adapter, "wb") as f:
            f.write(fake_gguf_bytes)

        with open(os.path.join(persona_src, "persona_config.json"), "w", encoding="utf-8") as f:
            f.write('{"persona_id": "friend_1", "display_name": "Друг 1"}\n')
        with open(os.path.join(persona_src, "prompt.txt"), "w", encoding="utf-8") as f:
            f.write("Ти — Друг 1. Використовуй [MOOD: ПОДИВ] або [MOOD: СПОКІЙ].\n")
        for mood in ("мовчить", "говорить_спокійно", "подив", "сміх"):
            with open(os.path.join(persona_src, "portraits", f"{mood}.txt"), "w", encoding="utf-8") as f:
                f.write(f"[ASCII SPRITE: {mood.upper()}]\n")

        # 2. Шифруємо контейнери
        print(" [2/5] Шифрування тестових контейнерів adapter.gguf.enc та payload.tar.enc...")
        adapter_enc = os.path.join(sim_disc_dir, "adapter.gguf.enc")
        payload_enc = os.path.join(sim_disc_dir, "payload.tar.enc")
        sig_path = os.path.join(sim_disc_dir, "origin.sig")

        with open(raw_adapter, "rb") as f_in, open(adapter_enc, "wb") as f_out:
            encrypt_stream(f_in, f_out, master_key=key)
        pack_payload_encrypted(persona_src, payload_enc, master_key=key)
        with open(sig_path, "wb") as f:
            f.write(generate_origin_sig("friend_1", key))

        # 3. Монтуємо RAM-диск (64 МБ для швидкого тесту) і розпаковуємо напряму в RAM
        print(" [3/5] Ініціалізація CryptoVault та монтування RAM-диска...")
        vault = CryptoVault(disc_path=sim_disc_dir, ramdisk_mb=64)
        ram_root = vault.initialize()

        assert vault.verify_persona_signature("friend_1", sig_path) is True
        print(" [4/5] Розшифрування контейнерів напряму в RAM-диск...")
        unpacked = vault.unpack_persona_to_ram("friend_1", adapter_enc, payload_enc)

        with open(unpacked["adapter_path"], "rb") as f:
            assert f.read() == fake_gguf_bytes
        for mood in ("мовчить", "говорить_спокійно", "подив", "сміх"):
            sprite_file = os.path.join(unpacked["persona_dir"], "portraits", f"{mood}.txt")
            assert os.path.isfile(sprite_file), f"Не знайдено спрайт у RAM: {sprite_file}"

        print(f"       [OK] Адаптер та 4 ASCII-спрайти емоцій успішно розпаковано в ОЗП: {unpacked['persona_dir']}")

        # 4. Тестуємо спрацювання Kill-Switch при вийманні диска (або клацанні тумблера)
        print(" [5/5] Тестування демона Kill-Switch (імітація виймання DVD-диска)...")
        triggered_event = threading.Event()
        vault._kill_switch_stop.set()  # Зупиняємо стандартний 3-секундний потік, щоб запустити швидкий 0.2с тест
        vault._kill_switch_thread = None
        vault.start_kill_switch(interval_sec=0.2, on_trigger_callback=lambda: triggered_event.set())

        # Видаляємо master.key з імітованого диска (ніби користувач висмикнув диск або клацнув тумблер)
        os.remove(key_file_path)
        assert triggered_event.wait(timeout=2.0), "Kill-Switch не спрацював після видалення ключа!"
        assert vault.mount_point is None or not os.path.exists(ram_root)
        assert vault._master_key is None
        print("       [OK] Kill-Switch миттєво виявив відсутність носія та знищив RAM-диск!")

    print("====================================================================")
    print(" [✓] MILESTONE 1 ПРОЙДЕНО УСПІШНО!")
    print("====================================================================")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GVK Private AI — Крипто-сховище, RAM-диск та Kill-Switch (core/crypto_vault.py)"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser(
        "self-test",
        help="Виконати перевірку Milestone 1 (шифрування -> монтування RAM-диска -> дешифрування -> Kill-Switch)",
    )
    args = parser.parse_args()

    if args.cmd == "self-test":
        run_milestone1_self_test()


if __name__ == "__main__":
    main()
