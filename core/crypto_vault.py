#!/usr/bin/env python3
"""
GVK Private AI - Crypto Vault, RAM Disk & Kill-Switch Daemon (core/crypto_vault.py)

Runtime security module enforcing the Zero-Plaintext-At-Rest principle by
keeping decrypted LoRA adapters, persona archives, and cryptographic keys
strictly in volatile memory.

Key Responsibilities:
  - Master Key Acquisition (32 bytes): auto-detect mounted `GVK_VAULT` optical
    or USB media (`vault/master.key`, ~0.01 s) or derive the key from a secret
    passphrase via Argon2id (256 MB RAM).
  - Cross-Platform RAM Disk Mounting (512-1024 MB for 16 GB RAM hosts):
    macOS (`hdiutil attach -nomount ram://...`, HFS+, mode 0700, Spotlight
    disabled), Linux (`tmpfs` under `/dev/shm/gvk_runtime_<pid>`, chmod 700),
    and Windows (ImDisk virtual RAM disk or isolated session directory with
    0x00 zero-wipe on exit).
  - Streaming AES-256-GCM Decryption (64 MB chunks) directly into the RAM disk
    for personal LoRA adapters (`adapter.gguf.enc`) and private persona archives
    (`payload.tar.enc`), while the public base model (`Qwen2.5-14B-Instruct-Q4_K_M.gguf`)
    is memory-mapped directly from SSD (`~/.gvk_ai/base/`) via `--mmap`.
  - Hardware Kill-Switch Daemon (3.0 s heartbeat): monitors `vault/master.key`
    on the physical media and triggers an emergency `SIGKILL` of `llama-server`,
    zero-wipes keys in RAM, detaches the RAM disk, and exits immediately if the
    disc is removed or the `GVK-DOCK // MK.I` kill switch is tripped.

Usage:
  python3 core/crypto_vault.py self-test
  python3 core/crypto_vault.py --help
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

# Add repository root to sys.path when running `python3 core/crypto_vault.py` directly
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

DEFAULT_RAMDISK_MB = 1024  # 1 GB for 3 LoRAs (~420 MB) + FAISS/prompts/portraits; 14B base uses --mmap
DEFAULT_BASE_MODEL_FILENAME = "Qwen2.5-14B-Instruct-Q4_K_M.gguf"
KILL_SWITCH_INTERVAL_SEC = 3.0


# =====================================================================
# 1. GVK_VAULT MEDIA DISCOVERY & MASTER KEY ACQUISITION
# =====================================================================
def find_base_model_path(
    custom_path: Optional[str] = None,
    gvk_home: Optional[str] = None,
) -> Optional[str]:
    """Locate the public base GGUF model for `llama-server --mmap`.

    Search priority:
      1. Explicitly provided `custom_path`.
      2. `GVK_BASE_MODEL` environment variable.
      3. `~/.gvk_ai/base/Qwen2.5-14B-Instruct-Q4_K_M.gguf` (or any `.gguf` in `~/.gvk_ai/base/`).
      4. Local repository paths (`base/Qwen2.5-14B-Instruct-Q4_K_M.gguf`, `base/*.gguf`, `base_model.gguf`).

    Args:
        custom_path: Optional explicit path to a GGUF model file.
        gvk_home: Optional override for the `~/.gvk_ai` home directory.

    Returns:
        Absolute path to the first non-empty GGUF model file found, or `None`.
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
    """Normalize a disc mount path (e.g., Windows `'D:'` -> `'D:\\'` before `abspath`)."""
    p = os.path.expanduser(raw_path.strip())
    if len(p) == 2 and p[1] == ":" and p[0].isalpha():
        p = p + "\\"
    return os.path.abspath(p)


def find_gvk_disc_path(explicit_path: Optional[str] = None) -> Optional[str]:
    """Locate a mounted `GVK_VAULT` optical disc or USB drive containing `vault/master.key`.

    Args:
        explicit_path: Optional explicit mount path to check first.

    Returns:
        Path to the detected media root directory, or `None` if not found.
    """
    candidates: List[str] = []
    if explicit_path:
        candidates.append(_normalize_disc_path(explicit_path))

    env_disc = os.environ.get("GVK_DISC_PATH")
    if env_disc:
        candidates.append(_normalize_disc_path(env_disc))

    sys_name = platform.system().lower()
    if sys_name == "darwin":
        # macOS mounts optical and removable volumes under /Volumes/<VolumeName>
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
    """Acquire the 32-byte master encryption key from hardware media or passphrase.

    Reads `vault/master.key` directly (~0.01 s) when `GVK_VAULT` media is
    mounted; otherwise prompts for (or uses) a secret master passphrase and
    derives the key via Argon2id.

    Args:
        disc_path: Optional explicit path to mounted `GVK_VAULT` media.
        passphrase: Optional secret passphrase for headless fallback derivation.
        allow_short: Allow short passphrases during automated tests.

    Returns:
        A 2-tuple `(key_buf, key_file_path)` where `key_buf` is a mutable
        `bytearray` suitable for in-place zero-wiping, and `key_file_path` is
        the monitored `vault/master.key` path on media (or `None` in passphrase mode).

    Raises:
        ValueError: If `vault/master.key` on media has an invalid byte length.
    """
    detected_disc = find_gvk_disc_path(disc_path)
    if detected_disc:
        key_file = os.path.join(detected_disc, "vault", "master.key")
        with open(key_file, "rb") as f:
            raw_key = f.read()
        if len(raw_key) != KEY_SIZE:
            raise ValueError(f"Corrupted master.key on media ({len(raw_key)} bytes instead of {KEY_SIZE}).")
        is_usb = (
            os.path.isdir(os.path.join(detected_disc, "base"))
            or os.path.isdir(os.path.join(detected_disc, "bin"))
            or os.path.isfile(os.path.join(detected_disc, "vault", "media_mode.json"))
        )
        mode_tag = "USB" if is_usb else "DVD"
        print(f" [OK] Hardware access key GVK_VAULT [{mode_tag}] read from media: {detected_disc}")
        return bytearray(raw_key), key_file

    # Fallback: interactive crypto challenge / manual master passphrase entry
    if passphrase is None:
        print("====================================================================")
        print(" [!] Physical GVK_VAULT media with access key not found.")
        print(" [*] Falling back to manual key derivation (Argon2id -> AES-256-GCM).")
        print("====================================================================")
        passphrase = getpass.getpass(" [?] Enter secret master passphrase to unlock personas -> ")

    print(" [*] Deriving 256-bit key via Argon2id (256 MB RAM)...")
    derived = derive_master_key(passphrase, allow_short=allow_short)
    return bytearray(derived), None


# =====================================================================
# 2. CROSS-PLATFORM RAM DISK & CRYPTO VAULT MANAGER
# =====================================================================
class CryptoVault:
    """Manage the encrypted runtime session, volatile RAM disk, and Kill-Switch.

    Handles mounting an isolated RAM disk (512-1024 MB across macOS, Linux, and
    Windows), resolving the public base model on SSD for `--mmap`, streaming
    AES-256-GCM decryption of persona secrets into RAM, background hardware
    media monitoring (Kill-Switch), and guaranteed zero-wipe cleanup on exit.

    Example:
        >>> with CryptoVault(disc_path="/Volumes/GVK_VAULT") as vault:
        ...     paths = vault.unpack_persona_to_ram("friend_1", adapter_enc, payload_enc)
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
        """Acquire the master key, mount the RAM disk, and start Kill-Switch if disc-backed.

        Returns:
            Absolute path to the mounted RAM disk (or isolated session directory).
        """
        self._master_key, self._monitored_key_path = acquire_master_key(
            disc_path=self.disc_path,
            passphrase=self._initial_passphrase,
            allow_short=self._allow_short,
        )
        # Immediately discard the plaintext passphrase reference from memory
        self._initial_passphrase = None

        self.mount_point = self.mount_ramdisk(self.ramdisk_mb)
        atexit.register(self.cleanup)

        if self._monitored_key_path:
            self.start_kill_switch()

        return self.mount_point

    # -----------------------------------------------------------------
    # RAM DISK MOUNTING (Linux / macOS / Windows)
    # -----------------------------------------------------------------
    def mount_ramdisk(self, size_mb: int) -> str:
        """Create and mount an isolated volatile storage area in RAM.

        No decrypted file inside this partition is written to persistent SSD
        storage when native RAM disk support is available.

        Args:
            size_mb: Requested RAM disk capacity in megabytes.

        Returns:
            Absolute path to the mounted RAM disk directory (or isolated fallback).
        """
        sys_name = platform.system().lower()
        pid = os.getpid()

        if sys_name == "linux":
            shm_base = "/dev/shm" if os.path.isdir("/dev/shm") else "/run/shm"
            if os.path.isdir(shm_base) and os.access(shm_base, os.W_OK):
                mount_dir = os.path.join(shm_base, f"gvk_runtime_{pid}")
                os.makedirs(mount_dir, mode=0o700, exist_ok=True)
                os.chmod(mount_dir, 0o700)
                print(f" [OK] RAM disk (Linux tmpfs) mounted at: {mount_dir}")
                return mount_dir

        elif sys_name == "darwin":
            # macOS: allocate an in-memory block device via `hdiutil attach -nomount ram://<sectors>`
            # 1 sector = 512 bytes -> size_mb * 2048 sectors
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

                    # Format block device as HFS+ and mount into a private 0700 directory
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

                    # Disable macOS Spotlight indexing on the RAM volume
                    open(os.path.join(mount_dir, ".metadata_never_index"), "wb").close()
                    subprocess.run(
                        ["mdutil", "-i", "off", mount_dir],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    print(f" [OK] RAM disk (macOS {size_mb} MB, device {device}) mounted at: {mount_dir}")
                    return mount_dir
            except Exception as exc:
                # Fall back to an isolated session directory if sandboxed without hdiutil access
                if self._macos_device:
                    subprocess.run(
                        ["hdiutil", "detach", self._macos_device, "-force"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    self._macos_device = None
                print(f" [!] Note: hardware hdiutil unavailable ({exc}); activated isolated session buffer.")

        elif sys_name == "windows":
            # Attempt to create a virtual RAM drive via ImDisk when installed
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
                            print(f" [OK] RAM disk (Windows ImDisk {size_mb} MB) mounted at: {mount_dir}")
                            return mount_dir
                        except Exception:
                            break

        # Universal isolated fallback (mode 0700 with forced 0x00 zero-wipe on cleanup)
        fallback_dir = tempfile.mkdtemp(prefix=f"gvk_ram_session_{pid}_")
        os.chmod(fallback_dir, 0o700)
        try:
            open(os.path.join(fallback_dir, ".metadata_never_index"), "wb").close()
        except OSError:
            pass
        print(f" [OK] Isolated session directory created at: {fallback_dir}")
        return fallback_dir

    # -----------------------------------------------------------------
    # STREAMING CONTAINER DECRYPTION TO RAM DISK
    # -----------------------------------------------------------------
    def _require_key(self) -> bytes:
        if not self._master_key or len(self._master_key) != KEY_SIZE:
            raise RuntimeError("CryptoVault is not initialized or master key has already been wiped!")
        return bytes(self._master_key)

    def decrypt_container_to_ram(self, enc_file_path: str, relative_ram_path: str) -> str:
        """Stream-decrypt an `.enc` container in 64 MB chunks directly into the RAM disk.

        Args:
            enc_file_path: Path to the encrypted `.enc` container (e.g., `adapter.gguf.enc`).
            relative_ram_path: Relative destination path inside `self.mount_point`.

        Returns:
            Absolute path to the decrypted file inside the RAM disk.

        Raises:
            RuntimeError: If the RAM disk is not mounted or the vault is uninitialized.
            FileNotFoundError: If `enc_file_path` does not exist.
        """
        if not self.mount_point:
            raise RuntimeError("RAM disk is not mounted!")
        if not os.path.isfile(enc_file_path):
            raise FileNotFoundError(f"Encrypted container not found: {enc_file_path}")

        key = self._require_key()
        dst_path = os.path.join(self.mount_point, relative_ram_path.lstrip("/\\"))
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        with open(enc_file_path, "rb") as f_in, open(dst_path, "wb") as f_out:
            os.chmod(dst_path, 0o600)
            decrypt_stream(f_in, f_out, master_key=key, label=f"Decrypting {os.path.basename(enc_file_path)}")

        return dst_path

    def unpack_persona_to_ram(self, persona_id: str, adapter_enc_path: str, payload_enc_path: str) -> dict:
        """Decrypt a persona's LoRA adapter and payload archive directly into the RAM disk.

        Args:
            persona_id: Target persona identifier (e.g., `"friend_1"`).
            adapter_enc_path: Path to encrypted `adapter.gguf.enc`.
            payload_enc_path: Path to encrypted `payload.tar.enc`.

        Returns:
            Dictionary containing RAM paths:
            - `persona_id`: Persona identifier.
            - `adapter_path`: `<ram>/adapters/<persona_id>.gguf`.
            - `persona_dir`: `<ram>/personas/<persona_id>/`.

        Raises:
            RuntimeError: If the RAM disk is not mounted.
        """
        if not self.mount_point:
            raise RuntimeError("RAM disk is not mounted!")

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
        """Verify the HMAC `origin.sig` signature for `persona_id`.

        Args:
            persona_id: Persona identifier bound to the signature.
            sig_file_path: Path to the 32-byte `.sig` file.

        Returns:
            `True` if the signature exists, has valid length, and matches the master key.
        """
        if not os.path.isfile(sig_file_path):
            return False
        with open(sig_file_path, "rb") as f:
            sig_bytes = f.read()
        if len(sig_bytes) != 32:
            return False
        return verify_origin_sig(persona_id, self._require_key(), sig_bytes)

    def resolve_base_model_path(self, custom_path: Optional[str] = None) -> Optional[str]:
        """Resolve the path to the public 14B base GGUF model for `llama-server --mmap`."""
        return find_base_model_path(custom_path=custom_path)

    # -----------------------------------------------------------------
    # HARDWARE KILL-SWITCH (BACKGROUND MEDIA MONITOR DAEMON)
    # -----------------------------------------------------------------
    def register_process_for_kill_switch(self, pid: int) -> None:
        """Register a child process PID (e.g., `llama-server`) for immediate termination on media removal."""
        if pid and pid not in self._protected_pids:
            self._protected_pids.append(pid)

    def start_kill_switch(
        self,
        interval_sec: float = KILL_SWITCH_INTERVAL_SEC,
        on_trigger_callback: Optional[Callable[[], None]] = None,
    ) -> None:
        """Start the background watchdog thread polling `vault/master.key` on media.

        Args:
            interval_sec: Polling interval in seconds (default 3.0 s).
            on_trigger_callback: Optional callback invoked instead of `os._exit(137)`
                when the Kill-Switch triggers (primarily for automated tests).
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
        """Execute an emergency Kill-Switch purge when GVK_VAULT media is disconnected.

        Terminates registered subprocesses (`SIGKILL`), zero-wipes the master key
        and RAM disk, and exits with status 137 unless `callback` is provided.
        """
        sys.stderr.write(
            "\n\n====================================================================\n"
            " [!] EMERGENCY RAM PURGE: LOST CONNECTION TO GVK_VAULT MEDIA!\n"
            " [!] Stopping llama-server and urgently wiping RAM data...\n"
            "====================================================================\n"
        )
        sys.stderr.flush()

        # 1. Immediate SIGKILL for all registered child processes (e.g., llama-server)
        had_pids = bool(self._protected_pids)
        for pid in list(self._protected_pids):
            try:
                os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))
            except OSError:
                pass
        if had_pids:
            time.sleep(0.25)

        # 2. Zero-wipe master key and unmount the RAM disk
        self.cleanup()

        if callback:
            callback()
            return

        # Force immediate process exit with code 137 (SIGKILL)
        os._exit(137)

    # -----------------------------------------------------------------
    # SECURE MEMORY ZERO-WIPE & RAM DISK UNMOUNTING
    # -----------------------------------------------------------------
    @staticmethod
    def _zero_wipe_directory(directory: str) -> None:
        """Overwrite all files inside `directory` with zero bytes (`0x00`) before deletion."""
        if not directory or not os.path.isdir(directory):
            return
        zero_buf = b"\x00" * (1024 * 1024)  # 1 MB zero buffer
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
        """Safely tear down the vault session.

        Stops the Kill-Switch thread, overwrites `master_key` in memory with
        zero bytes (`0x00`), zero-wipes all decrypted files on the RAM disk,
        and detaches/removes the RAM disk mount point.
        """
        with self._cleanup_lock:
            if self._is_cleaned_up:
                return
            self._is_cleaned_up = True

            self._kill_switch_stop.set()

            # 1. Zero-wipe master_key in RAM
            if self._master_key is not None:
                for i in range(len(self._master_key)):
                    self._master_key[i] = 0
                self._master_key = None

            # 2. Zero-wipe and unmount the RAM disk
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

                print(f" [OK] RAM disk zero-wiped and unmounted: {self.mount_point}")
                self.mount_point = None


# =====================================================================
# 3. AUTOMATED MILESTONE 1 SELF-TEST (ENCRYPT -> RAM DISK -> KILL-SWITCH)
# =====================================================================
def run_milestone1_self_test() -> None:
    """Run the Milestone 1 end-to-end security self-test.

    Simulates an inserted `GVK_VAULT` disc, encrypts a test LoRA adapter and
    persona payload, mounts a RAM disk, verifies signature and streaming
    decryption, and confirms that removing `master.key` triggers the Kill-Switch.
    """
    print("====================================================================")
    print(" [*] GVK Private AI - Milestone 1 Self-Test (core/crypto_vault.py)")
    print("====================================================================")

    with tempfile.TemporaryDirectory(prefix="gvk_sim_disc_") as sim_disc_dir:
        # 1. Simulate an inserted GVK_VAULT DVD disc
        vault_dir = os.path.join(sim_disc_dir, "vault")
        persona_src = os.path.join(sim_disc_dir, "src_friend_1")
        os.makedirs(vault_dir, exist_ok=True)
        os.makedirs(os.path.join(persona_src, "portraits"), exist_ok=True)

        print(" [1/5] Deriving Argon2id key and simulating GVK_VAULT DVD media...")
        key = derive_master_key("GVK_Milestone_1_Verification_Phrase_2026!")
        key_file_path = os.path.join(vault_dir, "master.key")
        with open(key_file_path, "wb") as f:
            f.write(key)

        # Create a synthetic LoRA adapter and private persona archive with emotion sprites
        raw_adapter = os.path.join(sim_disc_dir, "raw_adapter.gguf")
        fake_gguf_bytes = b"GGUF" + os.urandom(512 * 1024)  # 512 KB synthetic weights
        with open(raw_adapter, "wb") as f:
            f.write(fake_gguf_bytes)

        with open(os.path.join(persona_src, "persona_config.json"), "w", encoding="utf-8") as f:
            f.write('{"persona_id": "friend_1", "display_name": "Друг 1"}\n')
        with open(os.path.join(persona_src, "prompt.txt"), "w", encoding="utf-8") as f:
            f.write("Ти — Друг 1. Використовуй [MOOD: ПОДИВ] або [MOOD: СПОКІЙ].\n")
        for mood in ("мовчить", "говорить_спокійно", "подив", "сміх"):
            with open(os.path.join(persona_src, "portraits", f"{mood}.txt"), "w", encoding="utf-8") as f:
                f.write(f"[ASCII SPRITE: {mood.upper()}]\n")

        # 2. Encrypt containers
        print(" [2/5] Encrypting test containers adapter.gguf.enc and payload.tar.enc...")
        adapter_enc = os.path.join(sim_disc_dir, "adapter.gguf.enc")
        payload_enc = os.path.join(sim_disc_dir, "payload.tar.enc")
        sig_path = os.path.join(sim_disc_dir, "origin.sig")

        with open(raw_adapter, "rb") as f_in, open(adapter_enc, "wb") as f_out:
            encrypt_stream(f_in, f_out, master_key=key)
        pack_payload_encrypted(persona_src, payload_enc, master_key=key)
        with open(sig_path, "wb") as f:
            f.write(generate_origin_sig("friend_1", key))

        # 3. Mount RAM disk (64 MB for fast test) and unpack directly into RAM
        print(" [3/5] Initializing CryptoVault and mounting RAM disk...")
        vault = CryptoVault(disc_path=sim_disc_dir, ramdisk_mb=64)
        ram_root = vault.initialize()

        assert vault.verify_persona_signature("friend_1", sig_path) is True
        print(" [4/5] Decrypting containers directly into RAM disk...")
        unpacked = vault.unpack_persona_to_ram("friend_1", adapter_enc, payload_enc)

        with open(unpacked["adapter_path"], "rb") as f:
            assert f.read() == fake_gguf_bytes
        for mood in ("мовчить", "говорить_спокійно", "подив", "сміх"):
            sprite_file = os.path.join(unpacked["persona_dir"], "portraits", f"{mood}.txt")
            assert os.path.isfile(sprite_file), f"Sprite file not found in RAM: {sprite_file}"

        print(f"       [OK] Adapter and 4 ASCII emotion sprites unpacked into RAM: {unpacked['persona_dir']}")

        # 4. Verify Kill-Switch activation when media is removed (or toggle switch tripped)
        print(" [5/5] Testing Kill-Switch daemon (simulating DVD removal)...")
        triggered_event = threading.Event()
        vault._kill_switch_stop.set()  # Stop default 3 s loop to run a fast 0.2 s test
        vault._kill_switch_thread = None
        vault.start_kill_switch(interval_sec=0.2, on_trigger_callback=lambda: triggered_event.set())

        # Remove master.key from simulated disc to simulate media ejection
        os.remove(key_file_path)
        assert triggered_event.wait(timeout=2.0), "Kill-Switch did not trigger after key removal!"
        assert vault.mount_point is None or not os.path.exists(ram_root)
        assert vault._master_key is None
        print("       [OK] Kill-Switch detected missing media and purged the RAM disk!")

    print("====================================================================")
    print(" [OK] MILESTONE 1 PASSED SUCCESSFULLY!")
    print("====================================================================")


def main() -> None:
    """Parse CLI arguments and run CryptoVault commands."""
    parser = argparse.ArgumentParser(
        description="GVK Private AI - Crypto Vault, RAM Disk & Kill-Switch (core/crypto_vault.py)"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser(
        "self-test",
        help="Run Milestone 1 verification (encrypt -> mount RAM disk -> decrypt -> Kill-Switch)",
    )
    args = parser.parse_args()

    if args.cmd == "self-test":
        run_milestone1_self_test()


if __name__ == "__main__":
    main()
