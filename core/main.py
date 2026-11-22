#!/usr/bin/env python3
"""
GVK Private AI — Головна точка входу (core/main.py)

Поєднує:
  - Живе логування в термінал та у папку `logs/` (`logs/gvk_session_*.log`).
  - Автоматичне визначення шляхів для локального тестування (`personas/friend_1/`, `adapters/friend_1.gguf`)
    та продакшн-середовища користувача (`~/.gvk_ai/` + RAM-диск).
  - Ініціалізацію `GVKOrchestrator` (`core/orchestrator.py`) та термінального GUI (`core/menu.py`).

Приклади запуску:
  python3 core/main.py
  python3 core/main.py --fast
  python3 core/main.py --fast --start-server
  python3 core/main.py self-test
"""

import argparse
import datetime
import json
import logging
import os
import sys
from typing import List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.crypto_vault import CryptoVault, find_gvk_disc_path
from core.memory_rag import MemoryRAG, build_rag_index
from core.menu import (
    DEFAULT_AFK_SECONDS,
    DEFAULT_SMILE_SECONDS,
    DEFAULT_WAIT_SECONDS,
    LAST_WARNING_WAIT_SECONDS,
    PERSONA_SELECT_ITEMS,
    USERS_DICT,
    ChatSessionModal,
    GVKMenuApp,
    MainMenuModal,
    PersonaLoadingModal,
    detect_reply_emotion,
    run_self_test as run_menu_self_test,
)
from core.orchestrator import DEFAULT_RAG_HEADER, GVKOrchestrator, calculate_age
from updater import DEFAULT_GVK_HOME, detect_media_mode


# =====================================================================
# НАЛАШТУВАННЯ РЕАЛТАЙМ-ЛОГУВАННЯ (ТЕРМІНАЛ + ПАПКА LOGS/)
# =====================================================================
def setup_logging(repo_root: str = REPO_ROOT, gvk_home: str = DEFAULT_GVK_HOME) -> str:
    """
    Налаштовує одночасний вивід логів у реальному часі в термінал (stdout)
    та збереження у файл `logs/gvk_session_<timestamp>.log`.
    """
    logs_dir = os.path.join(repo_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_log_path = os.path.join(logs_dir, f"gvk_session_{timestamp}.log")
    latest_log_path = os.path.join(logs_dir, "gvk_latest.log")

    root_logger = logging.getLogger("gvk")
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # 1. Реалтайм-вивід у термінал
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    root_logger.addHandler(console_handler)

    # 2. Запис у файл сесії в папці logs/
    file_handler = logging.FileHandler(session_log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    root_logger.addHandler(file_handler)

    # 3. Дублювання в logs/gvk_latest.log
    latest_handler = logging.FileHandler(latest_log_path, mode="w", encoding="utf-8")
    latest_handler.setFormatter(fmt)
    root_logger.addHandler(latest_handler)

    # Якщо у користувача вже створено ~/.gvk_ai/logs, дублюємо лог і туди
    user_logs_dir = os.path.join(os.path.abspath(os.path.expanduser(gvk_home)), "logs")
    if os.path.isdir(user_logs_dir) and os.path.abspath(user_logs_dir) != os.path.abspath(logs_dir):
        try:
            user_log_path = os.path.join(user_logs_dir, f"gvk_session_{timestamp}.log")
            user_handler = logging.FileHandler(user_log_path, encoding="utf-8")
            user_handler.setFormatter(fmt)
            root_logger.addHandler(user_handler)
        except OSError:
            pass

    root_logger.info("Логування ініціалізовано. Файл логу: %s", session_log_path)
    return session_log_path


# =====================================================================
# АВТОМАТИЧНА САМОПЕРЕВІРКА (SELF-TEST)
# =====================================================================
def run_self_test() -> None:
    log_path = setup_logging()
    logger = logging.getLogger("gvk.main")
    logger.info("Запуск комплексного self-test (main.py + orchestrator.py + menu.py)...")

    # 1. Перевірка оркестратора та побудови системного промпту для friend_1
    orch = GVKOrchestrator(persona_id="friend_1")
    assert orch.config.get("persona_id") == "friend_1"
    assert orch.config.get("color") == "#78b068"
    expected_genitive = str(orch.config.get("genitive_name") or PERSONA_SELECT_ITEMS[0].get("genitive_name") or "")
    assert expected_genitive and orch.get_genitive_name() == expected_genitive
    loading_tips = orch.get_loading_tips()
    assert len(loading_tips) >= 3
    assert len(orch.get_warmup_questions()) == 2
    assert orch.config.get("emotional_states") == ["idle", "sleeping", "smiling", "talking"]
    assert "talking2.txt" in (orch.config.get("emotion_sprites", {}).get("talking") or [])

    cmd = orch.get_llama_server_command()
    assert "-ngl" not in cmd, f"Не повинно бути аргументу -ngl у команді: {cmd}"
    assert "--lora-scaled" in cmd, f"Очікувався аргумент --lora-scaled у команді: {cmd}"
    lora_idx = cmd.index("--lora-scaled")
    expected_scale = str(orch.config.get("lora_scaled", 1.0))
    assert cmd[lora_idx + 1].endswith(
        f":{expected_scale}"
    ), f"Очікувався масштаб :{expected_scale} для friend_1, отримано: {cmd[lora_idx + 1]}"

    warmup_payload = orch.build_warmup_payload(orch.get_warmup_questions()[0])
    assert warmup_payload.get("chat_template_kwargs") == {"enable_thinking": False}

    test_date = datetime.date(2026, 11, 17)
    age_on_test_date = calculate_age("01.01.2000", today=test_date)
    assert age_on_test_date == 26, f"Очікувався вік 26 на 17.11.2026, отримано {age_on_test_date}"

    # Неекспериментальний користувач [03] -> повний опис (якщо заданий у ростері/конфігу)
    u3_name = str(USERS_DICT[3]["name"])
    expected_u3_desc = orch.resolve_interlocutor_string(user_name_ua=u3_name, is_experimental=False)
    orch.set_interlocutor(user_id=3, user_name_ua=u3_name, is_experimental=False)
    sys_prompt_main = orch.build_system_prompt(today=test_date)
    expected_persona_age = calculate_age(str(orch.config.get("birth_date", "01.01.2000")), today=test_date)
    assert f"Сегодняшняя дата: 17.11.2026, тебе {expected_persona_age} " in sys_prompt_main
    assert f"Собеседник в этом диалоге: {expected_u3_desc}." in sys_prompt_main
    assert orch.resolve_interlocutor_string(user_name_ua=u3_name, is_experimental=True) == orch.translate_name_to_ru(u3_name)

    # Експериментальний користувач [07] -> лише перекладене ім'я
    u7_name = str(USERS_DICT[7]["name"])
    expected_u7_ru = orch.translate_name_to_ru(u7_name)
    orch.set_interlocutor(user_id=7, user_name_ua=u7_name, is_experimental=True)
    sys_prompt_exp = orch.build_system_prompt(today=test_date)
    assert f"Собеседник в этом диалоге: {expected_u7_ru}." in sys_prompt_exp

    # Перевірка додавання історії повідомлень та ефемерної RAG-ін'єкції (2-turn window)
    orch.history.append({"role": "user", "content": "привет бро"})
    orch.history.append({"role": "assistant", "content": "здарова шо ты"})
    assert orch.build_rag_query("го в рл") == "привет бро го в рл"

    payload_no_rag = orch.build_messages_payload(new_user_message="го в рл", today=test_date, include_rag=False)
    assert len(payload_no_rag) == 4
    assert payload_no_rag[0]["role"] == "system"
    assert payload_no_rag[1]["content"] == "привет бро"
    assert payload_no_rag[2]["content"] == "здарова шо ты"
    assert payload_no_rag[3]["content"] == "го в рл"

    # Перевірка ефемерної ін'єкції спогадів (з реальним memory.faiss або тимчасовим індексом, якщо faiss-cpu ще не встановлено)
    import tempfile

    with tempfile.TemporaryDirectory(prefix="gvk_orch_rag_") as tmp_rag_dir:
        active_rag = orch.rag
        if active_rag is None:
            sample_mem_path = os.path.join(tmp_rag_dir, "memories.txt")
            with open(sample_mem_path, "w", encoding="utf-8") as mf:
                mf.write(
                    "[MEMORY_01]\n"
                    "Категория: Ігри\n"
                    "Теги: рл, рокет ліга, катка, напарник\n"
                    "Содержание:\n"
                    "Партнер в игре рокет лига (РЛ) — постоянный тиммейт, катаем катки каждый вечер.\n"
                )
            sample_faiss_path = os.path.join(tmp_rag_dir, "memory.faiss")
            build_rag_index(sample_mem_path, sample_faiss_path, backend="ngram")
            orch.rag = MemoryRAG(sample_faiss_path)

        # 1) Привітання не ін'єктує спогадів
        orch.history.clear()
        greeting_msgs = orch.build_messages_payload(new_user_message="привет, как дела?", today=test_date)
        assert len(greeting_msgs) == 2, f"Привітання не повинно додавати RAG-блок: {greeting_msgs}"

        # 2) Тематичний запит ін'єктує ефемерний system-блок перед повідомленням користувача, не засмічуючи history
        orch.history.append({"role": "user", "content": "привет бро"})
        orch.history.append({"role": "assistant", "content": "здарова шо ты"})
        rag_msgs = orch.build_messages_payload(new_user_message="го в рокет лигу", today=test_date)
        assert len(rag_msgs) == 5, f"Очікувалось 5 повідомлень (з ефемерним RAG-блоком), отримано {len(rag_msgs)}"
        assert rag_msgs[3]["role"] == "system"
        assert DEFAULT_RAG_HEADER in rag_msgs[3]["content"]
        assert rag_msgs[4] == {"role": "user", "content": "го в рокет лигу"}
        assert len(orch.history) == 2, "self.history не повинна містити ефемерних RAG-блоків"

        # 3) Уточнююче питання на наступному кроці підтягує спогад завдяки 2-turn window
        orch.history.append(rag_msgs[-1])
        orch.history.append({"role": "assistant", "content": "та го через 10 мин"})
        followup_msgs = orch.build_messages_payload(new_user_message="а как часто вы катаете?", today=test_date)
        assert len(followup_msgs) == 7
        assert followup_msgs[5]["role"] == "system"
        assert DEFAULT_RAG_HEADER in followup_msgs[5]["content"]
        assert followup_msgs[6] == {"role": "user", "content": "а как часто вы катаете?"}

        orch.rag = active_rag

    # 2. Перевірка середньої колонки вибору персонажа та екрана завантаження в menu.py
    u2_name = str(USERS_DICT[2]["name"])
    p1_label = str(PERSONA_SELECT_ITEMS[0]["label"])
    test_colors = {str(orch.config.get("persona_id") or "friend_1"): str(orch.config.get("color") or "#78b068")}
    app = GVKMenuApp(wait_seconds=0.01, orchestrator=orch, persona_colors=test_colors, headless=True)
    app.selected_user_id = 2
    app.selected_user_name = u2_name
    app.show_main_menu()
    app.root.update()
    menu_modal = app.active_modal
    assert isinstance(menu_modal, MainMenuModal)
    assert menu_modal.persona_column_open is False

    menu_modal.open_persona_column()
    app.root.update()
    assert menu_modal.persona_column_open is True
    assert len(menu_modal.persona_labels) == 3

    app.show_persona_loading("friend_1", p1_label, background_task=lambda: None)
    app.root.update()
    loading_modal = app.active_modal
    assert isinstance(loading_modal, PersonaLoadingModal)
    assert loading_modal.title_label.cget("text") == f"Завантажуємо {orch.get_genitive_name()}..."
    assert loading_modal.tip_label.cget("text") in loading_tips
    assert loading_modal.tip_label.cget("fg") == "#78b068"
    assert loading_modal.bar_label.cget("fg") == "#78b068"
    assert loading_modal.format_loading_bar(0.5).endswith(" 50%")
    assert loading_modal.format_loading_bar(1.0).endswith("100%")

    unique_tips = list(dict.fromkeys(loading_tips))
    seen_cycle = [loading_modal._current_tip]
    for _ in range(len(unique_tips) - 1):
        loading_modal._current_tip = loading_modal._pick_random_tip()
        seen_cycle.append(loading_modal._current_tip)
    assert set(seen_cycle) == set(unique_tips) and len(seen_cycle) == len(unique_tips)

    loading_modal._finish()
    app.root.update()
    chat = app.active_modal
    assert isinstance(chat, ChatSessionModal)
    assert DEFAULT_AFK_SECONDS == 20.0
    assert DEFAULT_SMILE_SECONDS == 4.0

    all_portraits = orch.get_all_portraits()
    assert set(all_portraits.keys()) == {"idle", "sleeping", "smiling", "talking"}
    assert len(all_portraits["talking"]) == 2
    assert chat.current_emotion == "idle"
    assert chat.dossier_desc_label is not None and len(chat.dossier_desc_label.cget("text")) > 10
    assert chat.dossier_emotion_label is not None and chat.dossier_emotion_label.cget("text") == "Стан: [IDLE]"

    assert detect_reply_emotion("ХАХХАХПХП база") == ("ХАХХАХПХП база", "smiling")
    assert detect_reply_emotion("ок [MOOD:smiling]") == ("ок", "smiling")
    assert detect_reply_emotion("хз похуй") == ("хз похуй", "idle")

    # Перевірка talking -> smiling -> idle і AFK -> sleeping -> wake up (+ живий індикатор під портретом)
    chat.is_waiting_reply = True
    chat._start_talking_anim()
    assert chat.current_emotion == "talking"
    assert chat.dossier_emotion_label.cget("text") == "Стан: [TALKING]"
    assert chat.portrait_canvas.portrait_text == all_portraits["talking"][0]
    chat._schedule_talking_anim()
    assert chat.portrait_canvas.portrait_text == all_portraits["talking"][1]

    chat.is_waiting_reply = False
    chat._on_reply_received("ХАХХА імба")
    assert chat.current_emotion == "smiling"
    assert chat.dossier_emotion_label.cget("text") == "Стан: [SMILING]"
    assert chat.portrait_canvas.portrait_text == all_portraits["smiling"][0]

    chat._set_emotion("idle")
    chat._on_afk_timeout()
    assert chat.current_emotion == "sleeping"
    assert chat.dossier_emotion_label.cget("text") == "Стан: [SLEEPING]"
    assert chat.portrait_canvas.portrait_text == all_portraits["sleeping"][0]

    import tkinter as tk
    ev_wake = tk.Event()
    ev_wake.keysym = "a"
    ev_wake.char = "a"
    ev_wake.state = 0
    chat.handle_key(ev_wake)
    assert chat.current_emotion == "idle"
    assert chat.portrait_canvas.portrait_text == all_portraits["idle"][0]

    # Перевірка CTRL+N = Новий чат у зв'язці з оркестратором
    orch.history.append({"role": "user", "content": "тест"})
    orch.history.append({"role": "assistant", "content": "відповідь"})
    chat.append_message(u2_name, "тест", "#ffffff", is_user=True)
    assert app.footer_new_chat_label.cget("text") == "CTRL+N = Новий чат"
    app._on_ctrl_n()
    app.root.update()
    assert orch.history == []
    assert chat.chat_lines == []
    assert app.footer_status_label.cget("text") == "Новий чат розпочато."

    app.root.destroy()

    # 2b. Перевірка розпакування кількох активованих персонажів та реєстрації PID у Kill-Switch
    from unittest.mock import MagicMock, patch
    from tools.disc_packer import derive_master_key, encrypt_file, generate_origin_sig, pack_payload_encrypted, write_public_manifest

    with tempfile.TemporaryDirectory(prefix="gvk_vault_multi_") as tmp_vdir:
        test_key = derive_master_key("GVK_Main_SelfTest_MultiPersona_Key_2026!", allow_short=False)
        disc_root = os.path.join(tmp_vdir, "disc")
        home_root = os.path.join(tmp_vdir, "gvk_home")
        os.makedirs(os.path.join(disc_root, "vault"), exist_ok=True)
        os.makedirs(os.path.join(disc_root, "persona"), exist_ok=True)
        os.makedirs(os.path.join(home_root, "registry"), exist_ok=True)

        with open(os.path.join(disc_root, "vault", "master.key"), "wb") as kf:
            kf.write(test_key)

        fake_adapter = os.path.join(tmp_vdir, "a.gguf")
        with open(fake_adapter, "wb") as af:
            af.write(b"GGUF" + b"\x00" * 1024)

        for pid, target_enc_dir in (
            ("friend_1", os.path.join(home_root, "personas", "friend_1")),
            ("friend_2", os.path.join(disc_root, "persona")),
        ):
            os.makedirs(target_enc_dir, exist_ok=True)
            src_payload = os.path.join(tmp_vdir, f"src_{pid}")
            os.makedirs(os.path.join(src_payload, "portraits"), exist_ok=True)
            with open(os.path.join(src_payload, "persona_config.json"), "w", encoding="utf-8") as cf:
                json.dump({"persona_id": pid, "display_name": pid}, cf)
            with open(os.path.join(src_payload, "prompt.txt"), "w", encoding="utf-8") as pf:
                pf.write(f"System {pid}")
            with open(os.path.join(src_payload, "portraits", "idle.txt"), "w", encoding="utf-8") as sf:
                sf.write("⠀⣿")

            a_sha = encrypt_file(fake_adapter, os.path.join(target_enc_dir, "adapter.gguf.enc"), test_key)
            p_sha = pack_payload_encrypted(src_payload, os.path.join(target_enc_dir, "payload.tar.enc"), test_key)
            write_public_manifest(target_enc_dir, persona_id=pid, adapter_sha256=a_sha, payload_sha256=p_sha)
            sig = generate_origin_sig(pid, test_key)
            with open(os.path.join(home_root, "registry", f"{pid}.sig"), "wb") as sgf:
                sgf.write(sig)
            if target_enc_dir.startswith(disc_root):
                with open(os.path.join(target_enc_dir, "origin.sig"), "wb") as sgf:
                    sgf.write(sig)

        sim_vault = CryptoVault(disc_path=disc_root, ramdisk_mb=32)
        sim_ram = sim_vault.initialize()
        try:
            unpacked = unpack_activated_personas_to_ram(sim_vault, home_root, disc_root)
            assert unpacked == ["friend_1", "friend_2"], f"Очікувались обидва персонажі: {unpacked}"
            for pid in ("friend_1", "friend_2"):
                assert os.path.isfile(os.path.join(sim_ram, "adapters", f"{pid}.gguf"))
                assert os.path.isfile(os.path.join(sim_ram, "personas", pid, "prompt.txt"))

            orch_v = GVKOrchestrator(persona_id="friend_1", ram_root=sim_ram, vault=sim_vault)
            orch_v.paths["llama_server_path"] = "/bin/echo"
            orch_v.paths["base_model_path"] = fake_adapter
            mock_proc = MagicMock()
            mock_proc.pid = 424242
            mock_proc.poll.return_value = None
            with patch.object(orch_v, "is_server_healthy", side_effect=[False, True]), patch(
                "core.orchestrator.subprocess.Popen", return_value=mock_proc
            ):
                assert orch_v.ensure_server_running(wait_timeout=1.0) is True
            assert 424242 in sim_vault._protected_pids, "PID llama-server не зареєстровано у Kill-Switch!"
        finally:
            sim_vault.cleanup()

    # 3. Базовий self-test menu.py
    run_menu_self_test()
    logger.info("Комплексний self-test пройдено успішно! Лог збережено у: %s", log_path)


# =====================================================================
# РОЗПАКУВАННЯ АКТИВОВАНИХ ОСОБИСТОСТЕЙ У RAM-ДИСК
# =====================================================================
def _read_disc_manifest_persona_id(detected_disc: Optional[str]) -> Optional[str]:
    if not detected_disc:
        return None
    manifest_path = os.path.join(detected_disc, "persona", "manifest.json")
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            pid = str(data.get("persona_id") or "").strip()
            if pid:
                return pid
        except (OSError, json.JSONDecodeError):
            pass
    if os.path.isfile(os.path.join(detected_disc, "persona", "adapter.gguf.enc")):
        return "friend_1"
    return None


def _has_encrypted_persona_cache(gvk_home_abs: str) -> bool:
    reg_dir = os.path.join(gvk_home_abs, "registry")
    if os.path.isdir(reg_dir) and any(fname.endswith(".sig") for fname in os.listdir(reg_dir)):
        return True
    personas_dir = os.path.join(gvk_home_abs, "personas")
    if os.path.isdir(personas_dir):
        for pid in os.listdir(personas_dir):
            pdir = os.path.join(personas_dir, pid)
            if os.path.isfile(os.path.join(pdir, "adapter.gguf.enc")) and os.path.isfile(
                os.path.join(pdir, "payload.tar.enc")
            ):
                return True
    return False


def unpack_activated_personas_to_ram(
    vault: CryptoVault,
    gvk_home_abs: str,
    detected_disc: Optional[str] = None,
) -> List[str]:
    """
    Виявляє всі активовані зашифровані особистості (`~/.gvk_ai/registry/*.sig`,
    `~/.gvk_ai/personas/<id>/` та вставлений DVD) і розпаковує їх у RAM-диск.
    """
    log = logging.getLogger("gvk.main")
    candidate_ids: List[str] = []

    reg_dir = os.path.join(gvk_home_abs, "registry")
    if os.path.isdir(reg_dir):
        for fname in sorted(os.listdir(reg_dir)):
            if fname.endswith(".sig"):
                pid = fname[:-4].strip()
                if pid and pid not in candidate_ids:
                    candidate_ids.append(pid)

    cache_personas_dir = os.path.join(gvk_home_abs, "personas")
    if os.path.isdir(cache_personas_dir):
        for pid in sorted(os.listdir(cache_personas_dir)):
            pdir = os.path.join(cache_personas_dir, pid)
            if (
                os.path.isdir(pdir)
                and os.path.isfile(os.path.join(pdir, "adapter.gguf.enc"))
                and os.path.isfile(os.path.join(pdir, "payload.tar.enc"))
                and pid not in candidate_ids
            ):
                candidate_ids.append(pid)

    disc_pid = _read_disc_manifest_persona_id(detected_disc)
    if disc_pid and disc_pid not in candidate_ids:
        candidate_ids.append(disc_pid)

    candidate_ids.sort(key=lambda pid: (0 if pid == "friend_1" else 1, pid))

    unpacked: List[str] = []
    for pid in candidate_ids:
        cache_dir = os.path.join(cache_personas_dir, pid)
        disc_persona_dir = os.path.join(detected_disc, "persona") if (detected_disc and pid == disc_pid) else ""

        enc_dir: Optional[str] = None
        for d in (cache_dir, disc_persona_dir):
            if d and os.path.isfile(os.path.join(d, "adapter.gguf.enc")) and os.path.isfile(
                os.path.join(d, "payload.tar.enc")
            ):
                enc_dir = d
                break
        if not enc_dir:
            continue

        sig_candidates = [os.path.join(reg_dir, f"{pid}.sig")]
        if disc_persona_dir:
            sig_candidates.append(os.path.join(disc_persona_dir, "origin.sig"))

        sig_path = next((p for p in sig_candidates if os.path.isfile(p)), None)
        if sig_path is not None and not vault.verify_persona_signature(pid, sig_path):
            log.warning("Невалідний origin.sig для '%s' (%s), пропускаємо розпакування.", pid, sig_path)
            continue

        vault.unpack_persona_to_ram(
            pid,
            os.path.join(enc_dir, "adapter.gguf.enc"),
            os.path.join(enc_dir, "payload.tar.enc"),
        )
        unpacked.append(pid)

    if unpacked:
        log.info("Розпаковано у RAM-диск (%d): %s", len(unpacked), ", ".join(unpacked))
    return unpacked


# =====================================================================
# ТОЧКА ВХОДУ
# =====================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="GVK Private AI — Головна точка входу (core/main.py)")
    parser.add_argument(
        "cmd",
        nargs="?",
        choices=["run", "self-test"],
        default="run",
        help="Режим запуску: 'run' (за замовчуванням) або 'self-test'",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Швидкий режим (передає --fast у menu.py: таймери по 1–2 с)",
    )
    parser.add_argument(
        "--start-server",
        action="store_true",
        help="Автоматично запустити локальний llama-server під час старту",
    )
    parser.add_argument(
        "--disc-path",
        default=None,
        help="Опційний шлях до змонтованого DVD-диска GVK_VAULT",
    )
    args = parser.parse_args()

    if args.cmd == "self-test":
        run_self_test()
        return

    setup_logging()
    logger = logging.getLogger("gvk.main")
    logger.info("Запуск GVK Private AI (fast=%s, start_server=%s)...", args.fast, args.start_server)

    vault: Optional[CryptoVault] = None
    ram_root: Optional[str] = None
    local_persona_dir = os.path.join(REPO_ROOT, "personas", "friend_1")
    detected_disc = find_gvk_disc_path(args.disc_path) if (args.disc_path or not os.path.isdir(local_persona_dir)) else None
    gvk_home_abs = os.path.abspath(os.path.expanduser(DEFAULT_GVK_HOME))
    media_info = detect_media_mode(detected_disc)
    if detected_disc:
        logger.info(
            "Виявлено носій GVK_VAULT: %s [Режим: %s, base_gguf=%s, bins=%s]",
            detected_disc,
            media_info["mode"],
            media_info["has_base_model"],
            media_info["has_llama_bins"],
        )

    use_vault = bool(args.disc_path) or (
        not os.path.isdir(local_persona_dir)
        and (detected_disc is not None or _has_encrypted_persona_cache(gvk_home_abs))
    )

    unpacked_personas: List[str] = []
    if use_vault:
        try:
            vault = CryptoVault(disc_path=args.disc_path or detected_disc, ramdisk_mb=1024)
            ram_root = vault.initialize()
            unpacked_personas = unpack_activated_personas_to_ram(vault, gvk_home_abs, detected_disc)
            vault.start_kill_switch()
        except Exception as exc:
            logger.exception("Помилка ініціалізації CryptoVault / RAM-диска: %s", exc)
            if vault is not None:
                vault.cleanup()
                vault = None
                ram_root = None

    primary_persona = (
        "friend_1"
        if (os.path.isdir(local_persona_dir) or "friend_1" in unpacked_personas or not unpacked_personas)
        else unpacked_personas[0]
    )

    orchestrator: Optional[GVKOrchestrator] = None
    try:
        orchestrator = GVKOrchestrator(persona_id=primary_persona, ram_root=ram_root, vault=vault)
        if args.start_server:
            orchestrator.ensure_server_running()
    except Exception as exc:
        logger.exception("Попередження під час ініціалізації GVKOrchestrator: %s", exc)

    wait_sec = 1.0 if args.fast else float(DEFAULT_WAIT_SECONDS)
    last_warn_sec = 2.0 if args.fast else float(LAST_WARNING_WAIT_SECONDS)

    persona_colors = {}
    if orchestrator is not None and hasattr(orchestrator, "config"):
        pid = str(orchestrator.config.get("persona_id") or "")
        pcol = str(orchestrator.config.get("color") or "")
        if pid and pcol:
            persona_colors[pid] = pcol

    app = GVKMenuApp(
        wait_seconds=wait_sec,
        last_warning_seconds=last_warn_sec,
        orchestrator=orchestrator,
        persona_colors=persona_colors,
    )
    try:
        app.run()
    finally:
        if orchestrator is not None:
            orchestrator.shutdown()
        if vault is not None:
            vault.cleanup()
        logger.info("Сесію GVK Private AI завершено.")


if __name__ == "__main__":
    main()
