#!/usr/bin/env python3
"""
GVK Private AI — Оркестратор сесії та контексту (core/orchestrator.py)

Відповідає за:
  1. Універсальне визначення шляхів (RAM-диск -> локальна папка розробки `personas/` -> кеш `~/.gvk_ai`).
  2. Завантаження `persona_config.json`, `prompt.txt`, портретів (`portraits/*.txt`) та `memory.faiss`.
  3. Динамічну побудову системного промпту:
     - Базовий текст із `prompt.txt`
     - Сьогоднішня дата (`DD.MM.YYYY`) та точний вік персонажа на основі `birth_date` із `persona_config.json`
     - Ін'єкція співрозмовника (`Собеседник в этом диалоге: ...` або `Ты сейчас в групе с: ...`):
       * Для основної колонки (`experimental=False`): повний опис зі словника `interlocutor_descriptions`.
       * Для експериментальних колонок (`experimental=True`): лише ім'я російською зі словника `ua_to_ru_names`.
  4. Збереження історії попередніх повідомлень діалогу та взаємодію з `llama-server`.
"""

import datetime
import glob
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.memory_rag import DEFAULT_SIMILARITY_THRESHOLD, MemoryRAG, build_rag_index, register_persona_lexicon
from core.skill_runner import SkillRunner, strip_skill_tags
from updater import DEFAULT_GVK_HOME, find_base_model_path, find_llama_server

logger = logging.getLogger("gvk.orchestrator")

DEFAULT_RAG_TOP_K = 2
MAX_HISTORY_MESSAGES = 20
DEFAULT_RAG_HEADER = (
    "[Память по теме (если тебя прямо спрашивают об этом факте — отвечай точно по нему; "
    "если не спрашивают — не вставляй лишнего, отвечай коротко в своем обычном стиле)]:"
)
_GROUP_PREFIX_RE = re.compile(r"^\[[^\]]+\]:\s*")

# Резервний словник перекладу імен UA -> RU та описів (знеособлені заглушки без roster.json/persona_config.json)
DEFAULT_UA_TO_RU_NAMES: Dict[str, str] = {
    "Друг 1": "Друг 1",
    **{f"Учасник {idx:02d}": f"Участник {idx:02d}" for idx in range(1, 17)},
}

DEFAULT_INTERLOCUTOR_DESCRIPTIONS: Dict[str, str] = {}


def load_roster_mappings(
    persona_id: str = "friend_1",
    ram_root: Optional[str] = None,
    repo_root: str = REPO_ROOT,
    gvk_home: str = DEFAULT_GVK_HOME,
    persona_dir: Optional[str] = None,
) -> Tuple[Dict[str, str], Dict[str, str], Optional[str]]:
    """
    Завантажує реальні відповідності імен/описів із git-ignored `roster.json`
    (RAM-диск -> локальний `personas/roster.json` -> `persona_dir` -> `~/.gvk_ai`).
    """
    gvk_home_abs = os.path.abspath(os.path.expanduser(gvk_home))
    candidates: List[str] = []
    if ram_root:
        candidates.append(os.path.join(ram_root, "personas", "roster.json"))
        candidates.append(os.path.join(ram_root, "personas", persona_id, "roster.json"))
    candidates.append(os.path.join(repo_root, "personas", "roster.json"))
    if persona_dir:
        candidates.append(os.path.join(persona_dir, "roster.json"))
    candidates.append(os.path.join(repo_root, "personas", persona_id, "roster.json"))
    candidates.append(os.path.join(gvk_home_abs, "personas", "roster.json"))
    candidates.append(os.path.join(gvk_home_abs, "personas", persona_id, "roster.json"))

    ua_to_ru: Dict[str, str] = {}
    descriptions: Dict[str, str] = {}
    default_user_name: Optional[str] = None
    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Не вдалося прочитати roster.json (%s): %s", path, exc)
            continue
        if not isinstance(data, dict):
            continue
        raw_ua_to_ru = data.get("ua_to_ru_names")
        if isinstance(raw_ua_to_ru, dict):
            ua_to_ru.update({str(k): str(v) for k, v in raw_ua_to_ru.items()})
        raw_desc = data.get("interlocutor_descriptions")
        if isinstance(raw_desc, dict):
            descriptions.update({str(k): str(v) for k, v in raw_desc.items()})
        raw_users = data.get("users_dict")
        if isinstance(raw_users, dict):
            u2 = raw_users.get("2") or raw_users.get(2)
            if isinstance(u2, dict) and u2.get("name"):
                default_user_name = str(u2["name"])
        break
    return ua_to_ru, descriptions, default_user_name

PERSONAL_PROMPT_TEMPLATE = "Собеседник в этом диалоге: {description}."
GROUP_PROMPT_TEMPLATE = (
    "Ты сейчас в групе с: {description}. "
    "Сообщения участников подписаны в формате 'Имя: текст'. "
    "Отвечай от своего лица без префикса."
)

DEFAULT_WARMUP_QUESTIONS: List[str] = [
    "привет бро, шо ты, как дела?",
    "че делаешь сейчас, как день проходит?",
]


# =====================================================================
# ДОПОМІЖНІ ФУНКЦІЇ ДАТИ, ВІКУ ТА ШЛЯХІВ
# =====================================================================
def calculate_age(birth_date_str: str, today: Optional[datetime.date] = None) -> int:
    """Обчислює повну кількість років на дату `today` із рядка формату `DD.MM.YYYY`."""
    if today is None:
        today = datetime.date.today()
    day_s, month_s, year_s = birth_date_str.strip().split(".")
    birth = datetime.date(int(year_s), int(month_s), int(day_s))
    age = today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
    return max(0, age)


def russian_age_unit(age: int) -> str:
    """Повертає правильне узгодження ('год', 'года', 'лет') для числівника віку."""
    rem100 = age % 100
    if 11 <= rem100 <= 19:
        return "лет"
    rem10 = age % 10
    if rem10 == 1:
        return "год"
    if 2 <= rem10 <= 4:
        return "года"
    return "лет"


def ukrainian_age_unit(age: int) -> str:
    """Повертає правильне узгодження ('рік', 'роки', 'років') для українського UI."""
    rem100 = age % 100
    if 11 <= rem100 <= 19:
        return "років"
    rem10 = age % 10
    if rem10 == 1:
        return "рік"
    if 2 <= rem10 <= 4:
        return "роки"
    return "років"


def resolve_runtime_paths(
    persona_id: str = "friend_1",
    ram_root: Optional[str] = None,
    repo_root: str = REPO_ROOT,
    gvk_home: str = DEFAULT_GVK_HOME,
    disc_path: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """
    Уніфіковано знаходить шляхи як у режимі розробки (локальний репозиторій),
    так і в продакшні (`~/.gvk_ai` + RAM-диск + DVD/USB носій):
      - `persona_dir`: директорія з `persona_config.json`, `prompt.txt`, `portraits/`, `memory.faiss`
      - `adapter_path`: шлях до `<persona_id>.gguf`
      - `base_model_path`: шлях до `Qwen2.5-14B-Instruct-Q4_K_M.gguf`
      - `llama_server_path`: шлях до бінарника `llama-server`
    """
    gvk_home_abs = os.path.abspath(os.path.expanduser(gvk_home))

    # 1. Директорія персонажа (RAM -> локальне репо personas/<id> -> ~/.gvk_ai/personas/<id>)
    persona_candidates: List[str] = []
    if ram_root:
        persona_candidates.append(os.path.join(ram_root, "personas", persona_id))
    persona_candidates.append(os.path.join(repo_root, "personas", persona_id))
    persona_candidates.append(os.path.join(gvk_home_abs, "personas", persona_id))

    persona_dir: Optional[str] = None
    for cand in persona_candidates:
        if os.path.isdir(cand):
            persona_dir = os.path.abspath(cand)
            break

    # 2. Шлях до LoRA-адаптера (RAM -> adapters/<id>.gguf -> persona_config files.adapter -> локальні файли)
    cfg_adapter_name: Optional[str] = None
    if persona_dir:
        cfg_path = os.path.join(persona_dir, "persona_config.json")
        if os.path.isfile(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg_data = json.load(f)
                raw_adapter = (cfg_data.get("files") or {}).get("adapter")
                if raw_adapter:
                    cfg_adapter_name = str(raw_adapter).strip()
            except (OSError, json.JSONDecodeError):
                pass

    adapter_candidates: List[str] = []
    if ram_root:
        adapter_candidates.append(os.path.join(ram_root, "adapters", f"{persona_id}.gguf"))
        if cfg_adapter_name:
            adapter_candidates.append(os.path.join(ram_root, "adapters", cfg_adapter_name))
    adapter_candidates.append(os.path.join(repo_root, "adapters", f"{persona_id}.gguf"))
    if cfg_adapter_name:
        adapter_candidates.append(os.path.join(repo_root, "adapters", cfg_adapter_name))
    if persona_dir:
        adapter_candidates.append(os.path.join(persona_dir, f"{persona_id}.gguf"))
        if cfg_adapter_name:
            adapter_candidates.append(os.path.join(persona_dir, cfg_adapter_name))
        adapter_candidates.append(os.path.join(persona_dir, "adapter.gguf"))
    adapter_candidates.append(os.path.join(gvk_home_abs, "adapters", f"{persona_id}.gguf"))
    if persona_id == "friend_1":
        for gguf_path in sorted(glob.glob(os.path.join(repo_root, "*.gguf"))):
            gguf_name = os.path.basename(gguf_path).lower()
            if "qwen" not in gguf_name and "base_model" not in gguf_name:
                adapter_candidates.append(gguf_path)

    adapter_path: Optional[str] = None
    for cand in adapter_candidates:
        if os.path.isfile(cand):
            adapter_path = os.path.abspath(cand)
            break

    base_model_path = find_base_model_path(gvk_home=gvk_home_abs, repo_root=repo_root, disc_path=disc_path)
    llama_server_path = find_llama_server(gvk_home=gvk_home_abs, repo_root=repo_root, disc_path=disc_path)

    resolved = {
        "persona_id": persona_id,
        "persona_dir": persona_dir,
        "adapter_path": adapter_path,
        "base_model_path": base_model_path,
        "llama_server_path": llama_server_path,
    }
    logger.info("Визначено рантайм-шляхи для '%s': %s", persona_id, json.dumps(resolved, ensure_ascii=False))
    return resolved


# =====================================================================
# КЛАС ОРКЕСТРАТОРА (GVKOrchestrator)
# =====================================================================
class GVKOrchestrator:
    """
    Керує контекстом бесіди, системним промптом, історії повідомлень
    та викликами `llama-server` для обраного персонажа.
    """

    def __init__(
        self,
        persona_id: str = "friend_1",
        ram_root: Optional[str] = None,
        repo_root: str = REPO_ROOT,
        gvk_home: str = DEFAULT_GVK_HOME,
        vault: Optional[Any] = None,
    ) -> None:
        self.persona_id = persona_id
        self.ram_root = ram_root
        self.repo_root = repo_root
        self.gvk_home = gvk_home
        self.vault = vault

        vault_disc = getattr(vault, "disc_path", None) if vault is not None else None
        self.paths = resolve_runtime_paths(
            persona_id=persona_id,
            ram_root=ram_root,
            repo_root=repo_root,
            gvk_home=gvk_home,
            disc_path=vault_disc,
        )
        self.persona_dir = self.paths["persona_dir"]
        if not self.persona_dir:
            raise FileNotFoundError(f"Не знайдено директорію персонажа для '{persona_id}'.")

        self.config = self._load_persona_config()
        self.base_prompt = self._load_prompt_txt()
        self.rag: Optional[MemoryRAG] = self._load_rag_memory()
        self.last_rag_hits: List[Dict[str, Any]] = []
        self.skill_runner: SkillRunner = SkillRunner(
            self.persona_dir,
            persona_id=self.persona_id,
            persona_config=self.config,
        )

        roster_ua_to_ru, roster_descriptions, roster_default_user = load_roster_mappings(
            persona_id=self.persona_id,
            ram_root=self.ram_root,
            repo_root=self.repo_root,
            gvk_home=self.gvk_home,
            persona_dir=self.persona_dir,
        )
        self._roster_ua_to_ru: Dict[str, str] = roster_ua_to_ru
        self._roster_descriptions: Dict[str, str] = roster_descriptions
        self._staged_runtime_files: List[str] = []

        # Поточний стан сесії користувача
        self.user_id: int = 2
        self.user_name_ua: str = roster_default_user or "Учасник 02"
        self.is_experimental_user: bool = False
        self.chat_mode: str = "personal"  # "personal" | "group"
        self.group_members_ua: List[str] = []

        # Історія попередніх повідомлень у поточній розмові: [{"role": "user"|"assistant", "content": "..."}]
        self.history: List[Dict[str, str]] = []

        # Підпроцес llama-server (якщо запущений оркестратором)
        self._server_proc: Optional[subprocess.Popen] = None
        self._server_lock = threading.Lock()
        self._history_generation: int = 0

    # -----------------------------------------------------------------
    # ЗАВАНТАЖЕННЯ КОНФІГУ, ПРОМПТУ ТА ПОРТРЕТІВ
    # -----------------------------------------------------------------
    def _load_persona_config(self) -> Dict[str, Any]:
        cfg_path = os.path.join(self.persona_dir or "", "persona_config.json")
        if not os.path.isfile(cfg_path):
            logger.warning("Файл persona_config.json не знайдено (%s), використовуються значення за замовчуванням.", cfg_path)
            return {}
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(
            "Завантажено persona_config.json для '%s' (%s, колір=%s, станів емоцій=%d)",
            data.get("persona_id", self.persona_id),
            data.get("display_name", ""),
            data.get("color", ""),
            len(data.get("emotional_states", [])),
        )
        return data

    def _load_prompt_txt(self) -> str:
        prompt_rel = (self.config.get("files") or {}).get("prompt", "prompt.txt")
        prompt_path = os.path.join(self.persona_dir or "", prompt_rel)
        if not os.path.isfile(prompt_path):
            raise FileNotFoundError(f"Системний промпт не знайдено: {prompt_path}")
        with open(prompt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        logger.info("Завантажено prompt.txt (%d символів) із %s", len(text), prompt_path)
        return text

    def _load_rag_memory(self) -> Optional[MemoryRAG]:
        rag_rel = (self.config.get("files") or {}).get("rag_index", "memory.faiss")
        rag_path = os.path.join(self.persona_dir or "", rag_rel)

        # Якщо у корені data/memory.faiss є свіжіший індекс або в persona_dir немає memory.faiss — підхоплюємо / збираємо
        data_faiss = os.path.join(self.repo_root, "data", "memory.faiss")
        data_txt = os.path.join(self.repo_root, "data", "memories.txt")
        persona_txt = os.path.join(self.persona_dir or "", "memories.txt")

        chosen_path = rag_path if os.path.isfile(rag_path) else None
        if self.persona_id == "friend_1" and os.path.isfile(data_faiss):
            if not chosen_path or os.path.getmtime(data_faiss) > os.path.getmtime(chosen_path):
                chosen_path = data_faiss
        elif not chosen_path:
            src_txt = persona_txt if os.path.isfile(persona_txt) else (data_txt if os.path.isfile(data_txt) else None)
            if src_txt:
                try:
                    build_rag_index(src_txt, rag_path, backend="ngram")
                    chosen_path = rag_path
                except Exception as exc:
                    logger.warning("Не вдалося автоматично зібрати memory.faiss із %s: %s", src_txt, exc)

        if not chosen_path or not os.path.isfile(chosen_path):
            logger.warning("Файл індексу RAG не знайдено (%s), сесія без векторної пам'яті.", rag_path)
            return None

        try:
            register_persona_lexicon(self.config.get("rag_lexicon"))
            rag = MemoryRAG(chosen_path)
            logger.info(
                "Завантажено RAG-пам'ять (%d спогадів, dim=%d, backend=%s) із %s",
                len(rag.memories),
                rag.dimension,
                rag.index_backend,
                chosen_path,
            )
            return rag
        except Exception as exc:
            logger.warning("Не вдалося завантажити RAG-пам'ять із %s: %s", chosen_path, exc)
            return None

    def get_rag_status(self) -> str:
        """Повертає компактний рядок стану векторної пам'яті для досьє / логів."""
        if self.rag is None:
            return "пам'ять не підключена [IDLE]"
        backend_short = "ST" if self.rag.index_backend == "sentence-transformers" else self.rag.index_backend
        return f"{len(self.rag.memories)} спогадів [{backend_short}] [OK]"

    def get_portrait_text(self, emotion: str = "idle") -> str:
        """
        Повертає вміст ASCII-портрета для заданої емоції (`idle`, `sleeping`, `smiling`, `talking`).
        Якщо для однієї емоції задано декілька `.txt` файлів (наприклад `talking.txt`, `talking2.txt`),
        обирає один із них.
        """
        sprites_map = self.config.get("emotion_sprites") or {}
        candidates = sprites_map.get(emotion) or [f"{emotion}.txt"]
        if isinstance(candidates, str):
            candidates = [candidates]

        portraits_rel = (self.config.get("files") or {}).get("portraits_dir", "portraits")
        portraits_dir = os.path.join(self.persona_dir or "", portraits_rel)

        valid_files = [
            os.path.join(portraits_dir, fname)
            for fname in candidates
            if os.path.isfile(os.path.join(portraits_dir, fname))
        ]
        if not valid_files:
            fallback = os.path.join(portraits_dir, "idle.txt")
            if os.path.isfile(fallback):
                valid_files = [fallback]
            else:
                return ""

        chosen_path = valid_files[0] if len(valid_files) == 1 else random.choice(valid_files)
        with open(chosen_path, "r", encoding="utf-8") as f:
            return f.read().rstrip("\n")

    def get_all_portraits(self) -> Dict[str, List[str]]:
        """
        Попередньо завантажує всі ASCII-портрети для кожної емоції з `emotion_sprites`
        (наприклад `idle`, `sleeping`, `smiling`, `talking`: [`talking.txt`, `talking2.txt`]).
        """
        sprites_map = self.config.get("emotion_sprites") or {}
        portraits_rel = (self.config.get("files") or {}).get("portraits_dir", "portraits")
        portraits_dir = os.path.join(self.persona_dir or "", portraits_rel)

        states = list(dict.fromkeys(list(self.config.get("emotional_states") or []) + list(sprites_map.keys())))
        loaded: Dict[str, List[str]] = {}
        for emotion in states:
            candidates = sprites_map.get(emotion) or [f"{emotion}.txt"]
            if isinstance(candidates, str):
                candidates = [candidates]
            frames: List[str] = []
            for fname in candidates:
                fpath = os.path.join(portraits_dir, fname)
                if os.path.isfile(fpath):
                    with open(fpath, "r", encoding="utf-8") as f:
                        frames.append(f.read().rstrip("\n"))
            if frames:
                loaded[emotion] = frames
        return loaded

    def get_genitive_name(self) -> str:
        """Повертає ім'я персонажа в родовому відмінку (для 'Завантажуємо ...')."""
        return str(
            self.config.get("display_name_genitive")
            or self.config.get("genitive_name")
            or self.config.get("display_name")
            or self.persona_id
        )

    def get_loading_tips(self) -> List[str]:
        """Повертає список порад для екрана завантаження персонажа."""
        tips = self.config.get("loading_tips")
        if isinstance(tips, list) and tips:
            return [str(t) for t in tips if str(t).strip()]
        return ["loading tip 1", "loading tip 2", "loading tip 3"]

    def get_warmup_questions(self) -> List[str]:
        """Повертає 2 російськомовні фонові питання для прогріву моделі."""
        questions = self.config.get("warmup_questions")
        if isinstance(questions, list) and len(questions) >= 2:
            return [str(q) for q in questions[:2]]
        return list(DEFAULT_WARMUP_QUESTIONS)

    # -----------------------------------------------------------------
    # КОНФІГУРАЦІЯ КОРИСТУВАЧА ТА ПОБУДОВА СИСТЕМНОГО ПРОМПТУ
    # -----------------------------------------------------------------
    def set_interlocutor(
        self,
        user_id: int,
        user_name_ua: str,
        is_experimental: bool = False,
        chat_mode: str = "personal",
        reset_history: bool = True,
    ) -> None:
        """Зберігає вибраного у `menu.py` користувача та оновлює системний промпт."""
        self.user_id = user_id
        self.user_name_ua = user_name_ua.strip()
        self.is_experimental_user = bool(is_experimental)
        self.chat_mode = chat_mode
        if reset_history:
            self.clear_history(log_event=False)

        logger.info(
            "Встановлено співрозмовника: id=%d, name='%s', experimental=%s, mode='%s'",
            self.user_id,
            self.user_name_ua,
            self.is_experimental_user,
            self.chat_mode,
        )
        compiled_prompt = self.build_system_prompt()
        logger.info(
            "Зібраний системний промпт готовий (%d символів, mode=%s).",
            len(compiled_prompt),
            self.chat_mode,
        )

    def clear_history(self, log_event: bool = True) -> None:
        """Очищає історію поточної розмови та скидає стан RAG (для CTRL+N = Новий чат)."""
        self._history_generation += 1
        self.history.clear()
        self.last_rag_hits = []
        if log_event:
            logger.info(
                "Розпочато новий чат для '%s' (історію очищено, gen=%d).",
                self.persona_id,
                self._history_generation,
            )

    def translate_name_to_ru(self, name_ua: str) -> str:
        """Перекладає українське ім'я на російську за словником `ua_to_ru_names`."""
        ua_map: Dict[str, str] = {
            **DEFAULT_UA_TO_RU_NAMES,
            **self._roster_ua_to_ru,
            **(self.config.get("ua_to_ru_names") or {}),
        }
        cleaned = name_ua.strip()
        return ua_map.get(cleaned, cleaned)

    def resolve_interlocutor_string(
        self,
        user_name_ua: Optional[str] = None,
        is_experimental: Optional[bool] = None,
    ) -> str:
        """
        Формує опис співрозмовника:
          - Якщо користувач із основної колонки (`is_experimental=False`):
            повертає повний опис зі словника `interlocutor_descriptions`.
          - Якщо користувач із експериментальних колонок (`is_experimental=True`):
            повертає лише перекладене на російську ім'я.
        """
        name = (user_name_ua if user_name_ua is not None else self.user_name_ua).strip()
        exp = self.is_experimental_user if is_experimental is None else bool(is_experimental)
        ru_name = self.translate_name_to_ru(name)

        if exp:
            return ru_name

        desc_map: Dict[str, str] = {
            **DEFAULT_INTERLOCUTOR_DESCRIPTIONS,
            **self._roster_descriptions,
            **(self.config.get("interlocutor_descriptions") or {}),
        }
        return desc_map.get(name) or desc_map.get(ru_name) or ru_name

    def build_system_prompt(self, today: Optional[datetime.date] = None) -> str:
        """
        Збирає повний системний промпт:
          1. Текст із `prompt.txt`
          2. `Сегодняшняя дата: DD.MM.YYYY, тебе <age> <год/года/лет>.`
          3. `Собеседник в этом диалоге: <опис або ім'я>.` (або `Ты сейчас в групе с: ...` для групового режиму)
        """
        if today is None:
            today = datetime.date.today()

        today_str = today.strftime("%d.%m.%Y")
        birth_date_str = str(self.config.get("birth_date", "01.01.2000"))
        age = calculate_age(birth_date_str, today=today)
        age_unit = russian_age_unit(age)

        templates = self.config.get("prompt_templates") or {}
        date_age_tpl = templates.get(
            "date_age",
            "Сегодняшняя дата: {today_date}, тебе {age} {age_unit}.",
        )
        date_age_part = date_age_tpl.format(today_date=today_str, age=age, age_unit=age_unit)

        if self.chat_mode == "group":
            group_tpl = templates.get("group", GROUP_PROMPT_TEMPLATE)
            if self.group_members_ua:
                members_ru = [self.translate_name_to_ru(m) for m in self.group_members_ua]
                desc = ", ".join(members_ru)
            else:
                desc = self.translate_name_to_ru(self.user_name_ua)
            context_part = group_tpl.format(description=desc)
        else:
            personal_tpl = templates.get("personal", PERSONAL_PROMPT_TEMPLATE)
            desc = self.resolve_interlocutor_string()
            context_part = personal_tpl.format(description=desc)

        compiled = f"{self.base_prompt}\n{date_age_part} {context_part}"
        if self.skill_runner.available:
            skills_block = self.skill_runner.format_skills_for_prompt()
            if skills_block:
                compiled = f"{compiled}\n\n{skills_block}"
        return compiled

    def build_rag_query(self, new_user_message: str) -> str:
        """
        Формує 2-крокове вікно запиту до RAG:
        поєднує попередню репліку користувача з `self.history` та поточне повідомлення,
        щоб уточнюючі питання («а хто ще там був?») зберігали контекст теми.
        """
        cleaned_new = _GROUP_PREFIX_RE.sub("", (new_user_message or "").strip()).strip()
        if not cleaned_new:
            return ""

        prev_user_text = ""
        for msg in reversed(self.history):
            if msg.get("role") == "user":
                raw_prev = msg.get("content", "").strip()
                prev_user_text = _GROUP_PREFIX_RE.sub("", raw_prev).strip()
                if prev_user_text:
                    break

        if prev_user_text and prev_user_text.lower() != cleaned_new.lower():
            return f"{prev_user_text} {cleaned_new}"
        return cleaned_new

    def retrieve_memories_block(
        self,
        new_user_message: str,
        top_k: int = DEFAULT_RAG_TOP_K,
        threshold: Optional[float] = None,
    ) -> str:
        """
        Виконує пошук релевантних спогадів для поточного повідомлення з урахуванням
        2-крокового вікна діалогу. Спочатку бере прямі збіги за поточним повідомленням,
        а за наявності вільних слотів доповнює збігами за комбінованим 2-кроковим запитом.
        Повертає готовий текстовий блок для ефемерної ін'єкції або порожній рядок.
        """
        if self.rag is None or not new_user_message or not new_user_message.strip() or top_k <= 0:
            self.last_rag_hits = []
            return ""

        cleaned_new = _GROUP_PREFIX_RE.sub("", new_user_message.strip()).strip()
        window_query = self.build_rag_query(cleaned_new)

        direct_hits = self.rag.search(cleaned_new, top_k=top_k * 2, threshold=threshold)
        merged: Dict[Any, Dict[str, Any]] = {}
        for h in direct_hits:
            key = h.get("memory_id") or h.get("id")
            merged[key] = dict(h)

        top_direct_score = max((float(h.get("score", 0.0)) for h in direct_hits), default=0.0)
        eff_thresh = (
            getattr(self.rag, "effective_threshold", self.rag.threshold)
            if threshold is None
            else float(threshold)
        )

        if window_query and window_query != cleaned_new:
            window_hits = self.rag.search(window_query, top_k=top_k * 2, threshold=threshold)
            for wh in window_hits:
                key = wh.get("memory_id") or wh.get("id")
                discounted_score = round(float(wh.get("score", 0.0)) * 0.85, 4)
                if discounted_score < eff_thresh:
                    continue
                if key in merged:
                    # Посилюємо спогад, який одночасно збігається і з поточним питанням, і з попереднім ходом
                    boosted = max(float(merged[key]["score"]), discounted_score) + 0.10
                    merged[key]["score"] = round(boosted, 4)
                elif top_direct_score < 0.90:
                    # Доповнюємо контекстом попереднього ходу лише якщо поточне повідомлення не має сильного прямого збігу
                    cand = dict(wh)
                    cand["score"] = discounted_score
                    merged[key] = cand

        ranked_hits = sorted(merged.values(), key=lambda item: float(item.get("score", 0.0)), reverse=True)
        if len(ranked_hits) > 1 and float(ranked_hits[0].get("score", 0.0)) >= 0.90:
            min_secondary = float(ranked_hits[0]["score"]) * 0.78
            ranked_hits = [ranked_hits[0]] + [
                item for item in ranked_hits[1:] if float(item.get("score", 0.0)) >= min_secondary
            ]
        hits = ranked_hits[:top_k]
        self.last_rag_hits = list(hits)
        if not hits:
            return ""

        templates = self.config.get("prompt_templates") or {}
        header = str(templates.get("rag_header") or DEFAULT_RAG_HEADER)
        lines = [header]
        for hit in hits:
            lines.append(f"- {hit['text']}")

        logger.info(
            "RAG ін'єкція (%d спогадів для запиту '%s'): %s",
            len(hits),
            window_query,
            ", ".join(f"{h.get('memory_id')}({h.get('score')})" for h in hits),
        )
        return "\n".join(lines)

    def build_messages_payload(
        self,
        new_user_message: Optional[str] = None,
        today: Optional[datetime.date] = None,
        include_rag: bool = True,
    ) -> List[Dict[str, str]]:
        """
        Формує масив ChatML-повідомлень для `llama-server`:
          - `system`: базовий статичний системний промпт
          - уся попередня історія повідомлень у розмові (`self.history`, без RAG-блоків)
          - ефемерний `system`-блок зі спогадами RAG для поточного ходу (якщо знайдено збіги)
          - нове повідомлення користувача (якщо передано)
        """
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.build_system_prompt(today=today)}
        ]
        messages.extend(self.history[-MAX_HISTORY_MESSAGES:])
        if new_user_message is not None:
            if include_rag:
                memory_block = self.retrieve_memories_block(new_user_message)
                if memory_block:
                    messages.append({"role": "system", "content": memory_block})
            if self.chat_mode == "group":
                ru_sender = self.translate_name_to_ru(self.user_name_ua)
                formatted_user = f"[{ru_sender}]: {new_user_message}"
            else:
                formatted_user = new_user_message
            messages.append({"role": "user", "content": formatted_user})
        return messages

    # -----------------------------------------------------------------
    # ЗАПУСК ТА ВЗАЄМОДІЯ З LLAMA-SERVER
    # -----------------------------------------------------------------
    def _format_lora_path_for_cli(self, adapter_path: str) -> str:
        """
        Форматує шлях до LoRA-адаптера без двокрапки диска Windows (`D:`),
        оскільки `--lora-scaled FNAME:SCALE` у `llama-server` розбиває аргумент за `:`.
        """
        if not adapter_path:
            return ""
        try:
            rel = os.path.relpath(adapter_path, self.repo_root)
            if ":" not in rel:
                return rel
        except ValueError:
            pass

        if ":" not in adapter_path:
            return adapter_path

        if os.path.isfile(adapter_path):
            runtime_dir = os.path.join(self.repo_root, "adapters", "_runtime")
            os.makedirs(runtime_dir, exist_ok=True)
            staged_path = os.path.join(runtime_dir, os.path.basename(adapter_path))
            try:
                if os.path.exists(staged_path):
                    os.unlink(staged_path)
                os.link(adapter_path, staged_path)
            except OSError:
                shutil.copy2(adapter_path, staged_path)
            if staged_path not in self._staged_runtime_files:
                self._staged_runtime_files.append(staged_path)
            return os.path.join("adapters", "_runtime", os.path.basename(adapter_path))

        logger.warning(
            "LoRA-шлях містить ':' (%s), неможливо звести до відносного шляху.",
            adapter_path,
        )
        return adapter_path

    def get_llama_server_command(self) -> List[str]:
        """Повертає CLI-команду для запуску `llama-server` із параметрами `persona_config.json` (без `-ngl`)."""
        server_bin = self.paths.get("llama_server_path") or "llama-server"
        base_model = self.paths.get("base_model_path") or ""
        params = self.config.get("llama_params") or {}
        lora_scaled = self.config.get("lora_scaled", params.get("lora_scaled"))

        adapter = self.paths.get("adapter_path") or ""
        if not adapter:
            fallback_adapter = str((self.config.get("files") or {}).get("adapter") or f"{self.persona_id}.gguf")
            adapter = os.path.join("adapters", fallback_adapter)

        cmd = [
            server_bin,
            "-m",
            base_model,
            "-c",
            str(params.get("ctx_size", 4096)),
            "--temp",
            str(params.get("temp", 0.65)),
            "--top-p",
            str(params.get("top_p", 0.85)),
            "--repeat-penalty",
            str(params.get("repeat_penalty", 1.05)),
            "--port",
            str(params.get("port", 8080)),
            "--min-p",
            str(params.get("min_p", 0.08)),
            "--repeat-last-n",
            str(params.get("repeat_last_n", 256)),
            "--n-predict",
            str(params.get("n_predict", 128)),
            "--dry-multiplier",
            str(params.get("dry_multiplier", 0.8)),
        ]
        if params.get("presence_penalty") is not None:
            cmd.extend(["--presence-penalty", str(params["presence_penalty"])])
        if adapter:
            if lora_scaled is not None:
                safe_adapter = self._format_lora_path_for_cli(adapter)
                cmd.extend(["--lora-scaled", f"{safe_adapter}:{lora_scaled}"])
            else:
                cmd.extend(["--lora", adapter])
        return cmd

    def switch_active_lora(self, adapter_id: int = 0, scale: float = 1.0) -> bool:
        """Гарячий свічинг активного LoRA-адаптера через POST /lora-adapters (< 1 мс)."""
        port = int((self.config.get("llama_params") or {}).get("port", 8080))
        url = f"http://127.0.0.1:{port}/lora-adapters"
        payload = [{"id": int(adapter_id), "scale": float(scale)}]
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception as exc:
            logger.warning("Не вдалося перемкнути LoRA (id=%s, scale=%s): %s", adapter_id, scale, exc)
            return False

    def select_next_speakers(
        self,
        user_message: str,
        candidate_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        candidate_skill_runners: Optional[Dict[str, SkillRunner]] = None,
        max_bot_turns: int = 2,
    ) -> List[str]:
        """
        Арбітр конференції (Cross-Talk Engine):
          1. Прямий тег / `trigger_words` учасника (`P = 1.0`).
          2. Перехоплення за відповідністю запиту до скілів у `tools.py` (`Skill Match`).
          3. Спонтанна реакція (`Chattiness Roll` < `chattiness_prob`).
          4. Ліміт: максимум `max_bot_turns` (2–3) репліки між ботами поспіль.
        """
        pool = candidate_configs or {self.persona_id: self.config}
        runners = candidate_skill_runners or {self.persona_id: self.skill_runner}
        if not pool or max_bot_turns <= 0:
            return []

        lowered_msg = (user_message or "").lower()
        chosen: List[str] = []

        # 1. Прямий тригер (P = 1.0) — перевірка на межі слова, щоб короткі тригери ("рл") не спрацьовували всередині слів
        for pid, cfg in pool.items():
            triggers = [str(t).strip().lower() for t in (cfg.get("trigger_words") or []) if str(t).strip()]
            if any(re.search(rf"(?<!\w){re.escape(tr)}(?!\w)", lowered_msg) for tr in triggers):
                if pid not in chosen:
                    chosen.append(pid)

        # 2. Перехоплення за скілами (Skill Match)
        best_skill_pid: Optional[str] = None
        best_skill_score = 0.0
        for pid in pool:
            sr = runners.get(pid)
            if sr is not None and sr.available:
                m = sr.match_query(user_message)
                if m is not None and m.score > best_skill_score:
                    best_skill_score = m.score
                    best_skill_pid = pid
        if best_skill_pid and best_skill_pid not in chosen:
            chosen.append(best_skill_pid)

        # 3. Спонтанна реакція (Chattiness Roll)
        for pid, cfg in pool.items():
            if len(chosen) >= max_bot_turns:
                break
            if pid in chosen:
                continue
            prob = float(cfg.get("chattiness_prob", 0.35))
            if random.random() < prob:
                chosen.append(pid)

        # Fallback: хоча б один персонаж відповідає, якщо ніхто не спрацював
        if not chosen:
            chosen.append(self.persona_id if self.persona_id in pool else next(iter(pool.keys())))

        return chosen[:max_bot_turns]

    def get_persona_dossier(
        self,
        current_emotion: str = "idle",
        today: Optional[datetime.date] = None,
    ) -> Dict[str, str]:
        """Формує структурований короткий опис та статуси для відображення під портретом у TUI."""
        if today is None:
            today = datetime.date.today()
        display_name = str(self.config.get("display_name") or self.persona_id)
        short_name = str(self.config.get("short_name") or "")
        birth_date_str = str(self.config.get("birth_date", "01.01.2000"))
        try:
            age = calculate_age(birth_date_str, today=today)
            age_part = f", {age} {ukrainian_age_unit(age)}"
        except Exception:
            age_part = ""

        nick_part = f" ({short_name})" if short_name else ""
        title = f"{display_name}{nick_part}{age_part}"

        description = str(self.config.get("short_description") or "").strip()
        if not description:
            role = str(self.config.get("role") or "Учасник")
            location = str(self.config.get("location") or "").strip()
            description = f"{role} ({location})" if location else role

        style = str(self.config.get("style") or "").strip()
        skill_lines = self.skill_runner.get_dossier_skills_lines() if self.skill_runner.available else []
        return {
            "persona_id": self.persona_id,
            "title": title,
            "description": description,
            "style": style,
            "rag_status": self.get_rag_status(),
            "skill_status": self.skill_runner.get_dossier_status(),
            "skills_summary": "\n".join(skill_lines),
            "emotion_status": str(current_emotion or "idle").upper(),
        }

    def _log_chat_turn(
        self,
        user_text: str,
        reply_text: str,
        rag_hits: Optional[List[Dict[str, Any]]] = None,
        history_path: Optional[str] = None,
    ) -> None:
        """Зберігає обмін репліками в JSONL для подальшого аудиту та донавчання."""
        entry = {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "persona_id": self.persona_id,
            "user_name": self.user_name_ua,
            "chat_mode": self.chat_mode,
            "user_text": user_text,
            "rag_memory_ids": [h.get("memory_id") for h in (rag_hits or []) if h.get("memory_id")],
            "assistant_text": reply_text,
        }
        line = json.dumps(entry, ensure_ascii=False) + "\n"

        targets: List[str] = []
        if history_path:
            targets.append(history_path)
        else:
            targets.append(os.path.join(self.repo_root, "logs", "chat_history.jsonl"))
            user_logs = os.path.join(os.path.abspath(os.path.expanduser(self.gvk_home)), "logs")
            if os.path.isdir(user_logs):
                targets.append(os.path.join(user_logs, "chat_history.jsonl"))

        for path in targets:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
                with open(path, "a", encoding="utf-8") as f:
                    f.write(line)
            except OSError:
                pass

    def is_server_healthy(self, timeout: float = 1.5) -> bool:
        port = int((self.config.get("llama_params") or {}).get("port", 8080))
        url = f"http://127.0.0.1:{port}/health"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return resp.status == 200
        except Exception:
            return False

    def ensure_server_running(self, wait_timeout: float = 120.0) -> bool:
        """Запускає `llama-server`, якщо він ще не запущений на вказаному порту."""
        with self._server_lock:
            if self.is_server_healthy():
                logger.info("llama-server вже активний і відповідає на /health.")
                return True

            server_bin = self.paths.get("llama_server_path")
            base_model = self.paths.get("base_model_path")
            if not server_bin or not os.path.isfile(server_bin):
                logger.warning("Бінарник llama-server не знайдено (%s).", server_bin)
                return False
            if not base_model or not os.path.isfile(base_model):
                logger.warning("Базову модель не знайдено (%s).", base_model)
                return False

            cmd = self.get_llama_server_command()
            logger.info("Запуск llama-server: %s", " ".join(f'"{c}"' if " " in c else c for c in cmd))
            srv_log_path = os.path.join(self.repo_root, "logs", "llama_server.log")
            srv_log_handle = None
            try:
                os.makedirs(os.path.dirname(srv_log_path), exist_ok=True)
                srv_log_handle = open(srv_log_path, "a", encoding="utf-8")
            except OSError:
                srv_log_handle = None

            proc_env = dict(os.environ)
            if sys.platform.startswith("linux"):
                bin_dir = os.path.dirname(os.path.abspath(server_bin))
                prev_ld = proc_env.get("LD_LIBRARY_PATH", "")
                proc_env["LD_LIBRARY_PATH"] = f"{bin_dir}:{prev_ld}" if prev_ld else bin_dir

            popen_kwargs: Dict[str, Any] = {
                "cwd": self.repo_root,
                "env": proc_env,
                "stdout": srv_log_handle if srv_log_handle is not None else subprocess.DEVNULL,
                "stderr": subprocess.STDOUT if srv_log_handle is not None else subprocess.DEVNULL,
            }
            if sys.platform.startswith("win"):
                popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)

            try:
                self._server_proc = subprocess.Popen(cmd, **popen_kwargs)
            finally:
                if srv_log_handle is not None:
                    srv_log_handle.close()

            if (
                self.vault is not None
                and hasattr(self.vault, "register_process_for_kill_switch")
                and self._server_proc.pid
            ):
                self.vault.register_process_for_kill_switch(self._server_proc.pid)

            t0 = time.time()
            while time.time() - t0 < wait_timeout:
                if self._server_proc.poll() is not None:
                    tail = ""
                    try:
                        if os.path.isfile(srv_log_path):
                            with open(srv_log_path, "r", encoding="utf-8", errors="replace") as lf:
                                lines = lf.read().splitlines()[-20:]
                            tail = "\n".join(lines)[-1500:]
                    except OSError:
                        tail = ""
                    logger.error(
                        "Процес llama-server завершився достроково з кодом %s.%s",
                        self._server_proc.returncode,
                        f"\n{tail}" if tail else "",
                    )
                    return False
                if self.is_server_healthy(timeout=1.0):
                    logger.info("llama-server успішно запущено за %.1f с (PID=%d).", time.time() - t0, self._server_proc.pid)
                    return True
                time.sleep(0.5)

            logger.error("Час очікування запуску llama-server (%.0f с) вичерпано.", wait_timeout)
            return False

    def build_warmup_payload(self, question: str) -> Dict[str, Any]:
        """
        Формує JSON-пейлоад для фонового прогріву `/v1/chat/completions`
        із вимкненим режимом міркувань (`chat_template_kwargs: {"enable_thinking": false}`).
        """
        params = self.config.get("llama_params") or {}
        messages = [
            {"role": "system", "content": self.build_system_prompt()},
            {"role": "user", "content": question.strip()},
        ]
        return {
            "model": "qwen",
            "messages": messages,
            "temperature": float(params.get("temp", 0.52)),
            "top_p": float(params.get("top_p", 0.80)),
            "min_p": float(params.get("min_p", 0.10)),
            "repeat_penalty": float(params.get("repeat_penalty", 1.12)),
            "presence_penalty": float(params.get("presence_penalty", 0.35)),
            "repeat_last_n": int(params.get("repeat_last_n", 256)),
            "n_predict": int(params.get("n_predict", 160)),
            "max_tokens": int(params.get("n_predict", 160)),
            "dry_multiplier": float(params.get("dry_multiplier", 0.8)),
            "chat_template_kwargs": {
                "enable_thinking": False,
            },
            "stream": False,
        }

    def warmup_model(self, questions: Optional[List[str]] = None) -> List[str]:
        """
        Надсилає 2 фонові питання для прогріву моделі (з `enable_thinking: false`),
        не зберігаючи їх у `self.history` і не показуючи в чаті.
        """
        warmup_qs = questions if questions is not None else self.get_warmup_questions()
        params = self.config.get("llama_params") or {}
        port = int(params.get("port", 8080))
        url = f"http://127.0.0.1:{port}/v1/chat/completions"
        answers: List[str] = []

        for idx, q in enumerate(warmup_qs, start=1):
            payload = self.build_warmup_payload(q)
            req = urllib.request.Request(
                url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            t0 = time.time()
            try:
                logger.info("Фоновий прогрів [%d/%d]: %s", idx, len(warmup_qs), q)
                with urllib.request.urlopen(req, timeout=120) as resp:
                    raw_body = resp.read().decode("utf-8")
                data = json.loads(raw_body)
                reply = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
                )
                logger.info("Відповідь прогріву [%d/%d] за %.2f с: %s", idx, len(warmup_qs), time.time() - t0, reply)
                answers.append(reply)
            except Exception as exc:
                logger.warning("Помилка під час фонового прогріву [%d/%d]: %s", idx, len(warmup_qs), exc)
        return answers

    def prepare_session(
        self,
        user_id: int,
        user_name_ua: str,
        is_experimental: bool = False,
        chat_mode: str = "personal",
    ) -> bool:
        """
        Повна фонова підготовка сесії під час екрана завантаження:
          1. Оновлює співрозмовника та скидає історію чату.
          2. Запускає `llama-server` (якщо ще не запущений).
          3. Виконує 2 фонові запити прогріву без збереження в історію.
        """
        self.set_interlocutor(
            user_id=user_id,
            user_name_ua=user_name_ua,
            is_experimental=is_experimental,
            chat_mode=chat_mode,
            reset_history=True,
        )
        server_ok = self.ensure_server_running()
        if server_ok:
            self.warmup_model()
        return server_ok

    def _request_chat_completion(
        self,
        messages: List[Dict[str, str]],
        params: Dict[str, Any],
        port: int,
    ) -> str:
        payload = {
            "messages": messages,
            "temperature": float(params.get("temp", 0.52)),
            "top_p": float(params.get("top_p", 0.80)),
            "min_p": float(params.get("min_p", 0.10)),
            "repeat_penalty": float(params.get("repeat_penalty", 1.12)),
            "presence_penalty": float(params.get("presence_penalty", 0.35)),
            "repeat_last_n": int(params.get("repeat_last_n", 256)),
            "n_predict": int(params.get("n_predict", 160)),
            "max_tokens": int(params.get("n_predict", 160)),
            "dry_multiplier": float(params.get("dry_multiplier", 0.8)),
            "chat_template_kwargs": {"enable_thinking": False},
            "stream": False,
        }
        url = f"http://127.0.0.1:{port}/v1/chat/completions"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw_body = resp.read().decode("utf-8")
        data = json.loads(raw_body)
        return (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

    def send_message(self, user_text: str) -> str:
        """
        Додає повідомлення користувача до історії, надсилає запит у `llama-server`,
        за потреби виконує скіли `[SKILL: ...]`, логує хід у JSONL і повертає відповідь.
        """
        cleaned_user = user_text.strip()
        if not cleaned_user:
            return ""

        if self.skill_runner.available and self.skill_runner.is_slash_help_command(cleaned_user):
            return self.skill_runner.format_slash_help()

        slash_skill_res = None
        if self.skill_runner.available:
            slash_skill_res = self.skill_runner.execute_slash_command(cleaned_user)

        if not self.is_server_healthy():
            self.ensure_server_running()

        req_gen = self._history_generation
        messages = self.build_messages_payload(
            new_user_message=cleaned_user,
            include_rag=(slash_skill_res is None),
        )
        rag_snapshot = list(self.last_rag_hits)
        params = self.config.get("llama_params") or {}
        port = int(params.get("port", 8080))

        logger.info(
            "Повідомлення від '%s' (історія=%d реплік): %s",
            self.user_name_ua,
            len(self.history),
            cleaned_user,
        )

        t0 = time.time()
        active_skill_res = slash_skill_res
        try:
            if active_skill_res is None and self.skill_runner.available:
                active_skill_res = self.skill_runner.execute_for_query(cleaned_user, threshold=0.75)

            if active_skill_res is not None and active_skill_res.ok:
                trigger_kind = "Slash-скіл" if slash_skill_res is not None else "Авто-скіл за тригером"
                logger.info(
                    "%s '%s' (ok=%s, %.1f мс): %s",
                    trigger_kind,
                    active_skill_res.skill_name,
                    active_skill_res.ok,
                    active_skill_res.elapsed_ms,
                    active_skill_res.output,
                )
                skill_out = (active_skill_res.output or "").strip()
                reaction_messages = list(messages)
                # Не вставляємо проміжний role='system' (LoRA його ігнорує) — передаємо результат у фінальний user-запит
                reaction_messages[-1] = {
                    "role": "user",
                    "content": (
                        f"{cleaned_user}\n\n"
                        f"[Сработал инструмент. Точный результат: {skill_out}]\n"
                        "Напиши 1 короткую живую реакцию на этот результат в своем стиле (без повторения всего текста)."
                    ),
                }
                reaction_params = dict(params)
                reaction_params["n_predict"] = min(48, int(params.get("n_predict", 160)))
                try:
                    raw_comment = self._request_chat_completion(reaction_messages, params=reaction_params, port=port)
                    comment = strip_skill_tags(raw_comment).strip()
                except Exception as react_exc:
                    logger.warning("Не вдалося отримати реакцію моделі на скіл (%s) — показуємо прямий результат.", react_exc)
                    comment = ""

                if comment and comment.lower() not in skill_out.lower():
                    reply = f"{skill_out}\n\n{comment}"
                else:
                    reply = skill_out
            else:
                reply = self._request_chat_completion(messages, params=params, port=port)

                if self.skill_runner.available:
                    parsed_call = self.skill_runner.execute_tag(reply)
                    if parsed_call is not None:
                        logger.info(
                            "Виконано скіл '%s' (ok=%s, %.1f мс)",
                            parsed_call.skill_name,
                            parsed_call.ok,
                            parsed_call.elapsed_ms,
                        )
                        stripped = strip_skill_tags(reply)
                        if parsed_call.ok and parsed_call.output:
                            reply = f"{parsed_call.output}\n\n{stripped}".strip() if stripped else parsed_call.output
                        else:
                            reply = stripped or f"[Помилка скіла: {parsed_call.error}]"
                    else:
                        reply = strip_skill_tags(reply)

            dt = time.time() - t0
            if req_gen == self._history_generation:
                self.history.append(messages[-1])
                self.history.append({"role": "assistant", "content": reply})
                self._log_chat_turn(cleaned_user, reply, rag_hits=rag_snapshot)
                logger.info("Відповідь '%s' за %.2f с: %s", self.persona_id, dt, reply)
            else:
                logger.info("Ігноровано застарілу відповідь '%s' після скидання чату (CTRL+N).", self.persona_id)
            return reply
        except urllib.error.URLError as exc:
            logger.warning("llama-server недоступний на 127.0.0.1:%d (%s).", port, exc)
            if slash_skill_res is not None and slash_skill_res.ok and slash_skill_res.output:
                return slash_skill_res.output
            return f"[llama-server не запущений на порту {port}. Запустіть сервер для генерації відповіді.]"
        except Exception as exc:
            logger.exception("Помилка генерації відповіді: %s", exc)
            return f"[Помилка генерації: {exc}]"

    def shutdown(self) -> None:
        if self._server_proc is not None and self._server_proc.poll() is None:
            logger.info("Зупинка підпроцесу llama-server (PID=%d)...", self._server_proc.pid)
            self._server_proc.terminate()
            try:
                self._server_proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self._server_proc.kill()
            self._server_proc = None
        for staged in list(getattr(self, "_staged_runtime_files", [])):
            try:
                if os.path.exists(staged):
                    os.unlink(staged)
            except OSError:
                pass
        if hasattr(self, "_staged_runtime_files"):
            self._staged_runtime_files.clear()


# =====================================================================
# АВТОНОМНИЙ SELF-TEST ТА CLI
# =====================================================================
def run_self_test() -> None:
    import tempfile

    print("====================================================================")
    print(" [*] GVK Private AI — Self-Test оркестратора (core/orchestrator.py)")
    print("====================================================================")

    orch = GVKOrchestrator(persona_id="friend_1")
    cmd = orch.get_llama_server_command()
    assert "--lora-scaled" in cmd, f"Очікувався --lora-scaled у команді: {cmd}"
    assert "-ngl" not in cmd

    test_date = datetime.date(2026, 11, 17)
    orch.set_interlocutor(user_id=2, user_name_ua=orch.user_name_ua, is_experimental=False)
    sys_prompt = orch.build_system_prompt(today=test_date)
    expected_age = calculate_age(str(orch.config.get("birth_date", "01.01.2000")), today=test_date)
    assert f"Сегодняшняя дата: 17.11.2026, тебе {expected_age} " in sys_prompt

    dossier = orch.get_persona_dossier(current_emotion="idle", today=test_date)
    assert dossier["title"] and dossier["description"]
    assert "[OK]" in dossier["rag_status"], f"Очікувався робочий RAG у досьє: {dossier}"
    print(f" [1/4] Досьє та CLI OK: {dossier['title']} | {dossier['rag_status']}")

    rag_block = orch.retrieve_memories_block("го катати в рокет лигу")
    assert rag_block and orch.last_rag_hits
    print(f" [2/4] RAG-пам'ять ін'єктована: {len(orch.last_rag_hits)} спогадів")

    speakers = orch.select_next_speakers(
        "друг_1, зайди в рл",
        candidate_configs={
            "friend_1": {"trigger_words": ["друг_1", "рл"], "chattiness_prob": 0.1},
            "friend_2": {"trigger_words": ["місто_2"], "chattiness_prob": 0.9},
            "friend_3": {"trigger_words": ["гра_3"], "chattiness_prob": 0.9},
        },
        max_bot_turns=2,
    )
    assert speakers and speakers[0] == "friend_1" and len(speakers) <= 2
    subword_check = orch.select_next_speakers(
        "привіт, я в столиці",
        candidate_configs={
            "friend_2": {"trigger_words": ["столиці"], "chattiness_prob": 0.0},
            "friend_1": {"trigger_words": ["рл"], "chattiness_prob": 0.0},
        },
        candidate_skill_runners={"friend_2": None},
        max_bot_turns=2,
    )
    assert subword_check == ["friend_2"], f"Короткий тригер не повинен спрацьовувати всередині слів: {subword_check}"
    print(f" [3/4] Cross-Talk арбітр OK: {speakers}")

    with tempfile.TemporaryDirectory(prefix="gvk_orch_log_") as tmpdir:
        log_file = os.path.join(tmpdir, "chat_history.jsonl")
        orch._log_chat_turn("привіт", "здарова", rag_hits=orch.last_rag_hits, history_path=log_file)
        with open(log_file, "r", encoding="utf-8") as f:
            row = json.loads(f.readline())
        assert row["persona_id"] == "friend_1" and row["user_text"] == "привіт"
    print(" [4/4] JSONL-логер діалогів OK")

    print("====================================================================")
    print(" [✓] УСІ ТЕСТИ ORCHESTRATOR.PY ПРОЙДЕНО УСПІШНО!")
    print("====================================================================")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GVK Private AI — Оркестратор (core/orchestrator.py)")
    parser.add_argument("cmd", nargs="?", choices=["self-test"], default="self-test")
    args = parser.parse_args()
    if args.cmd == "self-test":
        run_self_test()
