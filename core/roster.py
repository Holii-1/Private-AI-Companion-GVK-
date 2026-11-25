#!/usr/bin/env python3
"""
GVK Private AI — Єдиний завантажувач приватного ростеру (core/roster.py)

Уніфікує пошук і парсинг єдиного файлу `personas/roster.json` для всіх модулів
(`core/menu.py`, `core/orchestrator.py`, `core/main.py`) без зайвих залежностей.

Порядок пошуку `roster.json`:
  1. RAM-диск (`<ram_root>/personas/roster.json`, `<ram_root>/personas/<persona_id>/roster.json`, ...)
  2. Локальний репозиторій (`<repo_root>/personas/roster.json`, `<persona_dir>/roster.json`, ...)
  3. Домашній кеш (`~/.gvk_ai/personas/roster.json`, ...)
"""

import glob
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("gvk.roster")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_roster_data(
    ram_root: Optional[str] = None,
    repo_root: str = REPO_ROOT,
    gvk_home: Optional[str] = None,
    persona_id: Optional[str] = None,
    persona_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Знаходить і завантажує єдиний приватний `roster.json` із RAM-диска,
    локальної папки `personas/` або `~/.gvk_ai/personas/`.
    Повертає словник JSON або `{}` за відсутності файлу.
    """
    home_abs = os.path.abspath(os.path.expanduser(gvk_home or os.environ.get("GVK_HOME") or "~/.gvk_ai"))
    candidates: List[str] = []

    def _add_root_candidates(root_dir: Optional[str]) -> None:
        if not root_dir:
            return
        p_base = os.path.join(root_dir, "personas")
        candidates.append(os.path.join(p_base, "roster.json"))
        if persona_id:
            candidates.append(os.path.join(p_base, persona_id, "roster.json"))
        candidates.append(os.path.join(p_base, "friend_1", "roster.json"))
        if os.path.isdir(p_base):
            candidates.extend(sorted(glob.glob(os.path.join(p_base, "friend_*", "roster.json"))))

    _add_root_candidates(ram_root)
    if repo_root:
        candidates.append(os.path.join(repo_root, "personas", "roster.json"))
    if persona_dir:
        candidates.append(os.path.join(persona_dir, "roster.json"))
    _add_root_candidates(repo_root)
    _add_root_candidates(home_abs)

    seen = set()
    for path in candidates:
        norm = os.path.abspath(path)
        if norm in seen:
            continue
        seen.add(norm)
        if not os.path.isfile(norm):
            continue
        try:
            with open(norm, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Не вдалося прочитати roster.json (%s): %s", norm, exc)
    return {}


def extract_roster_mappings(
    roster: Dict[str, Any],
) -> Tuple[Dict[str, str], Dict[str, str], Optional[str]]:
    """
    Витягує зі словника `roster.json`:
      - `ua_to_ru_names`: словник перекладу імен UA -> RU
      - `interlocutor_descriptions`: словник повних описів співрозмовників
      - `default_user_name`: ім'я користувача `[02]` за замовчуванням (або None)
    """
    if not isinstance(roster, dict):
        return {}, {}, None

    ua_to_ru: Dict[str, str] = {}
    raw_ua_to_ru = roster.get("ua_to_ru_names")
    if isinstance(raw_ua_to_ru, dict):
        ua_to_ru.update({str(k): str(v) for k, v in raw_ua_to_ru.items()})

    descriptions: Dict[str, str] = {}
    raw_desc = roster.get("interlocutor_descriptions")
    if isinstance(raw_desc, dict):
        descriptions.update({str(k): str(v) for k, v in raw_desc.items()})

    default_user_name: Optional[str] = None
    raw_users = roster.get("users_dict")
    if isinstance(raw_users, dict):
        u2 = raw_users.get("2") or raw_users.get(2)
        if isinstance(u2, dict) and u2.get("name"):
            default_user_name = str(u2["name"])

    return ua_to_ru, descriptions, default_user_name
