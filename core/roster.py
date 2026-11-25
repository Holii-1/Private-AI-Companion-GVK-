#!/usr/bin/env python3
"""
GVK Private AI - Unified Private Roster Loader (core/roster.py)

Centralizes discovery and parsing of the git-ignored `personas/roster.json`
file across all runtime modules (`core/menu.py`, `core/orchestrator.py`,
`core/main.py`) with zero extra dependencies.

Key Responsibilities:
  - Search for `roster.json` in priority order: RAM disk, local repository,
    and `~/.gvk_ai` home cache.
  - Extract UA-to-RU name mappings, interlocutor descriptions, and the
    default `[02]` user display name.

Usage:
  from core.roster import extract_roster_mappings, load_roster_data
  roster = load_roster_data(ram_root=ram_root)
  ua_to_ru, descriptions, default_user = extract_roster_mappings(roster)
"""

import glob
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("gvk.roster")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# =====================================================================
# ROSTER DISCOVERY & PARSING
# =====================================================================
def load_roster_data(
    ram_root: Optional[str] = None,
    repo_root: str = REPO_ROOT,
    gvk_home: Optional[str] = None,
    persona_id: Optional[str] = None,
    persona_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Locate and load the private `roster.json` file.

    Searches in priority order across the mounted RAM disk, the local
    repository `personas/` directory, and the `~/.gvk_ai/personas/` cache.

    Args:
        ram_root: Optional path to the mounted RAM disk root.
        repo_root: Path to the repository root directory.
        gvk_home: Optional override for the `~/.gvk_ai` home cache directory.
        persona_id: Optional persona identifier to check under `personas/<id>/`.
        persona_dir: Optional explicit persona directory path.

    Returns:
        Parsed JSON dictionary from the first readable `roster.json`, or `{}`
        if no valid roster file is found.

    Example:
        >>> roster = load_roster_data(persona_id="friend_1")
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
            logger.warning("Failed to read roster.json (%s): %s", norm, exc)
    return {}


def extract_roster_mappings(
    roster: Dict[str, Any],
) -> Tuple[Dict[str, str], Dict[str, str], Optional[str]]:
    """Extract name translations, interlocutor descriptions, and default user.

    Args:
        roster: Parsed `roster.json` dictionary.

    Returns:
        A 3-tuple `(ua_to_ru_names, interlocutor_descriptions, default_user_name)`:
        - `ua_to_ru_names`: Mapping of Ukrainian names to Russian equivalents.
        - `interlocutor_descriptions`: Mapping of user names to full prompt descriptions.
        - `default_user_name`: Display name for default user `[02]`, or `None`.

    Example:
        >>> ua_to_ru, descriptions, default_user = extract_roster_mappings(roster)
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
