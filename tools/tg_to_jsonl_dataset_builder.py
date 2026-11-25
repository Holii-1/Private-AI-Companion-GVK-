#!/usr/bin/env python3
"""
tg_to_jsonl_dataset_builder.py

Standalone pipeline to process raw Telegram chat export JSON files into an
instruction/chat JSONL dataset for LoRA / full fine-tuning of conversational models.

Key Pipeline Features:
1. Strict chat isolation: each chat file (personal or group) is segmented and processed independently.
2. Dual-mode support (Combined pipeline):
   - Personal chats (1-on-1): Clean dialogue without speaker prefixes.
   - Group chats: Senders mapped via USER_ID_MAP and prefixed as '[Имя Фамилия]: текст'
     while assistant turns remain pure, without prefix.
3. Telegram User ID mapping: Maps raw Telegram user IDs to full names and skips unmapped
   accounts (bots, strangers) with a console warning.
4. Assistant Text Integrity (Golden Rules):
   - Rule 1: If the assistant sent voice/media + text in the same assistant turn, naked placeholders
     are stripped, preserving the actual text.
   - Rule 2: If the assistant sent ONLY voice/media without text for the entire turn (a 'black hole'),
     the entire (User + Assistant) pair is dropped. If in the middle of a dialogue, the session
     is cleanly cut into two sub-sessions at that point to avoid any context leap.
   - User turns retain media placeholders ([Голосовое сообщение], [Фотография], etc.) as context.
5. Dynamic system prompt injection:
   - Personal: 'Собеседник в этом диалоге: {description}.'
   - Group: 'Ты сейчас в групе с: {description}. Сообщения участников подписаны в формате 'Имя: текст'. Отвечай от своего лица без префикса.'
6. Media placeholders: Converts raw Telegram media (photos, voice messages, video circles,
   stickers, files, forwarded messages) into structured textual placeholders instead of dropping them.
7. Orphan media reaction pruning: Strips useless assistant turns that react solely to invisible media
   (e.g., "не грузит", "покажи", "че за хуйня").
8. Discord / VoIP meta-talk filtering: Removes technical voice chat coordination sessions
   (e.g., "го в дс", "микро включи", "не слышу тебя").
9. Sliding window with overlap (chunking): Retains 2-4 preceding context turns across window cuts.
10. Global chronological ordering: All samples across all chats are strictly sorted earliest-first.
11. Temporal weighting: Replicates samples by epoch (pre-2022, 2022-2024, post-2024).
12. Externalized private config: Reads Telegram IDs and chat maps from a git-ignored JSON config
    (`--config data/builder_config_<persona>.json` or `--persona friend_2`).
"""

import argparse
import json
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# ==============================================================================
# 1. Configuration & Global Variables (Anonymized defaults; override via --config)
# ==============================================================================
REPO_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSISTANT_FROM_ID: str = "user000000001"

# Base directory where exported chat JSON files reside
RAW_CHATS_DIR: str = "./raw_chats_and_prompts"

# User ID Mapping: Maps Telegram userIDs to participant display names (load private map via --config)
USER_ID_MAP: Dict[str, str] = {
    "user000000002": "Friend 2",
    "user000000003": "Friend 3",
}

# Chat Configuration: maps filenames to chat type ("personal" | "group") and description
CHATS_CONFIG: Dict[str, Dict[str, str]] = {
    "personal_friend_2.json": {
        "type": "personal",
        "description": "Friend 2",
    },
    "group_trio.json": {
        "type": "group",
        "description": "Friend 2, Friend 3",
    },
}

# Backwards compatibility alias
INTERLOCUTOR_MAP = CHATS_CONFIG

OUTPUT_FILE: str = "./datasets/persona_dataset.jsonl"
SESSION_IDLE_TIMEOUT_MINUTES: int = 15
MAX_TURNS_PER_SAMPLE: int = 32
CHUNK_OVERLAP_TURNS: int = 4  # Overlap of 2-4 turns (2 full dialogue exchanges)

# System prompt templates
PERSONAL_PROMPT_TEMPLATE: str = "Собеседник в этом диалоге: {description}."
GROUP_PROMPT_TEMPLATE: str = (
    "Ты сейчас в групе с: {description}. "
    "Сообщения участников подписаны в формате 'Имя: текст'. "
    "Отвечай от своего лица без префикса."
)

# System prompt source file paths (checked in order of preference)
SYSTEM_PROMPT_CANDIDATES: List[str] = [
    "./raw_chats_and_prompts/training_system_prompt.txt",
    "./raw_chats_and_prompt/training_system_prompt.txt",
    "./personas/friend_1/prompt.txt",
]

# UTC Epoch boundaries for temporal weighting
EPOCH_2022_START: int = 1640995200  # 2022-01-01 00:00:00 UTC
EPOCH_2024_START: int = 1704067200  # 2024-01-01 00:00:00 UTC

# Temporal weighting multipliers (number of replications per epoch).
# These accept integer values from 0 to inf (a weight of 0 excludes that epoch from the final dataset).
WEIGHT_PRE_2022: int = 0
WEIGHT_2022_2024: int = 1
WEIGHT_POST_2024: int = 1


def load_builder_config(
    config_path: Optional[str] = None,
    persona_id: str = "friend_1",
) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Loads persona-specific dataset builder config (Telegram IDs, chats map, weights).
    Resolves explicit `--config`, or auto-discovers `data/builder_config_<persona>.json` /
    `raw_chats_and_prompts/<persona>/builder_config.json`.
    """
    resolved_cfg: Optional[str] = None
    if config_path:
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Builder config not found: {config_path}")
        resolved_cfg = os.path.abspath(config_path)
    else:
        candidates = [
            os.path.join(REPO_ROOT, "data", f"builder_config_{persona_id}.json"),
            os.path.join(REPO_ROOT, "raw_chats_and_prompts", persona_id, "builder_config.json"),
            os.path.join(REPO_ROOT, "personas", persona_id, "builder_config.json"),
        ]
        for cand in candidates:
            if os.path.isfile(cand):
                resolved_cfg = os.path.abspath(cand)
                break

    cfg: Dict[str, Any] = {
        "persona_id": persona_id,
        "assistant_from_id": ASSISTANT_FROM_ID,
        "raw_chats_dir": RAW_CHATS_DIR,
        "output_file": f"./datasets/{persona_id}_dataset_expanded.jsonl"
        if persona_id != "friend_1"
        else OUTPUT_FILE,
        "session_idle_timeout_minutes": SESSION_IDLE_TIMEOUT_MINUTES,
        "max_turns_per_sample": MAX_TURNS_PER_SAMPLE,
        "chunk_overlap_turns": CHUNK_OVERLAP_TURNS,
        "weight_pre_2022": WEIGHT_PRE_2022,
        "weight_2022_2024": WEIGHT_2022_2024,
        "weight_post_2024": WEIGHT_POST_2024,
        "system_prompt_candidates": [
            f"./raw_chats_and_prompts/{persona_id}/training_system_prompt.txt",
            f"./personas/{persona_id}/prompt.txt",
            *SYSTEM_PROMPT_CANDIDATES,
        ],
        "user_id_map": dict(USER_ID_MAP),
        "chats_config": dict(CHATS_CONFIG),
    }

    if resolved_cfg:
        with open(resolved_cfg, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise ValueError(f"Builder config must be a JSON object: {resolved_cfg}")
        cfg.update(loaded)

    return cfg, resolved_cfg

# ==============================================================================
# 2. Heuristic Regex Patterns for Data Cleaning
# ==============================================================================
# Media reaction heuristics: assistant turns that are uninformative without visual media
ORPHAN_MEDIA_REACTIONS = re.compile(
    r"^(бля\s+|а\s+|ну\s+|да\s+)?"
    r"(покажи(\s+что\s+это|\s+его)?|скинь\s+(еще\s+раз|еще|нормально)|не\s+(грузит|открывается)|"
    r"че\s+(это\s+)?за\s+(хуйня|дичь)|что\s+это\s+за\s+(хуйня|дичь)|вот\s+это(\s+тоже)?)\b",
    re.IGNORECASE,
)

MEDIA_PLACEHOLDER_PREFIXES: Tuple[str, ...] = (
    "[Фото",
    "[Видео",
    "[Пересланное",
    "[Стикер",
    "[Голосовое",
    "[Файл",
    "[Аудио",
    "[Геолокация",
    "[Опрос",
)

# Regex to detect lines that are purely naked media placeholders without human text
PLACEHOLDER_LINE_REGEX = re.compile(
    r"^\s*\[(Фотография|Голосовое сообщение|Видео-кружок|Стикер(\s+[^\]]+)?|Видео/GIF|Аудиосообщение|Файл|Геолокация|Опрос|Пересланное медиа/сообщение от [^\]]+)\]\s*$"
)

# Discord / VoIP coordination patterns (technical audio/connection crosstalk)
VOICE_METATALK_PATTERNS = re.compile(
    r"("
    r"\b((тебя|вас|меня)?\s*(не\s+слышу|не\s+слышно|плохо\s+слышно|еле\s+слышно)|слышно(\s+меня|\s+тебя|\s+ща|\s+щас|\s+сейчас|\s+вас)?)\b|"
    r"\b(микрофон|микро|наушник\w*|звук)\b.*\b(тихо|громко|фонит|шипит|настрой|сделай|поставь|включи|выключи|выруби|подруби|офф|сел|сели)\b|"
    r"\b(в\s+сети\s+отключилось|дискорд\s+(лагает|лежит|упал)|дропнуло(\s+из\s+войса)?)\b|"
    r"\b(замуть\w*|размуть\w*|размьють\w*|замьють\w*|мут\s+сними|перезайди\s+в\s+(дис|дс|войс|дискорд)|(зайди|го|прыгай|иди)\s+в\s+(дис|дс|войс|дискорд)|микро(фон)?\s*(включи|выключи|оффни|подруби|вруби))\b|"
    r"\b(ебать\s+как\s+громко(\s+стало)?|тут\s+громко\s+было|секунду\s+назад\s+норм\s+было)\b"
    r")",
    re.IGNORECASE,
)


# ==============================================================================
# 3. Data Structures
# ==============================================================================
@dataclass
class CleanMessage:
    from_id: str
    timestamp: int
    text: str
    role: str = ""


@dataclass
class Turn:
    role: str
    content: str
    timestamp: int


@dataclass
class DatasetSample:
    messages: List[Dict[str, str]]
    timestamp: int
    interlocutor: str
    session_index: int
    chat_type: str = "personal"


@dataclass
class ChatStats:
    filename: str
    chat_type: str
    friend_info: str
    total_raw_entries: int = 0
    valid_extracted_messages: int = 0
    media_placeholders_created: int = 0
    messages_joined_count: int = 0
    raw_sessions_count: int = 0
    voice_sessions_dropped: int = 0
    media_pairs_pruned: int = 0
    assistant_media_stripped: int = 0
    empty_assistant_pairs_dropped: int = 0
    active_sessions_count: int = 0
    samples_generated: int = 0
    unmapped_ids_count: int = 0


# ==============================================================================
# 4. Message Extraction & Media Placeholder Conversion
# ==============================================================================
def normalize_telegram_text(raw_text: Any) -> str:
    """
    Extract and concatenate text whether 'text' is a plain string or a list
    containing strings and rich-text dicts ([{"type": "...", "text": "..."}]).
    """
    if isinstance(raw_text, str):
        return raw_text.strip()
    if isinstance(raw_text, list):
        parts: List[str] = []
        for item in raw_text:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                val = item["text"]
                if isinstance(val, str):
                    parts.append(val)
        return "".join(parts).strip()
    return ""


def extract_message_content(entry: dict) -> Tuple[str, bool]:
    """
    Extract text or construct informative semantic placeholders for media.
    Returns (content_string, is_media_placeholder).
    """
    raw_text = normalize_telegram_text(entry.get("text", ""))

    # 1. Forwarded messages
    if entry.get("forwarded_from"):
        fwd_sender = entry.get("forwarded_from", "неизвестно")
        if raw_text:
            return f"[Пересланное сообщение от {fwd_sender}]: {raw_text}", True
        return f"[Пересланное медиа/сообщение от {fwd_sender}]", True

    # 2. Text messages (including text with media attached)
    if raw_text:
        return raw_text, False

    # 3. Media without caption (photos, voice messages, video notes, stickers, files)
    if entry.get("photo"):
        return "[Фотография]", True

    media_type = entry.get("media_type")
    if media_type == "voice_message":
        return "[Голосовое сообщение]", True
    elif media_type == "video_message":
        return "[Видео-кружок]", True
    elif media_type == "sticker":
        emoji = entry.get("sticker_emoji", "").strip()
        placeholder = f"[Стикер {emoji}]" if emoji else "[Стикер]"
        return placeholder, True
    elif media_type in ("video_file", "animation"):
        return "[Видео/GIF]", True
    elif media_type == "audio_file":
        return "[Аудиосообщение]", True
    elif entry.get("file"):
        return "[Файл]", True
    elif entry.get("location_information"):
        return "[Геолокация]", True
    elif entry.get("poll"):
        return "[Опрос]", True

    return "", False


def normalize_user_id(from_id: Any) -> str:
    """Normalize raw user ID representation to canonical 'user<numeric_id>' format."""
    if from_id is None:
        return ""
    s = str(from_id).strip()
    if s.startswith("user"):
        return s
    if s.isdigit():
        return f"user{s}"
    return s


def is_assistant_sender(from_id: Any, target_assistant_id: str) -> bool:
    """Check if message sender ID corresponds to the target assistant ID."""
    if from_id is None:
        return False
    return normalize_user_id(from_id) == normalize_user_id(target_assistant_id)


# ==============================================================================
# 5. Dialogue Hygiene & Filtering (Golden Rules for Assistant)
# ==============================================================================
def strip_naked_media_placeholders(text: str) -> str:
    """
    Remove lines that are purely naked media placeholders
    (voice message, photo without caption, sticker, etc.), preserving all real human text.
    """
    if not text:
        return ""
    clean_lines: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if PLACEHOLDER_LINE_REGEX.match(stripped):
            continue
        if any(
            stripped == p
            or (stripped.startswith(p) and stripped.endswith("]") and not stripped.startswith("[Пересланное сообщение"))
            for p in MEDIA_PLACEHOLDER_PREFIXES
        ):
            continue
        clean_lines.append(stripped)
    return "\n".join(clean_lines).strip()


def split_session_on_empty_assistant(
    turns: List[Turn],
) -> Tuple[List[List[Turn]], int, int]:
    """
    Enforce Golden Rules on Assistant turns:
    - Rule 1: If assistant had voice/media + text in the same turn, strip naked placeholders
      and keep only the real human text.
    - Rule 2: If assistant had ONLY voice/media without text for the entire turn (a 'black hole'),
      drop the entire (User + Assistant) pair. If in the middle of a dialogue, cut the session
      into distinct sub-sessions at this point so there is no context leap.
    Returns (list_of_valid_sub_sessions, media_stripped_count, empty_pairs_dropped_count).
    """
    valid_segments: List[List[Turn]] = []
    current_segment: List[Turn] = []
    media_stripped_count = 0
    empty_pairs_dropped_count = 0

    num_pairs = len(turns) // 2
    for k in range(num_pairs):
        u_turn = turns[2 * k]
        a_turn = turns[2 * k + 1]

        # Sanitize assistant turn by stripping naked placeholders
        cleaned_assistant_content = strip_naked_media_placeholders(a_turn.content)
        if cleaned_assistant_content != a_turn.content:
            media_stripped_count += 1

        if cleaned_assistant_content:
            # Assistant has real human text: pair is saved!
            current_segment.append(u_turn)
            current_segment.append(
                Turn(role=a_turn.role, content=cleaned_assistant_content, timestamp=a_turn.timestamp)
            )
        else:
            # Rule 2: Assistant had ONLY naked media / voice without text -> black hole!
            # Drop pair (User + Assistant) and cut session into two parts
            empty_pairs_dropped_count += 1
            if len(current_segment) >= 2:
                valid_segments.append(current_segment)
            current_segment = []

    if len(current_segment) >= 2:
        valid_segments.append(current_segment)

    return valid_segments, media_stripped_count, empty_pairs_dropped_count


def is_useless_media_reaction(assistant_text: str, prev_user_text: str) -> bool:
    """
    Determine if an assistant turn is an orphan reaction to invisible media
    (e.g., user sent [Фотография] or [Имя]: [Фотография] and assistant answered 'не грузит').
    """
    clean_user = prev_user_text.strip()
    # Strip sender prefix like '[Имя Фамилия]: ' if present
    stripped_user = re.sub(r"^\[[^\]]+\]:\s*", "", clean_user)
    if stripped_user.startswith(MEDIA_PLACEHOLDER_PREFIXES):
        if ORPHAN_MEDIA_REACTIONS.search(assistant_text.strip()):
            return True
    return False


def prune_useless_media_exchanges(turns: List[Turn]) -> Tuple[List[Turn], int]:
    """
    Remove turn pairs where the assistant responds with an orphan media reaction.
    Because turns strictly alternate (user, assistant, user, assistant),
    removing user-assistant pairs guarantees alternating role order is preserved.
    """
    if len(turns) < 2:
        return turns, 0

    cleaned: List[Turn] = []
    removed_pairs = 0
    i = 0
    while i < len(turns):
        if i + 1 < len(turns) and turns[i].role == "user" and turns[i + 1].role == "assistant":
            if is_useless_media_reaction(turns[i + 1].content, turns[i].content):
                removed_pairs += 1
                i += 2  # Discard both user turn and orphan assistant reaction
                continue
        cleaned.append(turns[i])
        i += 1

    # Ensure sequence still starts with "user" and ends with "assistant"
    while cleaned and cleaned[0].role != "user":
        cleaned.pop(0)
    while cleaned and cleaned[-1].role != "assistant":
        cleaned.pop()

    return cleaned, removed_pairs


def is_voice_garbage_session(turns: List[Turn]) -> bool:
    """
    Identify sessions consisting purely of technical voice chat / Discord coordination.
    - If a short session (<= 4 turns) contains any voice metatalk pattern, prune it.
    - If a longer session has >= 50% turns matching voice metatalk, prune it.
    """
    if not turns:
        return True
    voice_matches = sum(1 for t in turns if VOICE_METATALK_PATTERNS.search(t.content))
    if len(turns) <= 4:
        return voice_matches >= 1
    return (voice_matches / len(turns)) >= 0.5


# ==============================================================================
# 6. Sliding Window Chunking with Overlap
# ==============================================================================
def chunk_turns_into_samples(
    turns: List[Turn],
    max_turns: int,
    overlap_turns: int,
    system_prompt: str,
    interlocutor: str,
    session_index: int,
    chat_type: str,
) -> List[DatasetSample]:
    """
    Segment consolidated session turns into training samples using a sliding window.
    - Every sample begins at either the session start or with 2-4 preceding context turns.
    - Every sample strictly starts with 'user' and ends with 'assistant'.
    - An even number of turns is maintained for all samples.
    """
    if not turns or len(turns) < 2:
        return []

    effective_max = max(2, max_turns - (max_turns % 2))
    overlap = max(0, overlap_turns - (overlap_turns % 2))
    if overlap >= effective_max:
        overlap = max(0, effective_max - 2)
    step = effective_max - overlap

    samples: List[DatasetSample] = []
    n = len(turns)
    i = 0

    while i < n:
        j = min(i + effective_max, n)
        chunk = turns[i:j]

        # Enforce valid dialogue boundaries (starts with user, ends with assistant)
        while chunk and chunk[0].role != "user":
            chunk = chunk[1:]
        while chunk and chunk[-1].role != "assistant":
            chunk = chunk[:-1]

        if len(chunk) >= 2:
            payload = [{"role": "system", "content": system_prompt}] + [
                {"role": t.role, "content": t.content} for t in chunk
            ]
            samples.append(
                DatasetSample(
                    messages=payload,
                    timestamp=chunk[-1].timestamp,
                    interlocutor=interlocutor,
                    session_index=session_index,
                    chat_type=chat_type,
                )
            )

        if j >= n:
            break
        i += step

    return samples


# ==============================================================================
# 7. System Prompt Loading & Construction
# ==============================================================================
def load_base_system_prompt(candidate_paths: List[str]) -> Tuple[str, List[str]]:
    """
    Read the base system prompt from candidate text files and return (prompt, warnings).
    Handles plain text files as well as JSON-string encoded files.
    """
    warnings: List[str] = []
    resolved_path: Optional[str] = None

    for path in candidate_paths:
        if os.path.exists(path):
            resolved_path = path
            break

    if not resolved_path:
        msg = f"[Warning] Base system prompt file not found in candidates: {candidate_paths}. Using empty prompt."
        print(msg)
        warnings.append(msg)
        return "", warnings

    try:
        with open(resolved_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # Handle string files that might be JSON-quoted with escaped newlines
        if (content.startswith('"') and content.endswith('"')) or (
            content.startswith("'") and content.endswith("'")
        ):
            try:
                content = json.loads(content)
            except Exception:
                content = content[1:-1].replace(r"\n", "\n")
        elif r"\n" in content and "\n" not in content:
            content = content.replace(r"\n", "\n")

        print(f"[Info] Loaded base system prompt from: {resolved_path} ({len(content)} chars)")
        return content, warnings

    except Exception as e:
        msg = f"[Warning] Failed to read system prompt from {resolved_path}: {e}"
        print(msg)
        warnings.append(msg)
        return "", warnings


def build_system_prompt(base_prompt: str, chat_type: str, description: str) -> str:
    """
    Construct the final system prompt with dynamic interlocutor / group injection.
    - Personal: appends 'Собеседник в этом диалоге: {description}.'
    - Group: appends 'Ты сейчас в групе с: {description}. Сообщения участников подписаны в формате 'Имя: текст'. Отвечай от своего лица без префикса.'
    """
    if chat_type == "group":
        directive = GROUP_PROMPT_TEMPLATE.format(description=description)
    else:
        directive = PERSONAL_PROMPT_TEMPLATE.format(description=description)

    if base_prompt:
        return f"{base_prompt}\n\n{directive}"
    return directive


# ==============================================================================
# 8. Chat File Resolution & Strict Isolated Processing
# ==============================================================================
def resolve_chat_file_path(filename: str, base_dirs: List[str]) -> Optional[str]:
    """Find the existing path for a given chat filename."""
    if os.path.exists(filename):
        return filename
    for bdir in base_dirs:
        candidate = os.path.join(bdir, filename)
        if os.path.exists(candidate):
            return candidate
        candidate = os.path.join(bdir, os.path.basename(filename))
        if os.path.exists(candidate):
            return candidate
    return None


def process_single_chat_file(
    filename: str,
    chat_config: Dict[str, str],
    base_system_prompt: str,
    base_dirs: List[str],
    assistant_from_id: Optional[str] = None,
    user_id_map: Optional[Dict[str, str]] = None,
    session_idle_timeout_minutes: int = SESSION_IDLE_TIMEOUT_MINUTES,
    max_turns_per_sample: int = MAX_TURNS_PER_SAMPLE,
    chunk_overlap_turns: int = CHUNK_OVERLAP_TURNS,
) -> Tuple[List[DatasetSample], ChatStats, Dict[str, str], List[str]]:
    """
    Process one Telegram JSON export file in complete isolation:
    1. Parse raw messages and extract content with media placeholders.
    2. Check sender roles:
       - Assistant is always role='assistant' and has no prefix.
       - In personal chats: user messages are role='user' without prefix.
       - In group chats: user IDs are resolved against user_id_map to produce '[Имя Фамилия]: text'.
         Unmapped IDs are excluded and tracked for terminal warnings.
    3. Sort messages strictly within this chat file by timestamp.
    4. Segment dialogue sessions using session_idle_timeout_minutes.
    5. Consolidate consecutive messages from the same role.
    6. Enforce Golden Rules for Assistant:
       - Rule 1: Strip naked media placeholders when assistant turn has real text.
       - Rule 2: Drop pair and cut session when assistant turn has ONLY naked media / voice.
    7. Prune useless orphan media reactions and VoIP coordination sessions.
    8. Chunk sessions using sliding window with overlap.
    9. Attach friend-specific / group-specific context to system prompt.
    """
    eff_assistant_id = assistant_from_id or ASSISTANT_FROM_ID
    eff_user_map = user_id_map if user_id_map is not None else USER_ID_MAP
    chat_type = chat_config.get("type", "personal")
    interlocutor_desc = chat_config.get("description", "")

    stats = ChatStats(filename=filename, chat_type=chat_type, friend_info=interlocutor_desc)
    warnings: List[str] = []
    unmapped_senders: Dict[str, str] = {}

    resolved_path = resolve_chat_file_path(filename, base_dirs)
    if not resolved_path:
        msg = f"[Warning] Chat file not found: {filename} in {base_dirs}"
        print(msg)
        warnings.append(msg)
        return [], stats, unmapped_senders, warnings

    try:
        with open(resolved_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        msg = f"[Warning] Failed to load JSON from {resolved_path}: {e}"
        print(msg)
        warnings.append(msg)
        return [], stats, unmapped_senders, warnings

    raw_messages = data.get("messages", []) if isinstance(data, dict) else data
    if not isinstance(raw_messages, list) or len(raw_messages) == 0:
        msg = f"[Warning] File contains no messages: {resolved_path}"
        print(msg)
        warnings.append(msg)
        return [], stats, unmapped_senders, warnings

    stats.total_raw_entries = len(raw_messages)
    extracted_msgs: List[CleanMessage] = []

    for entry in raw_messages:
        if not isinstance(entry, dict) or entry.get("type") != "message":
            continue

        fid = entry.get("from_id")
        date_unix = entry.get("date_unixtime")
        if not fid or date_unix is None:
            continue

        try:
            timestamp = int(date_unix)
        except (ValueError, TypeError):
            continue

        content, is_media = extract_message_content(entry)
        if not content:
            continue

        # Check if sender is the assistant
        if is_assistant_sender(fid, eff_assistant_id):
            # Assistant turns are always clean, without speaker prefix
            role = "assistant"
            text = content
        else:
            role = "user"
            if chat_type == "group":
                norm_fid = normalize_user_id(fid)
                if norm_fid in eff_user_map:
                    sender_name = eff_user_map[norm_fid]
                    # Format user messages with [Имя Фамилия]: prefix in group mode
                    text = f"[{sender_name}]: {content}"
                else:
                    # Unmapped user in group chat: exclude message and record warning info
                    raw_from = entry.get("from") or str(fid)
                    unmapped_senders[norm_fid] = raw_from
                    stats.unmapped_ids_count += 1
                    continue
            else:
                # Personal chat: keep clean text without prefix
                text = content

        if is_media:
            stats.media_placeholders_created += 1

        extracted_msgs.append(
            CleanMessage(
                from_id=str(fid),
                timestamp=timestamp,
                text=text,
                role=role,
            )
        )

    stats.valid_extracted_messages = len(extracted_msgs)
    if stats.valid_extracted_messages == 0:
        msg = f"[Warning] 0 valid messages extracted from: {resolved_path}"
        print(msg)
        warnings.append(msg)
        return [], stats, unmapped_senders, warnings

    # Sort strictly within this chat file
    extracted_msgs.sort(key=lambda m: m.timestamp)

    # 1.1 Strict Session Segmentation
    timeout_sec = int(session_idle_timeout_minutes) * 60
    raw_sessions: List[List[CleanMessage]] = []
    current_session: List[CleanMessage] = [extracted_msgs[0]]

    for msg in extracted_msgs[1:]:
        if (msg.timestamp - current_session[-1].timestamp) > timeout_sec:
            raw_sessions.append(current_session)
            current_session = [msg]
        else:
            current_session.append(msg)

    if current_session:
        raw_sessions.append(current_session)

    stats.raw_sessions_count = len(raw_sessions)

    # Turn Consolidation, Hygiene Filtering & Pruning
    valid_sessions: List[List[Turn]] = []
    for session in raw_sessions:
        turns: List[Turn] = []
        for msg in session:
            if turns and turns[-1].role == msg.role:
                turns[-1].content = f"{turns[-1].content}\n{msg.text}"
                turns[-1].timestamp = msg.timestamp
                stats.messages_joined_count += 1
            else:
                turns.append(Turn(role=msg.role, content=msg.text, timestamp=msg.timestamp))

        # Enforce conversation starts with user and ends with assistant
        while turns and turns[0].role != "user":
            turns.pop(0)
        while turns and turns[-1].role != "assistant":
            turns.pop()

        if not turns or len(turns) < 2:
            continue

        # Golden Rules for Assistant:
        # Rule 1: Strip naked media when assistant has text
        # Rule 2: Drop pair and split session when assistant has ONLY naked media / voice
        sub_sessions, stripped_cnt, dropped_pairs_cnt = split_session_on_empty_assistant(turns)
        stats.assistant_media_stripped += stripped_cnt
        stats.empty_assistant_pairs_dropped += dropped_pairs_cnt

        for sub_session in sub_sessions:
            if len(sub_session) < 2:
                continue

            # Step 2: Prune useless orphan media reactions
            sub_session, pruned_count = prune_useless_media_exchanges(sub_session)
            stats.media_pairs_pruned += pruned_count
            if not sub_session or len(sub_session) < 2:
                continue

            # Part 2: Session-level VoIP meta-talk filtering
            if is_voice_garbage_session(sub_session):
                stats.voice_sessions_dropped += 1
                continue

            valid_sessions.append(sub_session)

    stats.active_sessions_count = len(valid_sessions)

    # Build interlocutor / group system prompt
    sample_system_prompt = build_system_prompt(base_system_prompt, chat_type, interlocutor_desc)

    # Step 1.5: Chunking with Overlap
    samples: List[DatasetSample] = []
    for s_idx, s_turns in enumerate(valid_sessions):
        chunk_samples = chunk_turns_into_samples(
            turns=s_turns,
            max_turns=int(max_turns_per_sample),
            overlap_turns=int(chunk_overlap_turns),
            system_prompt=sample_system_prompt,
            interlocutor=interlocutor_desc,
            session_index=s_idx,
            chat_type=chat_type,
        )
        samples.extend(chunk_samples)

    stats.samples_generated = len(samples)
    return samples, stats, unmapped_senders, warnings


# ==============================================================================
# 9. Temporal Weighting & Epoch Categorization
# ==============================================================================
def get_epoch_label(timestamp: int) -> str:
    """Categorize a UNIX timestamp into its temporal weighting bucket."""
    if timestamp < EPOCH_2022_START:
        return "pre_2022"
    elif timestamp < EPOCH_2024_START:
        return "2022_2024"
    else:
        return "post_2024"


def apply_temporal_weighting(
    samples: List[DatasetSample],
    weight_pre_2022: int = WEIGHT_PRE_2022,
    weight_2022_2024: int = WEIGHT_2022_2024,
    weight_post_2024: int = WEIGHT_POST_2024,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Apply epoch replication weights to samples while strictly preserving
    the chronological (monotonic non-decreasing timestamp) order.
    """
    weighted_records: List[Dict[str, Any]] = []
    epoch_counts: Dict[str, int] = {
        "pre_2022": 0,
        "2022_2024": 0,
        "post_2024": 0,
    }

    for sample in samples:
        epoch = get_epoch_label(sample.timestamp)
        record = {"messages": sample.messages}

        multiplier = int(weight_pre_2022)
        if epoch == "2022_2024":
            multiplier = int(weight_2022_2024)
        elif epoch == "post_2024":
            multiplier = int(weight_post_2024)

        if multiplier <= 0:
            continue

        for _ in range(multiplier):
            weighted_records.append(record)
            epoch_counts[epoch] += 1

    return weighted_records, epoch_counts


# ==============================================================================
# 10. Main Pipeline Execution (Combined Personal + Group)
# ==============================================================================
def run_build(
    config_path: Optional[str] = None,
    persona_id: str = "friend_1",
    raw_dir: Optional[str] = None,
    prompt_file: Optional[str] = None,
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    cfg, resolved_cfg_path = load_builder_config(config_path=config_path, persona_id=persona_id)
    if resolved_cfg_path:
        print(f"[Info] Loaded builder config: {resolved_cfg_path}")

    assistant_from_id = str(cfg.get("assistant_from_id") or ASSISTANT_FROM_ID)
    user_id_map = dict(cfg.get("user_id_map") or USER_ID_MAP)
    chats_config = dict(cfg.get("chats_config") or CHATS_CONFIG)
    eff_output_file = output_file or str(cfg.get("output_file") or OUTPUT_FILE)
    eff_raw_dir = raw_dir or str(cfg.get("raw_chats_dir") or RAW_CHATS_DIR)
    session_timeout = int(cfg.get("session_idle_timeout_minutes", SESSION_IDLE_TIMEOUT_MINUTES))
    max_turns = int(cfg.get("max_turns_per_sample", MAX_TURNS_PER_SAMPLE))
    overlap_turns = int(cfg.get("chunk_overlap_turns", CHUNK_OVERLAP_TURNS))
    w_pre = int(cfg.get("weight_pre_2022", WEIGHT_PRE_2022))
    w_mid = int(cfg.get("weight_2022_2024", WEIGHT_2022_2024))
    w_post = int(cfg.get("weight_post_2024", WEIGHT_POST_2024))

    prompt_candidates = list(cfg.get("system_prompt_candidates") or SYSTEM_PROMPT_CANDIDATES)
    if prompt_file:
        prompt_candidates = [prompt_file, *prompt_candidates]

    all_warnings: List[str] = []

    # 1. Load Base System Prompt
    base_system_prompt, prompt_warnings = load_base_system_prompt(prompt_candidates)
    all_warnings.extend(prompt_warnings)

    base_dirs = [eff_raw_dir, os.path.join(eff_raw_dir, persona_id), "./raw_chats_and_prompt", "."]

    # 2. Strict Chat Isolation: Process each chat file independently
    all_samples: List[DatasetSample] = []
    chat_statistics: List[ChatStats] = []
    all_unmapped_senders: Dict[str, str] = {}

    print("-" * 75)
    print(f"Processing Telegram chat files for '{persona_id}' in strict isolation...")
    print("-" * 75)

    for filename, chat_cfg in chats_config.items():
        file_samples, stats, unmapped_file, file_warnings = process_single_chat_file(
            filename=filename,
            chat_config=chat_cfg,
            base_system_prompt=base_system_prompt,
            base_dirs=base_dirs,
            assistant_from_id=assistant_from_id,
            user_id_map=user_id_map,
            session_idle_timeout_minutes=session_timeout,
            max_turns_per_sample=max_turns,
            chunk_overlap_turns=overlap_turns,
        )
        all_samples.extend(file_samples)
        chat_statistics.append(stats)
        all_unmapped_senders.update(unmapped_file)
        all_warnings.extend(file_warnings)

        ctype_tag = f"[{stats.chat_type}]"
        print(
            f"[*] {filename:42} {ctype_tag:10} | Raw: {stats.total_raw_entries:6} | Extracted: {stats.valid_extracted_messages:6} | "
            f"Sessions: {stats.active_sessions_count:4} | Samples: {stats.samples_generated:4}"
        )

    # 3. Global Chronological Sorting: Earliest first
    all_samples.sort(key=lambda s: s.timestamp)

    # 4. Temporal Weighting
    final_records, epoch_counts = apply_temporal_weighting(
        all_samples,
        weight_pre_2022=w_pre,
        weight_2022_2024=w_mid,
        weight_post_2024=w_post,
    )

    # 5. Write to JSONL
    output_dir = os.path.dirname(os.path.abspath(eff_output_file))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(eff_output_file, "w", encoding="utf-8") as out_f:
        for record in final_records:
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 6. Unmapped IDs Warning (if any unmapped accounts were found in group chats)
    if all_unmapped_senders:
        print("=" * 75)
        print("IDs were not mapped. Their names in the logs are:")
        for uid, uname in sorted(all_unmapped_senders.items()):
            print(f"  {uid}: {uname}")
        print("=" * 75)

    # 7. Detailed Summary Report
    print("=" * 75)
    print("Telegram Parser & Dataset Builder - Summary Report")
    print("=" * 75)
    total_raw = sum(s.total_raw_entries for s in chat_statistics)
    total_valid = sum(s.valid_extracted_messages for s in chat_statistics)
    total_media_placeholders = sum(s.media_placeholders_created for s in chat_statistics)
    total_joined = sum(s.messages_joined_count for s in chat_statistics)
    total_voice_dropped = sum(s.voice_sessions_dropped for s in chat_statistics)
    total_media_pruned = sum(s.media_pairs_pruned for s in chat_statistics)
    total_assistant_stripped = sum(s.assistant_media_stripped for s in chat_statistics)
    total_empty_pairs_dropped = sum(s.empty_assistant_pairs_dropped for s in chat_statistics)
    total_active_sessions = sum(s.active_sessions_count for s in chat_statistics)
    total_preweight_samples = len(all_samples)

    personal_samples = sum(s.samples_generated for s in chat_statistics if s.chat_type == "personal")
    group_samples = sum(s.samples_generated for s in chat_statistics if s.chat_type == "group")

    print(f"Total raw entries read                 : {total_raw}")
    print(f"Valid extracted messages               : {total_valid}")
    print(f"Media placeholders generated           : {total_media_placeholders}")
    print(f"Messages joined into turns             : {total_joined}")
    print(f"Orphan media reaction pairs pruned     : {total_media_pruned}")
    print(f"VoIP / Discord noise sessions dropped  : {total_voice_dropped}")
    print(f"Assistant media placeholders stripped  : {total_assistant_stripped} (Rule 1: voice/media + text kept)")
    print(f"Empty assistant pairs dropped          : {total_empty_pairs_dropped} (Rule 2: pure voice/media cut)")
    print(f"Total active dialogue sessions         : {total_active_sessions}")
    print(f"Total pre-weighting samples            : {total_preweight_samples}")
    print(f"  - Personal (1-on-1) samples          : {personal_samples}")
    print(f"  - Group chat samples                 : {group_samples}")
    print("-" * 75)
    print("Chat Isolation Breakdown:")
    for s in chat_statistics:
        print(f"  • {s.filename} [{s.chat_type}]:")
        print(f"      Context     : {s.friend_info}")
        print(f"      Extracted   : {s.valid_extracted_messages} msgs (Media placeholders: {s.media_placeholders_created})")
        print(f"      Sessions    : {s.active_sessions_count} active (Voice dropped: {s.voice_sessions_dropped}, Media pruned: {s.media_pairs_pruned})")
        print(f"      Sanitized   : {s.assistant_media_stripped} assistant turns stripped of naked media, {s.empty_assistant_pairs_dropped} empty assistant pairs dropped")
        print(f"      Samples     : {s.samples_generated}")
        if s.unmapped_ids_count > 0:
            print(f"      Unmapped    : {s.unmapped_ids_count} messages skipped from unmapped IDs")
    print("-" * 75)
    print("Samples generated per time epoch (post-weighting):")
    total_weighted = len(final_records)
    pct_pre = (epoch_counts["pre_2022"] / total_weighted * 100) if total_weighted else 0
    pct_mid = (epoch_counts["2022_2024"] / total_weighted * 100) if total_weighted else 0
    pct_post = (epoch_counts["post_2024"] / total_weighted * 100) if total_weighted else 0
    print(f"  Pre-2022 (< 2022-01-01, {w_pre}x)        : {epoch_counts['pre_2022']:5} ({pct_pre:5.1f}%)")
    print(f"  2022-2024 ({w_mid}x)                     : {epoch_counts['2022_2024']:5} ({pct_mid:5.1f}%)")
    print(f"  After 2024 (>= 2024-01-01, {w_post}x)     : {epoch_counts['post_2024']:5} ({pct_post:5.1f}%)")
    print("-" * 75)
    print(f"Total JSONL records written            : {total_weighted}")
    print("Chronologically sorted                 : YES (earliest first)")
    print(f"Output saved to                        : {eff_output_file}")
    if all_warnings:
        print("-" * 75)
        print(f"WARNINGS ({len(all_warnings)}):")
        for w in all_warnings:
            print(f"  {w}")
    print("=" * 75)
    return {
        "output_file": eff_output_file,
        "records": total_weighted,
        "epoch_counts": epoch_counts,
        "stats": chat_statistics,
    }


def run_self_test() -> None:
    print("====================================================================")
    print(" [*] GVK Private AI — Self-Test (tools/tg_to_jsonl_dataset_builder.py)")
    print("====================================================================")

    with tempfile.TemporaryDirectory(prefix="gvk_tg_builder_test_") as tmp_dir:
        prompt_path = os.path.join(tmp_dir, "prompt.txt")
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write("Ти — friend_1. Відповідай коротко.")

        personal_chat = {
            "messages": [
                {"type": "message", "from_id": "user002", "date_unixtime": "1710000000", "text": "привіт як справи"},
                {"type": "message", "from_id": "user001", "date_unixtime": "1710000010", "photo": "a.jpg", "text": "норм сиджу"},
                # Rule 2: black hole (assistant only photo) -> splits session cleanly
                {"type": "message", "from_id": "user002", "date_unixtime": "1710000020", "text": "шо робиш"},
                {"type": "message", "from_id": "user001", "date_unixtime": "1710000030", "photo": "b.jpg", "text": ""},
                {"type": "message", "from_id": "user002", "date_unixtime": "1710000040", "text": "го в катку"},
                {"type": "message", "from_id": "user001", "date_unixtime": "1710000050", "text": "залітаю"},
                # VoIP garbage session after 1 hour gap -> pruned
                {"type": "message", "from_id": "user002", "date_unixtime": "1710005000", "text": "го в дс"},
                {"type": "message", "from_id": "user001", "date_unixtime": "1710005010", "text": "не чую микро включи"},
            ]
        }
        group_chat = {
            "messages": [
                {"type": "message", "from_id": "user002", "date_unixtime": "1680000000", "text": "всім ку"},
                {"type": "message", "from_id": "user999", "from": "Stranger", "date_unixtime": "1680000005", "text": "спам"},
                {"type": "message", "from_id": "user001", "date_unixtime": "1680000010", "text": "дарова пацани"},
            ]
        }

        p_file = os.path.join(tmp_dir, "personal.json")
        g_file = os.path.join(tmp_dir, "group.json")
        with open(p_file, "w", encoding="utf-8") as f:
            json.dump(personal_chat, f)
        with open(g_file, "w", encoding="utf-8") as f:
            json.dump(group_chat, f)

        out_jsonl = os.path.join(tmp_dir, "out.jsonl")
        cfg_path = os.path.join(tmp_dir, "builder_config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "persona_id": "friend_1",
                    "assistant_from_id": "user001",
                    "raw_chats_dir": tmp_dir,
                    "output_file": out_jsonl,
                    "weight_pre_2022": 0,
                    "weight_2022_2024": 1,
                    "weight_post_2024": 2,
                    "system_prompt_candidates": [prompt_path],
                    "user_id_map": {"user002": "Friend 2", "user003": "Friend 3"},
                    "chats_config": {
                        "personal.json": {"type": "personal", "description": "Friend 2"},
                        "group.json": {"type": "group", "description": "Friend 2, Friend 3"},
                    },
                },
                f,
            )

        res = run_build(config_path=cfg_path, persona_id="friend_1")
        with open(out_jsonl, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        # Group (2023, weight 1 -> 1 sample) + 2 personal sub-sessions (2024, weight 2 -> 4 samples) = 5
        assert len(lines) == 5, f"Очікувалось 5 ChatML-записів, отримано {len(lines)}"
        assert lines[0]["messages"][1]["content"].startswith("[Friend 2]: "), lines[0]
        assert "норм сиджу" in lines[1]["messages"][2]["content"] and "[Фотография]" not in lines[1]["messages"][2]["content"]
        assert res["stats"][0].empty_assistant_pairs_dropped == 1
        assert res["stats"][0].assistant_media_stripped == 1
        assert res["stats"][0].voice_sessions_dropped == 1
        assert res["stats"][1].unmapped_ids_count == 1
        print(" [✓] Golden Rules 1 & 2, VoIP/Group фільтрація, конфіг і часове зважування OK!")
    print("====================================================================")
    print(" [✓] SELF-TEST DATASET BUILDER ПРОЙДЕНО УСПІШНО!")
    print("====================================================================")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GVK Private AI — Генератор ChatML-датасету з експортів Telegram"
    )
    parser.add_argument(
        "cmd",
        nargs="?",
        choices=["build", "self-test"],
        default="build",
        help="Режим: 'build' (за замовчуванням) або 'self-test'",
    )
    parser.add_argument("-c", "--config", default=None, help="Шлях до JSON-конфіга персони")
    parser.add_argument("-p", "--persona", default="friend_1", help="ID персони (friend_1, friend_2, friend_3)")
    parser.add_argument("--raw-dir", default=None, help="Папка з сирими експортами Telegram")
    parser.add_argument("--prompt-file", default=None, help="Шлях до системного промпту (.txt)")
    parser.add_argument("-o", "--output", default=None, help="Вихідний шлях до .jsonl датасету")
    args = parser.parse_args()

    if args.cmd == "self-test":
        run_self_test()
        return

    run_build(
        config_path=args.config,
        persona_id=args.persona,
        raw_dir=args.raw_dir,
        prompt_file=args.prompt_file,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
