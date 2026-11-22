#!/usr/bin/env python3
"""
GVK Private AI — Векторна пам'ять RAG (core/memory_rag.py)

Забезпечує спільний конвеєр ембедінгів, парсинг спогадів, збірку індексу
`memory.faiss` та швидкий гібридний (семантичний + лексичний) пошук спогадів
під час сесії оркестратора `core/orchestrator.py`.

Особливості:
  - Основний бекенд: `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`, 384 dim).
  - Автоматичний фолбек: детермінований 384-мірний N-gram / морфологічний feature-hashing
    енкодер, якщо `sentence-transformers` недоступний або середовище працює офлайн.
  - Підтримує як нативний індекс FAISS (`faiss.IndexFlatIP`), так і самодостатній
    бінарний контейнер `GVKRAG01` із вбудованим мета-трейлером `GVKMETA1`.
  - Безпосередньо приймає директорію персонажа (розпаковану з `payload.tar.enc` у RAM)
    або прямий шлях до файлу `memory.faiss`.

Приклади запуску:
  python3 core/memory_rag.py self-test
  python3 core/memory_rag.py query -i raw/friend_1_data -q "куди ми їздили минулого літа?"
"""

import argparse
import hashlib
import json
import math
import os
import re
import struct
import sys
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Запобігає конфліктам OpenMP/MPS між faiss та PyTorch на macOS ARM64
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# =====================================================================
# КОНСТАНТИ ТА НАЛАШТУВАННЯ
# =====================================================================
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_DIMENSION = 384
DEFAULT_SIMILARITY_THRESHOLD = 0.25
DEFAULT_ST_SIMILARITY_THRESHOLD = 0.72

_WHO_QUERY_RE = re.compile(
    r"\b(?:кто\s+так(?:ой|ая|ие)|хто\s+так(?:ий|а|і)|расскажи\s+про|розкажи\s+про|шо\s+за|что\s+за)\b",
    re.IGNORECASE,
)
_DOSSIER_CAT_PREFIXES: Tuple[str, ...] = ("друзья", "семья", "девушки", "личность")

RAG_MAGIC = b"GVKRAG01"
META_TRAILER_MAGIC = b"GVKMETA1"
RAG_FORMAT_VERSION = 1

_MODEL_CACHE: Dict[str, Any] = {}

_WORD_RE = re.compile(r"[a-zа-яіїєґ0-9]+")
_BULLET_RE = re.compile(r"^\s*(?:[-*•]+|\d+[.)])\s+")
_HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$")
_DIVIDER_RE = re.compile(r"^\s*[-*_=]{3,}\s*$")
_MEMORY_BLOCK_RE = re.compile(r"^\s*\[(MEMORY[_\-\s]*\d+|[A-ZА-ЯІЇЄҐ0-9_]{3,})\]\s*$", re.IGNORECASE)
_CATEGORY_LINE_RE = re.compile(r"^\s*(?:категория|категорія|category)\s*:\s*(.+?)\s*$", re.IGNORECASE)
_TAGS_LINE_RE = re.compile(r"^\s*(?:теги|тегі|ключевые слова|ключові слова|tags)\s*:\s*(.+?)\s*$", re.IGNORECASE)
_CONTENT_HEADER_RE = re.compile(r"^\s*(?:содержание|зміст|контент|content)\s*:\s*(.*?)\s*$", re.IGNORECASE)

_STOPWORDS = frozenset(
    {
        # Українські службові слова, займенники, вигуки
        "та", "це", "то", "те", "ті", "той", "така", "таке", "такі",
        "як", "що", "шо", "де", "коли", "куди", "звідки", "чому", "хто", "чий",
        "кого", "кому", "ким", "чого", "чим", "який", "яка", "яке", "які",
        "чи", "не", "ні", "так", "але", "або", "бо", "якщо", "щоб", "хоч", "хоча",
        "від", "до", "на", "по", "за", "із", "зі", "під", "над", "при", "про", "без", "для",
        "через", "після", "перед", "між", "біля", "коло", "серед", "проти",
        "ми", "ви", "він", "вона", "воно", "вони", "ти", "мене", "мені", "мною", "тебе", "тобі",
        "його", "її", "їх", "нам", "вам", "собі", "себе", "мій", "моя", "моє", "мої",
        "твій", "твоя", "твоє", "твої", "свій", "своя", "свої", "наш", "наша", "наші", "ваш",
        "вже", "ще", "там", "тут", "ось", "от", "ну", "же", "ж", "би", "б", "тобто", "просто", "дуже",
        "було", "бути", "був", "була", "були", "буде", "є", "має", "може",
        "привіт", "хай", "здарова", "справи", "норм", "нормально", "окей", "ага", "угу",
        "робиш", "робити", "сказав", "каже", "типу", "тіпа", "взагалі", "ваще", "розкажи", "скажи",
        # Російські службові слова, займенники, вигуки
        "и", "в", "во", "со", "ко", "об", "обо", "не", "ни", "что", "че", "чё", "как", "это", "то",
        "он", "она", "оно", "они", "мы", "вы", "ты", "я", "меня", "мне", "мной", "тебя", "тебе",
        "его", "ее", "её", "их", "нам", "вам", "себя", "себе", "мой", "моя", "мои", "твой", "твоя",
        "свой", "своя", "свои", "наш", "наша", "наши", "ваш", "все", "всё", "вся", "весь", "всех",
        "по", "из", "за", "от", "до", "при", "про", "над", "под", "для", "или", "но", "да", "нет",
        "же", "ли", "бы", "уж", "было", "быть", "был", "была", "были", "будет", "есть", "может",
        "уже", "еще", "ещё", "вот", "тут", "там", "где", "когда", "кто", "куда", "почему", "зачем",
        "кого", "кому", "кем", "чего", "чему", "чем", "какой", "какая", "какие", "какое",
        "очень", "просто", "вообще", "короче", "кароче", "типо", "привет", "дела", "хорошо",
        "ладно", "пока", "кент", "бро", "чел", "расскажи", "скажи", "знаешь", "помнишь",
    }
)

# Словник крос-мовних синонімів (UA <-> RU), де корені відрізняються більше ніж на і/и/е/є
_CANONICAL_STEM_ALIASES: Dict[str, str] = {
    "киив": "киев",
    "киев": "киев",
    "львив": "львов",
    "львов": "львов",
    "зир": "зерк",
    "дзерк": "зерк",
    "зерк": "зерк",
    "рик": "год",
    "рок": "год",
    "вчиш": "уч",
    "вчит": "уч",
    "вчув": "уч",
    "навч": "уч",
    "учус": "уч",
    "учит": "уч",
    "учил": "уч",
    "учеб": "уч",
    "изуч": "уч",
    "витч": "отчим",
    "отчим": "отчим",
    "видра": "отчисл",
    "вигнал": "отчисл",
    "отчис": "отчисл",
    "дивч": "девуш",
    "девуш": "девуш",
    "вирш": "стих",
    "стих": "стих",
    "готув": "готов",
    "готов": "готов",
    "машин": "авто",
    "водин": "авто",
    "водит": "авто",
    "воджу": "авто",
    "працю": "работ",
    "работ": "работ",
    "гуртож": "общеж",
    "общеж": "общеж",
    "мист": "город",
    "город": "город",
    "кент": "друг",
    "друз": "друг",
    "друг": "друг",
    "мешк": "жив",
    "изди": "езд",
    "поизд": "езд",
    "езд": "езд",
    "поезд": "езд",
    "путеш": "езд",
    "трип": "езд",
    "гонял": "езд",
    "зимо": "зим",
    "взим": "зим",
    "лито": "лет",
    "литн": "лет",
    "влит": "лет",
    "лето": "лет",
    "летн": "лет",
}

# Exact phonetic stems (for short words that could collide as prefixes)
_EXACT_STEM_ALIASES: Dict[str, str] = {
    "иж": "еда_food",
    "иш": "еда_food",
    "ист": "еда_food",
    "исти": "еда_food",
    "ест": "еда_food",
    "еда": "еда_food",
    "еду": "еда_food",
    "хава": "еда_food",
    "куша": "еда_food",
    "20": "2020",
    "21": "2021",
    "22": "2022",
    "23": "2023",
    "24": "2024",
    "25": "2025",
    "26": "2026",
}

# Синонімічне розширення запитів для пошуку UA <-> RU без втрати точності
_QUERY_EXPANSIONS: Dict[str, Tuple[str, ...]] = {
    "жив": ("перее", "посел", "общеж"),
    "уч": ("посту", "униве", "специ", "факул", "изуча", "лице", "школ"),
    "отчисл": ("лице", "вигон"),
    "еда_food": ("хава", "готов", "яичн", "макар", "куша"),
    "авто": ("прав", "вожде", "машин", "води", "рул"),
    "зим": ("январ", "февра", "декаб", "лыж"),
    "лет": ("июн", "июл", "авгус", "лагер"),
    "езд": ("путеш", "поезд", "трип", "гонял"),
}


def register_persona_lexicon(lexicon: Optional[Dict[str, Any]]) -> None:
    """
    Підключає приватні синоніми/аліаси з `persona_config.json` (`rag_lexicon`) у RAM
    під час сесії, щоб персональні імена й локації не зберігалися в публічному коді.
    """
    global _DOSSIER_CAT_PREFIXES
    if not isinstance(lexicon, dict):
        return

    extra_cats = lexicon.get("dossier_cat_prefixes")
    if isinstance(extra_cats, list):
        merged = list(_DOSSIER_CAT_PREFIXES)
        for c in extra_cats:
            c_low = str(c).strip().lower()
            if c_low and c_low not in merged:
                merged.append(c_low)
        _DOSSIER_CAT_PREFIXES = tuple(merged)

    extra_aliases = lexicon.get("canonical_stem_aliases")
    if isinstance(extra_aliases, dict):
        for k, v in extra_aliases.items():
            k_norm = str(k).strip().lower()
            v_norm = str(v).strip().lower()
            if k_norm and v_norm:
                _CANONICAL_STEM_ALIASES[k_norm] = v_norm

    extra_exp = lexicon.get("query_expansions")
    if isinstance(extra_exp, dict):
        for k, vals in extra_exp.items():
            k_norm = str(k).strip().lower()
            if not k_norm or not isinstance(vals, (list, tuple)):
                continue
            cur = list(_QUERY_EXPANSIONS.get(k_norm, ()))
            for v in vals:
                v_norm = str(v).strip().lower()
                if v_norm and v_norm not in cur:
                    cur.append(v_norm)
            _QUERY_EXPANSIONS[k_norm] = tuple(cur)


# =====================================================================
# ТЕКСТОВА НОРМАЛІЗАЦІЯ ТА КЛЮЧОВІ СЛОВА
# =====================================================================
def normalize_text(text: str) -> str:
    """Приводить текст до нижнього регістру та уніфікує варіативні літери."""
    lowered = text.lower().strip()
    return lowered.replace("ё", "е").replace("ґ", "г")


def normalize_slavic_phonetic(text: str) -> str:
    """
    Крос-мовна фонетична нормалізація (UA / RU / суржик):
    уніфікує і/ї/ы -> и, є/ё -> е, ґ -> г та прибирає м'який/твердий знаки всередині слів.
    Дозволяє українським запитам («Київ», «ліцей», «фізика», «рибалка»)
    точно знаходити російськомовні записи («Киев», «лицей», «физика», «рыбалка»).
    """
    s = normalize_text(text)
    s = s.replace("і", "и").replace("ї", "и").replace("ы", "и").replace("є", "е")
    s = s.replace("ь", "").replace("ъ", "").replace("'", "").replace("’", "")
    return s


def _stem_token(token: str) -> str:
    """Морфологічний + крос-мовний стемер для узгодження слов'янських відмінків і UA/RU коренів."""
    phon = normalize_slavic_phonetic(token)
    if not phon:
        return token
    if phon in _EXACT_STEM_ALIASES:
        return _EXACT_STEM_ALIASES[phon]
    if phon.isdigit():
        return phon
    for prefix, canonical in _CANONICAL_STEM_ALIASES.items():
        if phon.startswith(prefix):
            return canonical
    if len(phon) >= 6:
        return phon[:5]
    if len(phon) == 5:
        return phon[:4]
    if len(phon) == 4:
        return phon[:3]
    return phon


def _expand_query_stems(q_stems: set) -> set:
    """Повертає множину розширених синонімічних основ для запиту."""
    extra = set()
    for st in q_stems:
        if st in _QUERY_EXPANSIONS:
            extra.update(_QUERY_EXPANSIONS[st])
    return extra - q_stems


def extract_keywords(text: str) -> List[str]:
    """Витягує впорядковані унікальні ключові слова (довжина >= 2, без стоп-слів)."""
    norm = normalize_text(text)
    seen = set()
    keywords: List[str] = []
    for tok in _WORD_RE.findall(norm):
        if len(tok) < 2 or tok in _STOPWORDS:
            continue
        if len(tok) == 2 and normalize_slavic_phonetic(tok) in _STOPWORDS:
            continue
        if tok not in seen:
            seen.add(tok)
            keywords.append(tok)
    return keywords


def _extract_stems(text: str) -> set:
    return {_stem_token(kw) for kw in extract_keywords(text)}


# =====================================================================
# ДВОРІВНЕВИЙ ЕНКОДЕР ЕМБЕДІНГІВ
# =====================================================================
def _l2_normalize_vec(vec: List[float]) -> List[float]:
    norm_sq = sum(v * v for v in vec)
    if norm_sq <= 0.0:
        return vec
    inv_norm = 1.0 / math.sqrt(norm_sq)
    return [v * inv_norm for v in vec]


def _encode_ngram_vector(text: str, dim: int = DEFAULT_DIMENSION) -> List[float]:
    """
    Детермінований L2-нормалізований вектор ознак (слова, префікси, біграми,
    символьні 3/4-грами) через знаковий feature hashing (BLAKE2b).
    Забезпечує надійне семантико-морфологічне узгодження офлайн без залежностей.
    """
    vec = [0.0] * dim
    norm = normalize_text(text)
    raw_tokens = _WORD_RE.findall(norm)
    tokens = [t for t in raw_tokens if len(t) >= 2 and t not in _STOPWORDS]
    if not tokens:
        tokens = [t for t in raw_tokens if len(t) >= 2]
    if not tokens:
        return vec

    features: List[Tuple[str, float]] = []
    for tok in tokens:
        phon = normalize_slavic_phonetic(tok) or tok
        stem = _stem_token(tok)
        features.append((f"w:{phon}", 2.2))
        features.append((f"st:{stem}", 2.0))
        if len(phon) >= 4 and not phon.isdigit():
            features.append((f"p3:{phon[:3]}", 1.3))
            features.append((f"p4:{phon[:4]}", 1.6))
        if len(phon) >= 5:
            features.append((f"p5:{phon[:5]}", 1.6))
        if len(phon) >= 6:
            features.append((f"p6:{phon[:6]}", 1.6))

        padded = f"<{phon}>"
        for n, weight in ((3, 0.55), (4, 0.7)):
            if len(padded) >= n:
                for i in range(len(padded) - n + 1):
                    features.append((f"c{n}:{padded[i:i + n]}", weight))

    for i in range(len(tokens) - 1):
        features.append((f"bg:{tokens[i]}_{tokens[i + 1]}", 1.25))
        features.append((f"sbg:{_stem_token(tokens[i])}_{_stem_token(tokens[i + 1])}", 1.1))

    for feat, weight in features:
        digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if (digest[4] & 1) == 0 else -1.0
        vec[bucket] += sign * weight

    return _l2_normalize_vec(vec)


def _encode_ngram_vectors(texts: Sequence[str], dim: int = DEFAULT_DIMENSION) -> List[List[float]]:
    return [_encode_ngram_vector(t, dim=dim) for t in texts]


class EmbeddingEngine:
    """
    Уніфікований енкодер ембедінгів:
      - `sentence-transformers`: трансформерні ембедінги (кешуються у пам'яті).
      - `ngram`: швидкий детермінований фолбек без зовнішніх моделей.
      - `auto`: використовує `sentence-transformers`, а за їх відсутності чи офлайн-помилки
        автоматично перемикається на `ngram`.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        backend: str = "auto",
        dimension: int = DEFAULT_DIMENSION,
    ) -> None:
        if backend not in ("auto", "sentence-transformers", "ngram"):
            raise ValueError(f"Невідомий бекенд ембедінгів: {backend}")

        self.requested_backend = backend
        self.model_name = model_name
        self.dimension = dimension
        self.active_backend = "ngram"
        self._st_model = None

        if backend in ("auto", "sentence-transformers"):
            try:
                self._st_model = self._load_st_model(model_name)
                self.active_backend = "sentence-transformers"
                dim_fn = getattr(self._st_model, "get_embedding_dimension", None) or getattr(
                    self._st_model, "get_sentence_embedding_dimension", None
                )
                dim = dim_fn() if callable(dim_fn) else None
                if isinstance(dim, int) and dim > 0:
                    self.dimension = dim
            except Exception as exc:
                if backend == "sentence-transformers":
                    raise RuntimeError(
                        f"Не вдалося завантажити sentence-transformers ({model_name}): {exc}"
                    ) from exc
                self._st_model = None
                self.active_backend = "ngram"

    @staticmethod
    def _load_st_model(model_name: str) -> Any:
        if model_name in _MODEL_CACHE:
            return _MODEL_CACHE[model_name]
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(1)
        gvk_home = os.path.abspath(os.path.expanduser(os.environ.get("GVK_HOME") or "~/.gvk_ai"))
        short_name = model_name.split("/")[-1]
        local_emb_dir = os.path.join(gvk_home, "embeddings", short_name)
        load_target = (
            local_emb_dir
            if (os.path.isdir(local_emb_dir) and os.path.isfile(os.path.join(local_emb_dir, "config.json")))
            else model_name
        )
        try:
            model = SentenceTransformer(load_target, device="cpu", local_files_only=True)
        except Exception:
            model = SentenceTransformer(model_name, device="cpu")
        _MODEL_CACHE[model_name] = model
        return model

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        if self.active_backend == "sentence-transformers" and self._st_model is not None:
            raw = self._st_model.encode(
                list(texts),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return [[float(x) for x in row] for row in raw]
        return _encode_ngram_vectors(texts, dim=self.dimension)


# =====================================================================
# ПАРСИНГ СПОГАДІВ (.txt / .md / .json / .jsonl + [MEMORY_XX])
# =====================================================================
def _parse_tags_value(raw_tags: Any) -> List[str]:
    if not raw_tags:
        return []
    if isinstance(raw_tags, list):
        return [str(t).strip() for t in raw_tags if str(t).strip()]
    if isinstance(raw_tags, str):
        return [t.strip() for t in re.split(r"[;,]", raw_tags) if t.strip()]
    return []


def _parse_structured_memory_blocks(lines: Sequence[str]) -> List[Dict[str, Any]]:
    """
    Парсить деталізований формат спогадів із блоками `[MEMORY_XX]`,
    полями `Категория:`, `Теги:` та багаторядковим блоком `Содержание:`.
    """
    blocks: List[Dict[str, Any]] = []
    current_section = "general"
    current_block: Optional[Dict[str, Any]] = None

    def _flush() -> None:
        nonlocal current_block
        if current_block is None:
            return
        body = " ".join(current_block["body_lines"]).strip()
        if body:
            blocks.append(
                {
                    "memory_id": current_block["memory_id"],
                    "category": current_block["category"],
                    "tags": current_block["tags"],
                    "text": body,
                }
            )
        current_block = None

    for raw_line in lines:
        if _DIVIDER_RE.match(raw_line):
            continue

        block_match = _MEMORY_BLOCK_RE.match(raw_line)
        if block_match:
            _flush()
            current_block = {
                "memory_id": block_match.group(1).upper().replace(" ", "_"),
                "category": current_section,
                "tags": [],
                "body_lines": [],
                "in_content": False,
            }
            continue

        header_match = _HEADER_RE.match(raw_line)
        if header_match and (current_block is None or not current_block["in_content"]):
            _flush()
            current_section = header_match.group(1).strip() or "general"
            continue

        if current_block is None:
            continue

        stripped = raw_line.strip()
        if not stripped:
            continue

        if not current_block["in_content"]:
            cat_match = _CATEGORY_LINE_RE.match(raw_line)
            if cat_match:
                current_block["category"] = cat_match.group(1).strip() or current_block["category"]
                continue

            tags_match = _TAGS_LINE_RE.match(raw_line)
            if tags_match:
                current_block["tags"].extend(_parse_tags_value(tags_match.group(1)))
                continue

            content_match = _CONTENT_HEADER_RE.match(raw_line)
            if content_match:
                current_block["in_content"] = True
                inline_body = content_match.group(1).strip()
                if inline_body:
                    current_block["body_lines"].append(inline_body)
                continue

        current_block["body_lines"].append(stripped)

    _flush()
    return blocks


def _parse_markdown_memory_lines(lines: Sequence[str]) -> List[Dict[str, Any]]:
    """Парсить простий построковий або Markdown-формат (`# Розділ` + пункти списку)."""
    items: List[Dict[str, Any]] = []
    current_category = "general"
    last_was_bullet = False

    for raw_line in lines:
        if _DIVIDER_RE.match(raw_line):
            last_was_bullet = False
            continue

        header_match = _HEADER_RE.match(raw_line)
        if header_match:
            current_category = header_match.group(1).strip() or "general"
            last_was_bullet = False
            continue

        stripped = raw_line.strip()
        if not stripped:
            last_was_bullet = False
            continue

        if (
            last_was_bullet
            and (raw_line.startswith("  ") or raw_line.startswith("\t"))
            and not _BULLET_RE.match(raw_line)
            and items
        ):
            items[-1]["text"] = f"{items[-1]['text']} {stripped}"
            continue

        bullet_match = _BULLET_RE.match(raw_line)
        if bullet_match:
            cleaned = _BULLET_RE.sub("", raw_line).strip()
            last_was_bullet = True
        else:
            cleaned = stripped
            last_was_bullet = False

        if cleaned:
            items.append(
                {
                    "memory_id": "",
                    "category": current_category,
                    "tags": [],
                    "text": cleaned,
                }
            )

    return items


def parse_memories_file(input_path: str) -> List[Dict[str, Any]]:
    """
    Зчитує спогади з `.txt`, `.md`, `.json` або `.jsonl`.
    Автоматично розпізнає як структуровані блоки `[MEMORY_XX]` (`Категория:`, `Теги:`, `Содержание:`),
    так і звичайні списки з `# Розділ`, видаляє дублікати зі збереженням порядку.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Файл зі спогадами не знайдено: {input_path}")

    ext = os.path.splitext(input_path)[1].lower()
    raw_items: List[Dict[str, Any]] = []

    with open(input_path, "r", encoding="utf-8-sig") as f:
        if ext == ".json":
            data = json.load(f)
            if isinstance(data, dict):
                data = data.get("memories") or data.get("items") or []
            if not isinstance(data, list):
                raise ValueError("JSON-файл спогадів повинен містити список рядків або об'єктів.")
            for entry in data:
                if isinstance(entry, str):
                    raw_items.append({"memory_id": "", "text": entry, "category": "general", "tags": []})
                elif isinstance(entry, dict):
                    txt = entry.get("text") or entry.get("memory") or entry.get("content") or ""
                    cat = str(entry.get("category") or "general")
                    raw_items.append(
                        {
                            "memory_id": str(entry.get("memory_id") or ""),
                            "text": str(txt),
                            "category": cat,
                            "tags": _parse_tags_value(entry.get("tags")),
                        }
                    )
        elif ext == ".jsonl":
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                entry = json.loads(stripped)
                if isinstance(entry, str):
                    raw_items.append({"memory_id": "", "text": entry, "category": "general", "tags": []})
                elif isinstance(entry, dict):
                    txt = entry.get("text") or entry.get("memory") or entry.get("content") or ""
                    cat = str(entry.get("category") or "general")
                    raw_items.append(
                        {
                            "memory_id": str(entry.get("memory_id") or ""),
                            "text": str(txt),
                            "category": cat,
                            "tags": _parse_tags_value(entry.get("tags")),
                        }
                    )
        else:
            lines = f.readlines()
            if any(_MEMORY_BLOCK_RE.match(line) for line in lines):
                raw_items = _parse_structured_memory_blocks(lines)
            else:
                raw_items = _parse_markdown_memory_lines(lines)

    seen = set()
    records: List[Dict[str, Any]] = []
    for item in raw_items:
        text = " ".join(str(item["text"]).split())
        if not text:
            continue
        dedup_key = normalize_text(text)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        cat = str(item.get("category") or "general").strip() or "general"
        tags = _parse_tags_value(item.get("tags"))
        mem_id = str(item.get("memory_id") or "").strip() or f"MEM_{len(records) + 1:02d}"
        tag_kw_source = f"{cat if cat.lower() != 'general' else ''} {' '.join(tags)}".strip()
        kw_source = f"{tag_kw_source} {text}".strip()
        records.append(
            {
                "id": len(records),
                "memory_id": mem_id,
                "text": text,
                "category": cat,
                "tags": tags,
                "tag_keywords": extract_keywords(tag_kw_source) if tag_kw_source else [],
                "keywords": extract_keywords(kw_source),
            }
        )

    if not records:
        raise ValueError(f"У файлі '{input_path}' не знайдено жодного непорожнього спогаду.")

    return records


# =====================================================================
# ЗАПИС І ЧИТАННЯ ІНДЕКСУ (`memory.faiss` + `memories.json`)
# =====================================================================
def _pack_metadata_trailer(meta: Dict[str, Any]) -> bytes:
    json_bytes = json.dumps(meta, ensure_ascii=False).encode("utf-8")
    return json_bytes + struct.pack("<Q", len(json_bytes)) + META_TRAILER_MAGIC


def _extract_metadata_trailer(faiss_path: str) -> Tuple[Optional[Dict[str, Any]], int]:
    """
    Перевіряє, чи містить файл `memory.faiss` вбудований мета-трейлер `GVKMETA1`.
    Повертає `(metadata_dict_or_None, index_payload_byte_length)`.
    """
    file_size = os.path.getsize(faiss_path)
    footer_size = 8 + len(META_TRAILER_MAGIC)
    if file_size <= footer_size:
        return None, file_size

    with open(faiss_path, "rb") as f:
        f.seek(file_size - footer_size)
        raw_len = f.read(8)
        magic = f.read(len(META_TRAILER_MAGIC))
        if magic != META_TRAILER_MAGIC:
            return None, file_size

        json_len = struct.unpack("<Q", raw_len)[0]
        payload_end = file_size - footer_size - json_len
        if payload_end < 0:
            return None, file_size

        f.seek(payload_end)
        json_bytes = f.read(json_len)

    meta = json.loads(json_bytes.decode("utf-8"))
    return meta, payload_end


def _write_gvk_binary_index(
    output_path: str,
    vectors: Sequence[Sequence[float]],
    dim: int,
    trailer_bytes: bytes,
) -> None:
    count = len(vectors)
    pack_fmt = f"<{dim}f"
    with open(output_path, "wb") as f:
        f.write(RAG_MAGIC)
        f.write(struct.pack("<III", RAG_FORMAT_VERSION, count, dim))
        for vec in vectors:
            f.write(struct.pack(pack_fmt, *(float(x) for x in vec)))
        f.write(trailer_bytes)


def _read_gvk_binary_index(faiss_path: str, payload_end: int) -> Tuple[int, int, List[List[float]]]:
    with open(faiss_path, "rb") as f:
        magic = f.read(len(RAG_MAGIC))
        if magic != RAG_MAGIC:
            raise ValueError("Некоректний заголовок контейнера GVKRAG01.")
        version, count, dim = struct.unpack("<III", f.read(12))
        if version != RAG_FORMAT_VERSION:
            raise ValueError(f"Непідтримувана версія формату GVKRAG: {version}")
        expected_end = len(RAG_MAGIC) + 12 + count * dim * 4
        if expected_end > payload_end:
            raise ValueError("Файл memory.faiss пошкоджено або обрізано.")
        unpack_fmt = f"<{dim}f"
        row_bytes = dim * 4
        vectors: List[List[float]] = []
        for _ in range(count):
            vectors.append(list(struct.unpack(unpack_fmt, f.read(row_bytes))))
    return count, dim, vectors


def build_rag_index(
    input_path: str,
    output_faiss_path: str,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    backend: str = "auto",
) -> Dict[str, Any]:
    """
    Компілює текстовий файл спогадів у `memory.faiss` (+ `memories.json` поруч).
    Повертає словник із підсумковою статистикою збірки.
    """
    records = parse_memories_file(input_path)
    engine = EmbeddingEngine(model_name=model_name, backend=backend)
    texts: List[str] = []
    for rec in records:
        cat = str(rec.get("category", "general"))
        tags_str = ", ".join(rec.get("tags") or [])
        prefix_parts = []
        if cat and cat.lower() != "general":
            prefix_parts.append(cat)
        if tags_str:
            prefix_parts.append(f"Теги: {tags_str}")
        if prefix_parts:
            texts.append(f"{' | '.join(prefix_parts)}. {rec['text']}")
        else:
            texts.append(rec["text"])
    vectors = engine.encode(texts)
    dim = len(vectors[0]) if vectors else engine.dimension

    out_abs = os.path.abspath(output_faiss_path)
    out_dir = os.path.dirname(out_abs)
    os.makedirs(out_dir, exist_ok=True)
    memories_json_path = os.path.join(out_dir, "memories.json")

    metadata = {
        "version": RAG_FORMAT_VERSION,
        "model_name": engine.model_name,
        "backend": engine.active_backend,
        "dimension": dim,
        "count": len(records),
        "memories": records,
    }

    with open(memories_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        f.write("\n")

    trailer = _pack_metadata_trailer(metadata)
    index_format = "gvk-binary"

    try:
        import faiss
        import numpy as np

        if hasattr(faiss, "omp_set_num_threads"):
            faiss.omp_set_num_threads(1)
        arr = np.asarray(vectors, dtype=np.float32)
        index = faiss.IndexFlatIP(dim)
        index.add(arr)
        faiss.write_index(index, out_abs)
        with open(out_abs, "ab") as f:
            f.write(trailer)
        index_format = "faiss-indexflatip"
    except ImportError:
        _write_gvk_binary_index(out_abs, vectors, dim, trailer)

    return {
        "faiss_path": out_abs,
        "memories_json_path": memories_json_path,
        "count": len(records),
        "dimension": dim,
        "backend": engine.active_backend,
        "model_name": engine.model_name,
        "index_format": index_format,
        "faiss_bytes": os.path.getsize(out_abs),
        "json_bytes": os.path.getsize(memories_json_path),
    }


# =====================================================================
# РАНТАЙМ-ПОШУК СПОГАДІВ (MemoryRAG)
# =====================================================================
class MemoryRAG:
    """
    Швидкий рушій пошуку спогадів для `core/orchestrator.py`.
    Приймає або шлях до директорії персонажа (де лежить `memory.faiss`),
    або безпосередній шлях до `memory.faiss`.
    """

    def __init__(
        self,
        path: str,
        memories_json_path: Optional[str] = None,
        model_name: Optional[str] = None,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Шлях до пам'яті RAG не знайдено: {path}")

        if os.path.isdir(path):
            self.faiss_path = os.path.join(path, "memory.faiss")
            base_dir = path
        else:
            self.faiss_path = path
            base_dir = os.path.dirname(os.path.abspath(path))

        if not os.path.isfile(self.faiss_path):
            raise FileNotFoundError(f"Файл індексу memory.faiss не знайдено: {self.faiss_path}")

        self.threshold = float(threshold)
        trailer_meta, payload_end = _extract_metadata_trailer(self.faiss_path)

        json_candidate = memories_json_path or os.path.join(base_dir, "memories.json")
        if os.path.isfile(json_candidate):
            with open(json_candidate, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        elif trailer_meta is not None:
            self.metadata = trailer_meta
        else:
            raise FileNotFoundError(
                f"Не знайдено метадані спогадів (ні '{json_candidate}', ні вбудованого трейлера GVKMETA1)."
            )

        self.memories: List[Dict[str, Any]] = self.metadata.get("memories", [])
        self.dimension: int = int(self.metadata.get("dimension", DEFAULT_DIMENSION))
        self.index_backend: str = str(self.metadata.get("backend", "ngram"))
        self.model_name: str = str(model_name or self.metadata.get("model_name", DEFAULT_EMBEDDING_MODEL))

        self._mem_stems: List[set] = [
            {_stem_token(kw) for kw in (m.get("keywords") or extract_keywords(m["text"]))}
            for m in self.memories
        ]
        self._tag_stems: List[set] = [
            {
                _stem_token(kw)
                for kw in (
                    m.get("tag_keywords")
                    or extract_keywords(
                        f"{m.get('category', '')} {' '.join(m.get('tags') or [])}"
                    )
                )
            }
            for m in self.memories
        ]

        # Ініціалізуємо енкодер запитів ДО імпорту faiss, щоб уникнути конфлікту OpenMP на macOS
        requested = self.index_backend if self.index_backend in ("sentence-transformers", "ngram") else "auto"
        try:
            self.engine = EmbeddingEngine(
                model_name=self.model_name,
                backend=requested,
                dimension=self.dimension,
            )
        except Exception:
            # Офлайн-фолбек, якщо індекс побудовано з sentence-transformers, але модель недоступна локально
            self.engine = EmbeddingEngine(
                model_name=self.model_name,
                backend="ngram",
                dimension=self.dimension,
            )

        # Завантаження індексу (нативний FAISS або GVKRAG01)
        self._faiss_index = None
        self._vectors: Optional[List[List[float]]] = None

        with open(self.faiss_path, "rb") as f:
            header_magic = f.read(len(RAG_MAGIC))

        if header_magic == RAG_MAGIC:
            _, dim, vectors = _read_gvk_binary_index(self.faiss_path, payload_end)
            self.dimension = dim
            self._vectors = vectors
        else:
            try:
                import faiss
            except ImportError as exc:
                raise RuntimeError(
                    "Індекс створено у нативному форматі FAISS, але пакет 'faiss-cpu' не встановлено. "
                    "Встановіть залежності: pip install -r requirements.txt"
                ) from exc

            if hasattr(faiss, "omp_set_num_threads"):
                faiss.omp_set_num_threads(1)

            if payload_end < os.path.getsize(self.faiss_path):
                # Відділяємо чистий FAISS-потік без трейлера у тимчасовий файл для стабільності faiss.read_index
                with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as tmp:
                    tmp_path = tmp.name
                try:
                    with open(self.faiss_path, "rb") as src, open(tmp_path, "wb") as dst:
                        remaining = payload_end
                        buf_size = 1024 * 1024
                        while remaining > 0:
                            chunk = src.read(min(buf_size, remaining))
                            if not chunk:
                                break
                            dst.write(chunk)
                            remaining -= len(chunk)
                    self._faiss_index = faiss.read_index(tmp_path)
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            else:
                self._faiss_index = faiss.read_index(self.faiss_path)

    def _cosine_candidates(self, q_vec: List[float], candidate_k: int) -> List[Tuple[int, float]]:
        if not self.memories:
            return []

        if self._faiss_index is not None:
            import numpy as np

            q_arr = np.asarray([q_vec], dtype=np.float32)
            k = min(len(self.memories), candidate_k)
            scores, indices = self._faiss_index.search(q_arr, k)
            out: List[Tuple[int, float]] = []
            for idx, sc in zip(indices[0], scores[0]):
                idx_i = int(idx)
                if 0 <= idx_i < len(self.memories):
                    out.append((idx_i, float(sc)))
            return out

        assert self._vectors is not None
        scored: List[Tuple[int, float]] = []
        for idx, m_vec in enumerate(self._vectors):
            dot = sum(a * b for a, b in zip(q_vec, m_vec))
            scored.append((idx, dot))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:candidate_k]

    @property
    def effective_threshold(self) -> float:
        """Ефективний поріг релевантності з урахуванням активного бекенду ембедінгів."""
        if (
            self.index_backend == "sentence-transformers"
            and self.engine.active_backend == "sentence-transformers"
        ):
            return max(self.threshold, DEFAULT_ST_SIMILARITY_THRESHOLD)
        return self.threshold

    def search(
        self,
        query: str,
        top_k: int = 2,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Шукає до `top_k` найрелевантніших спогадів під текстовий запит.
        Відфільтровує нерелевантні привітання та записи нижче порогу `threshold`.
        """
        if not query or not query.strip() or not self.memories or top_k <= 0:
            return []

        eff_threshold = self.effective_threshold if threshold is None else float(threshold)
        q_stems = _extract_stems(query)
        extra_stems = _expand_query_stems(q_stems) if q_stems else set()

        # Порожній за змістом запит (напр. "привіт", "ок", "як справи?")
        if not q_stems and len(query.strip()) < 24:
            return []

        # Якщо індекс зібрано на sentence-transformers, але зараз доступний лише ngram-фолбек,
        # векторні простори не збігаються — переводимо оцінку на морфологічний ngram + лексику.
        vector_space_mismatch = (
            self.index_backend == "sentence-transformers"
            and self.engine.active_backend != "sentence-transformers"
        )
        using_st = (
            self.index_backend == "sentence-transformers"
            and self.engine.active_backend == "sentence-transformers"
        )

        if vector_space_mismatch:
            candidates = [(i, 0.0) for i in range(len(self.memories))]
        else:
            q_vec = self.engine.encode([query])[0]
            candidate_k = len(self.memories) if len(self.memories) <= 1000 else max(top_k * 10, 50)
            candidates = self._cosine_candidates(q_vec, candidate_k)

        is_who_query = bool(_WHO_QUERY_RE.search(query))

        ranked: List[Tuple[float, Dict[str, Any]]] = []
        for idx, cos_score in candidates:
            mem = self.memories[idx]
            if q_stems:
                mem_stems = self._mem_stems[idx]
                matched_stems = q_stems & mem_stems
                synonym_hits = extra_stems & mem_stems
                effective_matches = len(matched_stems) + 0.65 * len(synonym_hits)
                overlap = min(1.35, effective_matches / len(q_stems))
                tag_hits = len((q_stems | extra_stems) & self._tag_stems[idx])
                tag_bonus = min(tag_hits, 3) * 0.10
                total_matched_count = len(matched_stems) + (1 if synonym_hits else 0)
                multi_match_bonus = 0.08 if total_matched_count >= 2 else 0.0
            else:
                matched_stems = set()
                overlap = 0.0
                tag_bonus = 0.0
                multi_match_bonus = 0.0

            dossier_bonus = 0.0
            if is_who_query and matched_stems:
                cat_low = str(mem.get("category", "")).strip().lower()
                if cat_low.startswith(_DOSSIER_CAT_PREFIXES):
                    dossier_bonus = 0.22

            lexical_boost = 0.42 * overlap + tag_bonus + multi_match_bonus + dossier_bonus
            if using_st and lexical_boost == 0.0 and cos_score < 0.62:
                continue
            if vector_space_mismatch:
                final_score = overlap + tag_bonus + multi_match_bonus + dossier_bonus
            else:
                final_score = cos_score + lexical_boost

            if final_score >= eff_threshold:
                ranked.append(
                    (
                        final_score,
                        {
                            "id": mem.get("id", idx),
                            "memory_id": mem.get("memory_id", f"MEM_{idx:02d}"),
                            "text": mem["text"],
                            "score": round(float(final_score), 4),
                            "category": mem.get("category", "general"),
                        },
                    )
                )

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in ranked[:top_k]]

    def format_for_prompt(
        self,
        query: str,
        top_k: int = 2,
        threshold: Optional[float] = None,
        header: str = "[АКТУАЛЬНИЙ СПОГАД З ПАМ'ЯТІ]:",
    ) -> str:
        """Повертає готовий блок тексту для вставки в системний промпт, або ''."""
        hits = self.search(query, top_k=top_k, threshold=threshold)
        if not hits:
            return ""
        lines = [header]
        for hit in hits:
            lines.append(f"- {hit['text']}")
        return "\n".join(lines)


# =====================================================================
# АВТОНОМНИЙ ТЕСТ (SELF-TEST) ТА CLI
# =====================================================================
def run_self_test(backend: str = "ngram") -> None:
    print("====================================================================")
    print(" [*] GVK Private AI — Self-Test (core/memory_rag.py)")
    print("====================================================================")

    sample_memories = """# Подорожі
- Поїздка в гори: влітку 2025 ми їздили разом у Карпати, піднімалися на вершину та ночували в кемпінгу.
- Гірськолижний виїзд: взимку каталися на лижах на курорті, friend_1 впевнено їздить на лижах з дитинства.

# Навчання
- Викладач геометрії на першій парі завжди перевіряв складні теореми, а на перерві група розбирала задачі.
- Одногрупник клеїв стікери з кодом у коридорі на третьому поверсі корпусу.

# Ігри
- У Дота 2 мейн-герої friend_1 це Джаггернаут (Джаггер) та Міпо (Meepo) на 2к ммр.
"""

    with tempfile.TemporaryDirectory(prefix="gvk_rag_test_") as tmp_dir:
        mem_txt = os.path.join(tmp_dir, "memories.txt")
        with open(mem_txt, "w", encoding="utf-8") as f:
            f.write(sample_memories)

        faiss_path = os.path.join(tmp_dir, "memory.faiss")
        stats = build_rag_index(mem_txt, faiss_path, backend=backend)
        print(
            f" [1/4] Індекс зібрано: count={stats['count']}, dim={stats['dimension']}, "
            f"backend={stats['backend']}, format={stats['index_format']}"
        )
        assert stats["count"] == 5, f"Очікувалось 5 спогадів, отримано {stats['count']}"

        rag = MemoryRAG(tmp_dir)

        hits_mountain = rag.search("шо там було в карпатах?", top_k=1)
        print(f" [2/4] Запит 'Карпати' -> {hits_mountain}")
        assert hits_mountain and "Карпати" in hits_mountain[0]["text"]

        hits_geom = rag.search("хто перевіряв задачі з геометрії?", top_k=1)
        print(f" [3/4] Запит 'геометрія' -> {hits_geom}")
        assert hits_geom and "геометрії" in hits_geom[0]["text"]

        hits_greeting = rag.search("привіт, як справи?", top_k=2)
        print(f" [4/5] Запит-привітання -> {hits_greeting}")
        assert hits_greeting == [], f"Привітання не повинно повертати спогади: {hits_greeting}"

        hits_unrelated = rag.search("де купити хліб в магазині?", top_k=2)
        print(f" [5/5] Нерелевантний запит -> {hits_unrelated}")
        assert hits_unrelated == [], f"Нерелевантний запит не повинен повертати спогади: {hits_unrelated}"

        # Перевірка автономної роботи без memories.json (лише через трейлер у memory.faiss)
        os.remove(stats["memories_json_path"])
        rag_standalone = MemoryRAG(faiss_path)
        prompt_block = rag_standalone.format_for_prompt("на кому в доті граєш?", top_k=1)
        assert "Джаггернаут" in prompt_block

    # [6/6] Перевірка структурованого формату [MEMORY_XX] та крос-мовного пошуку (UA -> RU)
    structured_sample = """================================================================================
## РАЗДЕЛ 1. ТЕСТ
================================================================================

[MEMORY_01]
Категория: Друзья / friend_2
Теги: friend_2, университет, робототехника, компьютерная инженерия
Содержание:
Участник friend_2 переехал в другой город и поступил в технический университет на специальность Кибернетика и Робототехника.

[MEMORY_02]
Категория: Истории учебы / Физика
Теги: учебное заведение, сборник задач по физике, шкаф
Содержание:
В десятом классе одноклассники спрятали мой сборник задач по физике на верхнем шкафу.
"""
    with tempfile.TemporaryDirectory(prefix="gvk_rag_struct_") as tmp_dir2:
        struct_txt = os.path.join(tmp_dir2, "memories.txt")
        with open(struct_txt, "w", encoding="utf-8") as f:
            f.write(structured_sample)
        struct_faiss = os.path.join(tmp_dir2, "memory.faiss")
        struct_stats = build_rag_index(struct_txt, struct_faiss, backend=backend)
        assert struct_stats["count"] == 2, f"Очікувалось 2 блоки [MEMORY_XX], отримано {struct_stats['count']}"

        rag_struct = MemoryRAG(tmp_dir2)
        hits_ua_ru = rag_struct.search("де вчиться friend_2 в університеті?", top_k=1)
        print(f" [6/6] Структурований [MEMORY_XX] UA->RU -> {hits_ua_ru}")
        assert hits_ua_ru and hits_ua_ru[0]["memory_id"] == "MEMORY_01"

        hits_phys = rag_struct.search("історія зі збірником з фізики на шафі", top_k=1)
        assert hits_phys and hits_phys[0]["memory_id"] == "MEMORY_02"

    print("====================================================================")
    print(" [✓] УСІ ПЕРЕВІРКИ MEMORY_RAG ПРОЙДЕНО УСПІШНО!")
    print("====================================================================")


def main() -> None:
    parser = argparse.ArgumentParser(description="GVK Private AI — Runtime RAG пам'ять")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_test = sub.add_parser("self-test", help="Запустити автономний тест збірки та пошуку RAG")
    p_test.add_argument(
        "--backend",
        choices=("auto", "sentence-transformers", "ngram"),
        default="ngram",
        help="Бекенд для тесту (за замовчуванням ngram — працює миттєво й офлайн)",
    )

    p_query = sub.add_parser("query", help="Виконати пошук у зібраному індексі memory.faiss")
    p_query.add_argument("-i", "--index", required=True, help="Папка персонажа або шлях до memory.faiss")
    p_query.add_argument("-q", "--query", required=True, help="Текст запиту")
    p_query.add_argument("-k", "--top-k", type=int, default=2, help="Скільки спогадів повернути")
    p_query.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help="Мінімальний поріг релевантності",
    )

    args = parser.parse_args()
    if args.cmd == "self-test":
        run_self_test(backend=args.backend)
    elif args.cmd == "query":
        rag = MemoryRAG(args.index, threshold=args.threshold)
        results = rag.search(args.query, top_k=args.top_k, threshold=args.threshold)
        if not results:
            print("Релевантних спогадів не знайдено.")
            return
        for idx, hit in enumerate(results, 1):
            print(f"[{idx}] score={hit['score']:.4f} | category={hit['category']} | {hit['text']}")
        print("\n--- Блок для промпта ---")
        print(rag.format_for_prompt(args.query, top_k=args.top_k, threshold=args.threshold))


if __name__ == "__main__":
    main()
