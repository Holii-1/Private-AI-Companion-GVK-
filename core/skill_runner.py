#!/usr/bin/env python3
"""
GVK Private AI — Пісочниця індивідуальних скілів учасників (core/skill_runner.py)

Забезпечує безпечне виявлення, перевірку та виконання персональних Python-функцій
(`tools.py`) усередині ізольованого runtime (зазвичай із RAM-диска).

Основні властивості:
  1. Опціональність (Graceful Fallback): якщо `tools.py` ще не створено або він порожній,
     модуль працює штатно без помилок (`available=False`, статус `IDLE` для TUI).
  2. Статичний AST-фільтр (`validate_tools_ast`): до виконання коду перевіряє синтаксичне
     дерево, дозволяючи лише безпечні стандартні модулі та блокуючи файловий/мережевий/OS-доступ
     і небезпечні конструкції (`eval`, `exec`, `open`, `__globals__`, `__subclasses__`).
  3. Ізольоване виконання з тайм-аутом та лімітом токенів: кожен виклик скіла виконується
     в окремому обмеженому процесі з жорстким тайм-аутом (за замовч. 3.0 с) та обрізанням
     виводу до 2000 символів, щоб не зависнути й не переповнити контекст LLM.
  4. Безпечний парсер викликів LLM: розбирає теги `[SKILL: func(a=1, b="ok")]` через
     `ast.parse` + `ast.literal_eval` без використання `eval()`.

Приклади запуску:
  python3 core/skill_runner.py self-test
  python3 core/skill_runner.py inspect -d path/to/persona_dir
  python3 core/skill_runner.py run -d path/to/persona_dir -f calc_split --args '{"total": 1200, "people": 3}'
"""

import argparse
import ast
import builtins as _py_builtins
from contextlib import redirect_stdout
from dataclasses import dataclass, field
import importlib
import inspect
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


# =====================================================================
# КОНСТАНТИ ПІСОЧНИЦІ ТА ОБМЕЖЕННЯ
# =====================================================================
DEFAULT_TIMEOUT_SEC = 3.0
MAX_OUTPUT_CHARS = 2000
SKILL_MATCH_THRESHOLD = 0.35

ALLOWED_MODULES = frozenset(
    {
        "base64",
        "collections",
        "copy",
        "dataclasses",
        "datetime",
        "decimal",
        "enum",
        "functools",
        "hashlib",
        "itertools",
        "json",
        "math",
        "random",
        "re",
        "statistics",
        "string",
        "textwrap",
        "time",
        "typing",
    }
)

BLOCKED_CALL_NAMES = frozenset(
    {
        "__import__",
        "breakpoint",
        "compile",
        "delattr",
        "dir",
        "eval",
        "exec",
        "getattr",
        "globals",
        "input",
        "locals",
        "memoryview",
        "open",
        "setattr",
        "vars",
    }
)

SAFE_BUILTIN_NAMES = (
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "enumerate",
    "filter",
    "float",
    "frozenset",
    "int",
    "isinstance",
    "len",
    "list",
    "map",
    "max",
    "min",
    "pow",
    "print",
    "range",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
    "Exception",
    "IndexError",
    "KeyError",
    "RuntimeError",
    "TypeError",
    "ValueError",
    "ZeroDivisionError",
)

_SKILL_TAG_RE = re.compile(
    r"\[(?:SKILL|TOOL)\s*:\s*([A-Za-z_][A-Za-z0-9_]*)(?:\((.*?)\))?\s*\]",
    re.IGNORECASE | re.DOTALL,
)
_SLASH_CMD_RE = re.compile(r"^\s*/([a-zа-яіїєґ0-9_][a-zа-яіїєґ0-9_-]*)(?:\s+(.*)|\((.*)\))?\s*$", re.IGNORECASE | re.DOTALL)
_WORD_RE = re.compile(r"[a-zа-яіїєґ0-9_]+")
_MATCH_STOPWORDS = frozenset(
    {
        "для", "про", "под", "над", "при", "без", "что", "как", "кто", "хто", "где",
        "там", "тут", "это", "его", "она", "они", "или", "уже", "еще", "нет", "так",
        "все", "але", "або", "вид", "миж", "пид", "це", "чи", "шо", "че", "що",
    }
)
_SLASH_HELP_COMMANDS = frozenset({"skills", "скили", "скилы", "help", "команди", "команды"})


class SkillSecurityError(ValueError):
    """Помилка безпекової перевірки коду `tools.py` (AST або sandbox)."""


# =====================================================================
# СТРУКТУРИ ДАНИХ
# =====================================================================
@dataclass
class SkillSpec:
    name: str
    description: str
    triggers: List[str] = field(default_factory=list)
    parameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    signature_str: str = "()"
    short_desc: str = ""
    slash: str = ""
    slash_aliases: List[str] = field(default_factory=list)
    usage: str = ""


@dataclass
class SkillMatch:
    skill_name: str
    persona_id: str
    score: float
    matched_triggers: List[str] = field(default_factory=list)


@dataclass
class SkillCall:
    name: str
    args: List[Any]
    kwargs: Dict[str, Any]
    raw_tag: str


@dataclass
class SkillResult:
    ok: bool
    skill_name: str
    output: str
    error: Optional[str] = None
    elapsed_ms: float = 0.0
    truncated: bool = False


# =====================================================================
# ДЕКОРАТОР @skill ДЛЯ ФАЙЛІВ tools.py
# =====================================================================
def skill(
    triggers: Optional[Sequence[str]] = None,
    description: Optional[str] = None,
    name: Optional[str] = None,
    short_desc: Optional[str] = None,
    slash: Optional[str] = None,
    slash_aliases: Optional[Sequence[str]] = None,
    usage: Optional[str] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Декоратор для явної реєстрації скіла у `tools.py`.
    Усередині пісочниці доступний автоматично без потреби робити import.
    """

    def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        meta = {
            "name": name or func.__name__,
            "description": (description or inspect.getdoc(func) or "").strip(),
            "triggers": [str(t).strip() for t in (triggers or []) if str(t).strip()],
            "short_desc": (short_desc or "").strip(),
            "slash": (slash or "").strip(),
            "slash_aliases": [str(a).strip() for a in (slash_aliases or []) if str(a).strip()],
            "usage": (usage or "").strip(),
        }
        setattr(func, "_gvk_skill_meta", meta)
        return func

    return _decorator


# =====================================================================
# РІВЕНЬ 1-2: СТАТИЧНИЙ AST-ВАЛІДАТОР ТА ПІСОЧНИЦЯ ІМПОРТУ
# =====================================================================
def _normalize_for_match(text: str) -> str:
    s = text.lower().strip()
    s = s.replace("ё", "е").replace("ґ", "г").replace("і", "и").replace("ї", "и").replace("ы", "и").replace("є", "е")
    return s


def validate_tools_ast(source_code: str, filename: str = "tools.py") -> ast.Module:
    """
    Перевіряє синтаксичне дерево `tools.py` до виконання.
    Дозволяє лише модулі з `ALLOWED_MODULES` та блокує небезпечні виклики / dunder-атрибути.
    """
    try:
        tree = ast.parse(source_code, filename=filename, mode="exec")
    except SyntaxError as exc:
        raise SkillSecurityError(f"Синтаксична помилка у {filename} (рядок {exc.lineno}): {exc.msg}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_mod = alias.name.split(".")[0]
                if root_mod not in ALLOWED_MODULES:
                    raise SkillSecurityError(
                        f"Заборонений імпорт '{alias.name}' (рядок {node.lineno}). "
                        f"Дозволені модулі: {', '.join(sorted(ALLOWED_MODULES))}."
                    )

        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                raise SkillSecurityError(f"Заборонений відносний імпорт у {filename} (рядок {node.lineno}).")
            mod_name = (node.module or "").split(".")[0]
            if mod_name not in ALLOWED_MODULES:
                raise SkillSecurityError(
                    f"Заборонений імпорт із '{node.module}' (рядок {node.lineno})."
                )

        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                raise SkillSecurityError(
                    f"Заборонений доступ до службового dunder-атрибута '{node.attr}' (рядок {node.lineno})."
                )

        elif isinstance(node, ast.Name):
            if node.id in BLOCKED_CALL_NAMES:
                raise SkillSecurityError(
                    f"Заборонене використання небезпечного ідентифікатора '{node.id}' (рядок {node.lineno})."
                )

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_CALL_NAMES:
                raise SkillSecurityError(
                    f"Заборонений виклик '{node.func.id}()' у пісочниці (рядок {node.lineno})."
                )

    return tree


def _sandbox_import(
    name: str,
    globals_dict: Optional[Dict[str, Any]] = None,
    locals_dict: Optional[Dict[str, Any]] = None,
    fromlist: Tuple[str, ...] = (),
    level: int = 0,
) -> Any:
    if level != 0:
        raise ImportError("Відносні імпорти заборонені у пісочниці.")
    root_mod = name.split(".")[0]
    if root_mod not in ALLOWED_MODULES:
        raise ImportError(f"Модуль '{name}' заборонений у пісочниці GVK.")
    return importlib.__import__(name, globals_dict, locals_dict, fromlist, 0)


def build_sandbox_globals() -> Dict[str, Any]:
    """Формує ізольований простір імен `__builtins__` для виконання `tools.py`."""
    safe_builtins: Dict[str, Any] = {"__import__": _sandbox_import}
    for name in SAFE_BUILTIN_NAMES:
        safe_builtins[name] = getattr(_py_builtins, name)

    return {
        "__builtins__": safe_builtins,
        "__name__": "gvk_persona_tools",
        "skill": skill,
    }


def load_sandbox_namespace(tools_path: str) -> Dict[str, Any]:
    """Валідує AST та виконує `tools.py` у чистому sandbox-просторі імен."""
    with open(tools_path, "r", encoding="utf-8") as f:
        source_code = f.read()

    if not source_code.strip():
        return {}

    tree = validate_tools_ast(source_code, filename=os.path.basename(tools_path))
    code_obj = compile(tree, filename=os.path.basename(tools_path), mode="exec")
    namespace = build_sandbox_globals()
    exec(code_obj, namespace, namespace)  # noqa: S102 — код щойно перевірено AST-фільтром та ізольовано
    return namespace


# =====================================================================
# ПАРСЕР ТЕГІВ ВИКЛИКУ ВІДПОВІДІ LLM ([SKILL: func(a=1, b="x")])
# =====================================================================
def parse_skill_call(text: str) -> Optional[SkillCall]:
    """
    Безпечно парсить перший тег виклику `[SKILL: func_name(...)]` або `[TOOL: ...]`
    у згенерованому тексті моделі через `ast.parse` та `ast.literal_eval`.
    """
    if not text:
        return None
    match = _SKILL_TAG_RE.search(text)
    if not match:
        return None

    func_name = match.group(1).strip()
    raw_args = (match.group(2) or "").strip()
    call_expr = f"{func_name}({raw_args})"

    try:
        parsed = ast.parse(call_expr, mode="eval")
    except SyntaxError:
        return None

    if not isinstance(parsed.body, ast.Call):
        return None

    args: List[Any] = []
    kwargs: Dict[str, Any] = {}
    try:
        for arg_node in parsed.body.args:
            args.append(ast.literal_eval(arg_node))
        for kw_node in parsed.body.keywords:
            if kw_node.arg is None:
                return None  # Блокуємо **unpacking у виклику LLM
            kwargs[kw_node.arg] = ast.literal_eval(kw_node.value)
    except (ValueError, SyntaxError, TypeError):
        return None

    return SkillCall(name=func_name, args=args, kwargs=kwargs, raw_tag=match.group(0))


def strip_skill_tags(text: str) -> str:
    """Видаляє службові теги `[SKILL: ...]` / `[TOOL: ...]` з репліки перед показом у TUI."""
    if not text:
        return ""
    cleaned = _SKILL_TAG_RE.sub("", text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


# =====================================================================
# РІВЕНЬ 3: ОСНОВНИЙ КЛАС SKILL RUNNER
# =====================================================================
def _extract_param_schema(func: Callable[..., Any]) -> Tuple[Dict[str, Dict[str, Any]], str]:
    sig = inspect.signature(func)
    params: Dict[str, Dict[str, Any]] = {}
    parts: List[str] = []

    for p_name, param in sig.parameters.items():
        p_info: Dict[str, Any] = {"required": param.default is inspect.Parameter.empty}
        ann = param.annotation
        if ann is not inspect.Parameter.empty:
            p_info["type"] = getattr(ann, "__name__", str(ann))
        if param.default is not inspect.Parameter.empty:
            p_info["default"] = repr(param.default)

        params[p_name] = p_info
        piece = p_name
        if "type" in p_info:
            piece += f": {p_info['type']}"
        if "default" in p_info:
            piece += f" = {p_info['default']}"
        parts.append(piece)

    return params, f"({', '.join(parts)})"


class SkillRunner:
    """
    Менеджер скілів персонажа.
    Приймає шлях до папки персонажа (в RAM) або безпосередньо до `tools.py`.
    Якщо `tools.py` відсутній або порожній, безпечно працює в режимі без скілів.
    """

    def __init__(
        self,
        source_path: Optional[str] = None,
        persona_id: str = "persona",
        persona_config: Optional[Dict[str, Any]] = None,
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
        max_output_chars: int = MAX_OUTPUT_CHARS,
    ) -> None:
        self.persona_id = persona_id
        self.timeout_sec = float(timeout_sec)
        self.max_output_chars = int(max_output_chars)
        self.tools_path: Optional[str] = None
        self.persona_config: Dict[str, Any] = dict(persona_config or {})

        self.available: bool = False
        self.skills: Dict[str, SkillSpec] = {}
        self.load_error: Optional[str] = None

        if source_path:
            if os.path.isdir(source_path):
                cand_tools = os.path.join(source_path, "tools.py")
                if os.path.isfile(cand_tools):
                    self.tools_path = cand_tools
                cfg_path = os.path.join(source_path, "persona_config.json")
                if not self.persona_config and os.path.isfile(cfg_path):
                    try:
                        with open(cfg_path, "r", encoding="utf-8") as f:
                            loaded_cfg = json.load(f)
                        if isinstance(loaded_cfg, dict):
                            self.persona_config = loaded_cfg
                            self.persona_id = str(loaded_cfg.get("persona_id") or self.persona_id)
                    except (OSError, json.JSONDecodeError):
                        pass
            elif os.path.isfile(source_path):
                self.tools_path = source_path

        if self.tools_path and os.path.isfile(self.tools_path):
            self._load_skills(self.tools_path)

    def _load_skills(self, tools_path: str) -> None:
        namespace = load_sandbox_namespace(tools_path)
        if not namespace:
            self.available = False
            self.skills = {}
            return

        cfg_skills: Dict[str, dict] = {}
        for item in self.persona_config.get("skills") or []:
            if isinstance(item, dict) and item.get("name"):
                cfg_skills[str(item["name"])] = item

        loaded: Dict[str, SkillSpec] = {}
        for attr_name, obj in namespace.items():
            if attr_name.startswith("_") or attr_name == "skill":
                continue
            if not inspect.isfunction(obj):
                continue
            # Реєструємо тільки функції, визначені у самому `tools.py`
            if getattr(obj, "__module__", None) != "gvk_persona_tools":
                continue

            meta = getattr(obj, "_gvk_skill_meta", None) or {}
            skill_name = str(meta.get("name") or attr_name)
            cfg_entry = cfg_skills.get(skill_name, {})

            doc = str(meta.get("description") or cfg_entry.get("description") or inspect.getdoc(obj) or "").strip()
            if not doc:
                doc = f"Виконати скіл {skill_name}"

            raw_slash = str(cfg_entry.get("slash") or meta.get("slash") or "").strip()
            if raw_slash and not raw_slash.startswith("/"):
                raw_slash = f"/{raw_slash}"

            slash_aliases: List[str] = []
            for raw_alias in (
                ([raw_slash] if raw_slash else [])
                + list(cfg_entry.get("slash_aliases") or [])
                + list(meta.get("slash_aliases") or [])
                + [f"/{skill_name}"]
            ):
                a_clean = str(raw_alias).strip()
                if not a_clean:
                    continue
                if not a_clean.startswith("/"):
                    a_clean = f"/{a_clean}"
                if a_clean not in slash_aliases:
                    slash_aliases.append(a_clean)

            usage = str(cfg_entry.get("usage") or meta.get("usage") or raw_slash or f"/{skill_name}").strip()

            short_desc = str(cfg_entry.get("short_desc") or meta.get("short_desc") or "").strip()
            if not short_desc:
                short_desc = doc.split(".")[0].strip()[:36]

            triggers: List[str] = []
            for t in list(meta.get("triggers") or []) + list(cfg_entry.get("triggers") or []):
                t_clean = str(t).strip()
                if t_clean and t_clean not in triggers:
                    triggers.append(t_clean)

            params, sig_str = _extract_param_schema(obj)
            loaded[skill_name] = SkillSpec(
                name=skill_name,
                description=doc,
                triggers=triggers,
                parameters=params,
                signature_str=sig_str,
                short_desc=short_desc,
                slash=raw_slash or f"/{skill_name}",
                slash_aliases=slash_aliases,
                usage=usage,
            )

        self.skills = loaded
        self.available = bool(loaded)

    def list_skills(self) -> List[SkillSpec]:
        return list(self.skills.values())

    def get_dossier_status(self) -> str:
        """Повертає компактний рядок статусу скілів для панелі досьє TUI."""
        if not self.available or not self.skills:
            return "немає активних скілів [IDLE]"
        names = list(self.skills.keys())
        if len(names) == 1:
            return f"tools.{names[0]}() [OK]"
        return f"активних: {len(names)} [OK]"

    def get_dossier_skills_lines(self) -> List[str]:
        """Повертає список однорядкових описів скілів у форматі `як викликати - що робить`."""
        if not self.available or not self.skills:
            return []
        lines: List[str] = []
        for spec in self.skills.values():
            invoke = (spec.usage or spec.slash or f"/{spec.name}").strip()
            desc = (spec.short_desc or spec.name).strip()
            if desc.startswith(invoke) or desc.startswith("/"):
                lines.append(f"• {desc}")
            else:
                lines.append(f"• {invoke} - {desc}")
        return lines

    def is_slash_help_command(self, user_text: str) -> bool:
        """Перевіряє, чи ввів користувач команду перегляду списку скілів (`/skills`, `/скіли`, `/help`)."""
        if not user_text:
            return False
        m = _SLASH_CMD_RE.match(user_text.strip())
        if not m:
            return False
        cmd_norm = _normalize_for_match(m.group(1))
        rest = (m.group(2) or m.group(3) or "").strip()
        return cmd_norm in _SLASH_HELP_COMMANDS and not rest

    def format_slash_help(self) -> str:
        """Форматує список доступних slash-команд та підказку по ключових словах."""
        if not self.available or not self.skills:
            return "Для цього персонажа активні скіли відсутні."
        lines = ["Доступні скіли (через /команду або ключові слова в тексті):"]
        for spec in self.skills.values():
            invoke = (spec.usage or spec.slash or f"/{spec.name}").strip()
            desc = (spec.short_desc or spec.description).strip()
            lines.append(f"• {invoke} - {desc}")
        return "\n".join(lines)

    @staticmethod
    def _build_kwargs_from_text(spec: SkillSpec, raw_arg_text: str) -> Dict[str, Any]:
        """Витягує аргументи функції скіла з тексту після `/команди` або з природної фрази."""
        kwargs: Dict[str, Any] = {}
        param_items = list(spec.parameters.items())
        if not param_items:
            return kwargs

        text = (raw_arg_text or "").strip()
        num_matches = re.findall(r"\d+(?:[.,]\d+)?", text)
        num_idx = 0

        for idx, (p_name, p_meta) in enumerate(param_items):
            p_type = str(p_meta.get("type") or "").lower()
            p_default_raw = p_meta.get("default")
            is_int = p_type == "int"
            is_float = p_type == "float"
            if not (is_int or is_float) and isinstance(p_default_raw, str):
                try:
                    lit = ast.literal_eval(p_default_raw)
                    if isinstance(lit, int) and not isinstance(lit, bool):
                        is_int = True
                    elif isinstance(lit, float):
                        is_float = True
                except (ValueError, SyntaxError):
                    pass

            if is_int or is_float:
                if num_idx < len(num_matches):
                    raw_num = num_matches[num_idx].replace(",", ".")
                    num_idx += 1
                    try:
                        kwargs[p_name] = int(float(raw_num)) if is_int else float(raw_num)
                    except ValueError:
                        pass
            elif idx == 0:
                if text:
                    kwargs[p_name] = text
                break
        return kwargs

    def match_slash_command(self, user_text: str) -> Optional[Tuple[SkillSpec, str]]:
        """
        Перевіряє, чи починається повідомлення зі slash-команди зареєстрованого скіла
        (наприклад `/split 900 3` або `/ping`).
        """
        if not self.available or not self.skills or not user_text:
            return None
        m = _SLASH_CMD_RE.match(user_text.strip())
        if not m:
            return None
        cmd_norm = _normalize_for_match(m.group(1))
        arg_text = (m.group(2) if m.group(2) is not None else (m.group(3) or "")).strip()

        for spec in self.skills.values():
            candidates = list(spec.slash_aliases) + [spec.slash, f"/{spec.name}"]
            for alias in candidates:
                a_norm = _normalize_for_match(str(alias).lstrip("/"))
                if a_norm and cmd_norm == a_norm:
                    return spec, arg_text
        return None

    def execute_slash_command(self, user_text: str) -> Optional[SkillResult]:
        """Виконує скіл за явною slash-командою користувача (`/команда [аргументи]`)."""
        matched = self.match_slash_command(user_text)
        if matched is None:
            return None
        spec, arg_text = matched
        kwargs = self._build_kwargs_from_text(spec, arg_text)
        return self.execute(spec.name, kwargs=kwargs)

    def execute_for_query(
        self,
        user_text: str,
        threshold: float = 0.75,
    ) -> Optional[SkillResult]:
        """
        Якщо репліка користувача є slash-командою або явно збігається з тригером скіла
        (`score >= threshold`), автоматично витягує аргумент і виконує скіл у пісочниці.
        """
        if not self.available or not self.skills or not user_text or not user_text.strip():
            return None

        slash_res = self.execute_slash_command(user_text)
        if slash_res is not None:
            return slash_res

        match = self.match_query(user_text, threshold=threshold)
        if match is None or not match.matched_triggers:
            return None

        spec = self.skills.get(match.skill_name)
        if spec is None:
            return None

        kwargs = self._build_kwargs_from_text(spec, user_text)
        return self.execute(match.skill_name, kwargs=kwargs)

    @staticmethod
    def _token_matches_query(t_tok: str, q_tokens: Sequence[str]) -> bool:
        if len(t_tok) == 3:
            return any(qt == t_tok or (len(qt) <= 4 and qt.startswith(t_tok)) for qt in q_tokens)
        return any(qt.startswith(t_tok[:4]) or t_tok.startswith(qt[:4]) for qt in q_tokens)

    def match_query(
        self,
        query: str,
        threshold: float = SKILL_MATCH_THRESHOLD,
    ) -> Optional[SkillMatch]:
        """
        Оцінює, чи відповідає репліка користувача slash-команді, тригерам або опису скіла.
        Використовується арбітром `core/orchestrator.py` на етапі Skill Match.
        """
        if not self.available or not self.skills or not query or not query.strip():
            return None

        slash_hit = self.match_slash_command(query)
        if slash_hit is not None:
            spec, _ = slash_hit
            return SkillMatch(
                skill_name=spec.name,
                persona_id=self.persona_id,
                score=1.0,
                matched_triggers=[spec.slash or f"/{spec.name}"],
            )

        norm_q = _normalize_for_match(query)
        q_tokens = {tok for tok in _WORD_RE.findall(norm_q) if len(tok) >= 3 and tok not in _MATCH_STOPWORDS}

        best_match: Optional[SkillMatch] = None
        for spec in self.skills.values():
            matched_triggers: List[str] = []
            score = 0.0

            for trig in spec.triggers:
                t_norm = _normalize_for_match(trig)
                if not t_norm:
                    continue
                if re.search(rf"(?<![a-zа-яіїєґ0-9_]){re.escape(t_norm)}", norm_q):
                    matched_triggers.append(trig)
                    score = max(score, 0.90 if len(t_norm) >= 5 else 0.75)
                else:
                    t_tokens = {
                        tok for tok in _WORD_RE.findall(t_norm) if len(tok) >= 3 and tok not in _MATCH_STOPWORDS
                    }
                    if t_tokens and q_tokens:
                        stem_hits = sum(1 for tt in t_tokens if self._token_matches_query(tt, q_tokens))
                        if stem_hits > 0:
                            matched_triggers.append(trig)
                            if len(t_tokens) >= 2 and stem_hits == len(t_tokens):
                                score = max(score, 0.80)
                            else:
                                score = max(score, min(0.80, 0.45 + 0.15 * stem_hits))

            desc_tokens = {
                tok
                for tok in _WORD_RE.findall(_normalize_for_match(spec.description + " " + spec.name))
                if len(tok) >= 4 and tok not in _MATCH_STOPWORDS
            }
            if desc_tokens and q_tokens:
                overlap = sum(
                    1 for dt in desc_tokens if any(qt.startswith(dt[:4]) or dt.startswith(qt[:4]) for qt in q_tokens)
                )
                if overlap > 0:
                    score = max(score, min(0.65, 0.25 + 0.20 * overlap))

            if score >= threshold and (best_match is None or score > best_match.score):
                best_match = SkillMatch(
                    skill_name=spec.name,
                    persona_id=self.persona_id,
                    score=round(score, 4),
                    matched_triggers=matched_triggers,
                )

        return best_match

    def execute(
        self,
        skill_name: str,
        args: Optional[Sequence[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> SkillResult:
        """
        Виконує скіл в окремому ізольованому процесі з тайм-аутом `timeout_sec`
        та обмеженням довжини тексту `max_output_chars`.
        """
        t0 = time.time()
        if not self.available or not self.tools_path:
            return SkillResult(
                ok=False,
                skill_name=skill_name,
                output="",
                error="Скіли для цього персонажа не підключені.",
                elapsed_ms=0.0,
            )

        if skill_name not in self.skills:
            return SkillResult(
                ok=False,
                skill_name=skill_name,
                output="",
                error=f"Невідомий скіл '{skill_name}'.",
                elapsed_ms=0.0,
            )

        payload = json.dumps(
            {
                "tools_path": self.tools_path,
                "skill_name": skill_name,
                "args": list(args or []),
                "kwargs": dict(kwargs or {}),
            },
            ensure_ascii=False,
        )

        try:
            proc = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "_worker"],
                input=payload,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
            )
        except subprocess.TimeoutExpired:
            elapsed_ms = (time.time() - t0) * 1000.0
            return SkillResult(
                ok=False,
                skill_name=skill_name,
                output="",
                error=f"Перевищено ліміт часу виконання ({self.timeout_sec:.1f} с).",
                elapsed_ms=round(elapsed_ms, 2),
            )

        elapsed_ms = round((time.time() - t0) * 1000.0, 2)
        if proc.returncode != 0:
            err_msg = (proc.stderr or proc.stdout or f"Код завершення {proc.returncode}").strip()
            return SkillResult(
                ok=False,
                skill_name=skill_name,
                output="",
                error=err_msg[:400],
                elapsed_ms=elapsed_ms,
            )

        try:
            res_data = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return SkillResult(
                ok=False,
                skill_name=skill_name,
                output="",
                error="Некоректна відповідь воркера пісочниці.",
                elapsed_ms=elapsed_ms,
            )

        raw_out = str(res_data.get("output") or "")
        truncated = False
        if len(raw_out) > self.max_output_chars:
            suffix = "\n... [вивід скіла обрізано]"
            keep = max(0, self.max_output_chars - len(suffix))
            raw_out = raw_out[:keep].rstrip() + suffix
            truncated = True

        return SkillResult(
            ok=bool(res_data.get("ok")),
            skill_name=skill_name,
            output=raw_out,
            error=res_data.get("error"),
            elapsed_ms=elapsed_ms,
            truncated=truncated,
        )

    def execute_tag(self, llm_text: str) -> Optional[SkillResult]:
        """Якщо у тексті LLM є тег `[SKILL: ...]`, виконує його та повертає `SkillResult`."""
        call = parse_skill_call(llm_text)
        if call is None:
            return None
        return self.execute(call.name, args=call.args, kwargs=call.kwargs)

    def format_skills_for_prompt(self) -> str:
        """Формує інструкцію доступних скілів для системного промпта моделі."""
        if not self.available or not self.skills:
            return ""
        lines = [
            "[ДОСТУПНІ ІНСТРУМЕНТИ (СКІЛИ)]:",
            "Якщо для точної відповіді потрібен розрахунок або інструмент, "
            "поверни тег формату [SKILL: назва_функції(arg=значення)].",
        ]
        for spec in self.skills.values():
            slash_hint = f" [{spec.usage or spec.slash}]" if (spec.usage or spec.slash) else ""
            lines.append(f"- {spec.name}{spec.signature_str}{slash_hint}: {spec.description}")
        return "\n".join(lines)

    @staticmethod
    def format_result_for_prompt(result: SkillResult) -> str:
        """Формує блок із результатом роботи скіла для повторного ходу оркестратора."""
        if result.ok:
            return f"[РЕЗУЛЬТАТ СКІЛА {result.skill_name}]:\n{result.output}"
        return f"[ПОМИЛКА СКІЛА {result.skill_name}]: {result.error or 'невідома помилка'}"


# =====================================================================
# ВНУТРІШНІЙ ВОРКЕР ПІСОЧНИЦІ (ІЗОЛЬОВАНИЙ ПРОЦЕС)
# =====================================================================
def _format_return_value(ret_val: Any, printed: str) -> str:
    parts: List[str] = []
    if printed.strip():
        parts.append(printed.strip())
    if ret_val is not None:
        if isinstance(ret_val, (dict, list)):
            parts.append(json.dumps(ret_val, ensure_ascii=False, indent=2))
        else:
            parts.append(str(ret_val))
    return "\n".join(parts).strip()


def _run_worker_from_stdin() -> int:
    try:
        req = json.loads(sys.stdin.read())
        tools_path = req["tools_path"]
        skill_name = req["skill_name"]
        args = req.get("args") or []
        kwargs = req.get("kwargs") or {}

        ns = load_sandbox_namespace(tools_path)
        target_fn = None
        for attr_name, obj in ns.items():
            if not inspect.isfunction(obj) or getattr(obj, "__module__", None) != "gvk_persona_tools":
                continue
            meta = getattr(obj, "_gvk_skill_meta", None) or {}
            resolved_name = str(meta.get("name") or attr_name)
            if resolved_name == skill_name:
                target_fn = obj
                break

        if target_fn is None:
            sys.stdout.write(
                json.dumps({"ok": False, "output": "", "error": f"Функцію '{skill_name}' не знайдено."}, ensure_ascii=False)
            )
            return 0

        stdout_buf = io.StringIO()
        with redirect_stdout(stdout_buf):
            ret_val = target_fn(*args, **kwargs)

        formatted = _format_return_value(ret_val, stdout_buf.getvalue())
        sys.stdout.write(json.dumps({"ok": True, "output": formatted, "error": None}, ensure_ascii=False))
        return 0

    except Exception as exc:
        sys.stdout.write(
            json.dumps({"ok": False, "output": "", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False)
        )
        return 0


# =====================================================================
# АВТОНОМНИЙ SELF-TEST (ПЕРЕВІРКА ПІСОЧНИЦІ, ТАЙМ-АУТУ ТА FALLBACK)
# =====================================================================
def run_self_test() -> None:
    print("====================================================================")
    print(" [*] GVK Private AI — Self-Test пісочниці скілів (skill_runner.py)")
    print("====================================================================")

    with tempfile.TemporaryDirectory(prefix="gvk_skill_test_") as tmpdir:
        # 1. Режим без tools.py
        print(" [1/6] Перевірка режиму без tools.py (Graceful Fallback)...")
        empty_dir = os.path.join(tmpdir, "empty_persona")
        os.makedirs(empty_dir, exist_ok=True)
        idle_runner = SkillRunner(empty_dir, persona_id="friend_1")
        assert idle_runner.available is False
        assert idle_runner.list_skills() == []
        assert idle_runner.match_query("порахуй чек на трьох") is None
        assert idle_runner.format_skills_for_prompt() == ""
        assert "IDLE" in idle_runner.get_dossier_status()
        print(f"       [OK] Без tools.py працює стабільно: {idle_runner.get_dossier_status()}")

        # 2. Безпечні тестові скіли
        print(" [2/6] Перевірка завантаження та виконання безпечних скілів...")
        persona_dir = os.path.join(tmpdir, "persona_ok")
        os.makedirs(persona_dir, exist_ok=True)
        tools_file = os.path.join(persona_dir, "tools.py")
        with open(tools_file, "w", encoding="utf-8") as f:
            f.write(
                '''import math
import json

@skill(
    triggers=["чек", "розділити рахунок", "по скільки скинутись"],
    description="Ділить суму чека порівну між учасниками.",
    slash="/чек",
    slash_aliases=["/split"],
    usage="/чек <сума> [людей]",
    short_desc="ділить чек",
)
def calc_split(total: float, people: int = 2) -> str:
    if people <= 0:
        raise ValueError("Кількість людей має бути > 0")
    share = round(total / people, 2)
    return f"Разом {total} -> по {share} з кожного ({people} ос.)"

def analyze_code(snippet: str) -> str:
    """Рахує кількість рядків та базових конструкцій у фрагменті коду."""
    lines = [ln for ln in snippet.splitlines() if ln.strip()]
    return json.dumps({"non_empty_lines": len(lines), "sqrt_lines": round(math.sqrt(len(lines)), 2)}, ensure_ascii=False)

def infinite_loop() -> str:
    while True:
        pass

def crash_div(a: int = 1, b: int = 0) -> float:
    return a / b

def verbose_spam() -> str:
    return "A" * 5000
'''
            )

        runner = SkillRunner(persona_dir, persona_id="friend_1", timeout_sec=0.5, max_output_chars=300)
        assert runner.available is True
        assert "calc_split" in runner.skills and "analyze_code" in runner.skills
        assert "[OK]" in runner.get_dossier_status()

        res_calc = runner.execute("calc_split", kwargs={"total": 1200, "people": 3})
        assert res_calc.ok and "400.0" in res_calc.output, f"Невірний результат calc_split: {res_calc}"
        res_code = runner.execute("analyze_code", args=["a = 1\nb = 2\nc = 3\nd = 4"])
        assert res_code.ok and '"non_empty_lines": 4' in res_code.output
        print(f"       [OK] calc_split -> {res_calc.output}")

        # 3. Skill Match, slash-команди та парсер тегів [SKILL: ...]
        print(" [3/6] Перевірка Skill Match, slash-команд та парсера тегів LLM...")
        matched = runner.match_query("пацани, по скільки скинутись за піцу?")
        assert matched is not None and matched.skill_name == "calc_split"
        slash_res = runner.execute_slash_command("/чек 900 3")
        assert slash_res is not None and slash_res.ok and "300.0" in slash_res.output
        alias_res = runner.execute_for_query("/split 600 2")
        assert alias_res is not None and alias_res.ok and "300.0" in alias_res.output
        assert runner.is_slash_help_command("/skills") is True
        assert "/чек <сума> [людей] - ділить чек" in runner.get_dossier_skills_lines()[0]
        llm_reply = "Зараз гляну [SKILL: calc_split(total=900, people=3)] секунду."
        assert strip_skill_tags(llm_reply) == "Зараз гляну секунду."
        tag_res = runner.execute_tag(llm_reply)
        assert tag_res is not None and tag_res.ok and "300.0" in tag_res.output
        assert "[ДОСТУПНІ ІНСТРУМЕНТИ" in runner.format_skills_for_prompt()
        assert "[РЕЗУЛЬТАТ СКІЛА calc_split]" in runner.format_result_for_prompt(tag_res)
        print(f"       [OK] Skill Match (score={matched.score}), /чек, /split і теги працюють.")

        # 4. Блокування небезпечного коду AST-валідатором
        print(" [4/6] Перевірка блокування небезпечного коду AST-фільтром...")
        malicious_samples = [
            "import subprocess\ndef pwn(): return subprocess.check_output(['id'])",
            "from os import system\ndef pwn(): return system('ls')",
            "def pwn(): return open('/etc/passwd').read()",
            "def pwn(): return eval('1+1')",
            "def pwn(): return ().__class__.__base__.__subclasses__()",
            "def pwn(): return getattr('', '__class__')",
        ]
        for idx, sample in enumerate(malicious_samples, 1):
            blocked = False
            try:
                validate_tools_ast(sample, filename=f"bad_{idx}.py")
            except SkillSecurityError:
                blocked = True
            assert blocked, f"Небезпечний зразок #{idx} не був заблокований!"
        print(f"       [OK] Усі {len(malicious_samples)} небезпечних конструкцій заблоковано до виконання.")

        # 5. Захист від нескінченного циклу (Timeout)
        print(" [5/6] Перевірка тайм-ауту на нескінченному циклі while True...")
        res_timeout = runner.execute("infinite_loop")
        assert res_timeout.ok is False and res_timeout.error and "ліміт часу" in res_timeout.error
        print(f"       [OK] Завислий процес примусово зупинено ({res_timeout.elapsed_ms:.0f} мс).")

        # 6. Перехоплення винятків та обрізання надмірного виводу
        print(" [6/6] Перевірка перехоплення помилок та обрізання довгого виводу...")
        res_crash = runner.execute("crash_div")
        assert res_crash.ok is False and res_crash.error and "ZeroDivisionError" in res_crash.error
        res_spam = runner.execute("verbose_spam")
        assert res_spam.ok and res_spam.truncated is True and len(res_spam.output) <= 300
        print("       [OK] ZeroDivisionError ізольовано; довгий вивід обрізано до 300 симв.")

    print("====================================================================")
    print(" [✓] УСІ ТЕСТИ SKILL_RUNNER ПРОЙДЕНО УСПІШНО!")
    print("====================================================================")


# =====================================================================
# CLI ІНТЕРФЕЙС
# =====================================================================
def main(argv: Optional[List[str]] = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv and raw_argv[0] == "_worker":
        raise SystemExit(_run_worker_from_stdin())

    parser = argparse.ArgumentParser(
        description="GVK Private AI — Пісочниця скілів учасників (core/skill_runner.py)"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("self-test", help="Запустити повну самоперевірку пісочниці, AST-фільтра і тайм-аутів")

    p_insp = sub.add_parser("inspect", help="Показати список доступних скілів у папці персонажа або tools.py")
    p_insp.add_argument("-d", "--source", required=True, help="Шлях до папки персонажа або tools.py")

    p_run = sub.add_parser("run", help="Безпечно виконати скіл у пісочниці")
    p_run.add_argument("-d", "--source", required=True, help="Шлях до папки персонажа або tools.py")
    p_run.add_argument("-f", "--func", required=True, help="Назва функції скіла")
    p_run.add_argument("--args", default="{}", help="JSON-словник іменованих аргументів")
    p_run.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SEC, help="Ліміт часу в секундах")

    args = parser.parse_args(raw_argv)

    if args.cmd == "self-test":
        run_self_test()
    elif args.cmd == "inspect":
        runner = SkillRunner(args.source)
        print(f"Статус: {runner.get_dossier_status()}")
        for spec in runner.list_skills():
            tr = ", ".join(spec.triggers) if spec.triggers else "-"
            print(f" - {spec.name}{spec.signature_str}: {spec.description} [тригери: {tr}]")
    elif args.cmd == "run":
        runner = SkillRunner(args.source, timeout_sec=args.timeout)
        kwargs = json.loads(args.args) if args.args else {}
        res = runner.execute(args.func, kwargs=kwargs)
        if res.ok:
            print(res.output)
        else:
            print(f"[!] Помилка ({res.elapsed_ms:.1f} мс): {res.error}", file=sys.stderr)
            raise SystemExit(1)


if __name__ == "__main__":
    main()
