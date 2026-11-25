#!/usr/bin/env python3
"""
GVK Private AI - Persona Skill Execution Sandbox (core/skill_runner.py)

Provides safe discovery, static AST validation, trigger/slash matching, and
isolated subprocess execution of per-persona Python skill functions (`tools.py`),
typically unpacked into the runtime RAM disk.

Key Responsibilities:
  - Graceful Fallback: operate cleanly when `tools.py` is missing or empty
    (`available=False`, TUI dossier status `IDLE`).
  - Static AST Validator (`validate_tools_ast`): inspect the syntax tree prior
    to execution, whitelisting safe standard-library modules and blocking
    filesystem, network, OS, and unsafe constructs (`eval`, `exec`, `open`,
    `__globals__`, `__subclasses__`).
  - Isolated Subprocess Execution: run every skill call in a separate worker
    process with a hard timeout (default 3.0 s) and output truncation (default
    2000 chars) to prevent hangs and LLM context overflow.
  - Safe LLM Tag Parsing: parse `[SKILL: func(a=1, b="ok")]` tags via
    `ast.parse` and `ast.literal_eval` without invoking `eval()`.

Usage:
  python3 core/skill_runner.py self-test
  python3 core/skill_runner.py inspect -d path/to/persona_dir
  python3 core/skill_runner.py run -d path/to/persona_dir -f calc_split --args '{"total": 1200, "people": 3}'
  python3 core/skill_runner.py --help
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
# SANDBOX CONSTANTS & LIMITS
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
    """Raised when `tools.py` fails static AST or sandbox security validation."""


# =====================================================================
# DATA STRUCTURES
# =====================================================================
@dataclass
class SkillSpec:
    """Metadata and invocation schema for a registered persona skill."""

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
    """Result of matching a user query against a persona's available skills."""

    skill_name: str
    persona_id: str
    score: float
    matched_triggers: List[str] = field(default_factory=list)


@dataclass
class SkillCall:
    """Parsed `[SKILL: ...]` or `[TOOL: ...]` invocation extracted from LLM output."""

    name: str
    args: List[Any]
    kwargs: Dict[str, Any]
    raw_tag: str


@dataclass
class SkillResult:
    """Outcome of executing a skill inside the isolated worker subprocess."""

    ok: bool
    skill_name: str
    output: str
    error: Optional[str] = None
    elapsed_ms: float = 0.0
    truncated: bool = False


# =====================================================================
# @skill DECORATOR FOR tools.py FILES
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
    """Register a Python function in `tools.py` as an invocable persona skill.

    Injected automatically into the sandbox globals so `tools.py` can use
    `@skill(...)` without importing anything.

    Args:
        triggers: Optional natural-language trigger phrases for auto-matching.
        description: Full skill description shown to the LLM and `/skills` help.
        name: Optional override for the registered skill function name.
        short_desc: Compact one-line summary for the TUI dossier panel.
        slash: Primary slash command (e.g., `"/split"`).
        slash_aliases: Additional slash aliases.
        usage: Formatted usage hint (e.g., `"/split <total> [people]"`).

    Returns:
        Decorator attaching `_gvk_skill_meta` to the target function.
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
# LAYER 1-2: STATIC AST VALIDATOR & IMPORT SANDBOX
# =====================================================================
def _normalize_for_match(text: str) -> str:
    s = text.lower().strip()
    s = s.replace("ё", "е").replace("ґ", "г").replace("і", "и").replace("ї", "и").replace("ы", "и").replace("є", "е")
    return s


def validate_tools_ast(source_code: str, filename: str = "tools.py") -> ast.Module:
    """Validate the AST of `tools.py` before compilation and execution.

    Allows only modules in `ALLOWED_MODULES` and blocks relative imports,
    dunder attribute access, and dangerous built-in identifiers/calls.

    Args:
        source_code: Python source code of `tools.py`.
        filename: Filename label used in syntax and security error messages.

    Returns:
        Validated `ast.Module` node ready for `compile()`.

    Raises:
        SkillSecurityError: If parsing fails or any disallowed construct is detected.
    """
    try:
        tree = ast.parse(source_code, filename=filename, mode="exec")
    except SyntaxError as exc:
        raise SkillSecurityError(f"Syntax error in {filename} (line {exc.lineno}): {exc.msg}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_mod = alias.name.split(".")[0]
                if root_mod not in ALLOWED_MODULES:
                    raise SkillSecurityError(
                        f"Disallowed import '{alias.name}' (line {node.lineno}). "
                        f"Allowed modules: {', '.join(sorted(ALLOWED_MODULES))}."
                    )

        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                raise SkillSecurityError(f"Relative imports are disallowed in {filename} (line {node.lineno}).")
            mod_name = (node.module or "").split(".")[0]
            if mod_name not in ALLOWED_MODULES:
                raise SkillSecurityError(
                    f"Disallowed import from '{node.module}' (line {node.lineno})."
                )

        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                raise SkillSecurityError(
                    f"Disallowed access to dunder attribute '{node.attr}' (line {node.lineno})."
                )

        elif isinstance(node, ast.Name):
            if node.id in BLOCKED_CALL_NAMES:
                raise SkillSecurityError(
                    f"Disallowed use of restricted identifier '{node.id}' (line {node.lineno})."
                )

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_CALL_NAMES:
                raise SkillSecurityError(
                    f"Disallowed call '{node.func.id}()' in sandbox (line {node.lineno})."
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
        raise ImportError("Relative imports are disallowed in the sandbox.")
    root_mod = name.split(".")[0]
    if root_mod not in ALLOWED_MODULES:
        raise ImportError(f"Module '{name}' is disallowed in the GVK sandbox.")
    return importlib.__import__(name, globals_dict, locals_dict, fromlist, 0)


def build_sandbox_globals() -> Dict[str, Any]:
    """Build a restricted `__builtins__` globals dictionary for executing `tools.py`."""
    safe_builtins: Dict[str, Any] = {"__import__": _sandbox_import}
    for name in SAFE_BUILTIN_NAMES:
        safe_builtins[name] = getattr(_py_builtins, name)

    return {
        "__builtins__": safe_builtins,
        "__name__": "gvk_persona_tools",
        "skill": skill,
    }


def load_sandbox_namespace(tools_path: str) -> Dict[str, Any]:
    """Validate AST and execute `tools.py` inside a clean sandbox namespace.

    Args:
        tools_path: Path to the persona's `tools.py` file.

    Returns:
        Executed module namespace dictionary, or `{}` if the file is empty.
    """
    with open(tools_path, "r", encoding="utf-8") as f:
        source_code = f.read()

    if not source_code.strip():
        return {}

    tree = validate_tools_ast(source_code, filename=os.path.basename(tools_path))
    code_obj = compile(tree, filename=os.path.basename(tools_path), mode="exec")
    namespace = build_sandbox_globals()
    exec(code_obj, namespace, namespace)  # noqa: S102 - source validated by AST filter and sandboxed
    return namespace


# =====================================================================
# LLM SKILL-CALL TAG PARSER ([SKILL: func(a=1, b="x")])
# =====================================================================
def parse_skill_call(text: str) -> Optional[SkillCall]:
    """Safely parse the first `[SKILL: func(...)]` or `[TOOL: ...]` tag in LLM text.

    Uses `ast.parse` and `ast.literal_eval` to extract literal arguments without
    evaluating arbitrary Python expressions.

    Args:
        text: Raw assistant response text.

    Returns:
        Parsed `SkillCall` instance, or `None` if no valid tag is present.
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
                return None  # Block **kwargs unpacking in LLM-emitted calls
            kwargs[kw_node.arg] = ast.literal_eval(kw_node.value)
    except (ValueError, SyntaxError, TypeError):
        return None

    return SkillCall(name=func_name, args=args, kwargs=kwargs, raw_tag=match.group(0))


def strip_skill_tags(text: str) -> str:
    """Remove `[SKILL: ...]` and `[TOOL: ...]` service tags before rendering in the TUI."""
    if not text:
        return ""
    cleaned = _SKILL_TAG_RE.sub("", text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


# =====================================================================
# LAYER 3: SKILL RUNNER MANAGER
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
    """Manage discovery, matching, and sandboxed execution of a persona's skills.

    Accepts either a persona directory (typically on the RAM disk) or a direct
    path to `tools.py`. If `tools.py` is missing or empty, operates safely with
    `available=False`.

    Example:
        >>> runner = SkillRunner("personas/friend_1", persona_id="friend_1")
        >>> res = runner.execute("calc_split", kwargs={"total": 1200, "people": 3})
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
            # Only register functions defined directly inside `tools.py`
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
        """Return all registered `SkillSpec` definitions for this persona."""
        return list(self.skills.values())

    def get_dossier_status(self) -> str:
        """Return a compact localized skills status string for the TUI dossier panel."""
        if not self.available or not self.skills:
            return "немає активних скілів [IDLE]"
        names = list(self.skills.keys())
        if len(names) == 1:
            return f"tools.{names[0]}() [OK]"
        return f"активних: {len(names)} [OK]"

    def get_dossier_skills_lines(self) -> List[str]:
        """Return single-line skill summaries formatted as `<command> - <description>`."""
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
        """Return True if `user_text` is a bare skills help command (`/skills`, `/скіли`, `/help`)."""
        if not user_text:
            return False
        m = _SLASH_CMD_RE.match(user_text.strip())
        if not m:
            return False
        cmd_norm = _normalize_for_match(m.group(1))
        rest = (m.group(2) or m.group(3) or "").strip()
        return cmd_norm in _SLASH_HELP_COMMANDS and not rest

    def format_slash_help(self) -> str:
        """Format the localized list of available slash commands for the chat UI."""
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
        """Extract typed skill keyword arguments from trailing slash-command or natural-language text."""
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
        """Check whether `user_text` starts with a registered skill slash command.

        Args:
            user_text: Raw user chat message (e.g., `"/split 900 3"` or `"/ping"`).

        Returns:
            Tuple `(spec, arg_text)` on match, or `None`.
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
        """Execute a skill matched via an explicit user slash command (`/cmd [args]`)."""
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
        """Auto-execute a skill when `user_text` is a slash command or strongly matches a trigger.

        Args:
            user_text: Incoming user chat utterance.
            threshold: Minimum trigger-match score required for automatic execution.

        Returns:
            `SkillResult` if a skill was invoked, or `None`.
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
        """Score whether a user query matches a skill's slash command, triggers, or description.

        Used by the arbiter in `core/orchestrator.py` during the Skill Match stage.

        Args:
            query: Incoming user message.
            threshold: Minimum score required to report a match.

        Returns:
            Highest-scoring `SkillMatch` above `threshold`, or `None`.
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
        """Execute a skill in an isolated worker subprocess with timeout and output limits.

        Args:
            skill_name: Registered skill name to invoke.
            args: Optional positional arguments passed to the skill function.
            kwargs: Optional keyword arguments passed to the skill function.

        Returns:
            `SkillResult` with status, truncated output, error message, and latency in ms.
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
        """Parse and execute a `[SKILL: ...]` tag from `llm_text`, returning `SkillResult` or `None`."""
        call = parse_skill_call(llm_text)
        if call is None:
            return None
        return self.execute(call.name, args=call.args, kwargs=call.kwargs)

    def format_skills_for_prompt(self) -> str:
        """Format available skills as a localized prompt instruction block for the LLM."""
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
        """Format a skill execution result block for the orchestrator's follow-up LLM turn."""
        if result.ok:
            return f"[РЕЗУЛЬТАТ СКІЛА {result.skill_name}]:\n{result.output}"
        return f"[ПОМИЛКА СКІЛА {result.skill_name}]: {result.error or 'невідома помилка'}"


# =====================================================================
# INTERNAL SANDBOX WORKER (ISOLATED SUBPROCESS)
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
# STANDALONE SELF-TEST (SANDBOX, TIMEOUT & FALLBACK)
# =====================================================================
def run_self_test() -> None:
    """Run the standalone self-test suite for `SkillRunner` and the AST sandbox."""
    print("====================================================================")
    print(" [*] GVK Private AI - Skill Sandbox Self-Test (core/skill_runner.py)")
    print("====================================================================")

    with tempfile.TemporaryDirectory(prefix="gvk_skill_test_") as tmpdir:
        # 1. Missing tools.py mode (graceful fallback)
        print(" [1/6] Checking missing tools.py mode (Graceful Fallback)...")
        empty_dir = os.path.join(tmpdir, "empty_persona")
        os.makedirs(empty_dir, exist_ok=True)
        idle_runner = SkillRunner(empty_dir, persona_id="friend_1")
        assert idle_runner.available is False
        assert idle_runner.list_skills() == []
        assert idle_runner.match_query("порахуй чек на трьох") is None
        assert idle_runner.format_skills_for_prompt() == ""
        assert "IDLE" in idle_runner.get_dossier_status()
        print(f"       [OK] Graceful fallback works without tools.py: {idle_runner.get_dossier_status()}")

        # 2. Safe test skills
        print(" [2/6] Checking loading and execution of safe skills...")
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
        assert res_calc.ok and "400.0" in res_calc.output, f"Unexpected calc_split result: {res_calc}"
        res_code = runner.execute("analyze_code", args=["a = 1\nb = 2\nc = 3\nd = 4"])
        assert res_code.ok and '"non_empty_lines": 4' in res_code.output
        print(f"       [OK] calc_split -> {res_calc.output}")

        # 3. Skill Match, slash commands, and [SKILL: ...] tag parser
        print(" [3/6] Checking Skill Match, slash commands, and LLM tag parser...")
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
        print(f"       [OK] Skill Match (score={matched.score}), /чек, /split, and tags work.")

        # 4. Blocking unsafe code via the AST validator
        print(" [4/6] Checking AST validator blocks unsafe code...")
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
            assert blocked, f"Malicious sample #{idx} was not blocked!"
        print(f"       [OK] All {len(malicious_samples)} unsafe constructs were blocked before execution.")

        # 5. Protection against infinite loops (timeout)
        print(" [5/6] Checking timeout enforcement on an infinite while True loop...")
        res_timeout = runner.execute("infinite_loop")
        assert res_timeout.ok is False and res_timeout.error and "ліміт часу" in res_timeout.error
        print(f"       [OK] Hung worker process terminated ({res_timeout.elapsed_ms:.0f} ms).")

        # 6. Exception isolation and long-output truncation
        print(" [6/6] Checking exception isolation and output truncation...")
        res_crash = runner.execute("crash_div")
        assert res_crash.ok is False and res_crash.error and "ZeroDivisionError" in res_crash.error
        res_spam = runner.execute("verbose_spam")
        assert res_spam.ok and res_spam.truncated is True and len(res_spam.output) <= 300
        print("       [OK] ZeroDivisionError isolated; verbose output truncated to 300 chars.")

    print("====================================================================")
    print(" [OK] ALL SKILL_RUNNER SELF-TESTS PASSED SUCCESSFULLY!")
    print("====================================================================")


# =====================================================================
# CLI INTERFACE
# =====================================================================
def main(argv: Optional[List[str]] = None) -> None:
    """Parse CLI arguments and run skill inspection, execution, or self-test."""
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv and raw_argv[0] == "_worker":
        raise SystemExit(_run_worker_from_stdin())

    parser = argparse.ArgumentParser(
        description="GVK Private AI - Persona Skill Execution Sandbox (core/skill_runner.py)"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("self-test", help="Run full sandbox, AST filter, and timeout self-tests")

    p_insp = sub.add_parser("inspect", help="List available skills in a persona directory or tools.py")
    p_insp.add_argument("-d", "--source", required=True, help="Path to a persona directory or tools.py")

    p_run = sub.add_parser("run", help="Safely execute a skill inside the sandbox")
    p_run.add_argument("-d", "--source", required=True, help="Path to a persona directory or tools.py")
    p_run.add_argument("-f", "--func", required=True, help="Skill function name to invoke")
    p_run.add_argument("--args", default="{}", help="JSON object of keyword arguments")
    p_run.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SEC, help="Execution timeout in seconds")

    args = parser.parse_args(raw_argv)

    if args.cmd == "self-test":
        run_self_test()
    elif args.cmd == "inspect":
        runner = SkillRunner(args.source)
        print(f"Status: {runner.get_dossier_status()}")
        for spec in runner.list_skills():
            tr = ", ".join(spec.triggers) if spec.triggers else "-"
            print(f" - {spec.name}{spec.signature_str}: {spec.description} [triggers: {tr}]")
    elif args.cmd == "run":
        runner = SkillRunner(args.source, timeout_sec=args.timeout)
        kwargs = json.loads(args.args) if args.args else {}
        res = runner.execute(args.func, kwargs=kwargs)
        if res.ok:
            print(res.output)
        else:
            print(f"[!] Error ({res.elapsed_ms:.1f} ms): {res.error}", file=sys.stderr)
            raise SystemExit(1)


if __name__ == "__main__":
    main()
