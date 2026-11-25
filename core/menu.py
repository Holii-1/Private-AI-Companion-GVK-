#!/usr/bin/env python3
"""
GVK Private AI — Термінальний інтерфейс (core/menu.py)

Автономне вікно 1280x720 (720p, без зміни розміру) у стилі класичного терміналу:
  - Справжній чорний фон (#000000), єдиний моноширинний шрифт одного розміру,
    міжрядковий інтервал реального терміналу та вирівнювання по лівому краю.
  - Приглушена термінальна палітра (Cyan, Amber, Green, Purple, Red, Muted Grey).
  - Фіксовані позиції блоків без вертикальних стрибків при переносі рядків,
    відкритті `sudo` (`[17]`) або виклику підтвердження виходу `CTRL+C`.
  - Підтримка вводу секретного коду `sudo` на EN / DE / UA (`ігвщ`) / RU (`ыгвщ`) розкладках.

Приклади запуску:
  python3 core/menu.py
  python3 core/menu.py --fast
  python3 core/menu.py self-test
"""

import argparse
import glob
import json
import logging
import math
import os
import random
import re
import sys
import threading
import time

# Автоматичне підхоплення Tcl/Tk для портативного python-build-standalone (.venv із ~/.gvk_ai/python)
if "TCL_LIBRARY" not in os.environ or "TK_LIBRARY" not in os.environ:
    _base_pref = getattr(sys, "base_prefix", sys.prefix)
    for _sub in ("tcl", "lib"):
        _tcl_cand = os.path.join(_base_pref, _sub, "tcl8.6")
        _tk_cand = os.path.join(_base_pref, _sub, "tk8.6")
        if os.path.isdir(_tcl_cand) and "TCL_LIBRARY" not in os.environ:
            os.environ["TCL_LIBRARY"] = _tcl_cand
        if os.path.isdir(_tk_cand) and "TK_LIBRARY" not in os.environ:
            os.environ["TK_LIBRARY"] = _tk_cand

import tkinter as tk
from tkinter import font as tkfont
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("gvk.menu")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.roster import load_roster_data

# =====================================================================
# КОНСТАНТИ ВІКНА ТА ПРИГЛУШЕНОЇ ПАЛІТРИ ТЕРМІНАЛУ
# =====================================================================
APP_NAME = "GVK PRIVATE AI"
APP_HEADER = "GVK PRIVATE AI [Release 1.1.1 - Optical Funeral]"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
DEFAULT_WAIT_SECONDS = 15
SUDO_LECTURE_WAIT_SECONDS = 4
LAST_WARNING_WAIT_SECONDS = 60
DEFAULT_AFK_SECONDS = 20.0
DEFAULT_SMILE_SECONDS = 4.0
TALKING_FRAME_MS = 200
LINE_REVEAL_DELAY_MS = 600
LEFT_PAD = 16

BG_BLACK = "#000000"
TERM_WHITE = "#cccccc"
TERM_CYAN = "#5fafb8"
TERM_GREEN = "#78b068"
TERM_AMBER = "#c89d54"
TERM_PURPLE = "#9d7bb8"
TERM_RED = "#c75656"
TERM_MUTED = "#666666"

# Мапінг фізичних клавіш S-U-D-O для EN / DE / UA (ігвщ) / RU (ыгвщ)
SUDO_CHAR_MAP: Dict[str, str] = {
    "s": "s",
    "і": "s",
    "ы": "s",
    "u": "u",
    "г": "u",
    "d": "d",
    "в": "d",
    "o": "o",
    "щ": "o",
}
SUDO_KEYCODE_MAP: Dict[int, str] = {
    83: "s",  # VK_S
    85: "u",  # VK_U
    68: "d",  # VK_D
    79: "o",  # VK_O
}

# =====================================================================
# СЛОВНИК КОРИСТУВАЧІВ ТА ТЕКСТИ ЕКРАНІВ (ФОЛБЕК + ПРИВАТНИЙ ROSTER.JSON)
# =====================================================================
USERS_DICT: Dict[int, Dict[str, object]] = {
    1: {"name": "Учасник 01", "locked": True, "experimental": False},
    2: {"name": "Учасник 02", "locked": False, "experimental": False},
    3: {"name": "Учасник 03", "locked": False, "experimental": False},
    4: {"name": "Учасник 04", "locked": False, "experimental": False},
    5: {"name": "Учасник 05", "locked": False, "experimental": False},
    6: {"name": "Учасник 06", "locked": False, "experimental": False},
    7: {"name": "Учасник 07", "locked": False, "experimental": True},
    8: {"name": "Учасник 08", "locked": False, "experimental": True},
    9: {"name": "Учасник 09", "locked": False, "experimental": True},
    10: {"name": "Учасник 10", "locked": False, "experimental": True},
    11: {"name": "Учасник 11", "locked": False, "experimental": True},
    12: {"name": "Учасник 12", "locked": False, "experimental": True},
    13: {"name": "Учасник 13", "locked": False, "experimental": True},
    14: {"name": "Учасник 14", "locked": False, "experimental": True},
    15: {"name": "Учасник 15", "locked": False, "experimental": True},
    16: {"name": "Учасник 16", "locked": False, "experimental": True},
    17: {"name": "Введіть самостійно (російською):", "locked": False, "experimental": True, "custom": True},
}

MAIN_MENU_ITEMS: List[Dict[str, object]] = [
    {"id": 1, "label": "Написати комусь", "available": True},
    {"id": 2, "label": "Зібратися утрьох (недоступно)", "available": False},
    {"id": 3, "label": "Посидіти з вибраними людьми (недоступно)", "available": False},
    {"id": 4, "label": "Залетіти на спотік з усіма (недоступно)", "available": False},
]

PERSONA_SELECT_ITEMS: List[Dict[str, object]] = [
    {
        "id": 1,
        "persona_id": "friend_1",
        "label": "Друг 1",
        "genitive_name": "Друга 1",
        "loading_tips": ["loading tip 1", "loading tip 2", "loading tip 3"],
        "available": True,
        "color": TERM_GREEN,
    },
    {
        "id": 2,
        "persona_id": "friend_2",
        "label": "Друг 2 (недоступно)",
        "genitive_name": "Друга 2",
        "loading_tips": ["loading tip 1", "loading tip 2", "loading tip 3"],
        "available": False,
        "color": TERM_CYAN,
    },
    {
        "id": 3,
        "persona_id": "friend_3",
        "label": "Друг 3 (недоступно)",
        "genitive_name": "Друга 3",
        "loading_tips": ["loading tip 1", "loading tip 2", "loading tip 3"],
        "available": False,
        "color": TERM_AMBER,
    },
]

SPLASH_MESSAGE = (
    "© 2026 Розробники GVK Private AI.\n"
    "Усі майнові та немайнові авторські права захищено. \n"
    "Найменування «GVK»™ є торговою маркою та виключною інтелектуальною власністю розробників.\n\n"
    "ПОПЕРЕДЖЕННЯ ПРО РОБОТУ АЛГОРИТМУ:\n"
    "Програмний комплекс GVK функціонує на базі нейромережевих алгоритмів штучного інтелекту. "
    "Модель схильна до стохастичної генерації та алгоритмічних хиб (галюцинацій). "
    "Згенеровані судження, висновки чи описи, зокрема ті, що стосуються реальних фізичних осіб, "
    "є виключно машинною імітацією та можуть не відповідати дійсності. "
    'Система надається на умовах "як є" (as is).'
)

WARNING_SCREENS: List[Tuple[str, str]] = [
    (
        "САНІТАРНО-ЕПІДЕМІОЛОГІЧНЕ ЗАСТЕРЕЖЕННЯ",
        "Тривала взаємодія з моделлю може викликати когнітивний дисонанс, неконтрольовані напади сміху, "
        "фантомні спогади спільних чатів та гостру соціальну дезадаптацію. Міністерство охорони здоров'я "
        "України попереджає: надмірне споживання контенту згенерованого «GVK» шкодить вашому ментальному здоров'ю.",
    ),
    (
        "ОБМЕЖЕННЯ ЦІЛЬОВОГО ВИКОРИСТАННЯ",
        "Програмний комплекс GVK не розроблений, не призначений і не ліцензований для використання в умовах, "
        "що вимагають безвідмовної роботи, зокрема: під час управління об'єктами критичної інфраструктури, "
        "на атомних електростанціях, у системах управління повітряним рухом, системах життєзабезпечення, "
        "а також для прийняття будь-яких стратегічних рішень у приватній групі.",
    ),
    (
        "ЕКСПОРТНИЙ КОНТРОЛЬ ТА ОБМЕЖЕННЯ ОБІГУ",
        "Даний фізичний носій (оптичний диск або твердотільний USB-накопичувач) містить алгоритми нелінійної оптимізації та нейромережеві моделі, "
        "що класифікуються як технології подвійного призначення. Несанкціонований експорт, реекспорт або "
        "передача цього програмного забезпечення на території держав, що перебувають під санкціями ООН "
        "або РНБО України, суворо заборонені.",
    ),
    (
        "ОФІЦІЙНЕ ЗАСТЕРЕЖЕННЯ ТА ДЕКЛАРАЦІЯ ВІДПОВІДАЛЬНОСТІ",
        "ЦИМ ДОКУМЕНТОМ ОФІЦІЙНО ПОВІДОМЛЯЄМО: \n"
        "Раду національної безпеки і оборони України, Офіс Президента України, Службу безпеки України, "
        "Головне управління розвідки Міністерства оборони України, Міністерство внутрішніх справ України, "
        "Департамент кіберполіції Національної поліції України, Державне бюро розслідувань, Офіс Генерального "
        "прокурора, Державну службу спеціального зв'язку та захисту інформації України, а також усі інші "
        "уповноважені органи державної влади, що здійснюють діяльність у сфері забезпечення національної "
        "та інформаційної безпеки України на її суверенній території та поза її межами.\n\n"
        "Декларуємо, що будь-які згенеровані даним локальним програмно-апаратним комплексом (моделлю штучного "
        "інтелекту) висловлювання, текстові масиви чи семантичні конструкції, які можуть бути ідентифіковані "
        "або витлумачені як:\n"
        "– антидержавницькі, антиконституційні чи антидемократичні;\n"
        "– такі, що прямо чи опосередковано посягають на конституційний лад, державний суверенітет, "
        "незалежність, унітарний устрій та територіальну цілісність України в межах її міжнародно визнаних "
        "державних кордонів;\n"
        "– такі, що містять ознаки загрози національній безпеці, інформаційному простору або недоторканності держави,\n\n"
        "...Є ВИКЛЮЧНО критичною системною помилкою, випадковим алгоритмічним артефактом машинного навчання "
        "та результатом неконтрольованої стохастичної генерації (галюцинацією нейромережі), що виникла на "
        "етапі обробки неструктурованих історичних масивів даних.\n\n"
        "Усвідомлюючи відповідальність перед Богом, власною совістю, попередніми, нинішнім та прийдешніми "
        "поколіннями, розробники категорично та публічно відмежовуються від змісту вищезазначених генерацій. "
        "Дана нейромережева архітектура була спроєктована, локально розгорнута та передана виключно в "
        "ознайомчих цілях, у повній, беззастережній та суворій відповідності до Конституції України та "
        "чинного законодавства України.",
    ),
]


def load_roster_config(
    ram_root: Optional[str] = None,
    repo_root: str = REPO_ROOT,
    gvk_home: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Завантажує єдиний приватний `roster.json` через `core.roster.load_roster_data`
    (з RAM-диска, локального `personas/roster.json` або `~/.gvk_ai`).
    """
    return load_roster_data(
        ram_root=ram_root,
        repo_root=repo_root,
        gvk_home=gvk_home,
    )


def apply_roster_config(roster: Dict[str, Any]) -> None:
    """Накладає дані з `roster.json` поверх фолбек-заглушок модуля `core.menu`."""
    global SPLASH_MESSAGE
    if not roster:
        return

    raw_users = roster.get("users_dict")
    if isinstance(raw_users, dict) and raw_users:
        parsed_users: Dict[int, Dict[str, object]] = {}
        for k, v in raw_users.items():
            if isinstance(v, dict):
                try:
                    parsed_users[int(k)] = dict(v)
                except (TypeError, ValueError):
                    continue
        if parsed_users:
            USERS_DICT.clear()
            USERS_DICT.update(parsed_users)

    raw_personas = roster.get("persona_select_items")
    if isinstance(raw_personas, list) and raw_personas:
        PERSONA_SELECT_ITEMS[:] = [dict(item) for item in raw_personas if isinstance(item, dict)]

    raw_splash = roster.get("splash_message")
    if isinstance(raw_splash, str) and raw_splash.strip():
        SPLASH_MESSAGE = raw_splash

    raw_warnings = roster.get("warning_screens")
    if isinstance(raw_warnings, list) and raw_warnings:
        parsed_w = [
            (str(pair[0]), str(pair[1]))
            for pair in raw_warnings
            if isinstance(pair, (list, tuple)) and len(pair) >= 2
        ]
        if parsed_w:
            WARNING_SCREENS[:] = parsed_w


apply_roster_config(load_roster_config())

SUDO_LECTURE_MESSAGE = (
    "Сподіваємось, ви отримали звичну лекцію від місцевого системного адміністратора. \n"
    "Зазвичай усе зводиться до цих трьох правил:\n\n"
    "1) Поважайте приватність інших.\n"
    "2) Думайте, перш ніж друкувати.\n"
    "3) З великою владою приходить велика відповідальність."
)

LAUNCH_INSTRUCTION = (
    "Щоб продовжити, ви повинні підтвердити свою згоду з усіма вищезазначеними умовами.\n"
    "Для запуску алгоритму введіть наступну фразу (з урахуванням регістру та розділових знаків):"
)
LAUNCH_PHRASE_DISPLAY = (
    "«Я, перебуваючи у здоровому глузді та світлій пам'яті, добровільно беру на себе "
    "всі юридичні, моральні та кармічні наслідки взаємодії з даним алгоритмом.»"
)
LAUNCH_PHRASE_CORE = (
    "Я, перебуваючи у здоровому глузді та світлій пам'яті, добровільно беру на себе "
    "всі юридичні, моральні та кармічні наслідки взаємодії з даним алгоритмом."
)


def _normalize_apostrophes(text: str) -> str:
    """Уніфікує типографські апострофи до стандартного `'`, зберігаючи регістр і розділові знаки."""
    return text.replace("’", "'").replace("ʼ", "'").replace("`", "'")


def is_valid_launch_phrase(typed: str) -> bool:
    """Перевіряє точний посимвольний збіг фрази згоди (з лапками «» або без них)."""
    norm = _normalize_apostrophes(typed)
    return norm in (
        _normalize_apostrophes(LAUNCH_PHRASE_DISPLAY),
        _normalize_apostrophes(LAUNCH_PHRASE_CORE),
    )


def translate_sudo_key(event: tk.Event) -> Optional[str]:
    """
    Перетворює натиснуту клавішу в латинський символ з набору {'s', 'u', 'd', 'o'}
    незалежно від розкладки (English / German / Ukrainian / Russian).
    """
    ch = (event.char or "").lower()
    if ch in SUDO_CHAR_MAP:
        return SUDO_CHAR_MAP[ch]
    keycode = getattr(event, "keycode", 0)
    if keycode in SUDO_KEYCODE_MAP:
        return SUDO_KEYCODE_MAP[keycode]
    return None


def _has_non_text_modifier(event: tk.Event) -> bool:
    """Перевіряє, чи натиснуто Control/Command/Alt (ігнорує NumLock 0x8 на Windows)."""
    state = int(getattr(event, "state", 0) or 0)
    if sys.platform.startswith("win"):
        # На Windows 0x8 — це NumLock; 0x20000 — Alt; AltGr (0x4|0x20000) може друкувати символи
        is_ctrl = bool(state & 0x4)
        is_alt = bool(state & 0x20000)
        return is_ctrl != is_alt
    if sys.platform == "darwin":
        return bool(state & 0x4) or bool(state & 0x8) or bool(state & 0x10) or bool(state & 0x40000)
    return bool(state & 0x4) or bool(state & 0x8)


def _is_paste_event(event: tk.Event) -> bool:
    """Розпізнає Ctrl+V / Cmd+V незалежно від розкладки (EN / UA / RU / DE)."""
    state = int(getattr(event, "state", 0) or 0)
    if sys.platform.startswith("win"):
        has_mod = bool(state & 0x4) and not bool(state & 0x20000)
    elif sys.platform == "darwin":
        has_mod = bool(state & 0x4) or bool(state & 0x8) or bool(state & 0x10) or bool(state & 0x40000)
    else:
        has_mod = bool(state & 0x4)
    if not has_mod:
        return False
    ks = (getattr(event, "keysym", "") or "").lower()
    if ks in ("v", "cyrillic_em"):
        return True
    ch = getattr(event, "char", "") or ""
    if ch == "\x16":
        return True
    return sys.platform.startswith("win") and getattr(event, "keycode", 0) == 86


def _read_clipboard_text(widget: tk.Widget, multiline: bool = True) -> str:
    try:
        raw = widget.clipboard_get()
    except tk.TclError:
        return ""
    if not isinstance(raw, str):
        return ""
    cleaned = raw.replace("\r\n", "\n").replace("\r", "\n")
    if not multiline:
        cleaned = cleaned.replace("\n", " ").strip("\n")
    return cleaned


def make_term_label(
    parent: tk.Widget,
    font: tkfont.Font,
    text: str = "",
    fg: str = TERM_WHITE,
    bg: str = BG_BLACK,
    wraplength: int = 0,
    justify: str = tk.LEFT,
    anchor: str = "w",
) -> tk.Label:
    """Створює текстовий рядок зі строгим міжрядковим інтервалом реального терміналу (pady=0, bd=0)."""
    return tk.Label(
        parent,
        text=text,
        font=font,
        fg=fg,
        bg=bg,
        bd=0,
        padx=0,
        pady=0,
        highlightthickness=0,
        justify=justify,
        anchor=anchor,
        wraplength=wraplength,
    )


# =====================================================================
# ПЕРЕВИКОРИСТОВУВАНІ МОДАЛЬНІ КОМПОНЕНТИ
# =====================================================================
class TimedMessageModal(tk.Frame):
    """
    Перевикористовуваний модальний компонент для показу повідомлення або попередження:
      - Заголовок попередження розташований на один рядок вище самого повідомлення.
      - У рядку над футером виходу показується зворотний відлік `20s...` (або `40s...`),
        а після завершення — `Натисніть ENTER щоб продовжити`.
    """

    def __init__(
        self,
        parent: tk.Widget,
        body_text: str,
        wait_seconds: float,
        on_complete: Callable[[], None],
        term_font: tkfont.Font,
        warning_title: Optional[str] = None,
        title_color: str = TERM_AMBER,
        body_color: str = TERM_WHITE,
        require_enter: bool = True,
        on_status: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        super().__init__(parent, bg=BG_BLACK)
        self.wait_seconds = max(0.0, float(wait_seconds))
        self.on_complete = on_complete
        self.require_enter = require_enter
        self.on_status = on_status
        self.ready_for_enter = False
        self._timer_id: Optional[str] = None

        # Нижній рядок (тільки якщо не передано зовнішній on_status у футері)
        self.bottom_area = tk.Frame(self, bg=BG_BLACK)
        self.prompt_label = make_term_label(
            self.bottom_area,
            font=term_font,
            text="",
            fg=TERM_MUTED,
        )
        if self.on_status is None:
            self.bottom_area.pack(side=tk.BOTTOM, fill=tk.X)
            self.prompt_label.pack(anchor="w")

        # Центральний блок (посередині відносно висоти, текст зліва)
        self.center_area = tk.Frame(self, bg=BG_BLACK)
        self.center_area.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        block_frame = tk.Frame(self.center_area, bg=BG_BLACK)
        block_frame.place(x=0, rely=0.5, anchor="w")

        wrap_w = WINDOW_WIDTH - (LEFT_PAD * 2)
        if warning_title:
            # Заголовок рівно на один порожній рядок вище за саме повідомлення
            make_term_label(
                block_frame,
                font=term_font,
                text=warning_title + "\n",
                fg=title_color,
                wraplength=wrap_w,
            ).pack(anchor="w")

        make_term_label(
            block_frame,
            font=term_font,
            text=body_text,
            fg=body_color,
            wraplength=wrap_w,
        ).pack(anchor="w")

        # Якщо час >= 1 с, ведемо посекундний відлік "20s...", інакше швидкий таймер (для тестів)
        if self.wait_seconds >= 1.0:
            self._remaining_sec = int(math.ceil(self.wait_seconds))
            self._tick_countdown()
        else:
            ms = max(1, int(self.wait_seconds * 1000))
            self._set_status("1s...", TERM_MUTED)
            self._timer_id = self.after(ms, self._on_timer_done)

    def _set_status(self, text: str, fg: str = TERM_MUTED) -> None:
        self.prompt_label.config(text=text, fg=fg)
        if self.on_status is not None:
            self.on_status(text, fg)

    def _tick_countdown(self) -> None:
        if self._remaining_sec <= 0:
            self._on_timer_done()
            return
        self._set_status(f"{self._remaining_sec}s...", TERM_MUTED)
        self._remaining_sec -= 1
        self._timer_id = self.after(1000, self._tick_countdown)

    def _on_timer_done(self) -> None:
        self._timer_id = None
        if not self.require_enter:
            self.on_complete()
            return
        self.ready_for_enter = True
        self._set_status("Натисніть ENTER щоб продовжити", TERM_GREEN)

    def handle_key(self, event: tk.Event) -> None:
        if self.ready_for_enter and event.keysym in ("Return", "KP_Enter"):
            self.on_complete()

    def destroy(self) -> None:
        if self._timer_id is not None:
            try:
                self.after_cancel(self._timer_id)
            except Exception:
                pass
            self._timer_id = None
        super().destroy()


class TypingConfirmationModal(tk.Frame):
    """
    Перевикористовуваний модальний компонент для посимвольного вводу тексту
    з миготливим підкресленням `_` та перевіркою точної відповідності.
    Фіксована верхня точка прив'язки (`anchor="nw"`), щоб при переносі рядків текст не стрибав.
    """

    def __init__(
        self,
        parent: tk.Widget,
        instruction_text: str,
        target_display: str,
        validator: Callable[[str], bool],
        on_success: Callable[[str], None],
        term_font: tkfont.Font,
        on_status: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        super().__init__(parent, bg=BG_BLACK)
        self.validator = validator
        self.on_success = on_success
        self.on_status = on_status
        self.term_font = term_font
        self.typed_text = ""
        self.cursor_visible = True
        self._blink_id: Optional[str] = None

        wrap_w = WINDOW_WIDTH - (LEFT_PAD * 2)

        # Нижній рядок (якщо не передано зовнішній on_status у футері)
        self.status_label = make_term_label(
            self,
            font=term_font,
            text="",
            fg=TERM_MUTED,
        )
        if self.on_status is None:
            self.status_label.pack(side=tk.BOTTOM, anchor="w")

        # Центральна зона з фіксованою точкою прив'язки зверху-зліва (anchor="nw"),
        # щоб перенос введеного тексту на новий рядок ніколи не зсував блок вгору
        center_area = tk.Frame(self, bg=BG_BLACK)
        center_area.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        inner = tk.Frame(center_area, bg=BG_BLACK)
        inner.place(x=0, rely=0.38, anchor="nw")

        make_term_label(
            inner,
            font=term_font,
            text=instruction_text + "\n",
            fg=TERM_WHITE,
            wraplength=wrap_w,
        ).pack(anchor="w")

        make_term_label(
            inner,
            font=term_font,
            text=target_display + "\n",
            fg=TERM_AMBER,
            wraplength=wrap_w,
        ).pack(anchor="w")

        self.input_label = make_term_label(
            inner,
            font=term_font,
            text="> _",
            fg=TERM_WHITE,
            wraplength=wrap_w,
        )
        self.input_label.pack(anchor="w")

        self._schedule_blink()

    def _schedule_blink(self) -> None:
        self.cursor_visible = not self.cursor_visible
        self._refresh_display()
        self._blink_id = self.after(500, self._schedule_blink)

    def _is_prefix_matching(self) -> bool:
        norm_typed = _normalize_apostrophes(self.typed_text)
        norm_disp = _normalize_apostrophes(LAUNCH_PHRASE_DISPLAY)
        norm_core = _normalize_apostrophes(LAUNCH_PHRASE_CORE)
        return norm_disp.startswith(norm_typed) or norm_core.startswith(norm_typed)

    def _set_status(self, text: str, fg: str) -> None:
        self.status_label.config(text=text, fg=fg)
        if self.on_status is not None:
            self.on_status(text, fg)

    def _refresh_display(self) -> None:
        cursor = "_" if self.cursor_visible else " "
        self.input_label.config(text=f"> {self.typed_text}{cursor}")

        if not self.typed_text:
            self.input_label.config(fg=TERM_WHITE)
            self._set_status("", TERM_MUTED)
            return

        if self.validator(self.typed_text):
            self.input_label.config(fg=TERM_GREEN)
            self._set_status("Натисніть ENTER щоб продовжити", TERM_GREEN)
        elif self._is_prefix_matching():
            self.input_label.config(fg=TERM_WHITE)
            self._set_status("", TERM_MUTED)
        else:
            self.input_label.config(fg=TERM_RED)
            self._set_status("Текст не збігається зі зразком!", TERM_RED)

    def handle_key(self, event: tk.Event) -> None:
        if event.keysym in ("Return", "KP_Enter"):
            if self.validator(self.typed_text):
                self.on_success(self.typed_text)
            else:
                self._set_status(
                    "Введіть фразу точно за зразком (з урахуванням регістру та розділових знаків).",
                    TERM_RED,
                )
            return

        if event.keysym == "BackSpace":
            self.typed_text = self.typed_text[:-1]
            self._refresh_display()
            return

        if _is_paste_event(event):
            pasted = _read_clipboard_text(self, multiline=False)
            if pasted:
                self.typed_text += pasted
                self._refresh_display()
            return

        if event.char and event.char.isprintable() and not _has_non_text_modifier(event):
            self.typed_text += event.char
            self._refresh_display()

    def handle_paste(self) -> None:
        pasted = _read_clipboard_text(self, multiline=False)
        if pasted:
            self.typed_text += pasted
            self._refresh_display()

    def destroy(self) -> None:
        if self._blink_id is not None:
            try:
                self.after_cancel(self._blink_id)
            except Exception:
                pass
            self._blink_id = None
        super().destroy()


class UserSelectorModal(tk.Frame):
    """
    Модальний селектор користувача зі словника `users_dict`:
      - Фіксована верхня точка прив'язки (`anchor="nw"`), щоб додавання `[17]` не зсувало список.
      - Перша колонка строго зліва, друга — посередині, третя — справа, `[17]` — під другою колонкою.
      - Стрілка Вниз на `[12]` або `[16]` переходить на `[17]` (коли `sudo` активне);
        на `[17]`: Вгору -> `[12]`, Вліво -> `[06]`, Вправо -> `[16]`.
    """

    def __init__(
        self,
        parent: tk.Widget,
        users_dict: Dict[int, Dict[str, object]],
        on_select: Callable[[int, str, bool], None],
        term_font: tkfont.Font,
        initial_sudo: bool = False,
        on_status: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        super().__init__(parent, bg=BG_BLACK)
        self.users_dict = users_dict
        self.on_select = on_select
        self.on_status = on_status
        self.term_font = term_font

        self.sudo_unlocked = initial_sudo
        self.selected_id = 2
        self.custom_text = ""
        self.secret_buffer = ""
        self.digit_buffer = ""
        self._digit_timer_id: Optional[str] = None
        self._blink_id: Optional[str] = None
        self.cursor_visible = True

        self.item_labels: Dict[int, tk.Label] = {}

        # Статус-рядок (якщо не передано зовнішній on_status у футері)
        self.status_label = make_term_label(
            self,
            font=term_font,
            text="",
            fg=TERM_RED,
        )
        if self.on_status is None:
            self.status_label.pack(side=tk.BOTTOM, anchor="w")

        # Центральна область із фіксованою верхньою точкою (anchor="nw"),
        # щоб поява [17] розширювала список лише донизу без зсуву [01]..[06]
        self.center_area = tk.Frame(self, bg=BG_BLACK)
        self.center_area.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.content_frame = tk.Frame(self.center_area, bg=BG_BLACK)
        self.content_frame.place(x=0, rely=0.38, relwidth=1.0, anchor="nw")

        self._build_selector_ui()
        self._schedule_blink()

    def _max_selectable_id(self) -> int:
        return 17 if self.sudo_unlocked else 6

    def _build_selector_ui(self) -> None:
        for child in self.content_frame.winfo_children():
            child.destroy()
        self.item_labels.clear()

        grid_frame = tk.Frame(self.content_frame, bg=BG_BLACK)
        grid_frame.pack(fill=tk.X, anchor="w")
        grid_frame.grid_columnconfigure(0, weight=1, uniform="term_cols")
        grid_frame.grid_columnconfigure(1, weight=1, uniform="term_cols")
        grid_frame.grid_columnconfigure(2, weight=1, uniform="term_cols")

        # Заголовок лівої колонки
        make_term_label(
            grid_frame,
            font=self.term_font,
            text="Виберіть вашого користувача:",
            fg=TERM_WHITE,
        ).grid(row=0, column=0, sticky="w")

        if self.sudo_unlocked:
            make_term_label(
                grid_frame,
                font=self.term_font,
                text="Додатково (експериментально):",
                fg=TERM_PURPLE,
            ).grid(row=0, column=1, columnspan=2, sticky="w")

        # Колонка 1 (зліва): [01]..[06]
        for row_idx, uid in enumerate(range(1, 7), start=1):
            lbl = make_term_label(
                grid_frame,
                font=self.term_font,
                text=self._format_user_line(uid),
            )
            lbl.grid(row=row_idx, column=0, sticky="w")
            self.item_labels[uid] = lbl

        if self.sudo_unlocked:
            # Колонка 2 (посередині): [07]..[12]
            for row_idx, uid in enumerate(range(7, 13), start=1):
                lbl = make_term_label(
                    grid_frame,
                    font=self.term_font,
                    text=self._format_user_line(uid),
                )
                lbl.grid(row=row_idx, column=1, sticky="w")
                self.item_labels[uid] = lbl

            # Колонка 3 (справа): [13]..[18]
            for row_idx, uid in enumerate(range(13, 17), start=1):
                lbl = make_term_label(
                    grid_frame,
                    font=self.term_font,
                    text=self._format_user_line(uid),
                )
                lbl.grid(row=row_idx, column=2, sticky="w")
                self.item_labels[uid] = lbl
            # Порожній рядок висотою в 1 термінальний рядок + [17] під другою колонкою
            make_term_label(grid_frame, font=self.term_font, text="").grid(row=7, column=1, sticky="w")
            lbl_17 = make_term_label(
                grid_frame,
                font=self.term_font,
                text=self._format_user_line(17),
            )
            lbl_17.grid(row=8, column=1, columnspan=2, sticky="w")
            self.item_labels[17] = lbl_17

        self._refresh_highlights()

    def _format_user_line(self, uid: int) -> str:
        entry = self.users_dict[uid]
        name = str(entry["name"])
        if uid == 17:
            cursor = "_" if self.cursor_visible else " "
            return f"[17] {name} {self.custom_text}{cursor}"
        suffix = " (заблоковано)" if entry.get("locked") else ""
        return f"[{uid:02d}] {name}{suffix}"

    def _refresh_highlights(self) -> None:
        if 17 in self.item_labels:
            self.item_labels[17].config(text=self._format_user_line(17))

        for uid, lbl in self.item_labels.items():
            entry = self.users_dict[uid]
            is_locked = bool(entry.get("locked"))
            is_exp = bool(entry.get("experimental"))
            if uid == self.selected_id:
                active_bg = TERM_RED if is_locked else (TERM_PURPLE if is_exp else TERM_CYAN)
                lbl.config(bg=active_bg, fg=BG_BLACK)
            else:
                idle_fg = TERM_MUTED if is_locked else TERM_WHITE
                lbl.config(bg=BG_BLACK, fg=idle_fg)

    def _schedule_blink(self) -> None:
        self.cursor_visible = not self.cursor_visible
        if 17 in self.item_labels:
            self.item_labels[17].config(text=self._format_user_line(17))
        self._blink_id = self.after(500, self._schedule_blink)

    def _set_status(self, text: str, fg: str = TERM_RED) -> None:
        self.status_label.config(text=text, fg=fg)
        if self.on_status is not None:
            self.on_status(text, fg)

    def _reset_digit_buffer(self) -> None:
        self._digit_timer_id = None
        self.digit_buffer = ""

    def _handle_number_input(self, digit_char: str) -> None:
        if self._digit_timer_id is not None:
            try:
                self.after_cancel(self._digit_timer_id)
            except Exception:
                pass

        self.digit_buffer += digit_char
        max_id = self._max_selectable_id()

        val = int(self.digit_buffer)
        if 1 <= val <= max_id:
            self.selected_id = val
            self._set_status("")
            self._refresh_highlights()
        else:
            single_val = int(digit_char)
            if 1 <= single_val <= max_id:
                self.digit_buffer = digit_char
                self.selected_id = single_val
                self._set_status("")
                self._refresh_highlights()

        self._digit_timer_id = self.after(750, self._reset_digit_buffer)

    def handle_key(self, event: tk.Event) -> None:
        max_id = self._max_selectable_id()

        if event.keysym in ("Return", "KP_Enter"):
            entry = self.users_dict[self.selected_id]
            if entry.get("locked"):
                self._set_status("ПОМИЛКА: Користувач заблокований.", TERM_RED)
                return
            if self.selected_id == 17:
                cleaned = self.custom_text.strip()
                if not cleaned:
                    self._set_status("Введіть ім'я користувача перед натисканням ENTER.", TERM_AMBER)
                    return
                self.on_select(17, cleaned, self.sudo_unlocked)
                return

            self.on_select(self.selected_id, str(entry["name"]), self.sudo_unlocked)
            return

        if event.keysym == "Up":
            if self.sudo_unlocked and self.selected_id == 17:
                self.selected_id = 12
            elif self.selected_id <= 1:
                self.selected_id = max_id
            else:
                self.selected_id -= 1
            self._set_status("")
            self._refresh_highlights()
            return

        if event.keysym == "Down":
            if self.sudo_unlocked and self.selected_id in (12, 16):
                self.selected_id = 17
            elif self.selected_id >= max_id:
                self.selected_id = 1
            else:
                self.selected_id += 1
            self._set_status("")
            self._refresh_highlights()
            return

        if self.sudo_unlocked and event.keysym in ("Left", "Right"):
            if self.selected_id == 17:
                self.selected_id = 16 if event.keysym == "Right" else 6
                self._set_status("")
                self._refresh_highlights()
                return
            delta = 6 if event.keysym == "Right" else -6
            target = self.selected_id + delta
            if 1 <= target <= 16:
                self.selected_id = target
                self._set_status("")
                self._refresh_highlights()
            return

        if self.selected_id == 17:
            if event.keysym == "BackSpace":
                self.custom_text = self.custom_text[:-1]
                self._refresh_highlights()
                return
            if _is_paste_event(event):
                self.handle_paste()
                return
            if event.char and event.char.isprintable() and not _has_non_text_modifier(event):
                self.custom_text += event.char
                self.status_label.config(text="")
                self._refresh_highlights()
                return

        if event.char and event.char.isdigit():
            self._handle_number_input(event.char)
            return

        # Відстеження "sudo" на EN / DE / UA / RU розкладках клавіатури
        if not (event.state & 0x4):
            mapped = translate_sudo_key(event)
            if mapped:
                self.secret_buffer = (self.secret_buffer + mapped)[-8:]
                if self.secret_buffer.endswith("sudo") and not self.sudo_unlocked:
                    self.sudo_unlocked = True
                    self._build_selector_ui()
            elif event.char and event.char.isprintable():
                self.secret_buffer = ""

    def handle_paste(self) -> None:
        if self.selected_id != 17:
            return
        pasted = _read_clipboard_text(self, multiline=False)
        if pasted:
            self.custom_text += pasted
            self.status_label.config(text="")
            self._refresh_highlights()

    def destroy(self) -> None:
        for timer in (self._digit_timer_id, self._blink_id):
            if timer is not None:
                try:
                    self.after_cancel(timer)
                except Exception:
                    pass
        super().destroy()


class MainMenuModal(tk.Frame):
    """
    Головне меню GVK PRIVATE AI:
      - Під заголовком зліва: `Увійшли як: <Ім'я>` у кольорі Cyan (вибір за `0` або стрілкою Вгору).
      - Перша колонка (зліва): пункти режиму `[01]..[04]`.
      - При натисканні на `[01] Написати комусь` відкривається середня колонка з вибором персонажа:
        `[01] Друг 1`, `[02] Друг 2 (недоступно)`, `[03] Друг 3 (недоступно)`.
    """

    def __init__(
        self,
        parent: tk.Widget,
        user_name: str,
        menu_items: List[Dict[str, object]],
        on_switch_user: Callable[[], None],
        on_select_item: Callable[[int, str], None],
        term_font: tkfont.Font,
        persona_items: Optional[List[Dict[str, object]]] = None,
        on_select_persona: Optional[Callable[[str, str], None]] = None,
        on_status: Optional[Callable[[str, str], None]] = None,
        external_user_label: Optional[tk.Label] = None,
    ) -> None:
        super().__init__(parent, bg=BG_BLACK)
        self.user_name = user_name
        self.menu_items = menu_items
        self.persona_items = persona_items if persona_items is not None else PERSONA_SELECT_ITEMS
        self.on_switch_user = on_switch_user
        self.on_select_item = on_select_item
        self.on_select_persona = on_select_persona
        self.on_status = on_status
        self.term_font = term_font

        self.selected_index = 1
        self.persona_column_open = False
        self.active_column = 0  # 0 = ліва колонка режимів, 1 = середня колонка персонажів
        self.selected_persona_index = 1

        self.item_labels: Dict[int, tk.Label] = {}
        self.persona_labels: Dict[int, tk.Label] = {}

        if external_user_label is not None:
            self.user_label = external_user_label
            self.user_label.config(text=f"Увійшли як: {self.user_name}", fg=TERM_CYAN, bg=BG_BLACK)
        else:
            top_bar = tk.Frame(self, bg=BG_BLACK)
            top_bar.pack(side=tk.TOP, fill=tk.X)
            self.user_label = make_term_label(
                top_bar,
                font=term_font,
                text=f"Увійшли як: {self.user_name}",
                fg=TERM_CYAN,
            )
            self.user_label.pack(anchor="w")
        self.item_labels[0] = self.user_label

        # Статус-повідомлення (якщо не передано зовнішній on_status у футері)
        self.status_label = make_term_label(
            self,
            font=term_font,
            text="",
            fg=TERM_AMBER,
        )
        if self.on_status is None:
            self.status_label.pack(side=tk.BOTTOM, anchor="w")

        # Центральна 3-колонкова зона: Колонка 0 зліва, Колонка 1 посередині
        self.center_area = tk.Frame(self, bg=BG_BLACK)
        self.center_area.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.grid_frame = tk.Frame(self.center_area, bg=BG_BLACK)
        self.grid_frame.place(x=0, rely=0.44, relwidth=1.0, anchor="nw")
        self.grid_frame.grid_columnconfigure(0, weight=1, uniform="term_cols")
        self.grid_frame.grid_columnconfigure(1, weight=1, uniform="term_cols")
        self.grid_frame.grid_columnconfigure(2, weight=1, uniform="term_cols")

        self._build_columns()

    def _build_columns(self) -> None:
        for child in self.grid_frame.winfo_children():
            child.destroy()
        self.persona_labels.clear()
        for k in list(self.item_labels.keys()):
            if k != 0:
                del self.item_labels[k]

        # Ліва колонка (режими)
        for row_idx, item in enumerate(self.menu_items):
            idx = int(item["id"])
            label_text = f"[{idx:02d}] {item['label']}"
            lbl = make_term_label(
                self.grid_frame,
                font=self.term_font,
                text=label_text,
            )
            lbl.grid(row=row_idx, column=0, sticky="w")
            self.item_labels[idx] = lbl

        # Середня колонка (вибір співрозмовника при натисканні [01] Написати комусь)
        if self.persona_column_open:
            for row_idx, p_item in enumerate(self.persona_items):
                p_idx = int(p_item["id"])
                p_text = f"[{p_idx:02d}] {p_item['label']}"
                p_lbl = make_term_label(
                    self.grid_frame,
                    font=self.term_font,
                    text=p_text,
                )
                p_lbl.grid(row=row_idx, column=1, sticky="w")
                self.persona_labels[p_idx] = p_lbl

        self._refresh_highlights()

    def _refresh_highlights(self) -> None:
        if self.active_column == 0 and self.selected_index == 0:
            self.user_label.config(bg=TERM_CYAN, fg=BG_BLACK)
        else:
            self.user_label.config(bg=BG_BLACK, fg=TERM_CYAN)

        for item in self.menu_items:
            idx = int(item["id"])
            available = bool(item.get("available"))
            lbl = self.item_labels.get(idx)
            if not lbl:
                continue
            if self.active_column == 0 and idx == self.selected_index:
                lbl.config(
                    bg=TERM_CYAN if available else TERM_MUTED,
                    fg=BG_BLACK,
                )
            elif self.persona_column_open and idx == 1:
                lbl.config(bg=BG_BLACK, fg=TERM_CYAN)
            else:
                lbl.config(
                    bg=BG_BLACK,
                    fg=TERM_WHITE if available else TERM_MUTED,
                )

        for p_item in self.persona_items:
            p_idx = int(p_item["id"])
            available = bool(p_item.get("available"))
            p_color = str(p_item.get("color") or TERM_GREEN)
            p_lbl = self.persona_labels.get(p_idx)
            if not p_lbl:
                continue
            if self.active_column == 1 and p_idx == self.selected_persona_index:
                p_lbl.config(
                    bg=p_color if available else TERM_MUTED,
                    fg=BG_BLACK,
                )
            else:
                p_lbl.config(
                    bg=BG_BLACK,
                    fg=TERM_WHITE if available else TERM_MUTED,
                )

    def _set_status(self, text: str, fg: str = TERM_AMBER) -> None:
        self.status_label.config(text=text, fg=fg)
        if self.on_status is not None:
            self.on_status(text, fg)

    def open_persona_column(self) -> None:
        self.persona_column_open = True
        self.active_column = 1
        self.selected_persona_index = 1
        self._set_status("")
        self._build_columns()

    def close_persona_column(self) -> None:
        self.persona_column_open = False
        self.active_column = 0
        self._set_status("")
        self._build_columns()

    def handle_key(self, event: tk.Event) -> None:
        # Якщо активна середня колонка вибору персонажа
        if self.active_column == 1 and self.persona_column_open:
            max_p = len(self.persona_items)

            if event.keysym in ("Left", "Escape", "BackSpace"):
                self.close_persona_column()
                return

            if event.keysym == "Up":
                self.selected_persona_index = max_p if self.selected_persona_index <= 1 else self.selected_persona_index - 1
                self._set_status("")
                self._refresh_highlights()
                return

            if event.keysym == "Down":
                self.selected_persona_index = 1 if self.selected_persona_index >= max_p else self.selected_persona_index + 1
                self._set_status("")
                self._refresh_highlights()
                return

            if event.char == "0":
                self.close_persona_column()
                self.selected_index = 0
                self._refresh_highlights()
                return

            if event.char in ("1", "2", "3"):
                p_idx = int(event.char)
                if 1 <= p_idx <= max_p:
                    self.selected_persona_index = p_idx
                    self._set_status("")
                    self._refresh_highlights()
                return

            if event.keysym in ("Return", "KP_Enter"):
                chosen_p = next(
                    (it for it in self.persona_items if int(it["id"]) == self.selected_persona_index),
                    None,
                )
                if not chosen_p:
                    return
                if not chosen_p.get("available"):
                    self._set_status("Цей співрозмовник наразі недоступний.", TERM_AMBER)
                    return
                persona_id = str(chosen_p.get("persona_id", "friend_1"))
                label = str(chosen_p.get("label", ""))
                if self.on_select_persona:
                    self.on_select_persona(persona_id, label)
                else:
                    self.on_select_item(1, label)
                return
            return

        # Ліва колонка головного меню
        max_idx = len(self.menu_items)

        if event.keysym == "Up":
            self.selected_index = max_idx if self.selected_index <= 0 else self.selected_index - 1
            self._set_status("")
            self._refresh_highlights()
            return

        if event.keysym == "Down":
            self.selected_index = 0 if self.selected_index >= max_idx else self.selected_index + 1
            self._set_status("")
            self._refresh_highlights()
            return

        if event.keysym == "Right" and self.selected_index == 1:
            self.open_persona_column()
            return

        if event.char == "0":
            if self.selected_index == 0:
                self.on_switch_user()
                return
            self.selected_index = 0
            self._set_status("")
            self._refresh_highlights()
            return

        if event.char in ("1", "2", "3", "4"):
            idx = int(event.char)
            if 1 <= idx <= max_idx:
                self.selected_index = idx
                self._set_status("")
                self._refresh_highlights()
                if idx == 1:
                    self.open_persona_column()
                return

        if event.keysym in ("Return", "KP_Enter"):
            if self.selected_index == 0:
                self.on_switch_user()
                return

            chosen = next(
                (it for it in self.menu_items if int(it["id"]) == self.selected_index),
                None,
            )
            if not chosen:
                return
            if not chosen.get("available"):
                self._set_status("Цей режим наразі недоступний.", TERM_AMBER)
                return
            if int(chosen["id"]) == 1:
                self.open_persona_column()
                return
            self.on_select_item(int(chosen["id"]), str(chosen["label"]))


class PersonaLoadingModal(tk.Frame):
    """
    Модальний екран завантаження вибраного персонажа:
      - Посередині зверху: `Завантажуємо <родовий відмінок>...` (наприклад `Завантажуємо Друга 1...`)
      - Під ним (над смугою завантаження): випадково обрана порада (`loading_tips`), що змінюється кожні 10 с, у кольорі персонажа.
      - Посередині: смуга завантаження у кольорі персонажа з випадковою швидкістю (змінюється кожні 5 с), розрахованою на ~60 с до 100%.
      - У футері справа: час, що минув (`0s...`, `1s...`, ...).
      - Якщо сервер і 2 warmup-питання завершилися раніше (або сервер уже був запущений), смуга миттєво стрибає на 100% і переходить у чат.
    """

    def __init__(
        self,
        parent: tk.Widget,
        user_name: str,
        persona_name: str,
        genitive_name: str,
        loading_tips: List[str],
        persona_color: str,
        term_font: tkfont.Font,
        on_complete: Callable[[], None],
        background_task: Optional[Callable[[], None]] = None,
        bar_duration_seconds: float = 60.0,
        bar_width: int = 40,
        tip_interval_seconds: float = 10.0,
        speed_interval_seconds: float = 5.0,
        on_status: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        super().__init__(parent, bg=BG_BLACK)
        self.user_name = user_name
        self.persona_name = persona_name
        self.genitive_name = genitive_name
        self.loading_tips = list(
            dict.fromkeys(str(t) for t in loading_tips if str(t).strip())
        ) or ["loading tip 1"]
        self.persona_color = persona_color or TERM_GREEN
        self.term_font = term_font
        self.on_complete = on_complete
        self.background_task = background_task
        self.bar_duration_seconds = max(0.001, float(bar_duration_seconds))
        self.bar_width = max(10, int(bar_width))
        self.tip_interval_seconds = max(0.1, float(tip_interval_seconds))
        self.speed_interval_seconds = max(0.1, float(speed_interval_seconds))
        self.on_status = on_status

        self._start_time = time.monotonic()
        self._last_tick_time = self._start_time
        self._next_tip_time = self._start_time + self.tip_interval_seconds
        self._next_speed_time = self._start_time + self.speed_interval_seconds
        self._progress = 0.0
        self._current_rate = 1.0 / self.bar_duration_seconds
        self._tip_queue: List[str] = []
        self._current_tip = ""
        self._current_tip = self._pick_random_tip()
        self._timer_id: Optional[str] = None
        self._task_done = False
        self._completed = False

        wrap_w = WINDOW_WIDTH - (LEFT_PAD * 2)

        self.center_area = tk.Frame(self, bg=BG_BLACK)
        self.center_area.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        block_frame = tk.Frame(self.center_area, bg=BG_BLACK)
        block_frame.place(relx=0.5, rely=0.5, anchor="center")

        self.title_label = make_term_label(
            block_frame,
            font=term_font,
            text=f"Завантажуємо {self.genitive_name}...",
            fg=TERM_WHITE,
            wraplength=wrap_w,
            justify=tk.CENTER,
            anchor="center",
        )
        self.title_label.pack(anchor="center", pady=(0, 8))

        self.tip_label = make_term_label(
            block_frame,
            font=term_font,
            text=self._current_tip,
            fg=self.persona_color,
            wraplength=wrap_w,
            justify=tk.CENTER,
            anchor="center",
        )
        self.tip_label.pack(anchor="center", pady=(0, 8))

        self.bar_label = make_term_label(
            block_frame,
            font=term_font,
            text=self.format_loading_bar(0.0),
            fg=self.persona_color,
            justify=tk.CENTER,
            anchor="center",
        )
        self.bar_label.pack(anchor="center")

    def _pick_random_tip(self) -> str:
        if len(self.loading_tips) <= 1:
            return self.loading_tips[0]
        if not self._tip_queue:
            self._tip_queue = list(self.loading_tips)
            random.shuffle(self._tip_queue)
            if self._tip_queue[-1] == self._current_tip:
                self._tip_queue[0], self._tip_queue[-1] = self._tip_queue[-1], self._tip_queue[0]
        return self._tip_queue.pop()

    def _pick_random_rate(self, elapsed: float) -> float:
        rem_prog = max(0.0, 1.0 - self._progress)
        rem_time = max(self.speed_interval_seconds, self.bar_duration_seconds - elapsed)
        base_rate = rem_prog / rem_time
        return base_rate * random.uniform(0.4, 1.6)

    def start(self) -> None:
        """Запускає таймер відліку часу/смуги та фоновий потік підготовки моделі."""
        now = time.monotonic()
        self._start_time = now
        self._last_tick_time = now
        self._next_tip_time = now + self.tip_interval_seconds
        self._next_speed_time = now + self.speed_interval_seconds
        self._progress = 0.0
        self._current_rate = self._pick_random_rate(0.0)
        self._tip_queue.clear()
        self._current_tip = self._pick_random_tip()
        self.tip_label.config(text=self._current_tip)
        self._refresh_progress()
        if self.background_task is not None:
            def _worker() -> None:
                try:
                    self.background_task()
                except Exception as exc:
                    logger.exception("Помилка фонової підготовки персонажа: %s", exc)
                finally:
                    self._task_done = True

            threading.Thread(target=_worker, daemon=True).start()
        else:
            self._task_done = True
            self._timer_id = self.after(100, self._tick)
            return

        self._timer_id = self.after(100, self._tick)

    def format_loading_bar(self, ratio: float) -> str:
        clamped = min(1.0, max(0.0, ratio))
        filled = int(round(clamped * self.bar_width))
        pct = int(round(clamped * 100))
        bar_str = ("█" * filled) + ("░" * (self.bar_width - filled))
        return f"[{bar_str}] {pct:3d}%"

    def _set_status(self, text: str, fg: str = TERM_MUTED) -> None:
        if self.on_status is not None:
            self.on_status(text, fg)

    def _refresh_progress(self) -> None:
        now = time.monotonic()
        dt = max(0.0, now - self._last_tick_time)
        elapsed = max(0.0, now - self._start_time)
        self._last_tick_time = now

        if now >= self._next_tip_time:
            self._current_tip = self._pick_random_tip()
            self.tip_label.config(text=self._current_tip)
            while self._next_tip_time <= now:
                self._next_tip_time += self.tip_interval_seconds

        if now >= self._next_speed_time:
            self._current_rate = self._pick_random_rate(elapsed)
            while self._next_speed_time <= now:
                self._next_speed_time += self.speed_interval_seconds

        self._progress = min(0.99, self._progress + self._current_rate * dt)
        self.bar_label.config(text=self.format_loading_bar(self._progress))
        self._set_status(f"{int(elapsed)}s...", TERM_MUTED)

    def _tick(self) -> None:
        self._timer_id = None
        if not self.winfo_exists() or self._completed:
            return
        if self._task_done:
            self._finish()
            return
        self._refresh_progress()
        self._timer_id = self.after(100, self._tick)

    def _finish(self) -> None:
        if self._completed or not self.winfo_exists():
            return
        self._completed = True
        self._progress = 1.0
        self.bar_label.config(text=self.format_loading_bar(1.0))
        if self._timer_id is not None:
            try:
                self.after_cancel(self._timer_id)
            except Exception:
                pass
            self._timer_id = None
        self.on_complete()

    def handle_key(self, _event: tk.Event) -> None:
        pass

    def destroy(self) -> None:
        if self._timer_id is not None:
            try:
                self.after_cancel(self._timer_id)
            except Exception:
                pass
            self._timer_id = None
        super().destroy()


def compute_line_delay_ms(line: str) -> int:
    """Мапить довжину рядка у затримку появи наступного рядка в межах 0..1000 мс (80+ символів = 1 с)."""
    ratio = min(1.0, max(0.0, len(line) / 80.0))
    return int(round(ratio * 1000))


# Таблиця зміщень 8 точок Unicode Braille (U+2800..U+28FF): (біт, колонка 0..1, рядок 0..3)
_BRAILLE_DOT_BITS: Tuple[Tuple[int, int, int], ...] = (
    (0x01, 0, 0),
    (0x02, 0, 1),
    (0x04, 0, 2),
    (0x40, 0, 3),
    (0x08, 1, 0),
    (0x10, 1, 1),
    (0x20, 1, 2),
    (0x80, 1, 3),
)


class BraillePortraitCanvas(tk.Canvas):
    """
    Шрифто-незалежний рендерер ASCII/Braille-портретів на `tk.Canvas`:
      - Символи Unicode Braille (`U+2800`..`U+28FF`) декодуються у 2x4 сітку піксельних прямокутників,
        що гарантує однакову ширину всіх рядків незалежно від системних шрифтів.
      - Будь-які інші непробільні символи рендеряться у тій же фіксованій клітинці як запасний варіант.
      - Метод `set_portrait(text, color)` дозволяє миттєво змінювати емоцію / спрайт без зсуву макета.
    """

    def __init__(
        self,
        parent: tk.Widget,
        term_font: tkfont.Font,
        portrait_text: str = "",
        color: str = TERM_WHITE,
    ) -> None:
        super().__init__(parent, bg=BG_BLACK, bd=0, highlightthickness=0, width=1, height=1)
        self.term_font = term_font
        self.portrait_text = portrait_text
        self.portrait_color = color or TERM_WHITE

        self.cell_w = max(8, int(term_font.measure("M") or 10))
        self.cell_h = max(16, int(term_font.metrics("linespace") or 18))
        self.set_portrait(portrait_text, self.portrait_color)

    def set_portrait(self, portrait_text: str, color: Optional[str] = None) -> None:
        self.portrait_text = portrait_text or ""
        if color is not None:
            self.portrait_color = color or TERM_WHITE

        self.delete("all")
        lines = self.portrait_text.splitlines()
        if not lines:
            self.config(width=1, height=1)
            return

        max_cols = max((len(ln) for ln in lines), default=0)
        total_rows = len(lines)
        canvas_w = max(1, max_cols * self.cell_w)
        canvas_h = max(1, total_rows * self.cell_h)
        self.config(width=canvas_w, height=canvas_h)

        dx = self.cell_w / 2.0
        dy = self.cell_h / 4.0
        dw = max(2.0, dx * 0.65)
        dh = max(2.0, dy * 0.65)
        pad_x = (dx - dw) / 2.0
        pad_y = (dy - dh) / 2.0

        for row_idx, line in enumerate(lines):
            y0 = row_idx * self.cell_h
            for col_idx, ch in enumerate(line):
                x0 = col_idx * self.cell_w
                code = ord(ch)
                if 0x2800 <= code <= 0x28FF:
                    mask = code - 0x2800
                    if mask == 0:
                        continue
                    for bit, dot_col, dot_row in _BRAILLE_DOT_BITS:
                        if mask & bit:
                            rx = x0 + dot_col * dx + pad_x
                            ry = y0 + dot_row * dy + pad_y
                            self.create_rectangle(
                                rx,
                                ry,
                                rx + dw,
                                ry + dh,
                                fill=self.portrait_color,
                                outline=self.portrait_color,
                            )
                elif not ch.isspace():
                    self.create_text(
                        x0 + self.cell_w / 2.0,
                        y0 + self.cell_h / 2.0,
                        text=ch,
                        fill=self.portrait_color,
                        font=self.term_font,
                    )


_MOOD_TAG_RE = re.compile(r"\[MOOD:([a-zA-Z0-9_]+)\]", re.IGNORECASE)
_SMILE_LAUGH_RE = re.compile(r"(?:ХАХ|АХА|ПХП|ХПХ|АХП|ЛМАО|ЛОЛ)", re.IGNORECASE)
_SMILE_SLANG_WORDS = ("имба", "база", "кайф", "красава", "угар", "го")


def detect_reply_emotion(reply_text: str) -> Tuple[str, str]:
    """
    Визначає післявідповідну емоцію (`smiling` або `idle`) за легким евристичним аналізом
    та опційним тегом `[MOOD:<state>]` (який вирізається перед рендером).
    """
    explicit_mood: Optional[str] = None
    match = _MOOD_TAG_RE.search(reply_text)
    if match:
        explicit_mood = match.group(1).strip().lower()
    cleaned = _MOOD_TAG_RE.sub("", reply_text).strip()

    if explicit_mood in ("smiling", "idle", "sleeping", "talking"):
        return cleaned, explicit_mood

    if _SMILE_LAUGH_RE.search(cleaned) or ")" in cleaned:
        return cleaned, "smiling"

    tokens = re.findall(r"[a-zа-яіїєґ0-9]+", cleaned.lower())
    if any(tok in _SMILE_SLANG_WORDS for tok in tokens):
        return cleaned, "smiling"

    return cleaned, "idle"


class ChatSessionModal(tk.Frame):
    """
    Термінальний екран чату із вибраним персонажем (або кількома персонажами):
      - Верхня стрічка: зліва `Співрозмовник: ...` (або `Співрозмовники: ...`), справа `Увійшли як: ...`,
        відділена від основної області білою горизонтальною лінією.
      - Ліва частина: ASCII-портрет персонажа, відділений вертикальною білою лінією до нижньої стрічки.
      - Права частина: повідомлення співрозмовника зліва (ім'я у дужках `[Ім'я]` у кольорі персонажа),
        повідомлення користувача справа (`[Ім'я]` без двокрапки, вирівняні по правому краю),
        анімований індикатор `<Персонаж> друкує...` під останнім повідомленням,
        та поле вводу знизу з автоматичним перенесенням і `Shift+Enter`, відділене білими лініями.
    """

    def __init__(
        self,
        parent: tk.Widget,
        user_name: str,
        persona_name: str,
        persona_color: str,
        portrait_text: str,
        on_send_message: Callable[[str, Callable[[str], None]], None],
        on_back: Callable[[], None],
        term_font: tkfont.Font,
        persona_names: Optional[List[str]] = None,
        persona_colors: Optional[Dict[str, str]] = None,
        on_status: Optional[Callable[[str, str], None]] = None,
        emotion_sprites: Optional[Dict[str, List[str]]] = None,
        afk_seconds: float = DEFAULT_AFK_SECONDS,
        smile_seconds: float = DEFAULT_SMILE_SECONDS,
        dossier_info: Optional[Dict[str, str]] = None,
        on_new_chat: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(parent, bg=BG_BLACK)
        self.user_name = user_name
        self.persona_name = persona_name
        self.persona_names = persona_names if persona_names else [persona_name]
        self.persona_color = persona_color or TERM_WHITE
        self.persona_colors: Dict[str, str] = dict(persona_colors or {})
        self.persona_colors.setdefault(self.persona_name, self.persona_color)
        self.on_send_message = on_send_message
        self.on_back = on_back
        self.on_new_chat = on_new_chat
        self.on_status = on_status
        self.term_font = term_font
        self.afk_seconds = max(0.05, float(afk_seconds))
        self.smile_seconds = max(0.05, float(smile_seconds))
        self._session_generation = 0

        self.emotion_sprites: Dict[str, List[str]] = {
            k: list(v) for k, v in (emotion_sprites or {}).items() if v
        }
        if "idle" not in self.emotion_sprites and portrait_text:
            self.emotion_sprites["idle"] = [portrait_text]
        initial_portrait = (self.emotion_sprites.get("idle") or [portrait_text])[0] if (
            self.emotion_sprites.get("idle") or portrait_text
        ) else ""

        self.typed_text = ""
        self.cursor_visible = True
        self.is_waiting_reply = False
        self.current_emotion = "idle"
        self._talking_frame_idx = 0
        self._blink_id: Optional[str] = None
        self._typing_anim_id: Optional[str] = None
        self._talking_anim_id: Optional[str] = None
        self._smile_timer_id: Optional[str] = None
        self._afk_timer_id: Optional[str] = None
        self._line_reveal_id: Optional[str] = None
        self._scroll_anim_id: Optional[str] = None
        self._typing_dots = 0
        self._chat_wraplength = 520
        self._user_scrolled_up = False
        self.typing_label: Optional[tk.Label] = None
        # (speaker, text, color, is_user)
        self.chat_lines: List[Tuple[str, str, str, bool]] = []

        # Основна область між верхньою та нижньою білими лініями (які рендерить GVKMenuApp)
        body_area = tk.Frame(self, bg=BG_BLACK)
        body_area.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Ліва колонка: портрет персонажа + короткий опис під портретом (прилягає до краю і вертикальної лінії)
        self.left_pane = tk.Frame(body_area, bg=BG_BLACK)
        self.left_pane.pack(side=tk.LEFT, fill=tk.Y)

        self.portrait_holder = tk.Frame(self.left_pane, bg=BG_BLACK)
        self.portrait_holder.pack(side=tk.TOP, fill=tk.X, anchor="nw")
        self.portrait_canvas = BraillePortraitCanvas(
            self.portrait_holder,
            term_font=term_font,
            portrait_text=initial_portrait,
            color=self.persona_color,
        )
        if initial_portrait:
            self.portrait_canvas.pack(anchor="n", padx=0, pady=0)

        self.dossier_font = tkfont.Font(
            family=str(term_font.actual("family") or "Courier"),
            size=max(9, int(term_font.actual("size") or 12) - 1),
            weight="normal",
        )
        self.portrait_sep = tk.Frame(self.left_pane, bg=TERM_WHITE, height=1)
        self.dossier_frame = tk.Frame(self.left_pane, bg=BG_BLACK)
        self.dossier_title_label: Optional[tk.Label] = None
        self.dossier_desc_label: Optional[tk.Label] = None
        self.dossier_style_label: Optional[tk.Label] = None
        self.dossier_rag_label: Optional[tk.Label] = None
        self.dossier_skill_label: Optional[tk.Label] = None
        self.dossier_skills_list_label: Optional[tk.Widget] = None
        self.dossier_emotion_label: Optional[tk.Label] = None
        self._build_dossier_panel(dossier_info)

        # Вертикальна біла лінія від верхньої лінії до нижньої
        tk.Frame(body_area, bg=TERM_WHITE, width=1).pack(side=tk.LEFT, fill=tk.Y)

        # Права колонка: історія чату зверху + поле вводу знизу
        self.right_pane = tk.Frame(body_area, bg=BG_BLACK)
        self.right_pane.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Поле вводу знизу (відділене зверху білою лінією від чату)
        input_container = tk.Frame(self.right_pane, bg=BG_BLACK)
        input_container.pack(side=tk.BOTTOM, fill=tk.X)

        tk.Frame(input_container, bg=TERM_WHITE, height=1).pack(side=tk.TOP, fill=tk.X)

        self.input_label = make_term_label(
            input_container,
            font=term_font,
            text="> _",
            fg=TERM_WHITE,
            wraplength=580,
            anchor="nw",
        )
        self.input_label.pack(side=tk.TOP, fill=tk.X, anchor="nw", padx=LEFT_PAD, pady=(6, 6))

        line_step = max(16, int(self.term_font.metrics("linespace") or 18))
        self.chat_canvas = tk.Canvas(
            self.right_pane,
            bg=BG_BLACK,
            bd=0,
            highlightthickness=0,
            yscrollincrement=line_step,
        )
        self.chat_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=LEFT_PAD, pady=(6, 6))

        self.messages_frame = tk.Frame(self.chat_canvas, bg=BG_BLACK)
        self._canvas_window_id = self.chat_canvas.create_window(
            (0, 0), window=self.messages_frame, anchor="nw"
        )

        self.chat_canvas.bind("<Configure>", self._on_canvas_configure)
        self.right_pane.bind("<Configure>", self._on_right_pane_resize)
        self._schedule_blink()
        self._set_emotion("idle")
        self._reset_afk_timer()

    def _compute_left_pane_width(self) -> int:
        try:
            portrait_w = int(float(self.portrait_canvas.cget("width") or 0))
        except (TypeError, ValueError):
            portrait_w = 0
        has_portrait = bool(self.portrait_canvas.portrait_text) and portrait_w > 1
        return portrait_w if has_portrait else 460

    def _sync_left_pane_width(self) -> None:
        if not self.dossier_frame.winfo_manager():
            return
        pane_w = self._compute_left_pane_width()
        wrap_w = max(60, pane_w - 24)
        self.left_pane.config(width=pane_w)
        self.left_pane.pack_propagate(False)
        for lbl in (
            self.dossier_title_label,
            self.dossier_desc_label,
            self.dossier_style_label,
            self.dossier_rag_label,
            self.dossier_skill_label,
            self.dossier_emotion_label,
        ):
            if lbl is not None and lbl.winfo_exists():
                lbl.config(wraplength=wrap_w)

    def _build_dossier_panel(self, dossier_info: Optional[Dict[str, str]]) -> None:
        if not dossier_info:
            return
        pane_w = self._compute_left_pane_width()
        wrap_w = max(60, pane_w - 24)

        self.left_pane.config(width=pane_w)
        self.left_pane.pack_propagate(False)
        self.portrait_sep.pack(side=tk.TOP, fill=tk.X, pady=(2, 3))
        self.dossier_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=12, pady=(2, 2))

        title = str(dossier_info.get("title") or self.persona_name).strip()
        desc = str(dossier_info.get("description") or "").strip()
        style = str(dossier_info.get("style") or "").strip()
        rag_status = str(dossier_info.get("rag_status") or "").strip()
        skill_status = str(dossier_info.get("skill_status") or "").strip()
        skills_summary = str(dossier_info.get("skills_summary") or "").strip()
        emotion_status = str(dossier_info.get("emotion_status") or self.current_emotion).upper()

        self.dossier_title_label = make_term_label(
            self.dossier_frame,
            font=self.term_font,
            text=title,
            fg=self.persona_color,
            wraplength=wrap_w,
        )
        self.dossier_title_label.pack(anchor="w", pady=(0, 3))

        if desc:
            self.dossier_desc_label = make_term_label(
                self.dossier_frame,
                font=self.dossier_font,
                text=desc,
                fg=TERM_WHITE,
                wraplength=wrap_w,
            )
            self.dossier_desc_label.pack(anchor="w", pady=(0, 3))

        if style:
            self.dossier_style_label = make_term_label(
                self.dossier_frame,
                font=self.dossier_font,
                text=f"Стиль: {style}",
                fg="#9a9a9a",
                wraplength=wrap_w,
            )
            self.dossier_style_label.pack(anchor="w", pady=(0, 3))

        if rag_status:
            self.dossier_rag_label = make_term_label(
                self.dossier_frame,
                font=self.dossier_font,
                text=f"RAG: {rag_status}",
                fg=TERM_CYAN,
                wraplength=wrap_w,
            )
            self.dossier_rag_label.pack(anchor="w", pady=(0, 2))

        if skill_status:
            skill_fg = TERM_AMBER if "[OK]" in skill_status else TERM_MUTED
            self.dossier_skill_label = make_term_label(
                self.dossier_frame,
                font=self.dossier_font,
                text=f"Скіли: {skill_status}",
                fg=skill_fg,
                wraplength=wrap_w,
            )
            self.dossier_skill_label.pack(anchor="w", pady=(0, 2))

        if skills_summary:
            skills_box = tk.Frame(self.dossier_frame, bg=BG_BLACK)
            skills_box.pack(anchor="w", fill=tk.X, pady=(0, 3))
            for raw_line in skills_summary.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                if " - " in line:
                    cmd_part, desc_part = line.split(" - ", 1)
                    if self.dossier_font.measure(f"{cmd_part} — {desc_part}") <= wrap_w:
                        row = tk.Frame(skills_box, bg=BG_BLACK)
                        row.pack(anchor="w", fill=tk.X, pady=(0, 1))
                        make_term_label(
                            row,
                            font=self.dossier_font,
                            text=cmd_part,
                            fg="#e0b566",
                        ).pack(side=tk.LEFT, anchor="w")
                        make_term_label(
                            row,
                            font=self.dossier_font,
                            text=f" — {desc_part}",
                            fg=TERM_WHITE,
                        ).pack(side=tk.LEFT, anchor="w")
                        continue
                make_term_label(
                    skills_box,
                    font=self.dossier_font,
                    text=line.replace(" - ", " — ", 1),
                    fg=TERM_WHITE,
                    wraplength=wrap_w,
                ).pack(anchor="w", pady=(0, 1))
            self.dossier_skills_list_label = skills_box

        self.dossier_emotion_label = make_term_label(
            self.dossier_frame,
            font=self.term_font,
            text=f"Стан: [{emotion_status}]",
            fg=self.persona_color,
            wraplength=wrap_w,
        )
        self.dossier_emotion_label.pack(anchor="w", pady=(2, 0))

    def _set_emotion(self, emotion: str, frame_idx: int = 0) -> None:
        self.current_emotion = emotion
        if self.dossier_emotion_label is not None and self.dossier_emotion_label.winfo_exists():
            self.dossier_emotion_label.config(text=f"Стан: [{emotion.upper()}]")
        frames = self.emotion_sprites.get(emotion) or self.emotion_sprites.get("idle")
        if frames:
            self.set_portrait(frames[frame_idx % len(frames)])

    def _cancel_smile_timer(self) -> None:
        if self._smile_timer_id is not None:
            try:
                self.after_cancel(self._smile_timer_id)
            except Exception:
                pass
            self._smile_timer_id = None

    def _cancel_afk_timer(self) -> None:
        if self._afk_timer_id is not None:
            try:
                self.after_cancel(self._afk_timer_id)
            except Exception:
                pass
            self._afk_timer_id = None

    def _reset_afk_timer(self) -> None:
        self._cancel_afk_timer()
        if not self.winfo_exists() or self.is_waiting_reply:
            return
        self._afk_timer_id = self.after(int(round(self.afk_seconds * 1000)), self._on_afk_timeout)

    def _on_afk_timeout(self) -> None:
        self._afk_timer_id = None
        if not self.winfo_exists() or self.is_waiting_reply:
            return
        if self.current_emotion == "idle":
            self._set_emotion("sleeping")

    def _start_talking_anim(self) -> None:
        self._cancel_smile_timer()
        self._cancel_afk_timer()
        self._stop_talking_anim()
        self._talking_frame_idx = 0
        self._set_emotion("talking", self._talking_frame_idx)
        self._talking_anim_id = self.after(TALKING_FRAME_MS, self._schedule_talking_anim)

    def _schedule_talking_anim(self) -> None:
        prev_id = self._talking_anim_id
        self._talking_anim_id = None
        if prev_id is not None:
            try:
                self.after_cancel(prev_id)
            except Exception:
                pass
        if not self.is_waiting_reply or not self.winfo_exists():
            return
        self._talking_frame_idx += 1
        self._set_emotion("talking", self._talking_frame_idx)
        self._talking_anim_id = self.after(TALKING_FRAME_MS, self._schedule_talking_anim)

    def _stop_talking_anim(self) -> None:
        if self._talking_anim_id is not None:
            try:
                self.after_cancel(self._talking_anim_id)
            except Exception:
                pass
            self._talking_anim_id = None

    def _start_smile_hold(self) -> None:
        self._cancel_smile_timer()
        self._cancel_afk_timer()
        self._set_emotion("smiling")

        def _end_smile() -> None:
            self._smile_timer_id = None
            if not self.winfo_exists() or self.is_waiting_reply:
                return
            self._set_emotion("idle")
            self._reset_afk_timer()

        self._smile_timer_id = self.after(int(round(self.smile_seconds * 1000)), _end_smile)

    def _finish_reply_emotion(self, full_reply: str) -> None:
        self._stop_talking_anim()
        _, post_emotion = detect_reply_emotion(full_reply)
        if post_emotion == "smiling":
            self._start_smile_hold()
        else:
            self._set_emotion("idle")
            self._reset_afk_timer()

    def set_portrait(self, portrait_text: str, color: Optional[str] = None) -> None:
        """Оновлює портрет персонажа (для машини станів емоцій)."""
        self.portrait_canvas.set_portrait(portrait_text, color=color)
        is_packed = bool(self.portrait_canvas.winfo_manager())
        if portrait_text:
            if not is_packed:
                self.portrait_canvas.pack(anchor="nw", padx=0, pady=0)
        elif is_packed:
            self.portrait_canvas.pack_forget()
        self._sync_left_pane_width()

    def _set_status(self, text: str, fg: str = TERM_RED) -> None:
        if self.on_status is not None:
            self.on_status(text, fg)

    def _on_canvas_configure(self, event: tk.Event) -> None:
        if event.width <= 20:
            return
        self.chat_canvas.itemconfig(self._canvas_window_id, width=event.width)
        self._update_scroll_layout(scroll_to_bottom=not self._user_scrolled_up, smooth=False)

    def _stop_scroll_anim(self) -> None:
        if self._scroll_anim_id is not None:
            try:
                self.after_cancel(self._scroll_anim_id)
            except Exception:
                pass
            self._scroll_anim_id = None

    def _effective_canvas_height(self) -> int:
        canvas_h = int(self.chat_canvas.winfo_height())
        return canvas_h if canvas_h > 1 else 360

    def _update_scroll_layout(self, scroll_to_bottom: bool = True, smooth: bool = True) -> None:
        if not self.winfo_exists():
            return
        self.chat_canvas.update_idletasks()
        raw_w = int(self.chat_canvas.winfo_width())
        canvas_w = raw_w if raw_w > 1 else max(200, int(self._chat_wraplength))
        canvas_h = self._effective_canvas_height()
        content_h = max(0, self.messages_frame.winfo_reqheight())
        self.chat_canvas.itemconfig(self._canvas_window_id, width=canvas_w)

        if content_h < canvas_h:
            self.chat_canvas.coords(self._canvas_window_id, 0, canvas_h - content_h)
            self.chat_canvas.configure(scrollregion=(0, 0, canvas_w, canvas_h))
            self.chat_canvas.yview_moveto(0.0)
            self._user_scrolled_up = False
            return

        self.chat_canvas.coords(self._canvas_window_id, 0, 0)
        self.chat_canvas.configure(scrollregion=(0, 0, canvas_w, content_h))

        if scroll_to_bottom:
            self._scroll_to_bottom(smooth=smooth)

    def _scroll_to_bottom(self, smooth: bool = True) -> None:
        self._stop_scroll_anim()
        if not self.winfo_exists():
            return
        content_h = self.messages_frame.winfo_reqheight()
        canvas_h = self._effective_canvas_height()
        if content_h <= canvas_h:
            self.chat_canvas.yview_moveto(0.0)
            return

        target = max(0.0, (content_h - canvas_h) / float(content_h))
        if not smooth:
            self.chat_canvas.yview_moveto(1.0)
            return

        curr = float(self.chat_canvas.yview()[0])
        if abs(target - curr) < 0.002:
            self.chat_canvas.yview_moveto(1.0)
            return

        total_steps = 8

        def _step(step_idx: int) -> None:
            self._scroll_anim_id = None
            if not self.winfo_exists():
                return
            if step_idx >= total_steps:
                self.chat_canvas.yview_moveto(1.0)
                return
            t = step_idx / float(total_steps)
            eased = 1.0 - (1.0 - t) ** 2
            self.chat_canvas.yview_moveto(curr + (target - curr) * eased)
            self._scroll_anim_id = self.after(16, lambda: _step(step_idx + 1))

        self._scroll_anim_id = self.after(16, lambda: _step(1))
    def _on_right_pane_resize(self, event: tk.Event) -> None:
        if event.width <= 40:
            return
        chat_inner_width = max(200, event.width - 2 * LEFT_PAD)
        new_input_wrap = chat_inner_width
        new_chat_wrap = max(200, int(chat_inner_width * 0.75))
        self.input_label.config(wraplength=new_input_wrap)
        if abs(new_chat_wrap - self._chat_wraplength) > 12:
            self._chat_wraplength = new_chat_wrap
            self._render_messages(smooth_scroll=False)

    def _schedule_blink(self) -> None:
        self.cursor_visible = not self.cursor_visible
        self._refresh_input()
        self._blink_id = self.after(500, self._schedule_blink)

    def _refresh_input(self) -> None:
        cursor = "_" if self.cursor_visible else " "
        lines = self.typed_text.split("\n")
        formatted = "\n".join(f"> {ln}" for ln in lines) + cursor
        self.input_label.config(text=formatted)

    def _format_typing_text(self) -> str:
        return f"{self.persona_name} друкує{'.' * self._typing_dots}"

    def _start_typing_indicator(self) -> None:
        self._stop_typing_indicator()
        self._user_scrolled_up = False
        self._typing_dots = 0
        self._render_messages(smooth_scroll=True)
        self._typing_anim_id = self.after(350, self._schedule_typing_anim)

    def _schedule_typing_anim(self) -> None:
        if not self.is_waiting_reply or not self.winfo_exists():
            self._typing_anim_id = None
            return
        self._typing_dots = (self._typing_dots + 1) % 4
        if self.typing_label is not None and self.typing_label.winfo_exists():
            self.typing_label.config(text=self._format_typing_text())
        self._typing_anim_id = self.after(350, self._schedule_typing_anim)

    def _stop_typing_indicator(self) -> None:
        if self._typing_anim_id is not None:
            try:
                self.after_cancel(self._typing_anim_id)
            except Exception:
                pass
            self._typing_anim_id = None

    def _render_messages(self, smooth_scroll: bool = True) -> None:
        for child in self.messages_frame.winfo_children():
            child.destroy()
        self.typing_label = None

        visible = self.chat_lines[-200:]
        for speaker, text, color, is_user in visible:
            block = tk.Frame(self.messages_frame, bg=BG_BLACK)
            block.pack(anchor="e" if is_user else "w", pady=(0, 6))

            if is_user:
                make_term_label(
                    block,
                    font=self.term_font,
                    text=f"[{speaker}]",
                    fg=TERM_CYAN,
                    justify=tk.RIGHT,
                    anchor="e",
                ).pack(anchor="e")
                for user_line in text.split("\n"):
                    make_term_label(
                        block,
                        font=self.term_font,
                        text=user_line,
                        fg=TERM_WHITE,
                        wraplength=self._chat_wraplength,
                        justify=tk.LEFT,
                        anchor="e",
                    ).pack(anchor="e")
            else:
                name_color = self.persona_colors.get(speaker, color)
                make_term_label(
                    block,
                    font=self.term_font,
                    text=f"[{speaker}]",
                    fg=name_color,
                    justify=tk.LEFT,
                    anchor="w",
                ).pack(anchor="w")
                make_term_label(
                    block,
                    font=self.term_font,
                    text=text,
                    fg=TERM_WHITE,
                    wraplength=self._chat_wraplength,
                    justify=tk.LEFT,
                    anchor="w",
                ).pack(anchor="w")

        if self.is_waiting_reply:
            self.typing_label = make_term_label(
                self.messages_frame,
                font=self.term_font,
                text=self._format_typing_text(),
                fg=TERM_MUTED,
            )
            self.typing_label.pack(anchor="w")

        self._update_scroll_layout(scroll_to_bottom=not self._user_scrolled_up, smooth=smooth_scroll)

    def append_message(self, speaker: str, text: str, color: str, is_user: bool = False) -> None:
        self._user_scrolled_up = False
        self.chat_lines.append((speaker, text, color, is_user))
        self._render_messages(smooth_scroll=True)

    def reset_chat(self) -> None:
        """Очищає поточну розмову та запускає новий чистий чат (CTRL+N)."""
        self._session_generation += 1
        self._stop_typing_indicator()
        self._stop_talking_anim()
        self._cancel_smile_timer()
        self._cancel_afk_timer()
        self._stop_scroll_anim()
        if self._line_reveal_id is not None:
            try:
                self.after_cancel(self._line_reveal_id)
            except Exception:
                pass
            self._line_reveal_id = None

        self.is_waiting_reply = False
        self.typed_text = ""
        self._user_scrolled_up = False
        self.chat_lines.clear()

        if self.on_new_chat is not None:
            self.on_new_chat()

        self._set_emotion("idle")
        self._reset_afk_timer()
        self._render_messages(smooth_scroll=False)
        self._refresh_input()
        self._set_status("Новий чат розпочато.", TERM_GREEN)

    def _on_reply_received(self, reply_text: str, expected_gen: Optional[int] = None) -> None:
        if not self.winfo_exists():
            return
        if expected_gen is not None and expected_gen != self._session_generation:
            return
        if self._line_reveal_id is not None:
            try:
                self.after_cancel(self._line_reveal_id)
            except Exception:
                pass
            self._line_reveal_id = None

        cleaned_reply, _ = detect_reply_emotion(reply_text)
        raw_lines = cleaned_reply.strip("\r\n").splitlines()
        lines = raw_lines if raw_lines else [cleaned_reply]

        self._user_scrolled_up = False
        self.chat_lines.append((self.persona_name, lines[0], self.persona_color, False))
        if len(lines) == 1:
            self.is_waiting_reply = False
            self._stop_typing_indicator()
            self._finish_reply_emotion(reply_text)
            self._set_status("", TERM_MUTED)
            self._render_messages(smooth_scroll=True)
            self._refresh_input()
            return

        self.is_waiting_reply = True
        self._start_talking_anim()
        self._render_messages(smooth_scroll=True)

        def _reveal_next(idx: int) -> None:
            self._line_reveal_id = None
            if not self.winfo_exists():
                return
            self._user_scrolled_up = False
            if self.chat_lines:
                speaker, prev_text, color, is_user = self.chat_lines[-1]
                self.chat_lines[-1] = (speaker, f"{prev_text}\n{lines[idx]}", color, is_user)
            if idx + 1 < len(lines):
                self._render_messages(smooth_scroll=True)
                self._line_reveal_id = self.after(LINE_REVEAL_DELAY_MS, lambda: _reveal_next(idx + 1))
            else:
                self.is_waiting_reply = False
                self._stop_typing_indicator()
                self._finish_reply_emotion(reply_text)
                self._set_status("", TERM_MUTED)
                self._render_messages(smooth_scroll=True)
                self._refresh_input()

        self._line_reveal_id = self.after(LINE_REVEAL_DELAY_MS, lambda: _reveal_next(1))

    def handle_key(self, event: tk.Event) -> None:
        if not self.is_waiting_reply:
            if self.current_emotion == "sleeping":
                self._set_emotion("idle")
            if self.current_emotion == "idle":
                self._reset_afk_timer()

        if event.keysym == "Escape":
            self.on_back()
            return

        if event.keysym in ("Up", "Down"):
            content_h = self.messages_frame.winfo_reqheight()
            canvas_h = self._effective_canvas_height()
            if content_h > canvas_h:
                self._stop_scroll_anim()
                self.chat_canvas.yview_scroll(-1 if event.keysym == "Up" else 1, "units")
                self._user_scrolled_up = float(self.chat_canvas.yview()[1]) < 0.999
            return

        if event.keysym in ("Return", "KP_Enter"):
            if event.state & 0x1:
                self.typed_text += "\n"
                self._refresh_input()
                return
            if self.is_waiting_reply:
                self._set_status("Дочекайтеся відповіді!", TERM_RED)
                return
            msg = self.typed_text.strip()
            if not msg:
                return
            self.typed_text = ""
            self.is_waiting_reply = True
            self._user_scrolled_up = False
            self._set_status("", TERM_MUTED)
            self._cancel_smile_timer()
            self._cancel_afk_timer()
            if self.current_emotion != "idle":
                self._set_emotion("idle")
            self.chat_lines.append((self.user_name, msg, TERM_WHITE, True))
            self._start_typing_indicator()
            self._refresh_input()

            req_gen = self._session_generation

            def _deliver(reply: str) -> None:
                if self.winfo_exists() and req_gen == self._session_generation:
                    self.after(0, lambda: self._on_reply_received(reply, expected_gen=req_gen))

            self.on_send_message(msg, _deliver)
            return

        if event.keysym == "BackSpace":
            self.typed_text = self.typed_text[:-1]
            self._refresh_input()
            return

        if _is_paste_event(event):
            pasted = _read_clipboard_text(self, multiline=True)
            if pasted:
                self.typed_text += pasted
                self._refresh_input()
            return

        if event.char and event.char.isprintable() and not _has_non_text_modifier(event):
            self.typed_text += event.char
            self._refresh_input()

    def handle_paste(self) -> None:
        if not self.is_waiting_reply:
            if self.current_emotion == "sleeping":
                self._set_emotion("idle")
            if self.current_emotion == "idle":
                self._reset_afk_timer()
        pasted = _read_clipboard_text(self, multiline=True)
        if pasted:
            self.typed_text += pasted
            self._refresh_input()

    def destroy(self) -> None:
        self._stop_typing_indicator()
        self._stop_talking_anim()
        self._cancel_smile_timer()
        self._cancel_afk_timer()
        self._stop_scroll_anim()
        for timer in (self._blink_id, self._line_reveal_id):
            if timer is not None:
                try:
                    self.after_cancel(timer)
                except Exception:
                    pass
        self._blink_id = None
        self._line_reveal_id = None
        super().destroy()


class ConfirmExitOverlay(tk.Frame):
    """
    Перевикористовуваний дворядковий модальний блок підтвердження виходу (`CTRL+C`) внизу зліва:
      Рядок 1: "Ви дійсно хочете вийти?"
      Рядок 2: "Так Ні"
    Займає рівно ті ж 2 фіксовані рядки, що й стандартний футер, тому не зсуває текст на екрані.
    """

    def __init__(
        self,
        parent: tk.Widget,
        on_confirm_exit: Callable[[], None],
        on_cancel: Callable[[], None],
        term_font: tkfont.Font,
    ) -> None:
        super().__init__(parent, bg=BG_BLACK, bd=0, highlightthickness=0)
        self.on_confirm_exit = on_confirm_exit
        self.on_cancel = on_cancel
        self.choice_yes = False

        make_term_label(
            self,
            font=term_font,
            text="Вийти з GVK?  ",
            fg=TERM_RED,
        ).pack(anchor="w")

        options_row = tk.Frame(self, bg=BG_BLACK)
        options_row.pack(anchor="w")

        self.yes_label = make_term_label(
            options_row,
            font=term_font,
            text="Так",
            fg=TERM_WHITE,
        )
        self.yes_label.pack(side=tk.LEFT)

        make_term_label(
            options_row,
            font=term_font,
            text=" ",
            fg=TERM_WHITE,
        ).pack(side=tk.LEFT)

        self.no_label = make_term_label(
            options_row,
            font=term_font,
            text="Ні",
            fg=TERM_WHITE,
        )
        self.no_label.pack(side=tk.LEFT)

        self._refresh_choice()

    def _refresh_choice(self) -> None:
        if self.choice_yes:
            self.yes_label.config(bg=TERM_RED, fg=BG_BLACK)
            self.no_label.config(bg=BG_BLACK, fg=TERM_WHITE)
        else:
            self.yes_label.config(bg=BG_BLACK, fg=TERM_WHITE)
            self.no_label.config(bg=TERM_CYAN, fg=BG_BLACK)

    def handle_key(self, event: tk.Event) -> None:
        if event.keysym in ("Left", "Right", "Tab"):
            self.choice_yes = not self.choice_yes
            self._refresh_choice()
            return

        if event.keysym == "Escape":
            self.on_cancel()
            return

        if event.keysym in ("Return", "KP_Enter"):
            if self.choice_yes:
                self.on_confirm_exit()
            else:
                self.on_cancel()


# =====================================================================
# ГОЛОВНИЙ ДОДАТОК GVK MENU SHELL
# =====================================================================
def format_legend_pair(key: str, action: str, target_width: int) -> str:
    """Вирівнює пару `КЛАВІША = Дія` пробілами навколо `=` до заданої ширини у символах."""
    total_spaces = max(2, target_width - len(key) - len(action) - 1)
    left_spaces = total_spaces // 2
    right_spaces = total_spaces - left_spaces
    return f"{key}{' ' * left_spaces}={' ' * right_spaces}{action}"


class GVKMenuApp:
    """
    Головна оболонка `core/menu.py` (1280x720, без зміни розміру, класичний термінал).
    Зберігає вибраного користувача у `self.selected_user_name` та `self.selected_user_id`
    для подальшої передачі в `core/orchestrator.py`.
    """

    def __init__(
        self,
        wait_seconds: float = DEFAULT_WAIT_SECONDS,
        last_warning_seconds: Optional[float] = None,
        orchestrator: Optional[Any] = None,
        on_user_selected_callback: Optional[Callable[[int, str, bool], None]] = None,
        on_persona_selected_callback: Optional[Callable[[str, str], None]] = None,
        persona_items: Optional[List[Dict[str, object]]] = None,
        persona_colors: Optional[Dict[str, str]] = None,
        afk_seconds: float = DEFAULT_AFK_SECONDS,
        headless: bool = False,
    ) -> None:
        self.afk_seconds = max(0.05, float(afk_seconds))
        self.wait_seconds = max(0.0, float(wait_seconds))
        if last_warning_seconds is not None:
            self.last_warning_seconds = max(0.0, float(last_warning_seconds))
        elif self.wait_seconds == float(DEFAULT_WAIT_SECONDS):
            self.last_warning_seconds = float(LAST_WARNING_WAIT_SECONDS)
        else:
            self.last_warning_seconds = self.wait_seconds

        self.orchestrator = orchestrator
        if self.orchestrator is not None:
            apply_roster_config(
                load_roster_config(
                    ram_root=getattr(self.orchestrator, "ram_root", None),
                    repo_root=getattr(self.orchestrator, "repo_root", REPO_ROOT),
                    gvk_home=getattr(self.orchestrator, "gvk_home", None),
                )
            )
        else:
            apply_roster_config(load_roster_config())
        self.on_user_selected_callback = on_user_selected_callback
        self.on_persona_selected_callback = on_persona_selected_callback
        self.persona_items: List[Dict[str, object]] = (
            [dict(item) for item in persona_items]
            if persona_items is not None
            else [dict(item) for item in PERSONA_SELECT_ITEMS]
        )
        self.persona_colors: Dict[str, str] = {}
        for item in self.persona_items:
            pid = str(item.get("persona_id") or "")
            pcol = str(item.get("color") or "")
            if pid and pcol:
                self.persona_colors[pid] = pcol
        if persona_colors:
            for key, val in persona_colors.items():
                if val:
                    self.persona_colors[str(key)] = str(val)
            for item in self.persona_items:
                pid = str(item.get("persona_id") or "")
                if pid in self.persona_colors:
                    item["color"] = self.persona_colors[pid]
        if self.orchestrator is not None and hasattr(self.orchestrator, "config"):
            orch_pid = str(self.orchestrator.config.get("persona_id") or "")
            orch_col = str(self.orchestrator.config.get("color") or "")
            if orch_pid and orch_col:
                self.persona_colors.setdefault(orch_pid, orch_col)
                for item in self.persona_items:
                    if str(item.get("persona_id") or "") == orch_pid and not item.get("color"):
                        item["color"] = orch_col

        self.selected_user_id: Optional[int] = None
        self.selected_user_name: Optional[str] = None
        self.selected_user_experimental: bool = False
        self.selected_persona_id: Optional[str] = None
        self.sudo_mode: bool = False
        self.headless: bool = bool(headless)

        self.root = tk.Tk()
        if self.headless:
            self.root.withdraw()
        self.root.title(APP_NAME)
        self.root.configure(bg=BG_BLACK)
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.minsize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.root.maxsize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.root.resizable(False, False)

        # Єдиний моноширинний шрифт одного розміру для всього терміналу
        mono_family = "Consolas" if sys.platform.startswith("win") else "Courier"
        self.term_font = tkfont.Font(family=mono_family, size=12, weight="normal")

        # Головний внутрішній контейнер (без бокового padx, щоб лінії в чаті доходили до країв вікна)
        self.viewport = tk.Frame(self.root, bg=BG_BLACK)
        self.viewport.pack(fill=tk.BOTH, expand=True, pady=12)

        # Постійний верхній заголовок посередині: GVK PRIVATE AI [1.0.0] (просто білий)
        self.header_frame = tk.Frame(self.viewport, bg=BG_BLACK)
        self.header_frame.pack(side=tk.TOP, fill=tk.X, padx=LEFT_PAD)
        self.header_label = make_term_label(
            self.header_frame,
            font=self.term_font,
            text=APP_HEADER,
            fg=TERM_WHITE,
            anchor="center",
        )
        self.header_label.pack(anchor="center")

        # Другий рядок верхньої стрічки (Співрозмовник зліва, Увійшли як справа)
        self.sub_header_frame = tk.Frame(self.viewport, bg=BG_BLACK)
        self.sub_header_frame.pack(side=tk.TOP, fill=tk.X, padx=LEFT_PAD)
        self.top_peers_label = make_term_label(
            self.sub_header_frame,
            font=self.term_font,
            text="",
            fg=TERM_CYAN,
        )
        self.top_peers_label.pack(side=tk.LEFT, anchor="w")
        self.top_user_label = make_term_label(
            self.sub_header_frame,
            font=self.term_font,
            text="",
            fg=TERM_CYAN,
            anchor="e",
        )
        self.top_user_label.pack(side=tk.RIGHT, anchor="e")

        # Верхня біла лінія під верхньою стрічкою (прихована на екрані-заставці через BG_BLACK)
        self.top_white_line = tk.Frame(self.viewport, bg=BG_BLACK, height=1)
        self.top_white_line.pack(side=tk.TOP, fill=tk.X, pady=(4, 0))

        # Постійний дворядковий нижній бар:
        #   Ліворуч: слот CTRL+C = Вийти (2 рядки під "Вийти з GVK? / Так Ні")
        #   Праворуч: 2 рядки на всю решту ширини:
        #     Рядок 1: усі легенди клавіш В ОДИН РЯДОК
        #     Рядок 2: статус / відлік / помилки справа (не заважає 1-му рядку)
        self.footer_frame = tk.Frame(self.viewport, bg=BG_BLACK)
        self.footer_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=LEFT_PAD)

        # Елемент 1: CTRL+C = Вийти (або ConfirmExitOverlay тієї ж ширини)
        self.footer_col1_slot = tk.Frame(self.footer_frame, bg=BG_BLACK)
        self.footer_col1_slot.pack(side=tk.LEFT, anchor="nw", padx=(0, 16))

        self.default_col1_box = tk.Frame(self.footer_col1_slot, bg=BG_BLACK)
        self.default_col1_box.pack(anchor="nw")
        make_term_label(
            self.default_col1_box,
            font=self.term_font,
            text="CTRL+C = Вийти",
            fg=TERM_MUTED,
        ).pack(anchor="w")
        make_term_label(
            self.default_col1_box,
            font=self.term_font,
            text="",
            fg=TERM_MUTED,
        ).pack(anchor="w")

        self.footer_right_area = tk.Frame(self.footer_frame, bg=BG_BLACK)
        self.footer_right_area.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.footer_legend_frame = tk.Frame(self.footer_right_area, bg=BG_BLACK)
        self.footer_legend_frame.pack(side=tk.TOP, fill=tk.X, anchor="nw")

        # Динамічні легенди в 1-му рядку: компенсуємо пробілами лише коротші варіанти
        self._arrows_width = len("СТРІЛОЧКИ = Навігація")
        self._enter_width = len("ENTER = Підтвердити")

        self.footer_arrows_label = make_term_label(
            self.footer_legend_frame,
            font=self.term_font,
            text="СТРІЛОЧКИ = Навігація",
            fg=TERM_MUTED,
        )
        self.footer_arrows_label.pack(side=tk.LEFT, anchor="nw", padx=(0, 16))

        self.footer_enter_label = make_term_label(
            self.footer_legend_frame,
            font=self.term_font,
            text="ENTER = Підтвердити",
            fg=TERM_MUTED,
        )
        self.footer_enter_label.pack(side=tk.LEFT, anchor="nw", padx=(0, 16))

        # Додаткові легенди в 1-му рядку (лише в чаті)
        self.footer_chat_legends = tk.Frame(self.footer_legend_frame, bg=BG_BLACK)
        make_term_label(
            self.footer_chat_legends,
            font=self.term_font,
            text="ESC = Повернутися",
            fg=TERM_MUTED,
        ).pack(side=tk.LEFT, anchor="nw", padx=(0, 16))
        make_term_label(
            self.footer_chat_legends,
            font=self.term_font,
            text="SHIFT+ENTER = Новий рядок",
            fg=TERM_MUTED,
        ).pack(side=tk.LEFT, anchor="nw", padx=(0, 16))
        self.footer_new_chat_label = make_term_label(
            self.footer_chat_legends,
            font=self.term_font,
            text="CTRL+N = Новий чат",
            fg=TERM_MUTED,
        )
        self.footer_new_chat_label.pack(side=tk.LEFT, anchor="nw")

        # Правий статус/помилка/відлік на нижньому (2-му) рядку справа
        footer_row2 = tk.Frame(self.footer_right_area, bg=BG_BLACK)
        footer_row2.pack(side=tk.TOP, fill=tk.X)
        self.footer_status_label = make_term_label(
            footer_row2,
            font=self.term_font,
            text="",
            fg=TERM_RED,
            anchor="e",
            justify=tk.RIGHT,
        )
        self.footer_status_label.pack(side=tk.RIGHT, anchor="e")

        # Нижня біла лінія над нижньою стрічкою (піднята на 3px; прихована на заставці через BG_BLACK)
        self.bottom_white_line = tk.Frame(self.viewport, bg=BG_BLACK, height=1)
        self.bottom_white_line.pack(side=tk.BOTTOM, fill=tk.X, pady=(0, 3))

        # Динамічна робоча область між заголовком і футером
        self.stage_frame = tk.Frame(self.viewport, bg=BG_BLACK)
        self.stage_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.active_modal: Optional[tk.Widget] = None
        self.exit_overlay: Optional[ConfirmExitOverlay] = None
        self._last_paste_ts: float = 0.0

        # Перехоплення клавіш, CTRL+C, CTRL+N та вставки тексту
        self.root.bind("<Key>", self._on_key_press)
        self.root.bind("<Control-c>", self._on_ctrl_c)
        self.root.bind("<Control-C>", self._on_ctrl_c)
        self.root.bind("<Control-n>", self._on_ctrl_n)
        self.root.bind("<Control-N>", self._on_ctrl_n)
        self.root.bind("<Control-v>", self._on_paste)
        self.root.bind("<Control-V>", self._on_paste)
        self.root.bind("<<Paste>>", self._on_paste)
        if sys.platform == "darwin":
            self.root.bind("<Command-v>", self._on_paste)
            self.root.bind("<Command-V>", self._on_paste)
        self.root.protocol("WM_DELETE_WINDOW", self.show_exit_overlay)

        # Стартуємо з 20-секундного екрана авторських прав
        self.show_initial_splash()

    def set_footer_status(self, text: str = "", fg: str = TERM_RED) -> None:
        self.footer_status_label.config(text=text, fg=fg)

    def _set_modal(self, modal: tk.Widget, is_splash: bool = False) -> None:
        if self.active_modal is not None:
            self.active_modal.destroy()
        self.active_modal = modal
        self.set_footer_status("")

        line_color = BG_BLACK if is_splash else TERM_WHITE
        self.top_white_line.config(bg=line_color)
        self.bottom_white_line.config(bg=line_color)

        if isinstance(modal, ChatSessionModal):
            is_group = len(modal.persona_names) > 1
            peer_prefix = "Співрозмовники" if is_group else "Співрозмовник"
            peers_display = ", ".join(modal.persona_names)
            peers_fg = TERM_CYAN if is_group else modal.persona_color
            self.top_peers_label.config(text=f"{peer_prefix}: {peers_display}", fg=peers_fg)
            self.top_user_label.config(text=f"Увійшли як: {modal.user_name}", fg=TERM_CYAN, bg=BG_BLACK)
            modal.top_peers_label = self.top_peers_label
            modal.top_user_label = self.top_user_label
            self.active_modal.pack(fill=tk.BOTH, expand=True)
            self.footer_arrows_label.config(
                text=format_legend_pair("СТРІЛОЧКИ", "Гортати", self._arrows_width)
            )
            self.footer_enter_label.config(
                text=format_legend_pair("ENTER", "Надіслати", self._enter_width)
            )
            self.footer_chat_legends.pack(side=tk.LEFT, anchor="nw")
        elif isinstance(modal, PersonaLoadingModal):
            self.top_peers_label.config(text=f"Співрозмовник: {modal.persona_name}", fg=modal.persona_color)
            self.top_user_label.config(text=f"Увійшли як: {modal.user_name}", fg=TERM_CYAN, bg=BG_BLACK)
            self.active_modal.pack(fill=tk.BOTH, expand=True, padx=LEFT_PAD)
            self.footer_arrows_label.config(text="СТРІЛОЧКИ = Навігація")
            self.footer_enter_label.config(text="ENTER = Підтвердити")
            self.footer_chat_legends.pack_forget()
        elif isinstance(modal, MainMenuModal):
            self.top_peers_label.config(text="")
            self.top_user_label.config(text=f"Увійшли як: {modal.user_name}", fg=TERM_CYAN, bg=BG_BLACK)
            self.active_modal.pack(fill=tk.BOTH, expand=True, padx=LEFT_PAD)
            self.footer_arrows_label.config(text="СТРІЛОЧКИ = Навігація")
            self.footer_enter_label.config(text="ENTER = Підтвердити")
            self.footer_chat_legends.pack_forget()
        else:
            self.top_peers_label.config(text="")
            self.top_user_label.config(text="", fg=TERM_CYAN, bg=BG_BLACK)
            self.active_modal.pack(fill=tk.BOTH, expand=True, padx=LEFT_PAD)
            self.footer_arrows_label.config(text="СТРІЛОЧКИ = Навігація")
            self.footer_enter_label.config(text="ENTER = Підтвердити")
            self.footer_chat_legends.pack_forget()

    # -----------------------------------------------------------------
    # ЕТАПИ ОНБОРДИНГУ ТА ГОЛОВНОГО МЕНЮ
    # -----------------------------------------------------------------
    def show_initial_splash(self) -> None:
        modal = TimedMessageModal(
            self.stage_frame,
            body_text=SPLASH_MESSAGE,
            wait_seconds=self.wait_seconds,
            on_complete=self.show_user_selector,
            term_font=self.term_font,
            warning_title=None,
            body_color=TERM_WHITE,
            require_enter=False,
            on_status=self.set_footer_status,
        )
        self._set_modal(modal, is_splash=True)

    def show_user_selector(self) -> None:
        modal = UserSelectorModal(
            self.stage_frame,
            users_dict=USERS_DICT,
            on_select=self._on_user_selected,
            term_font=self.term_font,
            initial_sudo=self.sudo_mode,
            on_status=self.set_footer_status,
        )
        self._set_modal(modal)

    def _on_user_selected(self, user_id: int, user_name: str, sudo_unlocked: bool) -> None:
        self.selected_user_id = user_id
        self.selected_user_name = user_name
        self.sudo_mode = sudo_unlocked
        entry = USERS_DICT.get(user_id, {})
        self.selected_user_experimental = bool(entry.get("experimental", user_id >= 7))
        logger.info(
            "Користувач обрав профіль: [%02d] %s (experimental=%s, sudo=%s)",
            user_id,
            user_name,
            self.selected_user_experimental,
            sudo_unlocked,
        )
        if self.on_user_selected_callback is not None:
            self.on_user_selected_callback(user_id, user_name, self.selected_user_experimental)

        if self.sudo_mode:
            self.show_sudo_lecture()
        else:
            self.show_warning_step(0)

    def show_sudo_lecture(self) -> None:
        modal = TimedMessageModal(
            self.stage_frame,
            body_text=SUDO_LECTURE_MESSAGE,
            wait_seconds=min(self.wait_seconds, float(SUDO_LECTURE_WAIT_SECONDS)),
            on_complete=self.show_main_menu,
            term_font=self.term_font,
            warning_title=None,
            body_color=TERM_PURPLE,
            require_enter=False,
            on_status=self.set_footer_status,
        )
        self._set_modal(modal)

    def show_warning_step(self, index: int) -> None:
        if index >= len(WARNING_SCREENS):
            self.show_launch_confirmation()
            return

        title, body = WARNING_SCREENS[index]
        is_last = index == len(WARNING_SCREENS) - 1
        step_wait = self.last_warning_seconds if is_last else self.wait_seconds
        modal = TimedMessageModal(
            self.stage_frame,
            body_text=body,
            wait_seconds=step_wait,
            on_complete=lambda: self.show_warning_step(index + 1),
            term_font=self.term_font,
            warning_title=title,
            title_color=TERM_RED if is_last else TERM_AMBER,
            body_color=TERM_WHITE,
            require_enter=True,
            on_status=self.set_footer_status,
        )
        self._set_modal(modal)

    def show_launch_confirmation(self) -> None:
        modal = TypingConfirmationModal(
            self.stage_frame,
            instruction_text=LAUNCH_INSTRUCTION,
            target_display=LAUNCH_PHRASE_DISPLAY,
            validator=is_valid_launch_phrase,
            on_success=lambda _typed: self.show_main_menu(),
            term_font=self.term_font,
            on_status=self.set_footer_status,
        )
        self._set_modal(modal)

    def show_main_menu(self) -> None:
        modal = MainMenuModal(
            self.stage_frame,
            user_name=self.selected_user_name or "Невідомий",
            menu_items=MAIN_MENU_ITEMS,
            on_switch_user=self.show_user_selector,
            on_select_item=self._on_main_menu_action,
            term_font=self.term_font,
            persona_items=self.persona_items,
            on_select_persona=self._on_persona_selected,
            on_status=self.set_footer_status,
            external_user_label=self.top_user_label,
        )
        self._set_modal(modal)

    def _on_main_menu_action(self, item_id: int, label: str) -> None:
        logger.info("Обрано пункт головного меню: [%02d] %s", item_id, label)
        self.set_footer_status(f"Вибрано: [{item_id:02d}] {label}", TERM_GREEN)

    def _on_persona_selected(self, persona_id: str, persona_label: str) -> None:
        self.selected_persona_id = persona_id
        logger.info("Обрано персонажа для діалогу: %s (%s)", persona_id, persona_label)
        if self.on_persona_selected_callback is not None:
            self.on_persona_selected_callback(persona_id, persona_label)
        self.show_persona_loading(persona_id, persona_label)

    def _resolve_persona_color(self, persona_id: str, persona_label: str = "") -> str:
        if persona_id in self.persona_colors:
            return self.persona_colors[persona_id]
        if persona_label and persona_label in self.persona_colors:
            return self.persona_colors[persona_label]
        for item in self.persona_items:
            if str(item.get("persona_id") or "") == persona_id and item.get("color"):
                return str(item["color"])
        if self.orchestrator is not None and hasattr(self.orchestrator, "config"):
            orch_color = self.orchestrator.config.get("color")
            if orch_color:
                return str(orch_color)
        return TERM_WHITE

    def _resolve_persona_genitive(self, persona_id: str, persona_label: str = "") -> str:
        if self.orchestrator is not None:
            orch_pid = str(getattr(self.orchestrator, "config", {}).get("persona_id") or "")
            if not orch_pid or orch_pid == persona_id:
                if hasattr(self.orchestrator, "get_genitive_name"):
                    return str(self.orchestrator.get_genitive_name())
                cfg = getattr(self.orchestrator, "config", {}) or {}
                gen = cfg.get("display_name_genitive") or cfg.get("genitive_name")
                if gen:
                    return str(gen)
        for item in self.persona_items:
            if str(item.get("persona_id") or "") == persona_id and item.get("genitive_name"):
                return str(item["genitive_name"])
        return persona_label or persona_id

    def _resolve_persona_loading_tips(self, persona_id: str) -> List[str]:
        if self.orchestrator is not None:
            orch_pid = str(getattr(self.orchestrator, "config", {}).get("persona_id") or "")
            if not orch_pid or orch_pid == persona_id:
                if hasattr(self.orchestrator, "get_loading_tips"):
                    tips = self.orchestrator.get_loading_tips()
                    if tips:
                        return list(tips)
                cfg = getattr(self.orchestrator, "config", {}) or {}
                raw_tips = cfg.get("loading_tips")
                if isinstance(raw_tips, list) and raw_tips:
                    return [str(t) for t in raw_tips if str(t).strip()]
        for item in self.persona_items:
            if str(item.get("persona_id") or "") == persona_id:
                item_tips = item.get("loading_tips")
                if isinstance(item_tips, list) and item_tips:
                    return [str(t) for t in item_tips if str(t).strip()]
        return ["loading tip 1", "loading tip 2", "loading tip 3"]

    def show_persona_loading(
        self,
        persona_id: str,
        persona_label: str,
        persona_color: Optional[str] = None,
        background_task: Optional[Callable[[], None]] = None,
    ) -> None:
        resolved_color = persona_color or self._resolve_persona_color(persona_id, persona_label)
        genitive_name = self._resolve_persona_genitive(persona_id, persona_label)
        tips = self._resolve_persona_loading_tips(persona_id)
        prep_status = {"ok": True}

        if background_task is None and self.orchestrator is not None:
            def _default_bg_task() -> None:
                if hasattr(self.orchestrator, "prepare_session"):
                    default_u2 = str(USERS_DICT.get(2, {}).get("name") or "Учасник 02")
                    ok = self.orchestrator.prepare_session(
                        user_id=self.selected_user_id or 2,
                        user_name_ua=self.selected_user_name or default_u2,
                        is_experimental=self.selected_user_experimental,
                        chat_mode="personal",
                    )
                    if ok is False:
                        prep_status["ok"] = False

            background_task = _default_bg_task

        def _on_load_complete() -> None:
            warn = "" if prep_status["ok"] else "Увага: llama-server або базова модель недоступні!"
            self.show_chat_session(
                persona_id,
                persona_label,
                persona_color=resolved_color,
                skip_reset=True,
                startup_warning=warn,
            )

        modal = PersonaLoadingModal(
            self.stage_frame,
            user_name=self.selected_user_name or "Невідомий",
            persona_name=persona_label,
            genitive_name=genitive_name,
            loading_tips=tips,
            persona_color=resolved_color,
            term_font=self.term_font,
            on_complete=_on_load_complete,
            background_task=background_task,
            bar_duration_seconds=60.0,
            on_status=self.set_footer_status,
        )
        self._set_modal(modal)
        modal.start()

    def show_chat_session(
        self,
        persona_id: str,
        persona_label: str,
        persona_color: Optional[str] = None,
        skip_reset: bool = False,
        startup_warning: str = "",
    ) -> None:
        resolved_color = persona_color or self._resolve_persona_color(persona_id, persona_label)
        portrait_text = ""
        emotion_sprites: Dict[str, List[str]] = {}
        dossier_info: Optional[Dict[str, str]] = None

        if self.orchestrator is not None:
            try:
                if not skip_reset:
                    default_u2 = str(USERS_DICT.get(2, {}).get("name") or "Учасник 02")
                    self.orchestrator.set_interlocutor(
                        user_id=self.selected_user_id or 2,
                        user_name_ua=self.selected_user_name or default_u2,
                        is_experimental=self.selected_user_experimental,
                        chat_mode="personal",
                        reset_history=True,
                    )
                orch_pid = str(self.orchestrator.config.get("persona_id") or "")
                if not persona_color and (not orch_pid or orch_pid == persona_id):
                    orch_col = self.orchestrator.config.get("color")
                    if orch_col:
                        resolved_color = str(orch_col)
                if hasattr(self.orchestrator, "get_all_portraits"):
                    emotion_sprites = self.orchestrator.get_all_portraits()
                portrait_text = (emotion_sprites.get("idle") or [None])[0] or self.orchestrator.get_portrait_text("idle")
                if hasattr(self.orchestrator, "get_persona_dossier"):
                    dossier_info = self.orchestrator.get_persona_dossier(current_emotion="idle")
            except Exception as exc:
                logger.exception("Помилка ініціалізації сесії оркестратора: %s", exc)

        def _async_send(user_msg: str, deliver_cb: Callable[[str], None]) -> None:
            if self.orchestrator is None:
                deliver_cb(f"[Оркестратор не підключено. Повідомлення: {user_msg}]")
                return

            def _worker() -> None:
                reply = self.orchestrator.send_message(user_msg)
                deliver_cb(reply)

            threading.Thread(target=_worker, daemon=True).start()

        chat_colors = dict(self.persona_colors)
        chat_colors[persona_label] = resolved_color

        def _on_new_chat() -> None:
            if self.orchestrator is not None:
                if hasattr(self.orchestrator, "clear_history"):
                    self.orchestrator.clear_history()
                elif hasattr(self.orchestrator, "history"):
                    self.orchestrator.history.clear()

        modal = ChatSessionModal(
            self.stage_frame,
            user_name=self.selected_user_name or "Невідомий",
            persona_name=persona_label,
            persona_color=resolved_color,
            portrait_text=portrait_text,
            on_send_message=_async_send,
            on_back=self.show_main_menu,
            term_font=self.term_font,
            persona_colors=chat_colors,
            on_status=self.set_footer_status,
            emotion_sprites=emotion_sprites,
            afk_seconds=self.afk_seconds,
            dossier_info=dossier_info,
            on_new_chat=_on_new_chat,
        )
        self._set_modal(modal)
        if startup_warning:
            self.set_footer_status(startup_warning, TERM_AMBER)

    # -----------------------------------------------------------------
    # МОДАЛЬНИЙ ОВЕРЛЕЙ ВИХОДУ (CTRL+C) ТА НОВИЙ ЧАТ (CTRL+N)
    # -----------------------------------------------------------------
    def _on_ctrl_c(self, _event: Optional[tk.Event] = None) -> str:
        self.show_exit_overlay()
        return "break"

    def _on_ctrl_n(self, _event: Optional[tk.Event] = None) -> str:
        if self.exit_overlay is None and isinstance(self.active_modal, ChatSessionModal):
            self.active_modal.reset_chat()
            return "break"
        return ""

    def _on_paste(self, _event: Optional[tk.Event] = None) -> str:
        now = time.monotonic()
        if now - self._last_paste_ts < 0.08:
            return "break"
        self._last_paste_ts = now
        if self.exit_overlay is None and self.active_modal is not None and hasattr(self.active_modal, "handle_paste"):
            self.active_modal.handle_paste()  # type: ignore[attr-defined]
            return "break"
        return ""

    def show_exit_overlay(self) -> None:
        if self.exit_overlay is not None:
            return
        self.default_col1_box.pack_forget()
        self.exit_overlay = ConfirmExitOverlay(
            self.footer_col1_slot,
            on_confirm_exit=self.root.destroy,
            on_cancel=self.hide_exit_overlay,
            term_font=self.term_font,
        )
        self.exit_overlay.pack(anchor="nw")

    def hide_exit_overlay(self) -> None:
        if self.exit_overlay is not None:
            self.exit_overlay.destroy()
            self.exit_overlay = None
        self.default_col1_box.pack(anchor="nw")

    def _on_key_press(self, event: tk.Event) -> None:
        if (event.state & 0x4) and (
            event.keysym.lower() in ("c", "cyrillic_es")
            or (sys.platform.startswith("win") and getattr(event, "keycode", 0) == 67)
        ):
            self.show_exit_overlay()
            return

        if (event.state & 0x4) and (
            event.keysym.lower() in ("n", "cyrillic_te")
            or event.char == "\x0e"
            or (sys.platform.startswith("win") and getattr(event, "keycode", 0) == 78)
        ):
            if self.exit_overlay is None and isinstance(self.active_modal, ChatSessionModal):
                self.active_modal.reset_chat()
            return

        if self.exit_overlay is not None:
            self.exit_overlay.handle_key(event)
            return

        if _is_paste_event(event):
            self._on_paste(event)
            return

        if self.active_modal is not None and hasattr(self.active_modal, "handle_key"):
            self.active_modal.handle_key(event)  # type: ignore[attr-defined]

    def run(self) -> None:
        if not self.headless:
            self.root.update_idletasks()
            self.root.deiconify()
            self.root.lift()
            try:
                self.root.focus_force()
            except Exception:
                pass
        self.root.mainloop()


# =====================================================================
# АВТОМАТИЧНА САМОПЕРЕВІРКА (SELF-TEST)
# =====================================================================
def run_self_test() -> None:
    print("====================================================================")
    print(" [*] GVK Private AI — Самоперевірка core/menu.py")
    print("====================================================================")

    assert len(USERS_DICT) == 17, f"Очікувалось 17 записів у USERS_DICT, отримано {len(USERS_DICT)}"
    assert USERS_DICT[1]["locked"] is True
    assert len(WARNING_SCREENS) == 4
    assert is_valid_launch_phrase(LAUNCH_PHRASE_DISPLAY) is True
    assert is_valid_launch_phrase(LAUNCH_PHRASE_CORE) is True
    assert is_valid_launch_phrase(LAUNCH_PHRASE_CORE.lower()) is False

    app = GVKMenuApp(wait_seconds=0.01, headless=True)
    app.root.update()
    assert app.root.title() == APP_NAME
    assert app.header_label.cget("text") == APP_HEADER
    assert app.header_label.cget("fg") == TERM_WHITE

    # 1. Перевірка розблокування sudo українською розкладкою ("ігвщ") та переходів [12]/[16] <-> [17] і [17] -> Left -> [06]
    app.show_user_selector()
    app.root.update()
    selector = app.active_modal
    assert isinstance(selector, UserSelectorModal)
    assert len(selector.item_labels) == 6

    for ch in "ігвщ":
        ev = tk.Event()
        ev.keysym = ch
        ev.char = ch
        ev.keycode = 0
        ev.state = 0
        selector.handle_key(ev)
    assert selector.sudo_unlocked is True, "Не спрацювало розблокування sudo на українській розкладці (ігвщ)"
    assert len(selector.item_labels) == 17
    assert int(selector.item_labels[17].grid_info()["column"]) == 1

    ev_down = tk.Event()
    ev_down.keysym = "Down"
    ev_down.char = ""
    ev_down.state = 0

    ev_up = tk.Event()
    ev_up.keysym = "Up"
    ev_up.char = ""
    ev_up.state = 0

    ev_left = tk.Event()
    ev_left.keysym = "Left"
    ev_left.char = ""
    ev_left.state = 0

    ev_right = tk.Event()
    ev_right.keysym = "Right"
    ev_right.char = ""
    ev_right.state = 0

    # [06] -> Down -> [07]
    selector.selected_id = 6
    selector.handle_key(ev_down)
    assert selector.selected_id == 7, f"Очікувався перехід [06] -> [07], отримано {selector.selected_id}"

    # [12] -> Down -> [17] -> Up -> [12]
    selector.selected_id = 12
    selector.handle_key(ev_down)
    assert selector.selected_id == 17, f"Очікувався перехід [12] -> [17], отримано {selector.selected_id}"
    selector.handle_key(ev_up)
    assert selector.selected_id == 12, f"Очікувався перехід [17] -> [12], отримано {selector.selected_id}"

    # [16] -> Down -> [17] -> Up -> [12]
    selector.selected_id = 16
    selector.handle_key(ev_down)
    assert selector.selected_id == 17, f"Очікувався перехід [16] -> [17], отримано {selector.selected_id}"
    selector.handle_key(ev_up)
    assert selector.selected_id == 12, f"Очікувався перехід [17] -> [12], отримано {selector.selected_id}"

    # [17] -> Left -> [06], [17] -> Right -> [16]
    selector.selected_id = 17
    selector.handle_key(ev_left)
    assert selector.selected_id == 6, f"Очікувався перехід [17] -> [06] по Left, отримано {selector.selected_id}"
    selector.selected_id = 17
    selector.handle_key(ev_right)
    assert selector.selected_id == 16, f"Очікувався перехід [17] -> [16] по Right, отримано {selector.selected_id}"

    # [17] -> paste у кастомне поле імені
    selector.selected_id = 17
    app.root.clipboard_clear()
    app.root.clipboard_append("ТестовийКористувач")
    selector.handle_paste()
    assert selector.custom_text == "ТестовийКористувач", f"Не спрацював paste у [17]: {selector.custom_text!r}"

    # 2. Перевірка розблокування sudo російською розкладкою ("ыгвщ")
    app.sudo_mode = False
    app.show_user_selector()
    app.root.update()
    selector_ru = app.active_modal
    assert isinstance(selector_ru, UserSelectorModal)
    for ch in "ыгвщ":
        ev = tk.Event()
        ev.keysym = ch
        ev.char = ch
        ev.keycode = 0
        ev.state = 0
        selector_ru.handle_key(ev)
    assert selector_ru.sudo_unlocked is True, "Не спрацювало розблокування sudo на російській розкладці (ыгвщ)"

    # 3. Перевірка вибору користувача [02] та відліку часу
    user2_name = str(USERS_DICT[2]["name"])
    p1_name = str(PERSONA_SELECT_ITEMS[0]["label"]).replace(" (недоступно)", "")
    p3_name = str(PERSONA_SELECT_ITEMS[2]["label"]).replace(" (недоступно)", "")

    app._on_user_selected(2, user2_name, sudo_unlocked=False)
    assert app.selected_user_name == user2_name
    assert isinstance(app.active_modal, TimedMessageModal)

    # 4. Перевірка екрана підтвердження фрази та вставки з буфера обміну (з debounce від подвійного paste)
    app.show_launch_confirmation()
    app.root.update()
    assert isinstance(app.active_modal, TypingConfirmationModal)
    app.root.clipboard_clear()
    app.root.clipboard_append(LAUNCH_PHRASE_CORE)
    app._last_paste_ts = 0.0
    app._on_paste()
    app._on_paste()
    assert app.active_modal.typed_text == LAUNCH_PHRASE_CORE

    # Перевірка групового чату: Співрозмовники у кольорі TERM_CYAN
    group_modal = ChatSessionModal(
        app.stage_frame,
        user_name=user2_name,
        persona_name=p1_name,
        persona_color="#ffaa00",
        portrait_text="",
        on_send_message=lambda _m, _cb: None,
        on_back=app.show_main_menu,
        term_font=app.term_font,
        persona_names=[p1_name, p3_name],
    )
    app._set_modal(group_modal)
    app.root.update()
    assert app.top_peers_label.cget("text") == f"Співрозмовники: {p1_name}, {p3_name}"
    assert app.top_peers_label.cget("fg") == TERM_CYAN

    app.show_main_menu()
    app.root.update()
    assert isinstance(app.active_modal, MainMenuModal)
    assert app.active_modal.user_label.cget("text") == f"Увійшли як: {user2_name}"
    assert app.active_modal.user_label.cget("fg") == TERM_CYAN

    # 5. Перевірка оверлею виходу CTRL+C
    app.show_exit_overlay()
    app.root.update()
    assert app.exit_overlay is not None
    app.hide_exit_overlay()
    assert app.exit_overlay is None

    # 6. Перевірка чату: футер з ESC, вирівнювання багаторядкових повідомлень, анімація крапок та затримка 0..1 с
    assert compute_line_delay_ms("") == 0
    assert compute_line_delay_ms("a" * 40) == 500
    assert compute_line_delay_ms("a" * 80) == 1000
    assert compute_line_delay_ms("a" * 200) == 1000

    app.show_chat_session("friend_1", p1_name, persona_color="#ffaa00")
    app.root.update()
    chat = app.active_modal
    assert isinstance(chat, ChatSessionModal)
    assert chat.persona_color == "#ffaa00"
    assert chat.top_peers_label.cget("text") == f"Співрозмовник: {p1_name}"
    assert chat.top_peers_label.cget("fg") == "#ffaa00"
    assert chat.top_user_label.cget("text") == f"Увійшли як: {user2_name}"
    assert app.top_white_line.cget("bg") == TERM_WHITE
    assert app.bottom_white_line.cget("bg") == TERM_WHITE
    assert app.footer_arrows_label.cget("text") == format_legend_pair("СТРІЛОЧКИ", "Гортати", app._arrows_width)
    assert app.footer_enter_label.cget("text") == format_legend_pair("ENTER", "Надіслати", app._enter_width)
    assert app.footer_chat_legends.winfo_manager() == "pack"

    # Перевірка рендерера Braille-портрету та динамічної зміни емоції
    chat.set_portrait("⠀⣿\n⣿⠀", color="#ffaa00")
    app.root.update()
    assert chat.portrait_canvas.winfo_manager() == "pack"
    assert len(chat.portrait_canvas.find_all()) == 16
    assert int(chat.portrait_canvas.cget("width")) == 2 * chat.portrait_canvas.cell_w
    assert int(chat.portrait_canvas.cget("height")) == 2 * chat.portrait_canvas.cell_h

    # Перевірка Shift+Enter та вставки у чаті
    app.root.clipboard_clear()
    app.root.clipboard_append("вставлено")
    ev_paste = tk.Event()
    ev_paste.keysym = "v"
    ev_paste.char = "\x16"
    ev_paste.state = 0x4
    chat.handle_key(ev_paste)
    assert chat.typed_text == "вставлено"
    chat.typed_text = ""

    ev_shift_ret = tk.Event()
    ev_shift_ret.keysym = "Return"
    ev_shift_ret.char = "\r"
    ev_shift_ret.state = 0x1
    chat.typed_text = "рядок 1"
    chat.handle_key(ev_shift_ret)
    assert chat.typed_text == "рядок 1\n"
    assert chat.input_label.cget("text").startswith("> рядок 1\n> ")
    assert chat.is_waiting_reply is False
    chat.typed_text = ""

    chat.is_waiting_reply = True
    chat.append_message(user2_name, "привіт", TERM_WHITE, is_user=True)
    chat._start_typing_indicator()

    # Поки друг друкує, користувач може вводити текст, але Enter показує помилку "Дочекайтеся відповіді!"
    ev_char = tk.Event()
    ev_char.keysym = "a"
    ev_char.char = "a"
    ev_char.state = 0
    chat.handle_key(ev_char)
    assert chat.typed_text == "a"

    ev_ret = tk.Event()
    ev_ret.keysym = "Return"
    ev_ret.char = "\r"
    ev_ret.state = 0
    chat.handle_key(ev_ret)
    assert app.footer_status_label.cget("text") == "Дочекайтеся відповіді!"
    assert app.footer_status_label.cget("fg") == TERM_RED
    chat.typed_text = ""
    assert chat.typing_label is not None
    assert chat.typing_label.cget("text") == f"{p1_name} друкує"
    chat._schedule_typing_anim()
    assert chat.typing_label.cget("text") == f"{p1_name} друкує."
    chat._schedule_typing_anim()
    assert chat.typing_label.cget("text") == f"{p1_name} друкує.."
    chat._schedule_typing_anim()
    assert chat.typing_label.cget("text") == f"{p1_name} друкує..."
    chat._schedule_typing_anim()
    assert chat.typing_label.cget("text") == f"{p1_name} друкує"

    chat._on_reply_received("ок\nта ні")
    assert chat.chat_lines[-1][1] == "ок"
    assert chat.is_waiting_reply is True
    # Дочекаємось появи другого рядка ("та ні" -> ~63 мс)
    import time

    deadline = time.time() + 1.0
    while chat.is_waiting_reply and time.time() < deadline:
        app.root.update()
        time.sleep(0.01)
    assert chat.is_waiting_reply is False
    assert chat.chat_lines[-1][1] == "ок\nта ні"
    assert chat.typing_label is None
    assert app.footer_status_label.cget("text") == ""

    # Перевірка скролінгу чату при переповненні (Up/Down)
    for i in range(25):
        chat.append_message(p1_name, f"повідомлення {i}", "#ffaa00", is_user=False)
    app.root.update()
    assert chat.messages_frame.winfo_reqheight() > chat._effective_canvas_height()
    chat.handle_key(ev_up)
    app.root.update()
    assert chat._user_scrolled_up is True
    assert float(chat.chat_canvas.yview()[1]) < 0.999

    # Нове повідомлення від друга примусово прокручує чат донизу
    chat._on_reply_received("нове повідомлення знизу")
    assert chat._user_scrolled_up is False
    deadline = time.time() + 1.0
    while float(chat.chat_canvas.yview()[1]) < 0.999 and time.time() < deadline:
        app.root.update()
        time.sleep(0.01)
    assert float(chat.chat_canvas.yview()[1]) >= 0.999

    # Перевірка CTRL+N = Новий чат (підпис у футері + скидання історії на EN та UA розкладках)
    assert app.footer_new_chat_label.cget("text") == "CTRL+N = Новий чат"
    chat.typed_text = "недописаний текст"
    ev_ctrl_n_ua = tk.Event()
    ev_ctrl_n_ua.keysym = "Cyrillic_te"
    ev_ctrl_n_ua.char = "\x0e"
    ev_ctrl_n_ua.state = 0x4
    app._on_key_press(ev_ctrl_n_ua)
    app.root.update()
    assert chat.chat_lines == []
    assert chat.typed_text == ""
    assert chat.current_emotion == "idle"
    assert app.footer_status_label.cget("text") == "Новий чат розпочато."

    # Перевірка групового чату: Співрозмовники у кольорі TERM_CYAN
    group_modal = ChatSessionModal(
        app.stage_frame,
        user_name=user2_name,
        persona_name=p1_name,
        persona_color="#ffaa00",
        portrait_text="",
        on_send_message=lambda _m, _cb: None,
        on_back=app.show_main_menu,
        term_font=app.term_font,
        persona_names=[p1_name, p3_name],
    )
    app._set_modal(group_modal)
    app.root.update()
    assert app.top_peers_label.cget("text") == f"Співрозмовники: {p1_name}, {p3_name}"
    assert app.top_peers_label.cget("fg") == TERM_CYAN

    # Перевірка, що вертикальний розділювач прилягає впритул до правого краю портрета навіть із довгим текстом профілю
    dossier_modal = ChatSessionModal(
        app.stage_frame,
        user_name=user2_name,
        persona_name=p1_name,
        persona_color="#ffaa00",
        portrait_text=("⣿" * 20 + "\n") * 4,
        on_send_message=lambda _m, _cb: None,
        on_back=app.show_main_menu,
        term_font=app.term_font,
        dossier_info={
            "title": p1_name,
            "description": "Дуже довгий опис профілю " * 10,
            "style": "Дуже довгий стиль спілкування без розривів_одного_рядка_який_не_повинен_розсувати_панель",
            "skills_summary": "cmd_very_long_command - Дуже довгий опис команди скіла для перевірки переносу",
        },
    )
    app._set_modal(dossier_modal)
    app.root.update_idletasks()
    app.root.update()
    expected_portrait_w = int(float(dossier_modal.portrait_canvas.cget("width")))
    assert expected_portrait_w > 1
    assert dossier_modal.left_pane.winfo_reqwidth() == expected_portrait_w

    app.show_main_menu()
    app.root.update()
    assert app.footer_arrows_label.cget("text") == "СТРІЛОЧКИ = Навігація"
    assert app.footer_enter_label.cget("text") == "ENTER = Підтвердити"
    assert app.footer_chat_legends.winfo_manager() == ""

    app.show_initial_splash()
    assert app.top_white_line.cget("bg") == BG_BLACK
    assert app.bottom_white_line.cget("bg") == BG_BLACK

    app.root.destroy()
    print(" [✓] Усі компоненти core/menu.py успішно пройшли самоперевірку!")


def main() -> None:
    parser = argparse.ArgumentParser(description="GVK Private AI — Термінальне меню (core/menu.py)")
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
        help="Швидкий режим для розробки (таймери по 1 секунді замість 20/40 с)",
    )
    args = parser.parse_args()

    if args.cmd == "self-test":
        run_self_test()
        return

    wait_sec = 1.0 if args.fast else float(DEFAULT_WAIT_SECONDS)
    last_warn_sec = 2.0 if args.fast else float(LAST_WARNING_WAIT_SECONDS)
    app = GVKMenuApp(wait_seconds=wait_sec, last_warning_seconds=last_warn_sec)
    app.run()


if __name__ == "__main__":
    main()
