# GVK Private AI - Contribution & Documentation Style Guide

This guide defines the canonical documentation, docstring, comment, CLI, and terminal-output conventions for **GVK Private AI**. Follow these rules for all new files and updates so the repository remains uniform, searchable, and safe to maintain.

---

## 1. Language Boundary Policy

To avoid noisy, out-of-sync bilingual comments inside source code while keeping key user documentation accessible in Ukrainian, every surface in the repository follows a strict language boundary:

| Surface | Language | Scope & Rules |
| :--- | :--- | :--- |
| **Developer & Code Surface** | **English only** | Module headers, class & function docstrings, inline comments, section banners, Jupyter notebook markdown & cell headers, CLI `--help` (`argparse`), terminal/log debug outputs (`logger.*`, CLI `print()`, launcher shell output), and deep technical guides in `docs/`. |
| **Main Documentation** | **English + Ukrainian** | `README.md` is the primary English entry point, paired with `README.uk.md` for essential Ukrainian user documentation (Overview, Quick Start, Controls, Releases, and Navigation). Both files must cross-link at the top (`English \| Українська`). |
| **Runtime UI & LLM Prompts** | **Preserve original (RU / UA)** | **Do NOT translate or modify** Tkinter GUI labels/menus/dialogs/splash text (`core/menu.py`), persona prompts (`personas/*/prompt.txt`), `personas/roster.json`, LLM prompt-injection strings (`core/orchestrator.py`, `core/memory_rag.py`), RAG prompt headers, speaker prefixes, emotion-detection keywords, RU/UA stopwords/morphology/regexes, or keyboard-layout mappings (e.g., UA/RU `sudo` input). |

> **Repository root note:** Keep `CONTRIBUTING.md` at the repository root (not inside `docs/`) so GitHub automatically surfaces it in pull requests, issues, and the repository Community profile.

---

## 2. Branding & Prefix Convention

Every tracked script, Python module, launcher, and notebook MUST begin with the exact ASCII hyphen prefix:

```text
GVK Private AI - <Short Descriptive Title> (<relative/path>)
```

- Always use the ASCII hyphen (`-`), not an em-dash (`—`) or double slash (`//`).
- Always include the repository-relative path in parentheses for source files (e.g., `(core/main.py)`).

---

## 3. Canonical Templates

### 3.1 Python Module Header

```python
#!/usr/bin/env python3
"""
GVK Private AI - <Short Module Title> (<relative/path.py>)

<1-2 sentence overview of what this module owns and how it fits in the system.>

Key Responsibilities:
  - <Primary responsibility 1>
  - <Primary responsibility 2>

Usage:
  python3 <relative/path.py> [flags]
  python3 <relative/path.py> --help
"""
```

For package initializers (`__init__.py`), use a concise one-line docstring:

```python
"""GVK Private AI - Core Runtime Package (core/__init__.py)."""
```

### 3.2 Section Banners

Use 69-character `=` borders with concise uppercase English titles:

```python
# =====================================================================
# SECTION TITLE
# =====================================================================
```

### 3.3 Class & Function Docstrings (Google Style)

Write concise, imperative English docstrings for all public and helper classes/functions. Include `Args:`, `Returns:`, `Raises:`, and `Example:` sections whenever they clarify non-trivial inputs, outputs, or CLI/API usage.

```python
def build_rag_index(input_path: str, output_path: str) -> int:
    """Compile a memory text file into a FAISS vector index.

    Parses structured memory blocks, generates multilingual embeddings
    (with an offline n-gram fallback), and writes the FAISS index alongside
    its metadata JSON.

    Args:
        input_path: Path to source memories (`.txt`, `.md`, `.json`, or `.jsonl`).
        output_path: Destination path for `memory.faiss`.

    Returns:
        Total number of indexed memory chunks.

    Raises:
        FileNotFoundError: If `input_path` does not exist.

    Example:
        >>> build_rag_index("memories.txt", "personas/friend_1/memory.faiss")
    """
```

### 3.4 Shell & Batch Launchers (`run.sh`, `run.bat`)

**Bash (`run.sh`):**
```bash
#!/usr/bin/env bash
# =====================================================================
# GVK Private AI - Universal Zero-Install Launcher (macOS / Linux)
#
# Runs from a local clone or a GVK_VAULT DVD (no Git or Python needed).
#
# Usage:
#   ./run.sh
#   ./run.sh --fast
#   ./run.sh status
#   ./run.sh self-test
# =====================================================================
```

**Windows Batch (`run.bat`):**
```bat
@echo off
REM =====================================================================
REM GVK Private AI - Universal Zero-Install Launcher (Windows)
REM
REM Runs from a local clone or a GVK_VAULT DVD (no Git or Python needed).
REM
REM Usage:
REM   run.bat
REM   run.bat --fast
REM   run.bat status
REM   run.bat self-test
REM =====================================================================
```

### 3.5 Jupyter Notebooks (`notebooks/*.ipynb`)

- **Cell 0 (Markdown):** Start with `# GVK Private AI - <Notebook Title>` followed by **Overview**, **Prerequisites**, **Pipeline Stages**, and **Usage**.
- **Code Cells:** Start each cell with a uniform banner:
  ```python
  # =====================================================================
  # [Step N/Total] GVK Private AI - <Step Name>
  # =====================================================================
  ```
- Keep all markdown explanations, code comments, and diagnostic `print()` statements in clear English, while preserving any RU/UA test prompts used to evaluate the Russian/Ukrainian persona model.

### 3.6 CLI `--help`, Logging, and Terminal Debug Output

- All `argparse.ArgumentParser(description=...)` headers should start with `GVK Private AI - ...`.
- Write all `logger.debug/info/warning/error` messages and CLI `print()` diagnostics in clear, concise English.
- Keep consistent status prefixes in CLI scripts:
  - `[*]` — informational / in-progress step
  - `[OK]` — successful completion
  - `[!]` — warning or error
  - `[-]` — skipped step

---

## 4. Verification Checklist for Documentation & Cleanup PRs

When performing text-only / documentation cleanup:

1. **Zero behavior change:** Do not alter control flow, data structures, function signatures, cryptographic constants, or file paths.
2. **UI & prompt safety:** Verify that no Tkinter UI copy, LLM system prompts, RAG prompt headers, or RU/UA linguistic tokenizers/regexes were altered.
3. **Syntax check:** Compile all Python files cleanly:
   ```bash
   python3 -m py_compile core/*.py tools/*.py hardware/generate_3d_dock.py updater.py
   ```
4. **Header check:** Verify every tracked `.py`, `.sh`, `.bat`, and `.ipynb` file begins with `GVK Private AI -`.
5. **Self-tests:** Run the offline self-tests inside the project virtual environment:
   ```bash
   python3 core/main.py self-test
   ```
