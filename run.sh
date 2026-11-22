#!/usr/bin/env bash
# =====================================================================
# GVK Private AI — Універсальний Zero-Install лаунчер (macOS / Linux)
# Працює з локального репо або з DVD GVK_VAULT (навіть без Git та Python).
# =====================================================================
set -e

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GVK_HOME="${GVK_HOME:-${HOME}/.gvk_ai}"
REPO_URL="https://github.com/Holii-1/Private-AI-Companion-GVK-.git"
REPO_TARBALL_URL="https://github.com/Holii-1/Private-AI-Companion-GVK-/archive/refs/heads/main.tar.gz"
PORTABLE_PY_TAG="20250317"
PORTABLE_PY_VER="3.12.9"

CMD_NAME="${1:-}"
DISC_ARGS=()

has_real_git() {
    if [ "$(uname -s)" = "Darwin" ]; then
        # На чистій macOS /usr/bin/git — заглушка, що відкриває вікно Xcode CLT
        xcode-select -p >/dev/null 2>&1 || return 1
    fi
    command -v git >/dev/null 2>&1
}

export_portable_tk_env() {
    local py_root="${GVK_HOME}/python"
    if [ -d "${py_root}/lib/tcl8.6" ] && [ -z "${TCL_LIBRARY:-}" ]; then
        export TCL_LIBRARY="${py_root}/lib/tcl8.6"
    fi
    if [ -d "${py_root}/lib/tk8.6" ] && [ -z "${TK_LIBRARY:-}" ]; then
        export TK_LIBRARY="${py_root}/lib/tk8.6"
    fi
}

seed_repo_from_disc_bundle() {
    local target_dir="$1"
    if [ -f "${SCRIPT_DIR}/core_bundle/core/main.py" ] && [ -f "${SCRIPT_DIR}/core_bundle/updater.py" ]; then
        mkdir -p "${target_dir}"
        cp -R "${SCRIPT_DIR}/core_bundle/." "${target_dir}/"
        chmod -R u+w "${target_dir}" 2>/dev/null || true
        echo " [OK] Локальне ядро з DVD (core_bundle) розгорнуто у: ${target_dir}"
        return 0
    fi
    return 1
}

sync_repo_via_tarball() {
    local target_dir="$1"
    local tmp_dir archive extracted
    if ! command -v curl >/dev/null 2>&1 || ! command -v tar >/dev/null 2>&1; then
        if [ -f "${target_dir}/core/main.py" ]; then
            echo " [!] curl/tar недоступні — використовуємо локальну копію ядра."
            return 0
        fi
        echo " [X] Помилка: не знайдено git або curl+tar для завантаження ядра." >&2
        return 1
    fi

    tmp_dir="$(mktemp -d "${GVK_HOME}/repo_dl_XXXXXX")"
    archive="${tmp_dir}/repo.tar.gz"
    echo " [*] Завантаження ядра GVK Private AI (без git)..."
    if ! curl -fSL --retry 2 --connect-timeout 15 -o "${archive}" "${REPO_TARBALL_URL}"; then
        rm -rf "${tmp_dir}"
        if [ -f "${target_dir}/core/main.py" ]; then
            echo " [!] Немає мережі — використовуємо кешовану копію ядра."
            return 0
        fi
        echo " [X] Не вдалося завантажити ядро з GitHub. Перевірте інтернет на 1-й запуск." >&2
        return 1
    fi

    if ! tar -xzf "${archive}" -C "${tmp_dir}"; then
        rm -rf "${tmp_dir}"
        echo " [X] Помилка розпакування архіву репозиторію." >&2
        return 1
    fi

    extracted="$(find "${tmp_dir}" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
    if [ -z "${extracted}" ] || [ ! -f "${extracted}/updater.py" ]; then
        rm -rf "${tmp_dir}"
        echo " [X] Некоректний архів репозиторію (відсутній updater.py)." >&2
        return 1
    fi

    mkdir -p "${target_dir}"
    cp -R "${extracted}/." "${target_dir}/"
    rm -rf "${tmp_dir}"
    echo " [✓] Ядро GVK Private AI оновлено у: ${target_dir}"
    return 0
}

is_good_python() {
    local py_bin="$1"
    [ -n "${py_bin}" ] && [ -x "${py_bin}" ] || return 1
    if [[ "${py_bin}" == "${GVK_HOME}/python/"* ]]; then
        export_portable_tk_env
    fi
    "${py_bin}" -c "import sys, tkinter as tk; raise SystemExit(0 if sys.version_info >= (3, 10) and float(tk.TkVersion) >= 8.6 else 1)" >/dev/null 2>&1
}

bootstrap_portable_python() {
    local os_name arch_name triple py_url tmp_dir archive
    os_name="$(uname -s)"
    arch_name="$(uname -m)"
    triple=""

    if [ "${os_name}" = "Darwin" ]; then
        case "${arch_name}" in
            arm64|aarch64) triple="aarch64-apple-darwin" ;;
            x86_64)        triple="x86_64-apple-darwin" ;;
        esac
    elif [ "${os_name}" = "Linux" ]; then
        case "${arch_name}" in
            x86_64|amd64)  triple="x86_64-unknown-linux-gnu" ;;
            aarch64|arm64) triple="aarch64-unknown-linux-gnu" ;;
        esac
    fi

    if [ -z "${triple}" ]; then
        echo " [X] Непідтримувана архітектура (${os_name}/${arch_name}) для автозавантаження Python. Встановіть Python 3.12 вручну." >&2
        return 1
    fi
    if ! command -v curl >/dev/null 2>&1 || ! command -v tar >/dev/null 2>&1; then
        echo " [X] Потрібен curl + tar для автозавантаження портативного Python 3.12." >&2
        return 1
    fi

    mkdir -p "${GVK_HOME}"
    py_url="https://github.com/astral-sh/python-build-standalone/releases/download/${PORTABLE_PY_TAG}/cpython-${PORTABLE_PY_VER}%2B${PORTABLE_PY_TAG}-${triple}-install_only_stripped.tar.gz"
    tmp_dir="$(mktemp -d "${GVK_HOME}/py_dl_XXXXXX")"
    archive="${tmp_dir}/python.tar.gz"

    local usb_py_cand=""
    for cand_f in "${SCRIPT_DIR}/python_portable/"*"${triple}"*.tar.gz; do
        if [ -f "${cand_f}" ]; then
            usb_py_cand="${cand_f}"
            break
        fi
    done

    if [ -n "${usb_py_cand}" ]; then
        echo " [*] [USB-РЕЖИМ] Встановлення портативного Python ${PORTABLE_PY_VER} із носія: $(basename "${usb_py_cand}")..." >&2
        cp "${usb_py_cand}" "${archive}"
    else
        echo " [*] Python 3.10+ (з Tk >= 8.6) не знайдено — завантажуємо портативний Python ${PORTABLE_PY_VER} (~35 МБ)..." >&2
        if ! curl -fSL --retry 2 --connect-timeout 15 -o "${archive}" "${py_url}"; then
            rm -rf "${tmp_dir}"
            echo " [X] Не вдалося завантажити портативний Python. Перевірте інтернет або встановіть Python 3.12 вручну." >&2
            return 1
        fi
    fi

    if ! tar -xzf "${archive}" -C "${tmp_dir}" || [ ! -x "${tmp_dir}/python/bin/python3" ]; then
        rm -rf "${tmp_dir}"
        echo " [X] Помилка розпакування портативного Python." >&2
        return 1
    fi

    rm -rf "${GVK_HOME}/python"
    mv "${tmp_dir}/python" "${GVK_HOME}/python"
    rm -rf "${tmp_dir}"
    export_portable_tk_env
    echo " [✓] Портативний Python ${PORTABLE_PY_VER} встановлено у: ${GVK_HOME}/python" >&2
    printf '%s\n' "${GVK_HOME}/python/bin/python3"
    return 0
}

resolve_any_python() {
    local cand
    for cand in \
        "${GVK_HOME}/python/bin/python3" \
        "/opt/homebrew/bin/python3.12" \
        "/opt/homebrew/bin/python3.11" \
        "/opt/homebrew/bin/python3.13" \
        "/opt/homebrew/bin/python3" \
        "/usr/local/bin/python3.12" \
        "/usr/local/bin/python3" \
        "$(command -v python3 2>/dev/null || true)" \
        "${REPO_DIR}/.venv/bin/python3" \
        "${GVK_HOME}/.venv/bin/python3"; do
        if [ -n "${cand}" ] && [ -x "${cand}" ]; then
            printf '%s\n' "${cand}"
            return 0
        fi
    done
    return 1
}

resolve_system_python() {
    local cand
    for cand in \
        "${GVK_HOME}/python/bin/python3" \
        "/opt/homebrew/bin/python3.12" \
        "/opt/homebrew/bin/python3.11" \
        "/opt/homebrew/bin/python3.13" \
        "/opt/homebrew/bin/python3" \
        "/usr/local/bin/python3.12" \
        "/usr/local/bin/python3.11" \
        "/usr/local/bin/python3" \
        "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3" \
        "/Library/Frameworks/Python.framework/Versions/Current/bin/python3" \
        "$(command -v python3.12 2>/dev/null || true)" \
        "$(command -v python3.11 2>/dev/null || true)" \
        "$(command -v python3 2>/dev/null || true)" \
        "${REPO_DIR}/.venv/bin/python3" \
        "${GVK_HOME}/.venv/bin/python3"; do
        if is_good_python "${cand}"; then
            printf '%s\n' "${cand}"
            return 0
        fi
    done
    bootstrap_portable_python
}

resolve_venv_python() {
    local cand
    for cand in \
        "${REPO_DIR}/.venv/bin/python3" \
        "${GVK_HOME}/.venv/bin/python3"; do
        if [ -x "${cand}" ]; then
            printf '%s\n' "${cand}"
            return 0
        fi
    done
    resolve_system_python
}

if [ -f "${SCRIPT_DIR}/core/main.py" ] && [ -f "${SCRIPT_DIR}/updater.py" ]; then
    REPO_DIR="${SCRIPT_DIR}"
    echo "===================================================================="
    echo " [*] GVK Private AI — Локальний запуск: ${REPO_DIR}"
    echo "===================================================================="
    if [ -f "${SCRIPT_DIR}/persona/manifest.json" ]; then
        DISC_ARGS=(--disc-path "${SCRIPT_DIR}")
    fi
    if [ "${CMD_NAME}" != "self-test" ] && [ "${CMD_NAME}" != "--self-test" ] && [ "${CMD_NAME}" != "status" ]; then
        if [ -d "${REPO_DIR}/.git" ] && has_real_git; then
            if [ -z "$(git -C "${REPO_DIR}" status --porcelain 2>/dev/null || true)" ]; then
                echo " [*] Перевірка оновлень коду..."
                git -C "${REPO_DIR}" pull --ff-only || echo " [!] Не вдалося виконати git pull (офлайн), продовжуємо..."
            fi
        fi
    fi
else
    REPO_DIR="${GVK_HOME}/repo"
    DISC_ARGS=(--disc-path "${SCRIPT_DIR}")
    MEDIA_MODE="DVD"
    if [ -d "${SCRIPT_DIR}/base" ] || [ -d "${SCRIPT_DIR}/bin" ] || [ -f "${SCRIPT_DIR}/vault/media_mode.json" ]; then
        MEDIA_MODE="USB"
    fi
    echo "===================================================================="
    echo " [*] GVK_VAULT Виявлено: ${SCRIPT_DIR} [Режим: ${MEDIA_MODE}]"
    echo "===================================================================="
    mkdir -p "${GVK_HOME}"
    if [ ! -f "${REPO_DIR}/core/main.py" ]; then
        seed_repo_from_disc_bundle "${REPO_DIR}" || true
    fi
    if [ "${MEDIA_MODE}" = "USB" ] && [ -f "${REPO_DIR}/core/main.py" ]; then
        seed_repo_from_disc_bundle "${REPO_DIR}" || true
    elif has_real_git && { [ -d "${REPO_DIR}/.git" ] || [ ! -f "${REPO_DIR}/core/main.py" ]; }; then
        if [ ! -d "${REPO_DIR}/.git" ]; then
            echo " [*] Перший запуск: клонування ядра GVK Private AI..."
            git clone "${REPO_URL}" "${REPO_DIR}" || sync_repo_via_tarball "${REPO_DIR}"
        else
            echo " [*] Перевірка оновлень ядра та personas_encrypted/..."
            git -C "${REPO_DIR}" pull --ff-only || echo " [!] Не вдалося виконати git pull (офлайн або локальні зміни), продовжуємо..."
        fi
    else
        sync_repo_via_tarball "${REPO_DIR}"
    fi
    # Якщо на носії core_bundle свіжіший за завантажене з GitHub репо — поверх накладаємо файли з носія
    if [ -f "${SCRIPT_DIR}/core_bundle/core/main.py" ] && { [ ! -f "${REPO_DIR}/core/main.py" ] || [ "${SCRIPT_DIR}/core_bundle/core/main.py" -nt "${REPO_DIR}/core/main.py" ]; }; then
        seed_repo_from_disc_bundle "${REPO_DIR}" || true
    fi
fi

if [ "${CMD_NAME}" = "status" ]; then
    shift
    STAT_PY="$(resolve_any_python)" || {
        echo " [!] Python ще не встановлено (буде завантажено автоматично при першому запуску ./run.sh)." >&2
        exit 1
    }
    exec "${STAT_PY}" "${REPO_DIR}/updater.py" status --repo-root "${REPO_DIR}" "$@"
fi

SYS_PY="$(resolve_system_python)" || {
    echo " [X] Помилка: не вдалося знайти або завантажити Python 3.10+ (Tk >= 8.6)." >&2
    exit 1
}
export_portable_tk_env

if [ "${CMD_NAME}" = "self-test" ] || [ "${CMD_NAME}" = "--self-test" ]; then
    shift
    VENV_PY="$(resolve_venv_python)"
    exec "${VENV_PY}" "${REPO_DIR}/core/main.py" self-test "$@"
fi

if [ ${#DISC_ARGS[@]} -gt 0 ]; then
    "${SYS_PY}" "${REPO_DIR}/updater.py" sync --repo-root "${REPO_DIR}" "${DISC_ARGS[@]}"
else
    "${SYS_PY}" "${REPO_DIR}/updater.py" sync --repo-root "${REPO_DIR}"
fi

VENV_PY="$(resolve_venv_python)"

if [ -f "${REPO_DIR}/core/main.py" ]; then
    if [ ${#DISC_ARGS[@]} -gt 0 ]; then
        exec "${VENV_PY}" "${REPO_DIR}/core/main.py" "${DISC_ARGS[@]}" "$@"
    else
        exec "${VENV_PY}" "${REPO_DIR}/core/main.py" "$@"
    fi
else
    echo " [!] Модуль core/main.py ще не створено. Статус кешу:"
    exec "${SYS_PY}" "${REPO_DIR}/updater.py" status --repo-root "${REPO_DIR}"
fi
