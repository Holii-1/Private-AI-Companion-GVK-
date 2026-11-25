@echo off
REM =====================================================================
REM GVK Private AI - Universal Zero-Install Launcher (run.bat)
REM
REM Runs from a local repository clone or a GVK_VAULT DVD/USB media
REM on Windows (no pre-installed Git or Python required).
REM
REM Usage:
REM   run.bat
REM   run.bat --fast
REM   run.bat status
REM   run.bat self-test
REM =====================================================================
chcp 65001 >nul 2>&1
setlocal EnableExtensions

if not defined OMP_NUM_THREADS set "OMP_NUM_THREADS=1"
if not defined MKL_NUM_THREADS set "MKL_NUM_THREADS=1"
if not defined TOKENIZERS_PARALLELISM set "TOKENIZERS_PARALLELISM=false"
if not defined KMP_DUPLICATE_LIB_OK set "KMP_DUPLICATE_LIB_OK=TRUE"

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

if not defined GVK_HOME set "GVK_HOME=%USERPROFILE%\.gvk_ai"
set "REPO_URL=https://github.com/Holii-1/Private-AI-Companion-GVK-.git"
set "REPO_TARBALL_URL=https://github.com/Holii-1/Private-AI-Companion-GVK-/archive/refs/heads/main.tar.gz"
set "PORTABLE_PY_URL=https://github.com/astral-sh/python-build-standalone/releases/download/20250317/cpython-3.12.9%%2B20250317-x86_64-pc-windows-msvc-install_only_stripped.tar.gz"
set "DISC_PATH="
set "CMD_NAME=%~1"

if not exist "%SCRIPT_DIR%\core\main.py" goto :dvd_mode
if not exist "%SCRIPT_DIR%\updater.py" goto :dvd_mode

set "REPO_DIR=%SCRIPT_DIR%"
echo ====================================================================
echo  [*] GVK Private AI - Local Launch: %REPO_DIR%
echo ====================================================================
if exist "%SCRIPT_DIR%\persona\manifest.json" set "DISC_PATH=%SCRIPT_DIR%"
if /I "%CMD_NAME%"=="self-test" goto :resolve_py
if /I "%CMD_NAME%"=="--self-test" goto :resolve_py
if /I "%CMD_NAME%"=="status" goto :resolve_py
if not exist "%REPO_DIR%\.git" goto :resolve_py

where git >nul 2>&1
if errorlevel 1 goto :resolve_py

set "DIRTY_FILES="
for /f "delims=" %%i in ('git -C "%REPO_DIR%" status --porcelain 2^>nul') do set "DIRTY_FILES=1"
if not defined DIRTY_FILES (
    echo  [*] Checking repository updates via git pull...
    git -C "%REPO_DIR%" pull --ff-only || echo  [!] git pull skipped ^(offline^).
)
goto :resolve_py

:dvd_mode
set "REPO_DIR=%GVK_HOME%\repo"
set "DISC_PATH=%SCRIPT_DIR%"
set "MEDIA_MODE=DVD"
if exist "%SCRIPT_DIR%\base\*.gguf" set "MEDIA_MODE=USB"
if exist "%SCRIPT_DIR%\bin\*" set "MEDIA_MODE=USB"
if exist "%SCRIPT_DIR%\vault\media_mode.json" set "MEDIA_MODE=USB"
echo ====================================================================
echo  [*] GVK_VAULT Media Detected: %SCRIPT_DIR% [Mode: %MEDIA_MODE%]
echo ====================================================================
if not exist "%GVK_HOME%" mkdir "%GVK_HOME%"

if exist "%SCRIPT_DIR%\core_bundle\core\main.py" (
    if not exist "%REPO_DIR%" mkdir "%REPO_DIR%"
    xcopy "%SCRIPT_DIR%\core_bundle\*" "%REPO_DIR%\" /E /I /H /Y /Q >nul
    attrib -R "%REPO_DIR%\*.*" /S /D >nul 2>&1
    echo  [OK] Seeded local GVK core from %MEDIA_MODE% ^(core_bundle^).
    if /I "%MEDIA_MODE%"=="USB" goto :resolve_py
)

where git >nul 2>&1
if errorlevel 1 goto :download_repo_tarball

if exist "%REPO_DIR%\.git" (
    echo  [*] Checking repository updates via git pull...
    git -C "%REPO_DIR%" pull --ff-only || echo  [!] git pull skipped ^(offline or local changes^).
    goto :resolve_py
)
if exist "%REPO_DIR%\core\main.py" goto :download_repo_tarball

echo  [*] First launch: cloning GVK Private AI core...
git clone "%REPO_URL%" "%REPO_DIR%"
if not errorlevel 1 goto :resolve_py
echo  [!] git clone failed, trying direct archive download...

:download_repo_tarball
where curl.exe >nul 2>&1
if errorlevel 1 goto :repo_no_tools
where tar.exe >nul 2>&1
if errorlevel 1 goto :repo_no_tools

set "REPO_TMP=%GVK_HOME%\repo_dl_tmp"
set "REPO_ARCHIVE=%GVK_HOME%\repo_dl.tar.gz"
if exist "%REPO_TMP%" rmdir /S /Q "%REPO_TMP%"
if exist "%REPO_ARCHIVE%" del /F /Q "%REPO_ARCHIVE%"
mkdir "%REPO_TMP%"

echo  [*] Downloading GVK Private AI core archive ^(no Git required^)...
curl.exe -fSL --retry 2 --connect-timeout 15 -o "%REPO_ARCHIVE%" "%REPO_TARBALL_URL%"
if errorlevel 1 goto :repo_dl_failed

tar.exe -xzf "%REPO_ARCHIVE%" -C "%REPO_TMP%"
if errorlevel 1 goto :repo_dl_failed

set "EXTRACTED_DIR="
for /d %%D in ("%REPO_TMP%\*") do set "EXTRACTED_DIR=%%~fD"
if not defined EXTRACTED_DIR goto :repo_dl_failed
if not exist "%EXTRACTED_DIR%\updater.py" goto :repo_dl_failed

if not exist "%REPO_DIR%" mkdir "%REPO_DIR%"
xcopy "%EXTRACTED_DIR%\*" "%REPO_DIR%\" /E /I /H /Y /Q >nul
rmdir /S /Q "%REPO_TMP%" >nul 2>&1
del /F /Q "%REPO_ARCHIVE%" >nul 2>&1
echo  [OK] GVK Private AI core synced to: %REPO_DIR%
goto :resolve_py

:repo_dl_failed
if exist "%REPO_TMP%" rmdir /S /Q "%REPO_TMP%" >nul 2>&1
if exist "%REPO_ARCHIVE%" del /F /Q "%REPO_ARCHIVE%" >nul 2>&1
if exist "%REPO_DIR%\core\main.py" (
    echo  [!] Offline - using cached GVK core: %REPO_DIR%
    goto :resolve_py
)
echo  [!] Error: failed to download GVK core from GitHub. Internet is required on first launch.
call :fail 1
exit /b 1

:repo_no_tools
if exist "%REPO_DIR%\core\main.py" (
    echo  [!] Offline / no git or curl - using cached GVK core.
    goto :resolve_py
)
echo  [!] Error: neither git nor curl.exe/tar.exe found. Install Git for Windows and retry.
call :fail 1
exit /b 1

:resolve_py
set "SYS_PY="
set "SYS_FLAG="
if exist "%GVK_HOME%\python\tcl\tcl8.6" if not defined TCL_LIBRARY set "TCL_LIBRARY=%GVK_HOME%\python\tcl\tcl8.6"
if exist "%GVK_HOME%\python\tcl\tk8.6" if not defined TK_LIBRARY set "TK_LIBRARY=%GVK_HOME%\python\tcl\tk8.6"
if exist "%GVK_HOME%\python\lib\tcl8.6" if not defined TCL_LIBRARY set "TCL_LIBRARY=%GVK_HOME%\python\lib\tcl8.6"
if exist "%GVK_HOME%\python\lib\tk8.6" if not defined TK_LIBRARY set "TK_LIBRARY=%GVK_HOME%\python\lib\tk8.6"

if not exist "%GVK_HOME%\python\python.exe" goto :try_py_launcher
"%GVK_HOME%\python\python.exe" -c "import sys, tkinter as tk; raise SystemExit(0 if sys.version_info >= (3, 10) and float(tk.TkVersion) >= 8.6 else 1)" >nul 2>&1
if not errorlevel 1 (
    set "SYS_PY=%GVK_HOME%\python\python.exe"
    goto :sys_ready
)

:try_py_launcher
py -3 -c "import sys, tkinter as tk; raise SystemExit(0 if sys.version_info >= (3, 10) and float(tk.TkVersion) >= 8.6 else 1)" >nul 2>&1
if not errorlevel 1 (
    set "SYS_PY=py"
    set "SYS_FLAG=-3"
    goto :sys_ready
)

python -c "import sys, tkinter as tk; raise SystemExit(0 if sys.version_info >= (3, 10) and float(tk.TkVersion) >= 8.6 else 1)" >nul 2>&1
if not errorlevel 1 (
    set "SYS_PY=python"
    goto :sys_ready
)

if exist "%REPO_DIR%\.venv\Scripts\python.exe" (
    set "SYS_PY=%REPO_DIR%\.venv\Scripts\python.exe"
    goto :sys_ready
)
if exist "%GVK_HOME%\.venv\Scripts\python.exe" (
    set "SYS_PY=%GVK_HOME%\.venv\Scripts\python.exe"
    goto :sys_ready
)

:bootstrap_py
if /I "%CMD_NAME%"=="status" goto :py_missing
where tar.exe >nul 2>&1
if errorlevel 1 goto :py_missing

if not exist "%GVK_HOME%" mkdir "%GVK_HOME%"
set "PY_TMP=%GVK_HOME%\py_dl_tmp"
set "PY_ARCHIVE=%GVK_HOME%\py_portable.tar.gz"
if exist "%PY_TMP%" rmdir /S /Q "%PY_TMP%"
if exist "%PY_ARCHIVE%" del /F /Q "%PY_ARCHIVE%"
mkdir "%PY_TMP%"

set "USB_PY_ARCHIVE="
for %%F in ("%SCRIPT_DIR%\python_portable\*windows*.tar.gz") do set "USB_PY_ARCHIVE=%%~fF"
if defined USB_PY_ARCHIVE (
    echo  [*] [USB-MODE] Unpacking portable Python 3.12 from media: "%USB_PY_ARCHIVE%"...
    tar.exe -xzf "%USB_PY_ARCHIVE%" -C "%PY_TMP%"
    if not errorlevel 1 if exist "%PY_TMP%\python\python.exe" goto :py_extracted_ok
)

where curl.exe >nul 2>&1
if errorlevel 1 goto :py_missing

echo  [*] Python 3.10+ ^(with Tkinter^) not found - downloading portable Python 3.12 ^(~35 MB^)...
curl.exe -fSL --retry 2 --connect-timeout 15 -o "%PY_ARCHIVE%" "%PORTABLE_PY_URL%"
if errorlevel 1 goto :py_dl_failed

tar.exe -xzf "%PY_ARCHIVE%" -C "%PY_TMP%"
if errorlevel 1 goto :py_dl_failed
if not exist "%PY_TMP%\python\python.exe" goto :py_dl_failed

:py_extracted_ok

if exist "%GVK_HOME%\python" rmdir /S /Q "%GVK_HOME%\python"
move "%PY_TMP%\python" "%GVK_HOME%\python" >nul
rmdir /S /Q "%PY_TMP%" >nul 2>&1
del /F /Q "%PY_ARCHIVE%" >nul 2>&1

if exist "%GVK_HOME%\python\tcl\tcl8.6" set "TCL_LIBRARY=%GVK_HOME%\python\tcl\tcl8.6"
if exist "%GVK_HOME%\python\tcl\tk8.6" set "TK_LIBRARY=%GVK_HOME%\python\tcl\tk8.6"
if exist "%GVK_HOME%\python\lib\tcl8.6" if not defined TCL_LIBRARY set "TCL_LIBRARY=%GVK_HOME%\python\lib\tcl8.6"
if exist "%GVK_HOME%\python\lib\tk8.6" if not defined TK_LIBRARY set "TK_LIBRARY=%GVK_HOME%\python\lib\tk8.6"

set "SYS_PY=%GVK_HOME%\python\python.exe"
echo  [OK] Portable Python 3.12 installed to: %GVK_HOME%\python
goto :sys_ready

:py_dl_failed
if exist "%PY_TMP%" rmdir /S /Q "%PY_TMP%" >nul 2>&1
if exist "%PY_ARCHIVE%" del /F /Q "%PY_ARCHIVE%" >nul 2>&1

:py_missing
echo  [!] Error: Python 3.10+ not found and portable download failed. Check internet or install Python 3.12 from python.org.
call :fail 1
exit /b 1

:sys_ready
if /I "%CMD_NAME%"=="status" (
    "%SYS_PY%" %SYS_FLAG% "%REPO_DIR%\updater.py" status --repo-root "%REPO_DIR%"
    exit /b %errorlevel%
)

set "VENV_PY=%SYS_PY%"
set "VENV_FLAG=%SYS_FLAG%"
if exist "%REPO_DIR%\.venv\Scripts\python.exe" (
    set "VENV_PY=%REPO_DIR%\.venv\Scripts\python.exe"
    set "VENV_FLAG="
) else if exist "%GVK_HOME%\.venv\Scripts\python.exe" (
    set "VENV_PY=%GVK_HOME%\.venv\Scripts\python.exe"
    set "VENV_FLAG="
)

if /I "%CMD_NAME%"=="self-test" (
    "%VENV_PY%" %VENV_FLAG% "%REPO_DIR%\core\main.py" self-test
    exit /b %errorlevel%
)
if /I "%CMD_NAME%"=="--self-test" (
    "%VENV_PY%" %VENV_FLAG% "%REPO_DIR%\core\main.py" self-test
    exit /b %errorlevel%
)

if defined DISC_PATH (
    "%SYS_PY%" %SYS_FLAG% "%REPO_DIR%\updater.py" sync --repo-root "%REPO_DIR%" --disc-path "%DISC_PATH%"
) else (
    "%SYS_PY%" %SYS_FLAG% "%REPO_DIR%\updater.py" sync --repo-root "%REPO_DIR%"
)
if errorlevel 1 (
    call :fail %errorlevel%
    exit /b %errorlevel%
)

set "VENV_PY=%SYS_PY%"
set "VENV_FLAG=%SYS_FLAG%"
if exist "%REPO_DIR%\.venv\Scripts\python.exe" (
    set "VENV_PY=%REPO_DIR%\.venv\Scripts\python.exe"
    set "VENV_FLAG="
) else if exist "%GVK_HOME%\.venv\Scripts\python.exe" (
    set "VENV_PY=%GVK_HOME%\.venv\Scripts\python.exe"
    set "VENV_FLAG="
)

if exist "%REPO_DIR%\core\main.py" (
    if defined DISC_PATH (
        "%VENV_PY%" %VENV_FLAG% "%REPO_DIR%\core\main.py" --disc-path "%DISC_PATH%" %*
    ) else (
        "%VENV_PY%" %VENV_FLAG% "%REPO_DIR%\core\main.py" %*
    )
    if errorlevel 1 (
        call :fail %errorlevel%
        exit /b %errorlevel%
    )
    exit /b 0
)

echo  [!] core\main.py not found yet. Cache status:
"%SYS_PY%" %SYS_FLAG% "%REPO_DIR%\updater.py" status --repo-root "%REPO_DIR%"
exit /b %errorlevel%

:fail
echo.
if not defined GVK_NO_PAUSE pause
exit /b %~1
