@echo off
REM ============================================
REM shifter-tts — Скрипт установки (Windows)
REM ============================================

echo ==========================================
echo   shifter-tts - Installation
echo ==========================================
echo.

REM --- Проверка Python ---
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python не найден. Установите Python >= 3.10 с https://www.python.org/
    echo При установке отметьте "Add Python to PATH"
    pause
    exit /b 1
)

python --version | findstr "Python 3." >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Требуется Python >= 3.10, найден другой версии.
    pause
    exit /b 1
)

echo [OK] Python найден
echo.

REM --- Проверка GPU ---
where nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] NVIDIA GPU обнаружена
) else (
    echo [WARN] NVIDIA GPU не найдена. Приложение будет работать на CPU (медленно).
)

REM --- Создание виртуального окружения ---
set VENV_DIR=venv
if not exist "%VENV_DIR%" (
    echo.
    echo Creating virtual environment...
    python -m venv %VENV_DIR%
)

echo.
echo Activating virtual environment...
call %VENV_DIR%\Scripts\activate.bat

REM --- Обновление pip ---
echo Updating pip...
python -m pip install --upgrade pip setuptools wheel --quiet

REM --- Установка зависимостей ---
echo.
echo Installing dependencies (this may take a while)...
pip install -r requirements.txt --quiet

echo.
echo ==========================================
echo   Installation complete!
echo ==========================================
echo.
echo How to run:
echo   1. Activate environment: venv\Scripts\activate.bat
echo   2. Run the app: python -m tts_app
echo   3. Or: pip install -e . ^&^& shifter-tts
echo.
echo The model will download automatically on first launch.
echo.
echo Documentation: HELP.md
echo.
echo If model requires HuggingFace license:
echo   1. Register (free): https://huggingface.co/join
echo   2. Accept terms on model page
echo   3. Login in terminal:
echo      huggingface-cli login
echo   4. Download model:
echo      python download_model.py base
echo.
pause
