#!/bin/bash
# ============================================
# shifter-tts — Скрипт установки (Linux / macOS)
# ============================================

set -e  # Останавливаться при ошибках

echo "=========================================="
echo "  shifter-tts — Установка"
echo "=========================================="
echo ""

# --- Проверка Python ---
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 не найден. Установите Python >= 3.10"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "✅ Python $PYTHON_VERSION найден"

# --- Проверка GPU (CUDA) ---
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU обнаружена"
else
    echo "⚠️  NVIDIA GPU не найдена. Приложение будет работать на CPU (медленно)."
fi

# --- Создание виртуального окружения ---
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "📦 Создаю виртуальное окружение..."
    python3 -m venv "$VENV_DIR"
fi

echo ""
echo "🔧 Активирую виртуальное окружение..."
source "$VENV_DIR/bin/activate"

# --- Обновление pip ---
echo "📦 Обновляю pip..."
pip install --upgrade pip setuptools wheel -q

# --- Установка зависимостей ---
echo ""
echo "📦 Устанавливаю зависимости (это может занять время)..."
pip install -r requirements.txt -q

# --- Проверка установки PyTorch ---
python3 -c "import torch; print(f'✅ PyTorch {torch.__version__} OK')" 2>/dev/null || {
    echo ""
    echo "⚠️  PyTorch установлен, но проверка не прошла."
    echo "   Если у вас NVIDIA GPU, убедитесь что CUDA совместима:"
    echo "   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121"
}

# --- Проверка PyQt6 ---
python3 -c "from PyQt6.QtWidgets import QApplication; print('✅ PyQt6 OK')" 2>/dev/null || {
    echo ""
    echo "⚠️  PyQt6 не установлен корректно. Установите вручную:"
    echo "   pip install PyQt6"
}

echo ""
echo "=========================================="
echo "  ✅ Установка завершена!"
echo "=========================================="
echo ""
echo "📌 Как запустить:"
echo "   1. Активируйте окружение: source venv/bin/activate"
echo "   2. Запустите приложение: python -m tts_app"
echo "   3. Или: pip install -e . && shifter-tts"
echo ""
echo "📌 Для загрузки модели впервые — запустите приложение,"
echo "   модель скачается автоматически при первом запуске."
echo ""
echo "📖 Документация: HELP.md

echo ""
echo "🔑 Если модель требует лицензию HuggingFace:"
echo "   1. Регистрация (бесплатно): https://huggingface.co/join"
echo "   2. Примите условия на странице модели"
echo "   3. Авторизация в терминале:"
echo "      huggingface-cli login"
echo "   4. Скачайте модель:"
echo "      python download_model.py base""
echo ""
