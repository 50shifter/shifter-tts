#!/usr/bin/env python3
"""
shifter-tts — Загрузка моделей из HuggingFace Hub.

Запуск:
    python download_model.py              # базовая модель (voice cloning)
    python download_model.py custom       # кастомная версия с голосом
    python download_model.py voice_design # Voice Design
    python download_model.py tokenizer    # только токенизатор (для обучения)

Примечание:
    - Публичные модели скачиваются БЕСПЛАТНО без ключа
    - Если модель требует принятия лицензии — откроется браузер
    - Аккаунт на HuggingFace бесплатный: https://huggingface.co/join
"""

import argparse
import os
import sys


MODELS = {
    "base":      ("Qwen/Qwen3-TTS-12Hz-1.7B-Base",     "Базовая модель (voice cloning)"),
    "custom":    ("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "Кастомная версия с голосом"),
    "voice_design": ("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", "Voice Design (генерация голоса по описанию)"),
    "tokenizer": ("Qwen/Qwen3-TTS-Tokenizer-12Hz",     "Токенизатор 12 Гц (для обучения)"),
}


def download_model(model_id: str, cache_dir: str = None):
    """Скачать модель из HuggingFace."""
    print(f"📥 Скачивание модели: {model_id}")
    
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    
    try:
        from huggingface_hub import snapshot_download
        
        local_dir = snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            resume_download=True,
        )
        
        size_gb = 0
        for dirpath, _, filenames in os.walk(local_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.isfile(fp):
                    size_gb += os.path.getsize(fp) / (1024**3)
        
        print(f"✅ Модель сохранена в: {local_dir}")
        print(f"   Размер: ~{size_gb:.1f} ГБ")
        return local_dir
        
    except Exception as e:
        error_msg = str(e).lower()
        
        if "401" in error_msg or "unauthorized" in error_msg or "authentication" in error_msg:
            print("")
            print("=" * 60)
            print("  ⚠️  Требуется аккаунт HuggingFace")
            print("=" * 60)
            print()
            print(f"  Модель '{model_id}' требует принятия лицензии.")
            print()
            print("  1. Создайте бесплатный аккаунт:")
            print("     https://huggingface.co/join")
            print()
            print("  2. Примите лицензию на странице модели:")
            print(f"     https://huggingface.co/{model_id}")
            print()
            print("  3. Войдите в терминал:")
            print("     huggingface-cli login")
            print()
            print("  Или используйте токен (если уже есть аккаунт):")
            print("     huggingface-cli login --token YOUR_TOKEN_HERE")
            print("=" * 60)
        elif "gated" in error_msg or "access denied" in error_msg:
            print("")
            print("=" * 60)
            print("  ⚠️  Модель требует доступа (gated model)")
            print("=" * 60)
            print()
            print(f"  Перейдите на страницу модели и примите условия:")
            print(f"  https://huggingface.co/{model_id}")
            print()
            print("  После принятия — запустите снова.")
            print("=" * 60)
        else:
            print(f"❌ Ошибка при скачивании: {e}")
        
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Скачать модель shifter-tts из HuggingFace")
    parser.add_argument(
        "model",
        choices=list(MODELS.keys()),
        default="base",
        help="Какую модель скачать (по умолчанию: base)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Путь к кэшу HuggingFace (по умолчанию: ~/.cache/huggingface/hub)",
    )
    args = parser.parse_args()
    
    model_id = MODELS[args.model][0]
    description = MODELS[args.model][1]
    
    print(f"🎯 Модель: {model_id}")
    print(f"   Описание: {description}")
    print(f"   Страница: https://huggingface.co/{model_id}")
    print()
    
    download_model(model_id, args.cache_dir)


if __name__ == "__main__":
    main()
