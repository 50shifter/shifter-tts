# shifter-tts

Десктопное приложение для синтеза речи (TTS) на основе Qwen3-TTS с поддержкой клонирования голоса, калибровки атрибутов и транскрипции.

## Возможности

- Голосовые инструкции - [смеётся], [шёпотом], [громко], [пауза] прямо в тексте
- Voice Cloning - клонирование голоса по короткому референс-аудио
- Калибровка голоса - смешивание нескольких .pt векторов для точного контроля
- Транскрипция - Audio to текст через Faster Whisper (6 языков)
- Мониторинг GPU - VRAM, температура, загрузка в реальном времени
- Аудио из видео - извлечение звука из mp4, mov и других форматов
- Voice Design - генерация голоса по текстовому описанию

## Требования

- Python 3.10+
- NVIDIA GPU с 8 ГБ VRAM (для синтеза)
- NVIDIA GPU с 12 ГБ VRAM (для дообучения)
- Windows, Linux или macOS

## Быстрый запуск

### Автоматический (рекомендуется)

Запустите один из скриптов - всё установится и скачается самостоятельно:

```bash
# Windows
run.bat

# Linux / macOS
./run.sh
```

Скрипт создаст виртуальное окружение, установит зависимости и скачает модель. После этого запустится приложение.

### Установка только (без запуска)

Если нужен только pip-пакет:

```bash
# Windows
setup.bat

# Linux / macOS
./setup.sh

# Или через pip
pip install -e .
shifter-tts
```

### Ручная установка

```bash
python -m venv venv
source venv/bin/activate     # Linux/macOS
venv\Scripts\activate        # Windows

pip install -r requirements.txt
python -m tts_app
```

## Скачивание модели вручную

Модель также скачивается автоматически при первом запуске. Если нужен ручной контроль:

```bash
# Базовая модель (voice cloning)
python download_model.py base

# Кастомная версия
python download_model.py custom

# Voice Design (генерация по описанию)
python download_model.py voice_design

# Токенизатор (для обучения)
python download_model.py tokenizer
```

## Использование

### Голосовые инструкции

Вставляйте указания в квадратных скобках прямо в текст:

```
Привет [смеётся] как дела?
```

Поддерживаемые инструкции:

| Тип | Примеры |
|-----|---------|
| Эмоции | [смеётся], [хохочет], [грустно], [радостно], [сердито], [удивлённо], [плачет], [вздыхает] |
| Громкость | [шёпотом], [тихо], [громко], [кричит] |
| Паузы | [пауза], [длинная_пауза] |
| English | [laughing], [whispering], [shouting], [pause], [long_pause] |

Подробную памятку см. в HELP.md.

### Клонирование голоса

1. Загрузите короткое референсное аудио (10-20 секунд)
2. Введите текст для синтеза
3. Нажмите Synthesize

Вектор голоса можно сохранить как .pt файл и использовать повторно.

### Калибровка голоса

1. Загрузите несколько .pt файлов с разными голосовыми атрибутами
2. Настройте вес каждого (1.0 = полный, 0.5 = половина)
3. Нажмите Calibrate - появится новый микс
4. Сохраните результат как .pt и используйте при синтезе

### Транскрипция

- Форматы: WAV, MP3, OGG, FLAC, M4A
- Языки: English, Русский, German, French, Spanish

## Дообучение (Fine-tuning)

Требует GPU с 12+ ГБ VRAM.

```bash
# Подготовка данных
python finetuning/prepare_data.py --input_jsonl train.jsonl --output_jsonl processed.jsonl

# Дообучение
python finetuning/sft_12hz.py --train_data processed.jsonl
```

Примеры работы с моделью: examples/test_model_12hz_base.py, examples/test_model_12hz_custom_voice.py и другие.

## Структура проекта

```
shifter-tts/
├── tts_app/                  # GUI-приложение
│   ├── __main__.py           # python -m tts_app
│   ├── main.py               # Entry point (pip install)
│   ├── gui/window.py         # Главное окно PyQt6
│   └── modules/              # Модули: TTS, Whisper, GPU, видео
├── qwen_tts/                 # Библиотека ядра
│   ├── core/models/          # Модель Qwen3-TTS
│   ├── core/tokenizer_12hz/  # Токенизатор 12Hz
│   ├── core/tokenizer_25hz/  # Токенизатор 25Hz
│   ├── inference/            # Inference-код
│   └── cli/demo.py           # CLI-демо
├── examples/                 # Примеры скриптов
├── finetuning/               # Скрипты дообучения
│   ├── prepare_data.py       # Подготовка датасета
│   ├── sft_12hz.py          # Supervised fine-tuning
│   └── dataset.py            # Dataset классы
├── download_model.py         # Ручное скачивание моделей
├── run.py / run.bat / run.sh # Авто-установка + запуск
├── setup.py / setup.bat / setup.sh # Только установка (pip)
├── requirements.txt          # Зависимости
└── HELP.md                  # Памятка пользователя
```

## Частые вопросы

### Нужен ли ключ HuggingFace?

Нет, если модель публичная. Если требует принятия лицензии - зарегистрируйтесь на huggingface.co/join, примите условия на странице модели и войдите через терминал:

```bash
huggingface-cli login
```

### Модель не скачивается?

- Проверьте интернет-соединение
- Убедитесь, что вы приняли условия лицензии на странице модели HuggingFace
- Попробуйте: python download_model.py base

### Можно без GPU?

Да, но медленно. Для комфортной работы нужен NVIDIA GPU с 8 ГБ VRAM.

### Где хранится модель?

~/.cache/huggingface/hub/ - кэшируется и не скачивается повторно.

### Как запустить через venv?

```bash
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate       # Windows
python -m tts_app
```

## Лицензия

MIT
