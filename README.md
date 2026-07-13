# shifter-tts

Десктопное приложение для синтеза речи (TTS) на основе Qwen3-TTS с поддержкой клонирования голоса, калибровки атрибутов и транскрипции.

## Возможности

- **Голосовые инструкции** — `[смеётся]`, `[шёпотом]`, `[громко]`, `[пауза]` прямо в тексте
- **Voice Cloning** — клонирование голоса по короткому референс-аудио
- **Калибровка голоса** — смешивание нескольких `.pt` векторов для точного контроля
- **Транскрипция** — Audio → текст через Faster Whisper (6 языков)
- **Мониторинг GPU** — VRAM, температура, загрузка в реальном времени
- **Аудио из видео** — извлечение звука из mp4, mov и других форматов
- **Voice Design** — генерация голоса по текстовому описанию

## Требования

- Python 3.10+
- NVIDIA GPU с 8 ГБ VRAM (для синтеза)
- NVIDIA GPU с 12 ГБ VRAM (для дообучения)
- Windows, Linux или macOS

## Быстрый запуск

```bash
python -m tts_app
```

> Автоматическая установка (venv, зависимости, модель) доступна через:
> - **Windows:** `run.bat` или `python run.py`
> - **Linux/macOS:** `./run.sh`

## Установка

### Автоматическая

Запустите `run.py` — всё установится и скачается самостоятельно:

```bash
# Windows
run.bat

# Linux / macOS
./run.sh
```

**Что происходит:**

1. Создаётся виртуальное окружение `venv/`
2. Устанавливаются зависимости из `requirements.txt`
3. Скачивается модель (~4 ГБ)
4. Запускается приложение

### Ручная

```bash
python -m venv venv
source venv/bin/activate     # Linux/macOS
venv\Scripts\activate        # Windows

pip install -r requirements.txt
python -m tts_app
```

### Скачивание модели вручную

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
| Эмоции | `[смеётся]`, `[хохочет]`, `[грустно]`, `[радостно]`, `[сердито]`, `[удивлённо]`, `[плачет]`, `[вздыхает]` |
| Громкость | `[шёпотом]`, `[тихо]`, `[громко]`, `[кричит]` |
| Паузы | `[пауза]`, `[длинная_пауза]` |
| English | `[laughing]`, `[whispering]`, `[shouting]`, `[pause]`, `[long_pause]` |

> ⚠️ Инструкция влияет только на следующий фрагмент текста. Напишите полную памятку в [HELP.md](HELP.md).

### Клонирование голоса

1. Загрузите короткое референсное аудио (10–20 секунд)
2. Введите текст для синтеза
3. Нажмите **Synthesize**

### Калибровка голоса

1. Загрузите несколько `.pt` файлов с разными голосовыми атрибутами
2. Настройте вес каждого (1.0 = полный, 0.5 = половина)
3. Нажмите **Calibrate** → появится новый микс
4. Сохраните результат как `.pt` и используйте при синтезе

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

## Структура проекта

```
shifter-tts/
├── tts_app/                  # GUI-приложение
│   ├── __main__.py           # python -m tts_app
│   ├── gui/window.py         # Главное окно PyQt6
│   └── modules/              # Модули: TTS, Whisper, GPU, видео
├── qwen_tts/                 # Библиотека ядра
│   ├── core/                 # Модель, токенизатор
│   ├── inference/            # Inference-код
│   └── cli/                  # CLI-интерфейс
├── examples/                 # Примеры скриптов
├── finetuning/               # Скрипты дообучения
│   ├── prepare_data.py       # Подготовка датасета
│   └── sft_12hz.py           # Supervised fine-tuning
├── download_model.py         # Ручное скачивание моделей
├── run.py / run.bat / run.sh # Авто-установка + запуск
├── requirements.txt          # Зависимости
├── HELP.md                   # Памятка пользователя
└── README.md                 # Этот файл
```

## Частые вопросы

### Нужен ли ключ HuggingFace?

Нет, если модель публичная. Если требует принятия лицензии — зарегистрируйтесь на [huggingface.co/join](https://huggingface.co/join), примите условия на странице модели и войдите через терминал:

```bash
huggingface-cli login
```

### Модель не скачивается?

- Проверьте интернет-соединение
- Убедитесь, что вы приняли условия лицензии на странице модели HuggingFace
- Попробуйте: `python download_model.py base`

### Можно без GPU?

Да, но медленно. Для комфортной работы нужен NVIDIA GPU с 8 ГБ VRAM.

### Где хранится модель?

`~/.cache/huggingface/hub/` — кэшируется и не скачивается повторно.

### Как запустить через venv?

```bash
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate       # Windows
python -m tts_app
```

## Лицензия

MIT
