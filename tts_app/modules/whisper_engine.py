"""Whisper Engine — транскрибация аудио через faster-whisper."""

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from faster_whisper import WhisperModel


class TranscriptionSegment:
    """Один сегмент транскрипции."""

    def __init__(self, start: float, end: float, text: str, confidence: float):
        self.start = start
        self.end = end
        self.text = text
        self.confidence = confidence

    def to_srt(self, index: int) -> str:
        """Экспорт в формат SRT."""
        s_start = self._format_time(self.start)
        s_end = self._format_time(self.end)
        return f"{index}\n{s_start} --> {s_end}\n{self.text}\n"

    def to_txt(self, index: int) -> str:
        """Экспорт в TXT с таймкодами."""
        return f"[{self.start:.2f}s] {self.text}\n"

    @staticmethod
    def _format_time(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds * 1000) % 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def add_punctuation_from_gaps(segments: List[TranscriptionSegment]) -> List[TranscriptionSegment]:
    """Добавлять пунктуацию на основе пауз между сегментами.
    
    Правила:
      - Пауза 2-5 сек → точка (.)
      - Пауза >5 сек → многоточие (...)
      - Если текст уже заканчивается на знак препинания — не добавляем
    """
    if len(segments) <= 1:
        return segments

    PUNCTUATION_CHARS = frozenset(['.', ',', '!', '?', ':', ';', '-', ')', '"', "'"])
    
    for i in range(1, len(segments)):
        prev_end = segments[i - 1].end
        curr_start = segments[i].start
        gap = curr_start - prev_end

        # Пропускаем слишком маленькие паузы (шум, переходы)
        if gap < 1.5:
            continue

        # Проверяем, заканчивается ли предыдущий текст на знак препинания
        prev_text = segments[i - 1].text.rstrip()
        if prev_text and prev_text[-1] in PUNCTUATION_CHARS:
            continue

        # Добавляем пунктуацию
        if gap > 5.0:
            segments[i - 1].text = prev_text + "..."
        elif gap >= 2.0:
            segments[i - 1].text = prev_text + "."

    return segments


class WhisperEngine:
    """Обёртка над faster-whisper для транскрибации."""

    def __init__(self):
        self.model: Optional[WhisperModel] = None
        self._device = "cuda"
        self._compute_type = "float16"
        self._model_size = "medium"
        self._loading = False

    def set_log_callback(self, callback):
        """Установить коллбэк для логирования: callback(msg: str)"""
        self._log_callback = callback

    def _log(self, msg: str):
        if hasattr(self, "_log_callback") and self._log_callback:
            self._log_callback(msg)
        # Также пишем в файл лога с ротацией
        try:
            log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app.log")
            import logging
            logger = logging.getLogger(f"whisper_engine_{id(self)}")
            if not logger.handlers:
                handler = logging.FileHandler(log_file, encoding="utf-8")
                formatter = logging.Formatter("%(asctime)s [Whisper] %(message)s", datefmt="%H:%M:%S")
                handler.setFormatter(formatter)
                logger.addHandler(handler)
                logger.setLevel(logging.INFO)
            logger.info(msg)
        except Exception:
            pass

    def load_model(
        self,
        model_size: str = "medium",
        device: str = "auto",
        compute_type: str = "float16",
    ) -> Tuple[bool, str]:
        """Загрузить модель Whisper.

        Returns: (success: bool, message: str)
        """
        if self._loading:
            return False, "Модель уже загружается..."

        self._loading = True
        try:
            # Определяем устройство
            import torch
            if device in ("auto", "gpu", "cuda"):
                dev = "cuda" if torch.cuda.is_available() else "cpu"
                if dev == "cpu" and device != "auto":
                    return False, "GPU не обнаружен. Переключено на CPU."
            else:
                dev = "cpu"

            self._device = dev
            self._model_size = model_size
            self._compute_type = compute_type

            # Определяем вычислительный тип
            ct = compute_type
            if dev == "cpu":
                ct = "int8"  # На CPU int8 быстрее и точнее

            self._log(f"Загрузка Whisper: {model_size} | device={dev} | compute_type={ct}")

            # device_index: для CUDA — индекс GPU (0,1,...), для CPU — не нужен
            kwargs = {
                "device": dev,
                "compute_type": ct,
                "download_root": os.path.expanduser("~/.cache/huggingface/hub"),
            }
            if dev == "cuda":
                kwargs["device_index"] = 0

            self.model = WhisperModel(model_size, **kwargs)

            self._log(f"Whisper загружен: {model_size} на {dev}")
            return True, f"Модель Whisper ({model_size}) загружена на {dev}"

        except Exception as e:
            msg = f"Ошибка загрузки Whisper: {e}"
            self._log(msg)
            return False, msg

    def transcribe(
        self,
        audio_path: str,
        language: str = "ru",
        beam_size: int = 5,
        vad_filter: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[List[TranscriptionSegment], str]:
        """Транскрибировать аудиофайл.

        Args:
            audio_path: Путь к аудиофайлу
            language: Язык ("ru", "en", "auto")
            beam_size: Размер beam search
            vad_filter: Фильтр голосовых активностей
            progress_callback: callback(current, total)

        Returns: (segments, error_message_or_empty)
        """
        if self.model is None:
            return [], "Модель не загружена. Сначала загрузите модель."

        if not os.path.exists(audio_path):
            return [], f"Файл не найден: {audio_path}"

        try:
            self._log(f"Транскрипция: {audio_path} | lang={language}")

            # Запускаем транскрипцию
            lang_param = None if language == "auto" else language
            self._log(f"  Язык: {lang_param}, VAD: {vad_filter}")

            segments_iter, info = self.model.transcribe(
                audio_path,
                language=lang_param,
                beam_size=beam_size,
                vad_filter=vad_filter,
                word_timestamps=False,
            )

            # Если auto не распознал ничего — пробуем русский как fallback
            detected_lang = getattr(info, 'language', None)
            if language == "auto" and detected_lang:
                self._log(f"  Авто-язык: {detected_lang}")

            # Собираем сегменты из генератора
            segments = []
            for seg in segments_iter:
                try:
                    text = getattr(seg, 'text', '').strip()
                    start = float(getattr(seg, 'start', 0))
                    end = float(getattr(seg, 'end', 0))
                    avg_logprob = getattr(seg, 'avg_logprob', None)
                    confidence = float(avg_logprob) if avg_logprob is not None else 0.0
                    segments.append(TranscriptionSegment(
                        start=start,
                        end=end,
                        text=text,
                        confidence=confidence,
                    ))
                except Exception as seg_err:
                    self._log(f"Предупреждение: пропуск сегмента — {seg_err}")
                    continue

            # Добавляем пунктуацию на основе пауз между сегментами
            segments = add_punctuation_from_gaps(segments)

            duration = float(getattr(info, 'duration', 0)) if info else 0
            self._log(f"Транскрипция завершена: {len(segments)} сегментов, длительность: {duration:.1f}s")

            return segments, ""

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            msg = f"Ошибка транскрипции: {e}"
            self._log(msg)
            self._log(f"TRACEBACK:\n{tb}")
            return [], msg

    def export_txt(self, segments: List[TranscriptionSegment], output_path: str) -> Tuple[bool, str]:
        """Экспорт в TXT с таймкодами."""
        try:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                for i, seg in enumerate(segments):
                    f.write(seg.to_txt(i + 1))
            return True, f"Сохранено TXT: {output_path}"
        except Exception as e:
            return False, str(e)

    def export_srt(self, segments: List[TranscriptionSegment], output_path: str) -> Tuple[bool, str]:
        """Экспорт в SRT."""
        try:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                for i, seg in enumerate(segments):
                    f.write(seg.to_srt(i + 1))
            return True, f"Сохранено SRT: {output_path}"
        except Exception as e:
            return False, str(e)

    def unload_model(self):
        """Выгрузить модель из памяти."""
        if self.model is not None:
            del self.model
            self.model = None
            import torch
            torch.cuda.empty_cache()
