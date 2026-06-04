"""TTS Engine — обёртка над Qwen3-TTS для синтеза речи."""

import os
import re
import threading
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch

from qwen_tts import Qwen3TTSModel
from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem


# ======================== Парсер инструкций в скобках ========================

@dataclass
class TextSegment:
    """Сегмент текста: обычный текст или инструкция."""
    is_instruction: bool
    content: str  # текст или тип инструкции (смеётся, шёпотом и т.д.)
    audio: Optional[np.ndarray] = None  # заполняется после генерации


# Маппинг русских инструкций → параметры генерации
INSTRUCTION_PARAMS = {
    "смеётся":     {"temperature": 1.2, "top_k": 80, "desc": "смех"},
    "смеяться":    {"temperature": 1.2, "top_k": 80, "desc": "смех"},
    "хохочет":     {"temperature": 1.3, "top_k": 90, "desc": "громкий смех"},
    "хохотать":    {"temperature": 1.3, "top_k": 90, "desc": "громкий смех"},
    "шёпотом":     {"temperature": 0.5, "top_k": 20, "desc": "шёпот"},
    "шепчет":      {"temperature": 0.5, "top_k": 20, "desc": "шёпот"},
    "шёпот":       {"temperature": 0.5, "top_k": 20, "desc": "шёпот"},
    "кричит":      {"temperature": 1.3, "top_k": 90, "desc": "крик"},
    "кричать":     {"temperature": 1.3, "top_k": 90, "desc": "крик"},
    "громко":      {"temperature": 1.1, "top_k": 60, "desc": "громко"},
    "тихо":        {"temperature": 0.6, "top_k": 30, "desc": "тихо"},
    "пауза":       {"is_pause": True, "duration": 0.5, "desc": "пауза 0.5с"},
    "длинная_пауза":{"is_pause": True, "duration": 1.0, "desc": "пауза 1.0с"},
    "вздыхает":    {"temperature": 0.7, "top_k": 40, "desc": "вздох"},
    "вздыхать":    {"temperature": 0.7, "top_k": 40, "desc": "вздох"},
    "плачет":      {"temperature": 1.1, "top_k": 70, "desc": "плач"},
    "плакать":     {"temperature": 1.1, "top_k": 70, "desc": "плач"},
    "грустно":     {"temperature": 0.8, "top_k": 40, "desc": "грустный тон"},
    "радостно":    {"temperature": 1.1, "top_k": 60, "desc": "радостный тон"},
    "сердито":     {"temperature": 1.2, "top_k": 70, "desc": "сердитый тон"},
    "удивлённо":   {"temperature": 1.0, "top_k": 50, "desc": "удивлённый тон"},
    # Английские инструкции
    "laughing":    {"temperature": 1.2, "top_k": 80, "desc": "laughter"},
    "laughs":      {"temperature": 1.2, "top_k": 80, "desc": "laughter"},
    "whispering":  {"temperature": 0.5, "top_k": 20, "desc": "whisper"},
    "whispers":    {"temperature": 0.5, "top_k": 20, "desc": "whisper"},
    "shouting":    {"temperature": 1.3, "top_k": 90, "desc": "shout"},
    "shouts":      {"temperature": 1.3, "top_k": 90, "desc": "shout"},
    "pause":       {"is_pause": True, "duration": 0.5, "desc": "pause"},
    "long_pause":  {"is_pause": True, "duration": 1.0, "desc": "long pause"},
}

# Регулярное выражение для поиска инструкций в скобках: [смеётся], [шёпотом]
INSTRUCTION_PATTERN = re.compile(r'\[([^\]]+)\]')


def _parse_text_with_instructions(text: str) -> List[TextSegment]:
    """
    Разбить текст на сегменты: обычный текст и инструкции в скобках.
    
    Пример:
        "Приве́т [смеётся] как де́ла?"
        → [TextSegment(False, "Приве́т "),
           TextSegment(True, "смеётся"),
           TextSegment(False, " как де́ла?")]
    """
    segments: List[TextSegment] = []
    last_end = 0
    
    for match in INSTRUCTION_PATTERN.finditer(text):
        start, end = match.span()
        instruction = match.group(1).strip().lower()
        
        # Обычный текст перед инструкцией
        if start > last_end:
            normal_text = text[last_end:start]
            if normal_text.strip():  # пропускаем пустые сегменты
                segments.append(TextSegment(is_instruction=False, content=normal_text))
        
        # Инструкция
        segments.append(TextSegment(is_instruction=True, content=instruction))
        last_end = end
    
    # Обычный текст после последней инструкции
    if last_end < len(text):
        normal_text = text[last_end:]
        if normal_text.strip():
            segments.append(TextSegment(is_instruction=False, content=normal_text))
    
    # Если инструкций не найдено — весь текст как один сегмент
    if not segments:
        if text.strip():
            segments.append(TextSegment(is_instruction=False, content=text))
    
    return segments


# ======================== Обработка ударений ========================

def _normalize_stress_marks(text: str) -> str:
    """
    Удалить символы ударения (U+0301 combining acute accent) из текста.
    
    Модель Qwen3-TTS не поддерживает ударения — токенизатор BPE разбивает
    `е́` на два отдельных токена (`е` + `́`), что ломает произношение.
    
    Решение: удаляем combining-символы перед отправкой в модель,
    но сохраняем оригинальный текст для отображения в логе/UI.
    """
    # NFD разбивает `е́` на `е` + U+0301
    nfd = unicodedata.normalize('NFD', text)
    # Удаляем все combining marks (категория Mn — Mark, Nonspacing)
    cleaned = ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')
    return cleaned


def _has_stress_marks(text: str) -> bool:
    """Проверить, есть ли в тексте символы ударения."""
    for char in text:
        if unicodedata.category(char) == 'Mn':  # combining mark
            return True
    return False


class TTSEngine:
    """Обёртка над Qwen3-TTS для синтеза речи с voice cloning."""

    def __init__(self):
        self.model: Optional[Qwen3TTSModel] = None
        self._device = "cuda"
        self._dtype = torch.float16
        self._model_path = ""
        self._loading = False
        self._lock = threading.Lock()
        # Кэшированный вектор голоса (если загружен из .pt)
        self._cached_voice_vector: Optional[List[VoiceClonePromptItem]] = None

    def set_log_callback(self, callback):
        """Установить коллбэк для логирования: callback(msg: str)"""
        self._log_callback = callback

    def _log(self, msg: str):
        if hasattr(self, "_log_callback") and self._log_callback:
            self._log_callback(msg)
        # Также пишем в файл лога
        try:
            log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app.log")
            with open(log_file, "a", encoding="utf-8") as f:
                from datetime import datetime
                timestamp = datetime.now().strftime("%H:%M:%S")
                f.write(f"[{timestamp}] [TTS] {msg}\n")
        except Exception:
            pass

    def load_model(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        dtype: str = "float16",
    ) -> Tuple[bool, str]:
        """Загрузить модель Qwen3-TTS.

        Returns: (success: bool, message: str)
        """
        with self._lock:
            if self._loading:
                return False, "Модель уже загружается..."

            self._loading = True
            try:
                # Определяем устройство — обрабатываем cuda/gpu/auto
                if device in ("auto", "gpu", "cuda"):
                    self._device = "cuda" if torch.cuda.is_available() else "cpu"
                    if self._device == "cpu" and device != "auto":
                        return False, "GPU не обнаружен. Переключено на CPU."
                else:
                    self._device = "cpu"

                # Определяем dtype
                if dtype == "float16":
                    self._dtype = torch.float16
                elif dtype == "bfloat16":
                    self._dtype = torch.bfloat16
                else:
                    self._dtype = torch.float32

                # Путь к модели
                if model_path is None or model_path.strip() == "":
                    default_paths = [
                        os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base"),
                        os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-Base"),
                    ]
                    for p in default_paths:
                        if Path(p).exists():
                            model_path = p
                            break
                    else:
                        model_path = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

                self._model_path = model_path
                self._log(f"Загрузка модели: {model_path} | device={self._device} | dtype={dtype}")

                # Загружаем модель
                kwargs = {"device_map": self._device}
                if self._device == "cuda":
                    kwargs["torch_dtype"] = self._dtype

                self.model = Qwen3TTSModel.from_pretrained(model_path, **kwargs)
                self._log(f"Модель загружена успешно. Device: {self._device}")
                return True, f"Модель загружена ({model_path}) на {self._device}"

            except Exception as e:
                msg = f"Ошибка загрузки модели: {e}"
                self._log(msg)
                return False, msg
            finally:
                self._loading = False

    def load_voice_vector(self, path: str) -> Tuple[bool, str]:
        """Загрузить вектор голоса из .pt файла.

        Args:
            path: Путь к файлу .pt с сохранённым вектором голоса.

        Returns: (success: bool, message: str)
        """
        if self.model is None:
            return False, "Модель не загружена. Сначала загрузите модель."

        try:
            items = self.model.load_voice_vector(path)
            self._cached_voice_vector = items
            self._log(f"Вектор голоса загружен: {path} | items={len(items)}")
            return True, f"Вектор голоса загружен из {path}"
        except FileNotFoundError:
            return False, f"Файл не найден: {path}"
        except Exception as e:
            msg = f"Ошибка загрузки вектора голоса: {e}"
            self._log(msg)
            return False, msg

    def clear_voice_vector(self):
        """Очистить кэшированный вектор голоса."""
        self._cached_voice_vector = None
        self._log("Кэшированный вектор голоса очищен")

    def generate(
        self,
        text: str,
        ref_audio_path: Optional[str] = None,
        ref_text: Optional[str] = None,
        voice_vector_path: Optional[str] = None,
        x_vector_only_mode: bool = False,
        language: str = "auto",
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[Optional[np.ndarray], int, str]:
        # Маппинг языковых кодов — модель принимает полные названия
        lang_map = {
            "ru": "russian", "en": "english", "chinese": "chinese",
            "fr": "french", "de": "german", "it": "italian",
            "ja": "japanese", "ko": "korean", "pt": "portuguese",
            "es": "spanish", "auto": "auto",
        }
        language = lang_map.get(language.lower(), language)

        """Синтезировать речь.

        Поддерживаемые возможности:
          1. Ударения: символ U+0301 над гласной (автоматически удаляется перед моделью,
             т.к. BPE-токенизатор не поддерживает combining marks)
          2. Инструкции в скобках: [смеётся], [шёпотом], [пауза] и др.
             — каждый сегмент генерируется отдельно с подходящими параметрами

        Args:
            text: Текст для синтеза
            ref_audio_path: Путь к референсному аудио для voice cloning
            ref_text: Транскрипт референсного аудио
            voice_vector_path: Путь к .pt файлу с сохранённым вектором голоса
            x_vector_only_mode: Если True, использовать только speaker embedding
            language: Язык ("auto", "ru", "en")
            temperature: Температура генерации (по умолчанию 0.9)
            top_k: Top-k sampling (по умолчанию 50)
            top_p: Top-p sampling (по умолчанию 1.0)
            progress_callback: callback(current, total)

        Returns: (audio_array, sample_rate, error_message_or_empty)
        """
        if self.model is None:
            return None, 0, "Модель не загружена. Сначала загрузите модель."

        try:
            # Подготавливаем параметры генерации по умолчанию
            default_gen_kwargs = {
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "do_sample": True,
                "non_streaming_mode": True,
            }

            # Определяем источник voice cloning
            use_voice_vector = False
            voice_clone_prompt = None

            if voice_vector_path and os.path.exists(voice_vector_path):
                items = self.model.load_voice_vector(voice_vector_path)
                voice_clone_prompt = items
                use_voice_vector = True
                self._log(f"Используем вектор голоса из файла: {voice_vector_path}")
            elif self._cached_voice_vector is not None:
                voice_clone_prompt = self._cached_voice_vector
                use_voice_vector = True
                self._log("Используем кэшированный вектор голоса")
            elif ref_audio_path and os.path.exists(ref_audio_path):
                pass  # Будет обработано ниже
            else:
                raise ValueError(
                    "Не найден источник голоса. Необходимо:\n"
                    "  • Загрузить .pt файл вектора голоса, ИЛИ\n"
                    "  • Выбрать референсный аудиофайл (.wav/.mp3)"
                )

            # Проверяем наличие инструкций в скобках и ударений
            has_instructions = bool(INSTRUCTION_PATTERN.search(text))
            has_stress = _has_stress_marks(text)

            if has_instructions:
                segments = _parse_text_with_instructions(text)
                self._log(f"Найдено {len(segments)} сегментов (с инструкциями). Генерация по частям.")
                audio_parts: List[np.ndarray] = []
                total_segments = len(segments)

                for seg_idx, segment in enumerate(segments):
                    # Прогресс: 10% база + распределение между сегментами
                    seg_progress_base = 10 + int(70 * seg_idx / total_segments)

                    if segment.is_instruction:
                        instruction = segment.content.lower()
                        params = INSTRUCTION_PARAMS.get(instruction)

                        if params is None:
                            self._log(f"⚠ Неизвестная инструкция: [{instruction}] — пропускаем")
                            continue

                        if params.get("is_pause"):
                            # Пауза — генерируем тишину
                            duration = params.get("duration", 0.5)
                            sr_temp = 24000  # стандартная частота для паузы
                            silence = np.zeros(int(duration * sr_temp), dtype=np.float32)
                            audio_parts.append(silence)
                            self._log(f"  [{seg_idx+1}/{total_segments}] Пауза {duration}с")
                        else:
                            # Генерируем с особыми параметрами
                            seg_kwargs = dict(default_gen_kwargs)
                            seg_kwargs["temperature"] = params.get("temperature", temperature)
                            seg_kwargs["top_k"] = params.get("top_k", top_k)

                            # Для инструкций генерируем placeholder-текст,
                            # чтобы модель создала характерную интонацию
                            gen_text = f"{instruction}"

                            wavs, sr_seg = self._generate_segment(
                                text=gen_text,
                                voice_clone_prompt=voice_clone_prompt,
                                ref_audio_path=ref_audio_path,
                                ref_text=ref_text,
                                x_vector_only_mode=x_vector_only_mode,
                                language=language,
                                gen_kwargs=seg_kwargs,
                            )
                            if wavs is not None:
                                audio_parts.append(wavs)
                                self._log(f"  [{seg_idx+1}/{total_segments}] Инструкция: {params['desc']}")

                        seg_progress = min(seg_progress_base + 70 // total_segments, 85)
                        if progress_callback:
                            progress_callback(seg_progress, 100)

                    else:
                        # Обычный текст — удаляем ударения перед моделью
                        clean_text = _normalize_stress_marks(segment.content)

                        wavs, sr_seg = self._generate_segment(
                            text=clean_text,
                            voice_clone_prompt=voice_clone_prompt,
                            ref_audio_path=ref_audio_path,
                            ref_text=ref_text,
                            x_vector_only_mode=x_vector_only_mode,
                            language=language,
                            gen_kwargs=default_gen_kwargs,
                        )
                        if wavs is not None:
                            audio_parts.append(wavs)

                        seg_progress = min(seg_progress_base + 70 // total_segments, 85)
                        if progress_callback:
                            progress_callback(seg_progress, 100)

                # Склеиваем все части с короткими паузами между сегментами
                if audio_parts:
                    sr_final = 24000
                    # Пауза 0.15с между сегментами для естественности
                    pause_samples = int(0.15 * sr_final)
                    pause = np.zeros(pause_samples, dtype=np.float32)

                    final_audio = audio_parts[0]
                    for part in audio_parts[1:]:
                        final_audio = np.concatenate([final_audio, pause, part])

                    duration = len(final_audio) / sr_final if sr_final > 0 else 0
                    self._log(f"Генерация завершена (по сегментам). Длительность: {duration:.2f}s | SR: {sr_final}")
                    if progress_callback:
                        progress_callback(100, 100)
                    return final_audio, sr_final, ""
                else:
                    return None, 0, "Не удалось сгенерировать ни одного сегмента."

            elif has_stress:
                # Есть ударения, но нет инструкций — просто удаляем ударения и генерируем как обычно
                clean_text = _normalize_stress_marks(text)
                self._log(f"Ударения обнаружены. Удалены перед отправкой в модель.")

                wavs, sr = self._generate_segment(
                    text=clean_text,
                    voice_clone_prompt=voice_clone_prompt,
                    ref_audio_path=ref_audio_path,
                    ref_text=ref_text,
                    x_vector_only_mode=x_vector_only_mode,
                    language=language,
                    gen_kwargs=default_gen_kwargs,
                )

                if progress_callback:
                    progress_callback(100, 100)

                if wavs is not None:
                    duration = len(wavs) / sr if sr > 0 else 0
                    self._log(f"Генерация завершена. Длительность: {duration:.2f}s | SR: {sr}")
                    return wavs, sr, ""
                else:
                    return None, 0, "Ошибка генерации."

            else:
                # Обычная генерация без инструкций и ударений
                self._log(f"Генерация TTS: '{text[:50]}...' | lang={language}")

                if progress_callback:
                    progress_callback(10, 100)

                wavs, sr = self._generate_segment(
                    text=text,
                    voice_clone_prompt=voice_clone_prompt,
                    ref_audio_path=ref_audio_path,
                    ref_text=ref_text,
                    x_vector_only_mode=x_vector_only_mode,
                    language=language,
                    gen_kwargs=default_gen_kwargs,
                )

                if progress_callback:
                    progress_callback(100, 100)

                if wavs is not None:
                    duration = len(wavs) / sr if sr > 0 else 0
                    self._log(f"Генерация завершена. Длительность: {duration:.2f}s | SR: {sr}")
                    return wavs, sr, ""
                else:
                    return None, 0, "Ошибка генерации."

        except Exception as e:
            msg = f"Ошибка генерации TTS: {e}"
            self._log(msg)
            import traceback
            self._log(traceback.format_exc())
            return None, 0, msg

    def _generate_segment(
        self,
        text: str,
        voice_clone_prompt,
        ref_audio_path: Optional[str],
        ref_text: Optional[str],
        x_vector_only_mode: bool,
        language: str,
        gen_kwargs: dict,
    ) -> Tuple[Optional[np.ndarray], int]:
        """
        Сгенерировать аудио для одного сегмента текста.

        Returns: (audio_array or None, sample_rate)
        """
        try:
            if voice_clone_prompt is not None:
                wavs, sr = self.model.generate_voice_clone(
                    text=text,
                    language=language,
                    voice_clone_prompt=voice_clone_prompt,
                    x_vector_only_mode=x_vector_only_mode,
                    **gen_kwargs,
                )
            elif ref_audio_path and os.path.exists(ref_audio_path):
                wavs, sr = self.model.generate_voice_clone(
                    text=text,
                    language=language,
                    ref_audio=ref_audio_path,
                    ref_text=ref_text,
                    x_vector_only_mode=x_vector_only_mode,
                    **gen_kwargs,
                )
            else:
                return None, 0

            audio = wavs[0] if isinstance(wavs, list) else wavs
            return audio, sr

        except Exception as e:
            tb = traceback.format_exc()
            self._log(f"Ошибка генерации сегмента '{text[:30]}...': {e}")
            self._log(tb)
            return None, 0

    def extract_and_save_voice_vector(
        self,
        ref_audio_path: str,
        output_path: str,
        ref_text: Optional[str] = None,
        x_vector_only_mode: bool = False,
    ) -> Tuple[bool, str]:
        """Извлечь вектор голоса из референсного аудио и сохранить в .pt файл.

        Args:
            ref_audio_path: Путь к референсному аудиофайлу.
            output_path: Путь для сохранения .pt файла с вектором голоса.
            ref_text: Транскрипт референсного аудио (для ICL режима).
            x_vector_only_mode: Если True, сохранить только speaker embedding.

        Returns: (success: bool, message: str)
        """
        if self.model is None:
            return False, "Модель не загружена. Сначала загрузите модель."

        try:
            items = self.model.create_voice_clone_prompt(
                ref_audio=ref_audio_path,
                ref_text=ref_text,
                x_vector_only_mode=x_vector_only_mode,
            )
            saved_path = self.model.save_voice_vector(items, output_path)
            # Также кэшируем вектор
            self._cached_voice_vector = items
            self._log(f"Вектор голоса извлечён и сохранён: {saved_path}")
            return True, f"Вектор голоса сохранён в {saved_path}"
        except Exception as e:
            msg = f"Ошибка сохранения вектора голоса: {e}"
            self._log(msg)
            return False, msg

    def save_audio(self, audio: np.ndarray, sample_rate: int, output_path: str) -> Tuple[bool, str]:
        """Сохранить аудио в файл (WAV или MP3 через FFmpeg)."""
        try:
            # Создаём директорию если нужно
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

            if output_path.lower().endswith(".mp3"):
                # Сначала сохраняем временный WAV, потом конвертируем в MP3 через FFmpeg
                temp_wav = output_path.rsplit(".", 1)[0] + ".wav"
                sf.write(temp_wav, audio, sample_rate)
                
                # Конвертируем WAV → MP3 через FFmpeg
                import subprocess
                result = subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-i", temp_wav,
                        "-vn", "-acodec", "libmp3lame", "-q:a", "2", "-ar", "44100",
                        output_path
                    ],
                    capture_output=True, text=True, timeout=30,
                    encoding='utf-8', errors='replace'
                )
                
                # Удаляем временный WAV
                try:
                    os.remove(temp_wav)
                except Exception:
                    pass
                
                if result.returncode == 0 and os.path.isfile(output_path):
                    duration = len(audio) / sample_rate if sample_rate > 0 else 0
                    self._log(f"Сохранено: {output_path} ({duration:.2f}s, MP3)")
                    return True, f"Сохранено (MP3): {output_path}"
                else:
                    raise RuntimeError("FFmpeg конвертация не удалась")
            else:
                sf.write(output_path, audio, sample_rate)
                duration = len(audio) / sample_rate if sample_rate > 0 else 0
                self._log(f"Сохранено: {output_path} ({duration:.2f}s, WAV)")
                return True, f"Сохранено (WAV): {output_path}"

        except Exception as e:
            msg = f"Ошибка сохранения аудио: {e}"
            self._log(msg)
            return False, msg

    def unload_model(self):
        """Выгрузить модель из памяти."""
        with self._lock:
            if self.model is not None:
                del self.model
                self.model = None
                torch.cuda.empty_cache()
                self._log("Модель выгружена")
