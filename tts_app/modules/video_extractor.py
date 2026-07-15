"""Video/Audio extractor — извлечение аудио из видеофайлов через FFmpeg."""

import os
import subprocess
from pathlib import Path
from typing import Callable, Optional, Tuple

from tts_app.config import AUDIO_EXTENSIONS as SUPPORTED_AUDIO_EXTENSIONS
from tts_app.config import VIDEO_EXTENSIONS as SUPPORTED_VIDEO_EXTENSIONS


# Расширения импортируются из config.py
# SUPPORTED_VIDEO_EXTENSIONS и SUPPORTED_AUDIO_EXTENSIONS


class VideoExtractor:
    """Извлечение аудио из видеофайлов с помощью FFmpeg."""

    def __init__(self):
        self._ffmpeg_path = self._find_ffmpeg()
        self._log_callback = None

    def set_log_callback(self, callback: Callable[[str], None]):
        """Установить коллбэк для логирования."""
        self._log_callback = callback

    def _log(self, msg: str):
        if self._log_callback:
            self._log_callback(msg)

    @staticmethod
    def _find_ffmpeg() -> Optional[str]:
        """Найти путь к FFmpeg в системе."""
        from tts_app.config import _get_ffmpeg_default_paths as _default_paths

        # Проверяем PATH
        for name in ['ffmpeg', 'ffmpeg.exe']:
            try:
                result = subprocess.run(
                    ['where' if os.name == 'nt' else 'which', name],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0 and result.stdout.strip():
                    paths = [p.strip() for p in result.stdout.strip().split('\n')]
                    for p in paths:
                        if os.path.isfile(p) and os.access(p, os.X_OK):
                            return p
            except Exception:
                continue

        # Проверяем распространённые пути из config
        for p in _default_paths():
            if os.path.isfile(p):
                return p

        return None

    def is_ffmpeg_available(self) -> bool:
        """Проверить, доступен ли FFmpeg."""
        if self._ffmpeg_path and os.path.isfile(self._ffmpeg_path):
            try:
                result = subprocess.run(
                    [self._ffmpeg_path, '-version'],
                    capture_output=True, text=True, timeout=10
                )
                return result.returncode == 0
            except Exception:
                pass
        return False

    def get_ffmpeg_info(self) -> str:
        """Получить информацию о FFmpeg."""
        if not self.is_ffmpeg_available():
            return "FFmpeg не найден"
        try:
            result = subprocess.run(
                [self._ffmpeg_path, '-version'],
                capture_output=True, text=True, timeout=10
            )
            first_line = result.stdout.split('\n')[0] if result.stdout else "Unknown version"
            return f"{first_line} (путь: {self._ffmpeg_path})"
        except Exception as e:
            return f"FFmpeg найден, но ошибка проверки: {e}"

    def extract_audio(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        sample_rate: int = 24000,
        duration: Optional[float] = None,
        progress_callback=None,
    ) -> Tuple[bool, str]:
        """Извлечь аудио из видеофайла.

        Args:
            input_path: Путь к входному видеофайлу (.mp4, .avi, .mkv и др.)
            output_path: Путь для сохранения WAV файла. Если None — создаётся во временной папке.
            sample_rate: Частота дискретизации выходного аудио (по умолчанию 24000).
            duration: Ограничение длительности извлечения в секундах.
            progress_callback: callback(percent) для отслеживания прогресса.

        Returns:
            Tuple[success, message]
        """
        input_path = os.path.abspath(input_path)

        if not os.path.isfile(input_path):
            return False, f"Файл не найден: {input_path}"

        # Проверяем расширение файла
        ext = Path(input_path).suffix.lower()
        
        is_video = ext in SUPPORTED_VIDEO_EXTENSIONS
        is_audio = ext in SUPPORTED_AUDIO_EXTENSIONS

        if not is_video and not is_audio:
            return False, (
                f"Формат '{ext}' не поддерживается.\n\n"
                f"Поддерживаемые видео: {', '.join(sorted(SUPPORTED_VIDEO_EXTENSIONS))}\n"
                f"Поддерживаемое аудио: {', '.join(sorted(SUPPORTED_AUDIO_EXTENSIONS))}"
            )

        # Если это уже аудиофайл — просто конвертируем в WAV
        if is_audio and ext != '.wav':
            self._log(f"Конвертация аудио {ext} → WAV")
            return self._convert_audio_to_wav(input_path, output_path, sample_rate)

        # Извлекаем аудио из видео
        self._log(f"Извлечение аудио из видео: {Path(input_path).name}")

        if output_path is None:
            tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
            os.makedirs(tmp_dir, exist_ok=True)
            base_name = Path(input_path).stem
            # Убираем лишние символы из имени
            base_name = ''.join(c if c.isalnum() or c in '-_' else '_' for c in base_name)
            output_path = os.path.join(tmp_dir, f"{base_name}_audio.wav")

        # Создаём директорию если нужно
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Формируем команду FFmpeg
        cmd = [
            self._ffmpeg_path,
            '-y',                    # Перезаписывать без вопроса
            '-i', input_path,        # Входной файл
            '-vn',                   # Без видео (только аудио)
            '-acodec', 'pcm_s16le',  # Кодировщик PCM 16-bit
            '-ar', str(sample_rate), # Частота дискретизации
            '-ac', '1',              # Моно
        ]

        if duration:
            cmd.extend(['-t', str(duration)])

        cmd.append(output_path)

        try:
            self._log(f"FFmpeg команда: {' '.join(cmd[:5])}...")

            # Запускаем FFmpeg и отслеживаем прогресс
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # объединяем stderr с stdout для чтения прогресса
                universal_newlines=True,
                encoding='utf-8',
                errors='replace',
            )

            total_duration = None
            current_time = 0.0

            # stderr объединён с stdout (STDOUT), читаем из stdout
            for line in process.stdout:
                # Ищем общую длительность
                if total_duration is None and 'Duration:' in line:
                    parts = line.split('Duration:')
                    if len(parts) > 1:
                        time_str = parts[1].split(',')[0].strip()
                        total_duration = self._parse_time(time_str)
                        self._log(f"Длительность видео: {total_duration:.1f}s")

                # Ищем текущее время
                if 'time=' in line:
                    time_parts = line.split('time=')
                    if len(time_parts) > 1:
                        current_time = self._parse_time(time_parts[-1].split(' ')[0].strip())

                # Обновляем прогресс
                if progress_callback and total_duration and total_duration > 0:
                    percent = min(100, int(current_time / total_duration * 100))
                    progress_callback(percent)

            process.wait()

            if process.returncode != 0:
                return False, f"FFmpeg завершился с ошибкой (код {process.returncode})"

            if not os.path.isfile(output_path):
                return False, "Выходной файл не создан"

            # Проверяем размер файла
            file_size = os.path.getsize(output_path)
            if file_size < 100:  # Слишком маленький файл — ошибка
                return False, f"Получен пустой/корруптивный файл ({file_size} байт)"

            duration_sec = file_size / (sample_rate * 2)  # 16-bit mono
            self._log(f"Извлечено аудио: {duration_sec:.1f}s → {output_path}")
            return True, f"Аудио извлечено: {duration_sec:.1f}s → {output_path}"

        except FileNotFoundError:
            return False, "FFmpeg не найден. Установите FFmpeg для работы с видео."
        except subprocess.TimeoutExpired:
            process.kill()
            return False, "Превышено время обработки (таймаут)"
        except Exception as e:
            return False, f"Ошибка извлечения аудио: {e}"

    def _convert_audio_to_wav(
        self,
        input_path: str,
        output_path: Optional[str],
        sample_rate: int = 24000,
    ) -> Tuple[bool, str]:
        """Конвертировать аудиофайл в WAV."""
        if output_path is None:
            tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
            os.makedirs(tmp_dir, exist_ok=True)
            base_name = Path(input_path).stem
            base_name = ''.join(c if c.isalnum() or c in '-_' else '_' for c in base_name)
            output_path = os.path.join(tmp_dir, f"{base_name}_converted.wav")

        cmd = [
            self._ffmpeg_path,
            '-y',
            '-i', input_path,
            '-acodec', 'pcm_s16le',
            '-ar', str(sample_rate),
            '-ac', '1',
            output_path,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True, text=True, timeout=60,
                encoding='utf-8', errors='replace',
            )
            if result.returncode != 0:
                return False, f"Ошибка конвертации: {result.stderr[:200]}"

            if not os.path.isfile(output_path):
                return False, "Выходной файл не создан"

            duration = os.path.getsize(output_path) / (sample_rate * 2)
            self._log(f"Конвертировано: {duration:.1f}s → {output_path}")
            return True, f"Конвертировано: {Path(input_path).name} → {output_path}"

        except FileNotFoundError:
            return False, "FFmpeg не найден."
        except subprocess.TimeoutExpired:
            return False, "Превышено время конвертации."
        except Exception as e:
            return False, f"Ошибка конвертации: {e}"

    @staticmethod
    def _parse_time(time_str: str) -> float:
        """Парсить время из FFmpeg формата HH:MM:SS.mmm."""
        try:
            parts = time_str.split(':')
            if len(parts) == 3:
                h, m, s = parts
                return int(h) * 3600 + int(m) * 60 + float(s.replace(',', '.'))
            elif len(parts) == 2:
                m, s = parts
                return int(m) * 60 + float(s.replace(',', '.'))
        except Exception:
            pass
        return 0.0

    def get_supported_formats(self) -> dict:
        """Получить список поддерживаемых форматов."""
        return {
            'video': sorted(SUPPORTED_VIDEO_EXTENSIONS),
            'audio': sorted(SUPPORTED_AUDIO_EXTENSIONS),
        }
