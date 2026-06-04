"""Audio recorder — запись с микрофона через sounddevice."""

import os
import tempfile
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf


class AudioRecorder:
    """Запись аудио с микрофона и сохранение в WAV файл.
    
    Поддерживает непрерывную запись без ограничений по времени.
    Записанное аудио доступно через get_audio() для визуализации.
    """

    def __init__(self):
        self._stream = None
        self._audio_buffer: Optional[np.ndarray] = None
        self._sample_rate = 24000
        self._channels = 1
        self._log_callback = None
        self._is_recording = False

    def set_log_callback(self, callback: Callable[[str], None]):
        """Установить коллбэк для логирования."""
        self._log_callback = callback

    def _log(self, msg: str):
        if self._log_callback:
            self._log_callback(msg)

    def get_input_devices(self) -> list[dict]:
        """Получить список доступных устройств ввода (микрофонов)."""
        devices = sd.query_devices()
        input_devices = []
        for i, dev in enumerate(devices):
            if dev.get('max_input_channels', 0) > 0:
                input_devices.append({
                    'index': i,
                    'name': dev.get('name', f'Device {i}'),
                    'channels': dev.get('max_input_channels', 0),
                    'sample_rate': dev.get('default_samplerate', 44100),
                })
        return input_devices

    def get_default_device(self) -> Optional[int]:
        """Получить индекс устройства ввода по умолчанию."""
        try:
            info = sd.query_devices(kind='input')
            return info.get('index') if isinstance(info, dict) else None
        except Exception:
            return None

    def start_recording(self, sample_rate: int = 24000, device: Optional[int] = None):
        """Начать непрерывную запись с микрофона.
        
        Запись идёт в буфер в памяти. Остановить через stop_recording().
        Получить данные через get_audio().
        """
        self.stop_recording()  # Сначала останавливаем предыдущую
        
        if device is None:
            device = self.get_default_device()
        
        if device is None:
            raise RuntimeError("Микрофон не найден. Подключите микрофон.")

        dev_info = sd.query_devices(device)
        actual_sr = int(dev_info.get('default_samplerate', sample_rate))
        
        self._sample_rate = actual_sr
        self._audio_buffer = np.array([])
        self._is_recording = True
        
        def _callback(indata, frames, time_info, status):
            if status:
                self._log(f"Запись: {status}")
            if indata.ndim > 1:
                indata = np.mean(indata, axis=1)
            # Копируем данные в буфер
            self._audio_buffer = np.concatenate([self._audio_buffer, indata.copy()])

        self._stream = sd.InputStream(
            device=device,
            channels=self._channels,
            samplerate=actual_sr,
            callback=_callback,
            blocksize=1024,  # Баланс между отзывчивостью и нагрузкой на CPU
        )
        self._stream.start()
        
        dev_name = dev_info.get('name', 'Unknown')[:50]
        self._log(f"🎤 Запись начата: {dev_name} @ {actual_sr}Hz")

    def stop_recording(self) -> Optional[np.ndarray]:
        """Остановить запись и вернуть записанное аудио.
        
        Returns:
            numpy array или None если не записывали
        """
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        
        audio = self._audio_buffer.copy() if self._audio_buffer is not None else None
        self._is_recording = False
        
        if audio is not None and len(audio) > 0:
            duration = len(audio) / self._sample_rate if self._sample_rate > 0 else 0
            self._log(f"⏹ Запись остановлена: {duration:.1f}s @ {self._sample_rate}Hz, samples={len(audio)}")
        
        return audio

    def get_audio(self) -> Optional[np.ndarray]:
        """Получить текущий буфер записи (без остановки)."""
        if self._audio_buffer is not None:
            return self._audio_buffer.copy()
        return None

    def get_duration(self) -> float:
        """Получить длительность текущей записи в секундах."""
        if self._audio_buffer is not None and self._sample_rate > 0:
            return len(self._audio_buffer) / self._sample_rate
        return 0.0

    def get_peak_level(self) -> float:
        """Получить текущий пиковый уровень громкости (0.0 - 1.0)."""
        if self._audio_buffer is not None and len(self._audio_buffer) > 0:
            return float(np.max(np.abs(self._audio_buffer)))
        return 0.0

    def save_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        output_path: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Сохранить записанное аудио в WAV файл."""
        try:
            if output_path is None:
                tmp_dir = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "temp"
                )
                os.makedirs(tmp_dir, exist_ok=True)
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(
                    tmp_dir, f"recording_{timestamp}.wav"
                )

            # Нормализуем аудио
            audio = audio.astype(np.float32)
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio / peak

            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            sf.write(output_path, audio, sample_rate)
            
            duration = len(audio) / sample_rate if sample_rate > 0 else 0
            self._log(f"💾 Сохранено: {output_path} ({duration:.1f}s)")
            return True, f"Сохранено: {Path(output_path).name}"

        except Exception as e:
            msg = f"Ошибка сохранения аудио: {e}"
            self._log(msg)
            return False, msg
