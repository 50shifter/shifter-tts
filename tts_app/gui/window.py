"""Главное окно приложения TTS + Транскрипция."""

import os
import sys
import traceback
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QEvent, QUrl
from PyQt6.QtGui import QAction, QColor, QFont, QIcon, QPainter, QPen, QBrush, QDragEnterEvent, QDropEvent
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QPushButton, QProgressBar, QPlainTextEdit,
    QTabWidget, QTextEdit, QVBoxLayout, QWidget, QComboBox, QCheckBox,
    QGroupBox, QSlider, QSpinBox, QFrame, QSplitter, QSizePolicy,
    QDialog, QGridLayout, QSpacerItem, QSizePolicy,
)

from modules.tts_engine import TTSEngine
from modules.whisper_engine import WhisperEngine, TranscriptionSegment
from modules.gpu_manager import GPUManager
from modules.recorder import AudioRecorder
from modules.video_extractor import VideoExtractor


# ======================== Стили (QSS Dark Theme) ========================

DARK_THEME = """
    QMainWindow {
        background-color: #1a1a2e;
    }
    QTabWidget::pane {
        border: 1px solid #3d3d5c;
        background-color: #16213e;
        border-radius: 8px;
    }
    QTabBar::tab {
        background-color: #0f3460;
        color: #e0e0e0;
        padding: 10px 24px;
        margin-right: 2px;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        font-size: 13px;
        font-weight: bold;
    }
    QTabBar::tab:selected {
        background-color: #1a1a2e;
        color: #00d4ff;
    }
    QTabBar::tab:hover {
        background-color: #1a4a7a;
    }
    QPushButton {
        background-color: #0f3460;
        color: #ffffff;
        border: 1px solid #533483;
        border-radius: 6px;
        padding: 8px 16px;
        font-size: 13px;
        font-weight: bold;
    }
    QPushButton:hover {
        background-color: #533483;
        border-color: #00d4ff;
    }
    QPushButton:pressed {
        background-color: #00d4ff;
        color: #1a1a2e;
    }
    QPushButton:disabled {
        background-color: #2a2a3e;
        color: #666688;
        border-color: #333355;
    }
    QLineEdit, QComboBox {
        background-color: #0f3460;
        color: #ffffff;
        border: 1px solid #533483;
        border-radius: 4px;
        padding: 6px 10px;
        font-size: 12px;
    }
    QLineEdit:focus, QComboBox:focus {
        border-color: #00d4ff;
    }
    QTextEdit, QPlainTextEdit {
        background-color: #0a0a1a;
        color: #c8f7c5;
        border: 1px solid #3d3d5c;
        border-radius: 6px;
        padding: 8px;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 12px;
    }
    QProgressBar {
        border: 1px solid #3d3d5c;
        border-radius: 4px;
        text-align: center;
        background-color: #0a0a1a;
        height: 20px;
    }
    QProgressBar::chunk {
        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 #533483, stop:1 #00d4ff);
        border-radius: 3px;
    }
    QGroupBox {
        color: #e0e0e0;
        font-weight: bold;
        font-size: 13px;
        border: 1px solid #3d3d5c;
        border-radius: 6px;
        margin-top: 8px;
        padding-top: 12px;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px;
        color: #00d4ff;
    }
    QLabel {
        color: #e0e0e0;
        font-size: 12px;
    }
    QCheckBox {
        color: #e0e0e0;
        font-size: 12px;
        spacing: 6px;
    }
    QCheckBox::indicator {
        width: 16px;
        height: 16px;
        border: 1px solid #533483;
        border-radius: 3px;
        background-color: #0f3460;
    }
    QCheckBox::indicator:checked {
        background-color: #00d4ff;
        border-color: #00d4ff;
    }
    QSlider::groove:horizontal {
        height: 6px;
        background: #3d3d5c;
        border-radius: 3px;
    }
    QSlider::handle:horizontal {
        width: 16px;
        height: 16px;
        margin: -5px 0;
        background: qradialgradient(cx:0.5, cy:0.5, radius:0.5,
            fx:0.3, fy:0.3, stop:0 #00d4ff, stop:1 #533483);
        border-radius: 8px;
    }
    QSlider::tickmarks:horizontal {
        background: #533483;
    }
"""


# ======================== Диалог записи с микрофона ========================

class WaveformWidget(QWidget):
    """Виджет визуализации звуковой волны в реальном времени."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(120)
        self.setMaximumHeight(150)
        self._audio_data: Optional[np.ndarray] = None
        self._is_recording = False
        self.setFixedWidth(600)

    def set_audio_data(self, audio: np.ndarray):
        """Установить данные для отображения."""
        if audio is not None and len(audio) > 0:
            # Берём подвыборку для отображения (не более 1000 точек)
            max_points = 1000
            if len(audio) > max_points:
                step = len(audio) // max_points
                audio = audio[::step]
            self._audio_data = audio
        else:
            self._audio_data = None
        self.update()

    def set_recording_state(self, is_recording: bool):
        """Установить состояние записи."""
        self._is_recording = is_recording
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w = self.width()
        h = self.height()
        mid_y = h // 2

        # Фон
        painter.fillRect(0, 0, w, h, QColor(10, 10, 26))

        # Центральная линия
        painter.setPen(QPen(QColor(61, 61, 92), 1))
        painter.drawLine(0, mid_y, w, mid_y)

        if self._audio_data is not None and len(self._audio_data) > 0:
            # Рисуем волну
            data = self._audio_data
            step = max(1, len(data) // (w - 2))
            
            color = QColor(0, 212, 255) if not self._is_recording else QColor(239, 68, 68)
            painter.setPen(QPen(color, 2))

            for x in range(1, w - 1):
                idx = min(x * step, len(data) - 1)
                y = int(mid_y - data[idx] * (mid_y - 5))
                painter.drawPoint(x, y)
        else:
            # Пустое состояние
            painter.setPen(QPen(QColor(61, 61, 92), 1, Qt.PenStyle.DashLine))
            for x in range(0, w, 20):
                painter.drawLine(x, mid_y - 30, x, mid_y + 30)

        # Индикатор записи
        if self._is_recording:
            painter.setPen(QPen(QColor(239, 68, 68), 2))
            painter.drawEllipse(w - 30, 10, 12, 12)
            painter.setBrush(QColor(239, 68, 68))
            painter.drawEllipse(w - 28, 12, 8, 8)

        super().paintEvent(event)


class DropLineEdit(QLineEdit):
    """QLineEdit с поддержкой drag-and-drop файлов."""
    file_dropped = pyqtSignal(str)  # сигнал с путём к файлу

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Разрешить дроп файлов."""
        if event.mimeData().hasUrls():
            # Проверяем расширения файлов
            for url in event.mimeData().urls():
                ext = Path(url.toLocalFile()).suffix.lower()
                allowed = ['.wav', '.mp3', '.ogg', '.flac', '.pt', '.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm']
                if ext in allowed:
                    event.acceptProposedAction()
                    return
        super().dragEnterEvent(event)

    def dropEvent(self, event: QDropEvent):
        """Обработка дропа файла."""
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if os.path.isfile(file_path):
                    self.file_dropped.emit(file_path)
                    break
            event.acceptProposedAction()
        else:
            super().dropEvent(event)


class RecordingDialog(QDialog):
    """Диалоговое окно для записи голоса с микрофона.
    
    Без ограничений по времени. Визуализация волны в реальном времени.
    """

    finished_recorded = pyqtSignal(np.ndarray, int)
    finished_error = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🎙️ Запись голоса")
        self.setModal(True)
        self.setMinimumSize(650, 320)
        
        self.recorder = AudioRecorder()
        self._is_recording = False
        self._audio_data: Optional[np.ndarray] = None
        self._sample_rate = 0
        self._update_timer: Optional[QTimer] = None

        self._setup_ui()
        self._detect_microphone()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Заголовок
        title_label = QLabel("🎙️ Запись голоса с микрофона")
        title_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #00d4ff; padding: 4px;"
        )
        layout.addWidget(title_label)

        # Инструкция
        instr_label = QLabel(
            "Говорите в микрофон. Нажмите \"Остановить\" когда закончите.\n"
            "Без ограничений по времени."
        )
        instr_label.setStyleSheet("color: #e0e0e0; font-size: 12px;")
        layout.addWidget(instr_label)

        # Визуализация волны
        self.waveform = WaveformWidget()
        layout.addWidget(self.waveform)

        # Таймер и уровень громкости
        info_layout = QHBoxLayout()
        
        self.timer_label = QLabel("⏱ 0.0s")
        self.timer_label.setStyleSheet(
            "font-size: 24px; font-weight: bold; color: #4ade80;"
        )
        info_layout.addWidget(self.timer_label)

        info_layout.addStretch()

        self.level_label = QLabel("🔊 --")
        self.level_label.setStyleSheet("font-size: 12px; color: #e0e0e0;")
        info_layout.addWidget(self.level_label)

        layout.addLayout(info_layout)

        # Статус устройства
        self.device_label = QLabel("")
        self.device_label.setStyleSheet("color: #fbbf24; font-size: 11px;")
        layout.addWidget(self.device_label)

        # Кнопки
        btn_layout = QHBoxLayout()
        
        self.record_button = QPushButton("⏺ Начать запись")
        self.record_button.setFixedHeight(45)
        self.record_button.setStyleSheet("""
            QPushButton {
                background-color: #ef4444;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 8px;
            }
            QPushButton:hover { background-color: #dc2626; }
        """)
        self.record_button.clicked.connect(self._toggle_recording)
        btn_layout.addWidget(self.record_button)

        cancel_btn = QPushButton("✖ Отмена")
        cancel_btn.setFixedHeight(45)
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #3d3d5c;
                color: white;
                font-size: 14px;
                border-radius: 8px;
            }
            QPushButton:hover { background-color: #533483; }
        """)
        cancel_btn.clicked.connect(self._cancel)
        btn_layout.addWidget(cancel_btn)

        layout.addLayout(btn_layout)

    def _detect_microphone(self):
        """Определить доступные микрофоны."""
        devices = self.recorder.get_input_devices()
        if not devices:
            self.device_label.setText("⚠ Микрофон не найден!")
            self.device_label.setStyleSheet("color: #ef4444; font-size: 11px;")
            self.record_button.setEnabled(False)
            return

        default_idx = self.recorder.get_default_device()
        if default_idx is not None:
            for dev in devices:
                if dev['index'] == default_idx:
                    name = dev['name'][:60]
                    self.device_label.setText(f"🎤 Микрофон: {name}")
                    self.device_label.setStyleSheet("color: #4ade80; font-size: 11px;")
                    break
            else:
                self.device_label.setText(
                    f"🎤 Доступно микрофонов: {len(devices)}"
                )
        else:
            names = [d['name'][:30] for d in devices]
            self.device_label.setText(f"🎤 Микрофоны: {' | '.join(names)}")

    def _toggle_recording(self):
        """Переключить запись."""
        if not self._is_recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        """Начать запись."""
        try:
            self.recorder.start_recording(sample_rate=24000)
            self._is_recording = True
            
            # Обновляем UI
            self.record_button.setText("⏹ Остановить")
            self.record_button.setStyleSheet("""
                QPushButton {
                    background-color: #fbbf24;
                    color: #1a1a2e;
                    font-size: 14px;
                    font-weight: bold;
                    border-radius: 8px;
                }
                QPushButton:hover { background-color: #f59e0b; }
            """)
            self.waveform.set_recording_state(True)

            # Таймер обновления
            self._update_timer = QTimer(self)
            self._update_timer.timeout.connect(self._on_update_tick)
            self._update_timer.start(100)  # Обновление каждые 100мс

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось начать запись:\n{e}")

    def _stop_recording(self):
        """Остановить запись."""
        if self._update_timer:
            self._update_timer.stop()
            self._update_timer = None

        audio = self.recorder.stop_recording()
        self._is_recording = False

        # Обновляем UI
        self.record_button.setText("⏺ Начать запись")
        self.record_button.setStyleSheet("""
            QPushButton {
                background-color: #ef4444;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 8px;
            }
            QPushButton:hover { background-color: #dc2626; }
        """)
        self.waveform.set_recording_state(False)

        if audio is not None and len(audio) > 0:
            self._audio_data = audio
            self._sample_rate = self.recorder._sample_rate
            self.waveform.set_audio_data(audio)
            self.accept()
        else:
            QMessageBox.warning(self, "Внимание", "Запись пуста. Попробуйте ещё раз.")

    def _cancel(self):
        """Отменить запись."""
        if self._is_recording:
            self.recorder.stop_recording()
        if self._update_timer:
            self._update_timer.stop()
            self._update_timer = None
        self.reject()

    def _on_update_tick(self):
        """Тик обновления UI во время записи."""
        audio = self.recorder.get_audio()
        if audio is not None:
            # Обновляем визуализацию (берём последние N точек для производительности)
            max_points = 2000
            if len(audio) > max_points:
                audio = audio[-max_points:]
            self.waveform.set_audio_data(audio)
            
            # Обновляем таймер
            duration = self.recorder.get_duration()
            mins = int(duration // 60)
            secs = duration % 60
            if mins > 0:
                self.timer_label.setText(f"⏱ {mins}:{secs:05.2f}")
            else:
                self.timer_label.setText(f"⏱ {secs:.1f}s")
            
            # Обновляем уровень громкости
            peak = self.recorder.get_peak_level()
            if peak > 0.8:
                color = "#ef4444"
            elif peak > 0.5:
                color = "#fbbf24"
            else:
                color = "#4ade80"
            self.level_label.setText(f"🔊 {int(peak*100)}%")
            self.level_label.setStyleSheet(
                f"font-size: 12px; color: {color}; font-weight: bold;"
            )

    def get_recorded_audio(self):
        """Получить записанное аудио."""
        return self._audio_data, self._sample_rate


class ExtractAudioThread(QThread):
    """Фоновый поток для извлечения аудио из видеофайла."""
    progress = pyqtSignal(int)
    finished_success = pyqtSignal(str, str)  # output_path, message
    finished_error = pyqtSignal(str)

    def __init__(self, extractor: VideoExtractor, input_path: str):
        super().__init__()
        self.extractor = extractor
        self.input_path = input_path

    def run(self):
        try:
            ok, msg = self.extractor.extract_audio(
                input_path=self.input_path,
                sample_rate=24000,
                progress_callback=lambda p: self.progress.emit(p),
            )
            if ok:
                # msg содержит путь к файлу
                parts = msg.split('→')
                output_path = parts[-1].strip() if len(parts) > 1 else ""
                self.finished_success.emit(output_path, msg)
            else:
                self.finished_error.emit(msg)
        except Exception as e:
            tb = traceback.format_exc()
            log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app.log")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"\n=== EXTRACT ERROR ===\n{tb}\n")
            self.finished_error.emit(str(e))


# ======================== Поток генерации TTS ========================

class TTSThread(QThread):
    """Фоновый поток для синтеза речи."""
    progress = pyqtSignal(int, int)
    finished_audio = pyqtSignal(np.ndarray, int)
    finished_error = pyqtSignal(str)
    finished_done = pyqtSignal()

    def __init__(self, engine: TTSEngine, text: str, ref_audio: Optional[str],
                 ref_text: Optional[str], voice_vector_path: Optional[str],
                 language: str, temperature: float,
                 top_k: int, top_p: float):
        super().__init__()
        self.engine = engine
        self.text = text
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self.voice_vector_path = voice_vector_path
        self.language = language
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self._cancel_requested = False

    def cancel(self):
        """Запросить отмену генерации."""
        self._cancel_requested = True

    def is_cancelled(self):
        """Проверить, запрошена ли отмена."""
        return self._cancel_requested

    def run(self):
        try:
            audio, sr, err = self.engine.generate(
                text=self.text,
                ref_audio_path=self.ref_audio,
                ref_text=self.ref_text,
                voice_vector_path=self.voice_vector_path,
                language=self.language,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                progress_callback=lambda cur, total: self.progress.emit(cur, total),
            )
            if err:
                self.finished_error.emit(err)
            else:
                self.finished_audio.emit(audio, sr)
        except Exception as e:
            tb = traceback.format_exc()
            log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app.log")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"\n=== TTS ERROR ===\n{tb}\n")
            print(f"[ERROR] TTS generation failed:\n{tb}", file=sys.stderr)
            self.finished_error.emit(f"TTS упал: {e}\n\nСмотри app.log.")
        finally:
            try:
                self.finished_done.emit()
            except Exception:
                pass


# ======================== Поток транскрипции ========================

class TranscribeThread(QThread):
    """Фоновый поток для транскрибации."""
    progress = pyqtSignal(int, int)
    finished_segments = pyqtSignal(list)
    finished_error = pyqtSignal(str)
    finished_done = pyqtSignal()

    def __init__(self, engine: WhisperEngine, audio_path: str, language: str, vad_filter: bool = False):
        super().__init__()
        self.engine = engine
        self.audio_path = audio_path
        self.language = language
        self.vad_filter = vad_filter

    def run(self):
        try:
            segments, err = self.engine.transcribe(
                audio_path=self.audio_path,
                language=self.language,
                vad_filter=self.vad_filter,
                progress_callback=lambda cur, total: self.progress.emit(cur, total),
            )
            if err:
                self.finished_error.emit(err)
            else:
                self.finished_segments.emit(segments)
        except Exception as e:
            tb = traceback.format_exc()
            log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app.log")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"\n=== TRANSCRIBE ERROR ===\n{tb}\n")
            print(f"[ERROR] Transcription failed:\n{tb}", file=sys.stderr)
            self.finished_error.emit(f"Транскрипция упала: {e}\n\nСмотри app.log для деталей.")
        finally:
            try:
                self.finished_done.emit()
            except Exception:
                pass


# ======================== Статус-бар ========================

class StatusLabel(QLabel):
    """Метка статуса с цветовой индикацией."""

    def set_status(self, text: str, color: str = "#e0e0e0"):
        self.setText(text)
        self.setStyleSheet(f"color: {color}; font-size: 12px; font-weight: bold;")


# ======================== Главное окно ========================

class MainWindow(QMainWindow):
    """Главное окно приложения."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Qwen3-TTS Studio")
        self.setMinimumSize(900, 700)
        self.resize(1100, 800)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Инициализация движков
        self.tts_engine = TTSEngine()
        self.whisper_engine = WhisperEngine()
        self.gpu_manager = GPUManager()
        self.recorder = AudioRecorder()
        self.video_extractor = VideoExtractor()

        # Настройки
        self.settings = {
            "device": "auto",       # auto, gpu, cpu
            "dtype": "float16",     # float16, bfloat16, float32
            "tts_model_path": "",
            "whisper_model_size": "medium",
            "default_language": "ru",  # язык по умолчанию (ru, en, auto)
            "auto_clean_gpu": False,
        }

        # Состояние
        self.current_audio: Optional[np.ndarray] = None
        self.current_sr: int = 0
        self.current_segments: list = []
        self.tts_thread: Optional[TTSThread] = None
        self.transcribe_thread: Optional[TranscribeThread] = None
        # Временные файлы записи и извлечения
        self._temp_recording_path: Optional[str] = None
        self._temp_extracted_audio_path: Optional[str] = None
        # Аудиоплеер для воспроизведения синтезированного аудио
        self.audio_player: Optional[QMediaPlayer] = None
        self.audio_output: Optional[QAudioOutput] = None
        self._audio_file_path: Optional[str] = None  # временный файл для воспроизведения

        # Настройка коллбэков логирования
        self.tts_engine.set_log_callback(self._log)
        self.whisper_engine.set_log_callback(self._log)
        self.gpu_manager.set_log_callback(self._log)
        self.recorder.set_log_callback(self._log)
        self.video_extractor.set_log_callback(self._log)

        self._setup_ui()
        self._apply_theme()

    def _setup_ui(self):
        """Создать весь UI."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(8)

        # Заголовок
        header = QHBoxLayout()
        title_label = QLabel("🎙️ Qwen3-TTS Studio")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #00d4ff; padding: 4px;")
        header.addWidget(title_label)
        header.addStretch()

        self.status_label = StatusLabel("● Готово")
        self.status_label.set_status("● Готово", "#4ade80")
        header.addWidget(self.status_label)
        main_layout.addLayout(header)

        # Табы
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # --- Вкладка TTS ---
        self._setup_tts_tab()
        # --- Вкладка Транскрипция ---
        self._setup_transcribe_tab()
        # --- Вкладка GPU ---
        self._setup_gpu_tab()
        # --- Вкладка Настройки ---
        self._setup_settings_tab()

        # Лог-панель
        log_group = QGroupBox("📋 Лог операций")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        log_layout.addWidget(self.log_text)
        main_layout.addWidget(log_group)

    # ======================== Вкладка TTS ========================

    def _setup_tts_tab(self):
        """Создать вкладку синтеза речи."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)
        layout.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Текст для синтеза
        text_group = QGroupBox("Текст для синтеза")
        text_layout = QVBoxLayout(text_group)
        self.tts_text_edit = QTextEdit()
        self.tts_text_edit.setPlaceholderText(
            "Введите текст для синтеза...\n\n"
            "Инструкции в скобках:\n"
            "  [смеётся] [шёпотом] [кричит] [пауза] [громко]\n"
            "  [грустно] [радостно] [сердито] [вздыхает]\n\n"
            "Ударения: ́ над гласной (напр. \"Приве́т, как де́ла?\")\n"
            "⚠ Ударения удаляются перед моделью — для отображения в логе.\n\n"
            "⚠ Нужен референсный аудиофайл для voice cloning!"
        )
        self.tts_text_edit.setMaximumHeight(150)
        self.tts_text_edit.setMinimumWidth(600)  # Делаем поле шире
        self.tts_text_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        text_layout.addWidget(self.tts_text_edit)
        layout.addWidget(text_group)

        # Референсное аудио
        ref_group = QGroupBox("Voice Cloning (референс)")
        ref_layout = QVBoxLayout(ref_group)

        # Строка выбора референсного аудио
        ref_audio_row = QHBoxLayout()
        self.ref_audio_path = DropLineEdit()
        self.ref_audio_path.setPlaceholderText("Путь к референсному аудио (.wav/.mp3) для voice cloning")
        self.ref_audio_path.setReadOnly(True)
        self.ref_audio_path.file_dropped.connect(self._on_ref_audio_dropped)
        self.ref_audio_path.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        ref_audio_row.addWidget(self.ref_audio_path)

        btn_browse_ref = QPushButton("📂 Выбрать")
        btn_browse_ref.clicked.connect(self._browse_ref_audio)
        ref_audio_row.addWidget(btn_browse_ref)

        # Кнопка записи с микрофона
        self.btn_record = QPushButton("🎤 Записать")
        self.btn_record.setFixedWidth(100)
        self.btn_record.setStyleSheet("""
            QPushButton {
                background-color: #ef4444;
                color: white;
                font-weight: bold;
                border-radius: 6px;
            }
            QPushButton:hover { background-color: #dc2626; }
        """)
        self.btn_record.clicked.connect(self._start_recording)
        ref_audio_row.addWidget(self.btn_record)

        # Кнопка выбора видеофайла (извлечь аудио)
        self.btn_video_extract = QPushButton("🎬 Видео")
        self.btn_video_extract.setFixedWidth(100)
        self.btn_video_extract.setStyleSheet("""
            QPushButton {
                background-color: #533483;
                color: white;
                font-weight: bold;
                border-radius: 6px;
            }
            QPushButton:hover { background-color: #7c3aed; }
        """)
        self.btn_video_extract.clicked.connect(self._browse_video_file)
        ref_audio_row.addWidget(self.btn_video_extract)

        ref_layout.addLayout(ref_audio_row)

        # Кнопка сохранения вектора голоса
        save_vector_row = QHBoxLayout()
        self.voice_vector_path = DropLineEdit()
        self.voice_vector_path.setPlaceholderText("Или: путь к .pt файлу с сохранённым вектором голоса")
        self.voice_vector_path.setReadOnly(True)
        self.voice_vector_path.file_dropped.connect(self._on_voice_vector_dropped)
        self.voice_vector_path.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        save_vector_row.addWidget(self.voice_vector_path)

        btn_browse_vector = QPushButton("📂 Выбрать .pt")
        btn_browse_vector.clicked.connect(self._browse_voice_vector)
        save_vector_row.addWidget(btn_browse_vector)

        # Кнопка удаления вектора голоса
        self.btn_remove_vector = QPushButton("✖ Убрать .pt")
        self.btn_remove_vector.setFixedWidth(120)
        self.btn_remove_vector.setStyleSheet("""
            QPushButton {
                background-color: #3d3d5c;
                color: #e0e0e0;
                font-weight: bold;
                border-radius: 6px;
            }
            QPushButton:hover { background-color: #ef4444; color: white; }
        """)
        self.btn_remove_vector.clicked.connect(self._remove_voice_vector)
        save_vector_row.addWidget(self.btn_remove_vector)

        ref_layout.addLayout(save_vector_row)

        # Кнопка извлечения и сохранения вектора
        btn_extract_save_vector = QPushButton("💾 Сохранить вектор голоса")
        btn_extract_save_vector.clicked.connect(self._save_voice_vector)
        ref_layout.addWidget(btn_extract_save_vector)

        layout.addWidget(ref_group)

        # Референсный текст
        self.ref_text_edit = QLineEdit()
        self.ref_text_edit.setPlaceholderText(
            "Транскрипт референсного аудио (для ICL режима, необязательно)"
        )
        self.ref_text_edit.setMinimumWidth(400)  # Делаем поле шире
        self.ref_text_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout.addWidget(self.ref_text_edit)

        # Параметры генерации
        params_group = QGroupBox("Параметры генерации")
        params_layout = QHBoxLayout(params_group)

        # Язык
        lang_label = QLabel("Язык:")
        lang_label.setToolTip(
            "🌐 Язык речи для синтеза.\n"
            "• auto — модель определит язык автоматически\n"
            "• ru — русский язык\n"
            "• en — английский язык"
        )
        self.tts_language = QComboBox()
        self.tts_language.addItems(["auto", "ru", "en"])
        # Устанавливаем язык по умолчанию из настроек
        default_lang = self.settings.get("default_language", "ru")
        lang_idx = {"auto": 0, "ru": 1, "en": 2}.get(default_lang, 1)
        self.tts_language.setCurrentIndex(lang_idx)
        params_layout.addWidget(lang_label)
        params_layout.addWidget(self.tts_language)

        # Температура
        temp_label = QLabel("Температура:")
        temp_label.setToolTip(
            "🌡️ Насколько \"креативной\" будет речь.\n"
            "• 0.5–0.7 — сбалансированно (рекомендуется)\n"
            "• < 0.5 — монотонно, но стабильно\n"
            "• > 0.8 — эмоционально, возможны артефакты"
        )
        self.tts_temperature = QSlider(Qt.Orientation.Horizontal)
        self.tts_temperature.setRange(10, 200)
        self.tts_temperature.setValue(90)
        self.tts_temperature.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.tts_temperature.setTickInterval(25)
        self.temp_value_label = QLabel("0.90")
        self.tts_temperature.valueChanged.connect(
            lambda v: self.temp_value_label.setText(f"{v/100:.2f}")
        )
        params_layout.addWidget(temp_label)
        params_layout.addWidget(self.tts_temperature)
        params_layout.addWidget(self.temp_value_label)

        # Top-K
        topk_label = QLabel("Top-K:")
        topk_label.setToolTip(
            "🔢 Сколько вариантов следующего звука рассматривает модель.\n"
            "• 30–50 — чёткая речь (рекомендуется)\n"
            "• < 20 — очень предсказуемо, монотонно\n"
            "• > 70 — живее, но риск ошибок произношения"
        )
        self.tts_top_k = QSpinBox()
        self.tts_top_k.setRange(1, 200)
        self.tts_top_k.setValue(50)
        params_layout.addWidget(topk_label)
        params_layout.addWidget(self.tts_top_k)

        # Top-P
        topp_label = QLabel("Top-P:")
        topp_label.setToolTip(
            "📊 Какой процент самых вероятных звуков учитывать.\n"
            "• 0.8–0.9 — сбалансированно (рекомендуется)\n"
            "• < 0.7 — стабильно, но однообразно\n"
            "• > 0.95 — живее, возможен шум"
        )
        self.tts_top_p = QSlider(Qt.Orientation.Horizontal)
        self.tts_top_p.setRange(50, 100)
        self.tts_top_p.setValue(100)
        self.topp_value_label = QLabel("1.00")
        self.tts_top_p.valueChanged.connect(
            lambda v: self.topp_value_label.setText(f"{v/100:.2f}")
        )
        params_layout.addWidget(topp_label)
        params_layout.addWidget(self.tts_top_p)
        params_layout.addWidget(self.topp_value_label)

        layout.addWidget(params_group)

        # Кнопки генерации
        gen_btn_layout = QHBoxLayout()
        gen_btn_layout.setContentsMargins(0, 0, 0, 0)
        
        self.btn_generate = QPushButton("🎤 Генерировать", objectName="generateBtn")
        self.btn_generate.setObjectName("generateBtn")
        self.btn_generate.setFixedHeight(48)
        self.btn_generate.setMinimumWidth(200)
        self.btn_generate.setStyleSheet("""
            QPushButton#generateBtn {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #533483, stop:1 #0f3460);
                font-size: 15px;
                padding: 10px 24px;
            }
        """)
        self.btn_generate.clicked.connect(self._generate_tts)
        gen_btn_layout.addWidget(self.btn_generate)

        # Кнопка отмены генерации (скрыта по умолчанию)
        self.btn_cancel = QPushButton("⏹ Отмена")
        self.btn_cancel.setObjectName("cancelBtn")
        self.btn_cancel.setFixedHeight(48)
        self.btn_cancel.setStyleSheet("""
            QPushButton#cancelBtn {
                background-color: #ef4444;
                color: white;
                font-size: 15px;
                padding: 10px 24px;
                border-radius: 6px;
            }
            QPushButton#cancelBtn:hover { background-color: #dc2626; }
        """)
        self.btn_cancel.setVisible(False)
        self.btn_cancel.clicked.connect(self._cancel_tts_generation)
        gen_btn_layout.addWidget(self.btn_cancel)
        gen_btn_layout.addStretch()

        layout.addLayout(gen_btn_layout)

        # Прогресс TTS
        self.tts_progress = QProgressBar()
        self.tts_progress.setVisible(False)
        layout.addWidget(self.tts_progress)

        # Результат
        result_group = QGroupBox("Результат")
        result_layout = QHBoxLayout(result_group)

        self.tts_result_label = QLabel("Нет результата")
        self.tts_result_label.setStyleSheet("color: #888; font-style: italic;")
        result_layout.addWidget(self.tts_result_label)
        result_layout.addStretch()

        # Кнопки воспроизведения
        play_btn_layout = QHBoxLayout()

        self.btn_play = QPushButton("▶ Воспроизвести")
        self.btn_play.setFixedWidth(140)
        self.btn_play.setStyleSheet("""
            QPushButton {
                background-color: #22c55e;
                color: white;
                font-weight: bold;
                border-radius: 6px;
            }
            QPushButton:hover { background-color: #16a34a; }
        """)
        self.btn_play.setEnabled(False)
        self.btn_play.clicked.connect(self._toggle_playback)
        play_btn_layout.addWidget(self.btn_play)

        self.btn_stop = QPushButton("⏹ Стоп")
        self.btn_stop.setFixedWidth(90)
        self.btn_stop.setStyleSheet("""
            QPushButton {
                background-color: #ef4444;
                color: white;
                font-weight: bold;
                border-radius: 6px;
            }
            QPushButton:hover { background-color: #dc2626; }
        """)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_playback)
        play_btn_layout.addWidget(self.btn_stop)

        # Индикатор позиции воспроизведения
        self.playback_position_label = QLabel("0:00 / 0:00")
        self.playback_position_label.setStyleSheet("color: #888; font-size: 11px;")
        play_btn_layout.addWidget(self.playback_position_label)
        play_btn_layout.addStretch()

        result_layout.addLayout(play_btn_layout)

        # Кнопки сохранения
        save_btn_layout = QHBoxLayout()

        btn_save_wav = QPushButton("💾 WAV")
        btn_save_wav.clicked.connect(lambda: self._save_audio(".wav"))
        save_btn_layout.addWidget(btn_save_wav)

        btn_save_mp3 = QPushButton("🎵 MP3")
        btn_save_mp3.clicked.connect(lambda: self._save_audio(".mp3"))
        save_btn_layout.addWidget(btn_save_mp3)

        result_layout.addLayout(save_btn_layout)

        layout.addWidget(result_group)

        # Drag & Drop
        self.tts_text_edit.setAcceptRichText(False)
        tab.setLayout(layout)
        self.tabs.addTab(tab, "🎙️ Синтез речи (TTS)")

    def _start_recording(self):
        """Открыть диалог записи с микрофона."""
        # Запись не требует загрузки модели — открываем сразу
        dialog = RecordingDialog(self)
        result = dialog.exec()

        if result == QDialog.DialogCode.Accepted:
            audio, sr = dialog.get_recorded_audio()
            if audio is not None and len(audio) > 0:
                # Сохраняем в временный файл
                tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
                os.makedirs(tmp_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self._temp_recording_path = os.path.join(
                    tmp_dir, f"recording_{timestamp}.wav"
                )

                ok, msg = self.recorder.save_audio(audio, sr, self._temp_recording_path)
                if ok:
                    self.ref_audio_path.setText(self._temp_recording_path)
                    duration = len(audio) / sr if sr > 0 else 0
                    QMessageBox.information(
                        self, "Запись завершена",
                        f"✅ Записано: {duration:.1f} сек\n{msg}"
                    )
                    self._log(f"Запись с микрофона: {self._temp_recording_path}")
                else:
                    QMessageBox.critical(self, "Ошибка", msg)
            else:
                QMessageBox.warning(self, "Внимание", "Запись пуста. Попробуйте ещё раз.")
        else:
            self._log("Запись отменена пользователем")

    def _on_ref_audio_dropped(self, file_path: str):
        """Обработка дропа файла на поле референсного аудио."""
        ext = Path(file_path).suffix.lower()
        audio_exts = ['.wav', '.mp3', '.ogg', '.flac']
        video_exts = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm']

        if ext in audio_exts:
            self._cleanup_temp_files()
            self.ref_audio_path.setText(file_path)
            self._log(f"Перетащен аудиофайл: {Path(file_path).name}")
        elif ext in video_exts:
            if not self.video_extractor.is_ffmpeg_available():
                QMessageBox.warning(
                    self, "FFmpeg не найден",
                    "Для видеофайлов необходим FFmpeg.\nУстановите: winget install Gyan.FFmpeg"
                )
                return
            self._set_status("⏳ Извлечение аудио из видео...", "#fbbf24")
            self.btn_video_extract.setEnabled(False)
            self._extract_thread = ExtractAudioThread(self.video_extractor, file_path)
            self._extract_thread.progress.connect(self._on_extract_progress)
            self._extract_thread.finished_success.connect(self._on_extract_success)
            self._extract_thread.finished_error.connect(self._on_extract_error)
            self._extract_thread.start()
        else:
            QMessageBox.warning(
                self, "Неподдерживаемый формат",
                f"На поле аудио можно перетаскивать:\n"
                f"  Аудио: {', '.join(audio_exts)}\n"
                f"  Видео: {', '.join(video_exts)}\nПолучен: {ext}"
            )

    def _on_voice_vector_dropped(self, file_path: str):
        """Обработка дропа файла на поле вектора голоса."""
        ext = Path(file_path).suffix.lower()
        if ext == '.pt':
            self.voice_vector_path.setText(file_path)
            self._log(f"Перетащен .pt файл: {Path(file_path).name}")
        else:
            QMessageBox.warning(
                self, "Неподдерживаемый формат",
                f"На поле вектора голоса можно перетаскивать только .pt файлы.\nПолучен: {ext}"
            )

    def _browse_ref_audio(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выбрать референсное аудио", "",
            "Audio Files (*.wav *.mp3 *.ogg *.flac);;All Files (*)"
        )
        if path:
            # Очищаем временные файлы если были
            for tmp_path in [self._temp_recording_path, self._temp_extracted_audio_path]:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
            self._temp_recording_path = None
            self._temp_extracted_audio_path = None
            self.ref_audio_path.setText(path)

    def _browse_video_file(self):
        """Выбрать видеофайл и извлечь из него аудио."""
        video_exts = ' '.join(f'*{e}' for e in ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm'])
        path, _ = QFileDialog.getOpenFileName(
            self, "Выбрать видеофайл (извлечь аудио)", "",
            f"Video Files ({video_exts});;All Files (*)"
        )
        if not path:
            return

        # Проверяем FFmpeg
        if not self.video_extractor.is_ffmpeg_available():
            QMessageBox.warning(
                self, "FFmpeg не найден",
                "Для извлечения аудио из видео необходим FFmpeg.\n\n"
                "Установите его через:\n"
                "  winget install Gyan.FFmpeg\n\n"
                "Или выберите WAV/MP3 файл напрямую."
            )
            return

        self._set_status("⏳ Извлечение аудио из видео...", "#fbbf24")
        self.btn_video_extract.setEnabled(False)

        # Запускаем в потоке
        self._extract_thread = ExtractAudioThread(self.video_extractor, path)
        self._extract_thread.progress.connect(self._on_extract_progress)
        self._extract_thread.finished_success.connect(self._on_extract_success)
        self._extract_thread.finished_error.connect(self._on_extract_error)
        self._extract_thread.start()

    def _on_extract_progress(self, percent: int):
        """Обновить прогресс извлечения."""
        self._set_status(f"⏳ Извлечение аудио... {percent}%", "#fbbf24")

    def _on_extract_success(self, output_path: str, message: str):
        """Извлечение завершено успешно."""
        self._temp_extracted_audio_path = output_path
        self.ref_audio_path.setText(output_path)
        self._set_status("✅ Аудио извлечено", "#4ade80")
        QMessageBox.information(self, "Успех", f"{message}\n\nФайл готов к использованию.")
        self.btn_video_extract.setEnabled(True)

    def _on_extract_error(self, error_msg: str):
        """Ошибка извлечения."""
        self._set_status("❌ Ошибка извлечения", "#ef4444")
        QMessageBox.critical(self, "Ошибка", f"Не удалось извлечь аудио:\n{error_msg}")
        self.btn_video_extract.setEnabled(True)

    def _browse_voice_vector(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выбрать вектор голоса (.pt)", "",
            "Voice Vector Files (*.pt);;All Files (*)"
        )
        if path:
            self.voice_vector_path.setText(path)

    def _remove_voice_vector(self):
        """Убрать загруженный .pt файл и очистить кэш."""
        current = self.voice_vector_path.text().strip()
        if not current:
            self._log("Вектор голоса уже не выбран")
            return

        # Очищаем поле и кэш движка
        self.voice_vector_path.setText("")
        self.tts_engine.clear_voice_vector()
        self._log(f"Вектор голоса убран: {Path(current).name}")
        self._set_status("✅ Вектор голоса убран", "#4ade80")

    def _save_voice_vector(self):
        """Извлечь и сохранить вектор голоса из референсного аудио."""
        ref_audio = self.ref_audio_path.text().strip()
        if not ref_audio or not os.path.exists(ref_audio):
            QMessageBox.warning(
                self, "Внимание",
                "Сначала выберите референсный аудиофайл (.wav/.mp3)."
            )
            return

        # Проверяем загружена ли модель
        if self.tts_engine.model is None:
            ok, msg = self._load_tts_model()
            if not ok:
                QMessageBox.warning(self, "Ошибка", f"Не удалось загрузить TTS модель:\n{msg}")
                return

        ref_text = self.ref_text_edit.text() or None

        # Запрашиваем путь сохранения
        default_name = Path(ref_audio).stem + "_voice.pt"
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить вектор голоса", default_name,
            "Voice Vector Files (*.pt)"
        )
        if not path:
            return

        self._set_status("⏳ Извлечение и сохранение вектора голоса...", "#fbbf24")

        ok, msg = self.tts_engine.extract_and_save_voice_vector(
            ref_audio_path=ref_audio,
            output_path=path,
            ref_text=ref_text,
            x_vector_only_mode=False,
        )

        if ok:
            QMessageBox.information(self, "Успех", msg)
            self._set_status("✅ Вектор голоса сохранён", "#4ade80")
            # Автоматически загружаем вектор для использования
            self.voice_vector_path.setText(path)
        else:
            QMessageBox.critical(self, "Ошибка", msg)
            self._set_status("❌ Ошибка сохранения вектора", "#ef4444")

    def _generate_tts(self):
        """Запустить генерацию TTS."""
        text = self.tts_text_edit.toPlainText().strip()
        if not text:
            self._set_status("⚠ Введите текст", "#fbbf24")
            return

        ref_audio = self.ref_audio_path.text() or None
        voice_vector = self.voice_vector_path.text() or None

        # Проверяем, что хотя бы один источник голоса указан
        if not ref_audio and not voice_vector:
            QMessageBox.warning(
                self, "Внимание",
                "Для синтеза речи необходим:\n"
                "  • Референсный аудиофайл (.wav/.mp3), ИЛИ\n"
                "  • Сохранённый вектор голоса (.pt файл)"
            )
            return

        ref_text = self.ref_text_edit.text() or None
        language = self.tts_language.currentText()
        temperature = self.tts_temperature.value() / 100.0
        top_k = self.tts_top_k.value()
        top_p = self.tts_top_p.value() / 100.0

        # Проверяем загружена ли модель
        if self.tts_engine.model is None:
            ok, msg = self._load_tts_model()
            if not ok:
                QMessageBox.warning(self, "Ошибка", f"Не удалось загрузить TTS модель:\n{msg}")
                return

        # UI состояние
        self.tts_progress.setVisible(True)
        self.tts_progress.setValue(0)
        self.btn_generate.setEnabled(False)
        self.btn_generate.setText("⏳ Генерация...")
        self.btn_cancel.setVisible(True)

        self._set_status("🔄 Синтез речи...", "#fbbf24")

        # Запускаем поток
        self.tts_thread = TTSThread(
            self.tts_engine, text, ref_audio, ref_text,
            voice_vector, language, temperature, top_k, top_p,
        )
        self.tts_thread.progress.connect(self._on_tts_progress)
        self.tts_thread.finished_audio.connect(self._on_tts_finished)
        self.tts_thread.finished_error.connect(self._on_tts_error)
        self.tts_thread.finished_done.connect(self._on_tts_done)
        self.tts_thread.start()

    def _cancel_tts_generation(self):
        """Отменить текущую генерацию TTS."""
        if self.tts_thread and self.tts_thread.isRunning():
            self.tts_thread.cancel()
            self._set_status("⏹ Генерация отменена", "#ef4444")
            # Ждём завершения потока
            self.tts_thread.wait(2000)
            self.btn_cancel.setVisible(False)
            self.btn_generate.setEnabled(True)
            self.btn_generate.setText("🎤 Генерировать")
            self.tts_progress.setVisible(False)
            self._log("Генерация TTS отменена пользователем")

    def _on_tts_progress(self, current: int, total: int):
        self.tts_progress.setValue(current)

    def _on_tts_finished(self, audio: np.ndarray, sr: int):
        self.current_audio = audio
        self.current_sr = sr
        duration = len(audio) / sr if sr > 0 else 0
        self.tts_result_label.setText(f"✅ Готово! Длительность: {duration:.2f}s | SR: {sr}Hz")
        self.tts_result_label.setStyleSheet("color: #4ade80; font-weight: bold;")
        self._log(f"TTS синтез завершён: {duration:.2f}s @ {sr}Hz")

        # Включаем кнопки воспроизведения и сохраняем для воспроизведения
        self.btn_play.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.playback_position_label.setText("0:00 / 0:00")

    def _on_tts_error(self, error_msg: str):
        QMessageBox.critical(self, "Ошибка TTS", error_msg)
        self._set_status("❌ Ошибка синтеза", "#ef4444")
        self.btn_cancel.setVisible(False)

    def _on_tts_done(self):
        self.tts_progress.setVisible(False)
        self.btn_generate.setEnabled(True)
        self.btn_generate.setText("🎤 Генерировать")
        self.btn_cancel.setVisible(False)

        # Принудительная очистка GPU кэша после каждой генерации
        self.gpu_manager.clear_cache()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        self._log("GPU кэш очищен после генерации")

    def _save_audio(self, ext: str):
        if self.current_audio is None:
            QMessageBox.warning(self, "Внимание", "Нет результата для сохранения.")
            return

        default_name = f"output{ext}"
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить аудио", default_name,
            "Audio Files (*.wav *.mp3)"
        )
        if path:
            ok, msg = self.tts_engine.save_audio(self.current_audio, self.current_sr, path)
            if ok:
                QMessageBox.information(self, "Успех", msg)
                self._log(msg)
            else:
                QMessageBox.critical(self, "Ошибка", msg)

    # ======================== Аудиоплеер ========================

    def _init_audio_player(self):
        """Инициализировать аудиоплеер (один раз)."""
        if self.audio_player is not None:
            return

        try:
            self.audio_output = QAudioOutput()
            self.audio_player = QMediaPlayer()
            self.audio_player.setAudioOutput(self.audio_output)

            # Сигнал изменения позиции (для обновления индикатора)
            self.audio_player.positionChanged.connect(self._on_playback_position_changed)
            self.audio_player.durationChanged.connect(self._on_playback_duration_changed)
            self.audio_player.playbackStateChanged.connect(self._on_playback_state_changed)
        except Exception as e:
            self._log(f"Ошибка инициализации плеера: {e}")

    def _toggle_playback(self):
        """Переключить воспроизведение / паузу."""
        if self.audio_player is None:
            self._init_audio_player()

        if self.audio_player is None:
            QMessageBox.critical(self, "Ошибка", "Не удалось инициализировать аудиоплеер.")
            return

        state = self.audio_player.playbackState()
        if state == QMediaPlayer.PlaybackState.PlayingState:
            # Пауза
            self.audio_player.pause()
        else:
            # Воспроизведение
            self._prepare_audio_file()
            self.audio_player.play()

    def _stop_playback(self):
        """Остановить воспроизведение."""
        if self.audio_player is not None:
            self.audio_player.stop()

    def _prepare_audio_file(self):
        """Сохранить текущее аудио во временный файл для воспроизведения."""
        if self.current_audio is None or self.current_sr == 0:
            return

        # Удаляем старый временный файл если есть
        if self._audio_file_path and os.path.exists(self._audio_file_path):
            try:
                os.remove(self._audio_file_path)
            except Exception:
                pass

        # Создаём временный WAV файл
        tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
        os.makedirs(tmp_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._audio_file_path = os.path.join(tmp_dir, f"playback_{timestamp}.wav")

        try:
            import soundfile as sf
            sf.write(self._audio_file_path, self.current_audio, self.current_sr)

            if self.audio_player is not None and self._audio_file_path:
                self.audio_player.setSource(QUrl.fromLocalFile(self._audio_file_path))
        except Exception as e:
            self._log(f"Ошибка подготовки аудио для воспроизведения: {e}")

    def _on_playback_position_changed(self, position: int):
        """Обновить индикатор позиции."""
        duration = self.audio_player.duration() if self.audio_player else 0
        self.playback_position_label.setText(
            f"{self._format_time(position)} / {self._format_time(duration)}"
        )

    def _on_playback_duration_changed(self, duration: int):
        """Обновить индикатор длительности."""
        position = self.audio_player.position() if self.audio_player else 0
        self.playback_position_label.setText(
            f"{self._format_time(position)} / {self._format_time(duration)}"
        )

    def _on_playback_state_changed(self, state: QMediaPlayer.PlaybackState):
        """Обновить состояние кнопок воспроизведения."""
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.btn_play.setText("⏸ Пауза")
            self.btn_stop.setEnabled(True)
        elif state == QMediaPlayer.PlaybackState.PausedState:
            self.btn_play.setText("▶ Продолжить")
        else:  # StoppedState или NullState
            self.btn_play.setText("▶ Воспроизвести")
            self.btn_stop.setEnabled(False)
            if self.audio_player and self.audio_player.position() > 0:
                pos = self.audio_player.position()
                dur = self.audio_player.duration()
                self.playback_position_label.setText(
                    f"{self._format_time(pos)} / {self._format_time(dur)}"
                )

    @staticmethod
    def _format_time(ms: int) -> str:
        """Форматировать миллисекунды в ММ:СС."""
        if ms <= 0:
            return "0:00"
        total_secs = ms // 1000
        mins = total_secs // 60
        secs = total_secs % 60
        return f"{mins}:{secs:02d}"

    def _cleanup_temp_audio(self):
        """Очистить временный файл воспроизведения."""
        if self._audio_file_path and os.path.exists(self._audio_file_path):
            try:
                os.remove(self._audio_file_path)
            except Exception:
                pass
            self._audio_file_path = None

    # ======================== Вкладка Транскрипция ========================

    def _setup_transcribe_tab(self):
        """Создать вкладку транскрибции."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        # Выбор файла
        file_group = QGroupBox("Аудиофайл")
        file_layout = QHBoxLayout(file_group)

        self.transcribe_audio_path = DropLineEdit()
        self.transcribe_audio_path.setPlaceholderText("Путь к аудиофайлу (.wav, .mp3, .ogg)")
        self.transcribe_audio_path.setReadOnly(True)
        self.transcribe_audio_path.file_dropped.connect(self._on_transcribe_audio_dropped)
        self.transcribe_audio_path.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        file_layout.addWidget(self.transcribe_audio_path)

        btn_browse_trans = QPushButton("📂 Выбрать")
        btn_browse_trans.clicked.connect(self._browse_transcribe_audio)
        file_layout.addWidget(btn_browse_trans)

        layout.addWidget(file_group)

        # Параметры
        params_group = QGroupBox("Параметры транскрипции")
        params_layout = QHBoxLayout(params_group)

        lang_label = QLabel("Язык:")
        lang_label.setToolTip(
            "🌐 Язык аудио для транскрипции.\n"
            "• auto — модель определит язык автоматически\n"
            "• ru — русский язык\n"
            "• en — английский язык"
        )
        self.transcribe_language = QComboBox()
        self.transcribe_language.addItems(["auto", "ru", "en"])
        # Устанавливаем язык по умолчанию из настроек
        default_lang = self.settings.get("default_language", "ru")
        lang_idx = {"auto": 0, "ru": 1, "en": 2}.get(default_lang, 1)
        self.transcribe_language.setCurrentIndex(lang_idx)
        self.transcribe_language.setMinimumWidth(80)
        params_layout.addWidget(lang_label)
        params_layout.addWidget(self.transcribe_language)

        beam_label = QLabel("Beam Size:")
        beam_label.setToolTip(
            "🔍 Сколько вариантов текста проверяет модель при транскрипции.\n"
            "• 3–5 — быстро и достаточно точно (рекомендуется)\n"
            "• 7–10 — точнее, но медленнее (для шумной речи)\n"
            "• 1 — максимально быстро, но больше ошибок"
        )
        self.transcribe_beam = QSpinBox()
        self.transcribe_beam.setRange(1, 10)
        self.transcribe_beam.setValue(5)
        params_layout.addWidget(beam_label)
        params_layout.addWidget(self.transcribe_beam)

        self.vad_check = QCheckBox("VAD фильтр (шумоподавление)")
        self.vad_check.setChecked(False)  # По умолчанию выключен — лучше результат
        self.vad_check.setToolTip("Фильтрует тишину и шум. Может убрать полезные сегменты.")
        params_layout.addWidget(self.vad_check)
        params_layout.addStretch()

        layout.addWidget(params_group)

        # Кнопка транскрипции
        btn_transcribe = QPushButton("🎧 Транскрибировать", objectName="transcribeBtn")
        btn_transcribe.setObjectName("transcribeBtn")
        btn_transcribe.setFixedHeight(48)
        btn_transcribe.setMinimumWidth(200)
        btn_transcribe.setStyleSheet("""
            QPushButton#transcribeBtn {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #533483, stop:1 #0f3460);
                font-size: 15px;
                padding: 10px 24px;
            }
        """)
        btn_transcribe.clicked.connect(self._transcribe)
        layout.addWidget(btn_transcribe, 0, Qt.AlignmentFlag.AlignCenter)

        # Прогресс транскрипции
        self.transcribe_progress = QProgressBar()
        self.transcribe_progress.setVisible(False)
        layout.addWidget(self.transcribe_progress)

        # Результат
        result_group = QGroupBox("Результат транскрипции")
        result_layout = QVBoxLayout(result_group)

        self.transcribe_text_edit = QTextEdit()
        self.transcribe_text_edit.setReadOnly(True)
        self.transcribe_text_edit.setPlaceholderText(
            "Здесь появится чистый текст транскрипции.\n"
            "Пунктуация расставляется автоматически по паузам:\n"
            "  2-5 сек → точка (.)\n"
            "  >5 сек → многоточие (...)"
        )
        self.transcribe_text_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        result_layout.addWidget(self.transcribe_text_edit)

        export_layout = QHBoxLayout()
        btn_save_txt = QPushButton("📄 TXT")
        btn_save_txt.clicked.connect(lambda: self._export_transcription(".txt"))
        export_layout.addWidget(btn_save_txt)

        btn_save_srt = QPushButton("🎬 SRT")
        btn_save_srt.clicked.connect(lambda: self._export_transcription(".srt"))
        export_layout.addWidget(btn_save_srt)

        btn_clear_gpu = QPushButton("🧹 Очистить GPU кэш")
        btn_clear_gpu.clicked.connect(self._clear_gpu_after_transcribe)
        export_layout.addWidget(btn_clear_gpu)
        export_layout.addStretch()

        result_layout.addLayout(export_layout)
        layout.addWidget(result_group)

        tab.setLayout(layout)
        self.tabs.addTab(tab, "🎧 Транскрипция")

    def _on_transcribe_audio_dropped(self, file_path: str):
        """Обработка дропа файла на поле транскрипции."""
        ext = Path(file_path).suffix.lower()
        audio_exts = ['.wav', '.mp3', '.ogg', '.flac']
        video_exts = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm']

        if ext in audio_exts:
            self.transcribe_audio_path.setText(file_path)
            self._log(f"Перетащен аудиофайл для транскрипции: {Path(file_path).name}")
        elif ext in video_exts:
            if not self.video_extractor.is_ffmpeg_available():
                QMessageBox.warning(
                    self, "FFmpeg не найден",
                    "Для видеофайлов необходим FFmpeg.\nУстановите: winget install Gyan.FFmpeg"
                )
                return
            # Извлекаем аудио и подставляем путь
            tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
            os.makedirs(tmp_dir, exist_ok=True)
            base_name = Path(file_path).stem
            base_name = ''.join(c if c.isalnum() or c in '-_' else '_' for c in base_name)
            output_path = os.path.join(tmp_dir, f"{base_name}_transcribe_audio.wav")
            
            self._set_status("⏳ Извлечение аудио из видео...", "#fbbf24")
            ok, msg = self.video_extractor.extract_audio(
                input_path=file_path,
                output_path=output_path,
                sample_rate=16000,  # Whisper использует 16kHz
            )
            if ok:
                self.transcribe_audio_path.setText(output_path)
                self._set_status("✅ Аудио извлечено", "#4ade80")
                self._log(f"Перетащен видеофайл для транскрипции: {Path(file_path).name}")
            else:
                QMessageBox.critical(self, "Ошибка", f"Не удалось извлечь аудио:\n{msg}")
        else:
            QMessageBox.warning(
                self, "Неподдерживаемый формат",
                f"На поле транскрипции можно перетаскивать:\n"
                f"  Аудио: {', '.join(audio_exts)}\n"
                f"  Видео: {', '.join(video_exts)}\nПолучен: {ext}"
            )

    def _browse_transcribe_audio(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выбрать аудиофайл", "",
            "Audio Files (*.wav *.mp3 *.ogg *.flac)"
        )
        if path:
            self.transcribe_audio_path.setText(path)

    def _transcribe(self):
        audio_path = self.transcribe_audio_path.text().strip()
        if not audio_path:
            self._set_status("⚠ Выберите аудиофайл", "#fbbf24")
            return

        language = self.transcribe_language.currentText()
        beam_size = self.transcribe_beam.value()
        vad_checked = self.vad_check.isChecked()

        # Проверяем загружена ли модель
        if self.whisper_engine.model is None:
            ok, msg = self._load_whisper_model()
            if not ok:
                QMessageBox.warning(self, "Ошибка", f"Не удалось загрузить Whisper:\n{msg}")
                return

        # UI состояние
        self.transcribe_progress.setVisible(True)
        self.transcribe_progress.setValue(0)
        btn = [w for w in self.findChildren(QPushButton) if w.objectName() == "transcribeBtn"]
        if btn:
            btn[0].setEnabled(False)
            btn[0].setText("⏳ Транскрипция...")

        self._set_status(f"🔄 Транскрипция... (VAD={'вкл' if vad_checked else 'выкл'})", "#fbbf24")

        # Запускаем поток
        self.transcribe_thread = TranscribeThread(
            self.whisper_engine, audio_path, language, vad_filter=vad_checked,
        )
        self.transcribe_thread.progress.connect(self._on_transcribe_progress)
        self.transcribe_thread.finished_segments.connect(self._on_transcribe_finished)
        self.transcribe_thread.finished_error.connect(self._on_transcribe_error)
        self.transcribe_thread.finished_done.connect(self._on_transcribe_done)
        self.transcribe_thread.start()

    def _on_transcribe_progress(self, current: int, total: int):
        self.transcribe_progress.setValue(current)

    def _on_transcribe_finished(self, segments: list):
        self.current_segments = segments
        # Чистый текст без таймкодов — основной вывод
        clean_text = "\n".join(seg.text for seg in segments) if segments else ""
        self.transcribe_text_edit.setPlainText(clean_text)

        # Также показываем чистый текст в лог
        self._log(f"Транскрипция: {len(segments)} сегментов")
        if clean_text:
            self._log(clean_text[:500])
        status_color = "#4ade80" if segments else "#fbbf24"
        status_msg = f"✅ Транскрипция: {len(segments)} сегментов" if segments else "⚠ Нет сегментов"
        self._set_status(status_msg, status_color)

    def _on_transcribe_error(self, error_msg: str):
        QMessageBox.critical(self, "Ошибка транскрипции", error_msg)
        self._set_status("❌ Ошибка", "#ef4444")

    def _on_transcribe_done(self):
        self.transcribe_progress.setVisible(False)
        btn = [w for w in self.findChildren(QPushButton) if w.objectName() == "transcribeBtn"]
        if btn:
            btn[0].setEnabled(True)
            btn[0].setText("🎧 Транскрибировать")

    def _export_transcription(self, ext: str):
        if not self.current_segments:
            QMessageBox.warning(self, "Внимание", "Нет результатов для экспорта.")
            return

        default_name = f"transcription{ext}"
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить транскрипцию", default_name,
            "Files (*.* )"
        )
        if path:
            # TXT: чистый текст без таймкодов
            if ext == ".txt":
                try:
                    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                    with open(path, "w", encoding="utf-8") as f:
                        for seg in self.current_segments:
                            f.write(seg.text + "\n")
                    msg = f"Сохранён чистый TXT: {path}"
                    QMessageBox.information(self, "Успех", msg)
                    self._log(msg)
                except Exception as e:
                    QMessageBox.critical(self, "Ошибка", str(e))
            elif ext == ".srt":
                ok, msg = self.whisper_engine.export_srt(self.current_segments, path)
                if ok:
                    QMessageBox.information(self, "Успех", msg)
                    self._log(msg)
                else:
                    QMessageBox.critical(self, "Ошибка", msg)

    def _clear_gpu_after_transcribe(self):
        result = self.gpu_manager.clear_cache()
        self._log(result)
        self._update_gpu_display()

    # ======================== Вкладка GPU ========================

    def _setup_gpu_tab(self):
        """Создать вкладку управления GPU."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        # Мониторинг
        monitor_group = QGroupBox("📊 Мониторинг GPU")
        monitor_layout = QHBoxLayout(monitor_group)

        self.gpu_vram_label = QLabel("VRAM: -- / -- MiB")
        self.gpu_vram_label.setMinimumWidth(140)
        monitor_layout.addWidget(self.gpu_vram_label)

        self.gpu_temp_label = QLabel("Температура: --°C")
        self.gpu_temp_label.setMinimumWidth(120)
        monitor_layout.addWidget(self.gpu_temp_label)

        self.gpu_util_label = QLabel("Загрузка: --%")
        self.gpu_util_label.setMinimumWidth(100)
        monitor_layout.addWidget(self.gpu_util_label)

        self.gpu_power_label = QLabel("Мощность: --W")
        self.gpu_power_label.setMinimumWidth(120)
        monitor_layout.addWidget(self.gpu_power_label)

        layout.addWidget(monitor_group)

        # Torch VRAM
        torch_group = QGroupBox("🔥 PyTorch VRAM")
        torch_layout = QHBoxLayout(torch_group)

        self.torch_alloc_label = QLabel("Выделено: -- MiB")
        torch_layout.addWidget(self.torch_alloc_label)

        self.torch_reserved_label = QLabel("Зарезервировано: -- MiB")
        torch_layout.addWidget(self.torch_reserved_label)

        layout.addWidget(torch_group)

        # Кнопки управления
        control_group = QGroupBox("🎮 Управление")
        control_layout = QVBoxLayout(control_group)

        btn_clear_cache = QPushButton("🧹 Очистить CUDA кэш")
        btn_clear_cache.clicked.connect(self._clear_gpu_cache)
        control_layout.addWidget(btn_clear_cache)

        btn_force_release = QPushButton("💥 Принудительно освободить VRAM")
        btn_force_release.clicked.connect(self._force_release_vram)
        control_layout.addWidget(btn_force_release)

        self.auto_clean_check = QCheckBox("Автоматически чистить кэш после каждой операции")
        self.auto_clean_check.setChecked(False)
        self.auto_clean_check.stateChanged.connect(self._on_auto_clean_changed)
        control_layout.addWidget(self.auto_clean_check)

        layout.addWidget(control_group)

        # Кнопки выгрузки моделей
        unload_group = QGroupBox("📦 Выгрузка моделей")
        unload_layout = QHBoxLayout(unload_group)

        btn_unload_tts = QPushButton("Выгрузить TTS модель")
        btn_unload_tts.clicked.connect(self._unload_tts_model)
        unload_layout.addWidget(btn_unload_tts)

        btn_unload_whisper = QPushButton("Выгрузить Whisper модель")
        btn_unload_whisper.clicked.connect(self._unload_whisper_model)
        unload_layout.addWidget(btn_unload_whisper)

        layout.addWidget(unload_group)

        # Таймер обновления GPU (всегда активен)
        self.gpu_timer = QTimer()
        self.gpu_timer.timeout.connect(self._update_gpu_display)
        self.gpu_timer.start(1000)  # Обновление каждую секунду для реального времени

        tab.setLayout(layout)
        self.tabs.addTab(tab, "🖥️ Управление GPU")

    def _update_gpu_display(self):
        """Обновить отображение статистики GPU."""
        stats = self.gpu_manager.get_gpu_stats()
        torch_vram = self.gpu_manager.get_torch_vram()

        if stats:
            self.gpu_vram_label.setText(f"VRAM: {stats['used']}/{stats['total']} MiB")
            self.gpu_temp_label.setText(f"Температура: {stats['temp']}°C")
            self.gpu_util_label.setText(f"Загрузка: {stats['util']}%")
            self.gpu_power_label.setText(f"Мощность: {stats['power']:.1f}W")

            # Цветовая индикация температуры
            if stats['temp'] > 80:
                self.gpu_temp_label.setStyleSheet("color: #ef4444; font-weight: bold;")
            elif stats['temp'] > 65:
                self.gpu_temp_label.setStyleSheet("color: #fbbf24;")
            else:
                self.gpu_temp_label.setStyleSheet("")

        if torch_vram["reserved"] > 0:
            self.torch_alloc_label.setText(f"Выделено: {torch_vram['allocated']} MiB")
            self.torch_reserved_label.setText(f"Зарезервировано: {torch_vram['reserved']} MiB")

    def _clear_gpu_cache(self):
        result = self.gpu_manager.clear_cache()
        QMessageBox.information(self, "GPU Кэш", result)
        self._update_gpu_display()

    def _force_release_vram(self):
        result = self.gpu_manager.force_release_vram()
        QMessageBox.information(self, "VRAM", result)
        self._update_gpu_display()

    def _on_auto_clean_changed(self, state):
        self.settings["auto_clean_gpu"] = (state == 2)
        self.gpu_manager.auto_clean = self.settings["auto_clean_gpu"]

    def _unload_tts_model(self):
        if QMessageBox.question(
            self, "Подтверждение",
            "Выгрузить TTS модель из памяти?\nЭто потребует повторной загрузки при следующем использовании.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        ) == QMessageBox.StandardButton.Yes:
            self.tts_engine.unload_model()
            self._log("TTS модель выгружена")
            self._set_status("TTS модель выгружена", "#fbbf24")

    def _unload_whisper_model(self):
        if QMessageBox.question(
            self, "Подтверждение",
            "Выгрузить Whisper модель из памяти?\nЭто потребует повторной загрузки при следующем использовании.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        ) == QMessageBox.StandardButton.Yes:
            self.whisper_engine.unload_model()
            self._log("Whisper модель выгружена")
            self._set_status("Whisper модель выгружена", "#fbbf24")

    # ======================== Вкладка Настройки ========================

    def _setup_settings_tab(self):
        """Создать вкладку настроек."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        # Устройство
        device_group = QGroupBox("Устройство вычислений")
        device_layout = QHBoxLayout(device_group)

        dev_label = QLabel("Режим:")
        dev_label.setToolTip(
            "💻 Какое устройство использовать для вычислений.\n"
            "• auto — автоматически выберет GPU если есть\n"
            "• gpu (CUDA) — принудительно использует видеокарту NVIDIA\n"
            "• cpu — только процессор (медленнее, но работает везде)"
        )
        self.settings_device = QComboBox()
        self.settings_device.addItems(["auto", "gpu (CUDA)", "cpu"])
        idx = {"auto": 0, "gpu": 1, "cpu": 2}.get(self.settings["device"], 0)
        self.settings_device.setCurrentIndex(idx)
        self.settings_device.setMinimumWidth(150)
        device_layout.addWidget(dev_label)
        device_layout.addWidget(self.settings_device)
        layout.addWidget(device_group)

        # Точность
        dtype_group = QGroupBox("Точность вычислений")
        dtype_layout = QHBoxLayout(dtype_group)

        dtype_label = QLabel("Тип:")
        dtype_label.setToolTip(
            "🔢 Точность вычислений (влияет на качество и память).\n"
            "• float16 — быстро, мало памяти (рекомендуется для GPU)\n"
            "• bfloat16 — стабильнее float16, нужен новый GPU (Ampere+)\n"
            "• float32 — максимальное качество, но много памяти"
        )
        self.settings_dtype = QComboBox()
        self.settings_dtype.addItems(["float16", "bfloat16", "float32"])
        idx = {"float16": 0, "bfloat16": 1, "float32": 2}.get(self.settings["dtype"], 0)
        self.settings_dtype.setCurrentIndex(idx)
        self.settings_dtype.setMinimumWidth(120)
        dtype_layout.addWidget(dtype_label)
        dtype_layout.addWidget(self.settings_dtype)
        layout.addWidget(dtype_group)

        # Язык по умолчанию
        lang_default_group = QGroupBox("Язык по умолчанию")
        lang_default_layout = QHBoxLayout(lang_default_group)

        lang_def_label = QLabel("Язык:")
        lang_def_label.setToolTip(
            "🌐 Язык, который будет выбран автоматически при запуске.\n"
            "• ru — русский язык\n"
            "• en — английский язык\n"
            "• auto — определять автоматически (по умолчанию)"
        )
        self.settings_default_lang = QComboBox()
        self.settings_default_lang.addItems(["auto", "ru", "en"])
        idx = {"auto": 0, "ru": 1, "en": 2}.get(self.settings.get("default_language", "ru"), 1)
        self.settings_default_lang.setCurrentIndex(idx)
        self.settings_default_lang.setMinimumWidth(80)
        lang_default_layout.addWidget(lang_def_label)
        lang_default_layout.addWidget(self.settings_default_lang)
        layout.addWidget(lang_default_group)

        # Путь к TTS модели
        tts_path_group = QGroupBox("Путь к модели Qwen3-TTS")
        tts_path_layout = QHBoxLayout(tts_path_group)

        self.tts_model_path_edit = QLineEdit()
        self.tts_model_path_edit.setText(self.settings["tts_model_path"])
        self.tts_model_path_edit.setPlaceholderText(
            "~/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base"
        )
        self.tts_model_path_edit.setToolTip(
            "📁 Путь к папке с моделью Qwen3-TTS.\n"
            "• Оставьте пустым для автоматической загрузки из HuggingFace\n"
            "• Или укажите путь к скачанной модели локально"
        )
        self.tts_model_path_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tts_path_layout.addWidget(self.tts_model_path_edit)

        btn_browse_tts = QPushButton("📂")
        btn_browse_tts.setFixedWidth(40)
        btn_browse_tts.clicked.connect(self._browse_tts_model)
        tts_path_layout.addWidget(btn_browse_tts)
        layout.addWidget(tts_path_group)

        # Размер Whisper модели
        whisper_size_group = QGroupBox("Модель Whisper")
        whisper_size_layout = QHBoxLayout(whisper_size_group)

        ws_label = QLabel("Размер:")
        ws_label.setToolTip(
            "📦 Размер модели Whisper (влияет на точность и скорость).\n"
            "• tiny — очень быстро, низкая точность\n"
            "• base — быстро, средняя точность\n"
            "• small — хорошо для большинства задач\n"
            "• medium — высокая точность (рекомендуется)\n"
            "• large — максимальная точность, но медленно и много памяти"
        )
        self.settings_whisper_size = QComboBox()
        self.settings_whisper_size.addItems(["tiny", "base", "small", "medium", "large"])
        idx = {"tiny": 0, "base": 1, "small": 2, "medium": 3, "large": 4}.get(
            self.settings["whisper_model_size"], 3)
        self.settings_whisper_size.setCurrentIndex(idx)
        self.settings_whisper_size.setMinimumWidth(80)
        whisper_size_layout.addWidget(ws_label)
        whisper_size_layout.addWidget(self.settings_whisper_size)
        layout.addWidget(whisper_size_group)

        # Кнопки загрузки моделей
        load_group = QGroupBox("Загрузка моделей")
        load_layout = QVBoxLayout(load_group)

        btn_load_tts = QPushButton("📥 Загрузить TTS модель")
        btn_load_tts.clicked.connect(self._load_tts_model_ui)
        load_layout.addWidget(btn_load_tts)

        btn_load_whisper = QPushButton("📥 Загрузить Whisper модель")
        btn_load_whisper.clicked.connect(self._load_whisper_model_ui)
        load_layout.addWidget(btn_load_whisper)

        layout.addWidget(load_group)

        # Кнопка сохранения настроек
        save_btn = QPushButton("💾 Сохранить настройки и перезапустить модели")
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #533483, stop:1 #0f3460);
                font-size: 14px;
                padding: 10px 24px;
            }
        """)
        save_btn.clicked.connect(self._save_settings)
        layout.addWidget(save_btn)

        layout.addStretch()
        tab.setLayout(layout)
        self.tabs.addTab(tab, "⚙️ Настройки")

    def _browse_tts_model(self):
        path = QFileDialog.getExistingDirectory(self, "Выбрать папку с моделью TTS")
        if path:
            self.tts_model_path_edit.setText(path)

    def _load_tts_model_ui(self) -> bool:
        model_path = self.tts_model_path_edit.text().strip() or None
        device = self.settings_device.currentText()
        dtype = self.settings_dtype.currentText()

        # Маппинг текста в значения
        dev_map = {"auto": "auto", "gpu (CUDA)": "gpu", "cpu": "cpu"}
        d_type_map = {
            "float16": "float16", "bfloat16": "bfloat16", "float32": "float32"
        }

        ok, msg = self.tts_engine.load_model(
            model_path=model_path,
            device=dev_map.get(device, "auto"),
            dtype=d_type_map.get(dtype, "float16"),
        )
        if ok:
            QMessageBox.information(self, "Успех", msg)
            self._set_status("✅ TTS модель загружена", "#4ade80")
        else:
            QMessageBox.critical(self, "Ошибка", msg)
            self._set_status("❌ Ошибка загрузки TTS", "#ef4444")
        return ok

    def _load_tts_model(self):
        """Загрузить модель TTS (вызывается из генерации)."""
        model_path = self.tts_model_path_edit.text().strip() or None
        device = self.settings_device.currentText()
        dtype = self.settings_dtype.currentText()

        dev_map = {"auto": "auto", "gpu (CUDA)": "gpu", "cpu": "cpu"}
        d_type_map = {
            "float16": "float16", "bfloat16": "bfloat16", "float32": "float32"
        }

        ok, msg = self.tts_engine.load_model(
            model_path=model_path,
            device=dev_map.get(device, "auto"),
            dtype=d_type_map.get(dtype, "float16"),
        )
        if ok:
            self._set_status("✅ TTS модель загружена", "#4ade80")
        else:
            self._set_status("❌ Ошибка загрузки TTS", "#ef4444")
        return ok, msg

    def _load_whisper_model_ui(self) -> bool:
        model_size = self.settings_whisper_size.currentText()
        device = self.settings_device.currentText()

        dev_map = {"auto": "auto", "gpu (CUDA)": "gpu", "cpu": "cpu"}

        ok, msg = self.whisper_engine.load_model(
            model_size=model_size,
            device=dev_map.get(device, "auto"),
        )
        if ok:
            QMessageBox.information(self, "Успех", msg)
            self._set_status("✅ Whisper загружен", "#4ade80")
        else:
            QMessageBox.critical(self, "Ошибка", msg)
            self._set_status("❌ Ошибка загрузки Whisper", "#ef4444")
        return ok

    def _load_whisper_model(self):
        """Загрузить модель Whisper (вызывается из транскрипции)."""
        model_size = self.settings_whisper_size.currentText()
        device = self.settings_device.currentText()

        dev_map = {"auto": "auto", "gpu (CUDA)": "gpu", "cpu": "cpu"}

        ok, msg = self.whisper_engine.load_model(
            model_size=model_size,
            device=dev_map.get(device, "auto"),
        )
        if ok:
            self._set_status("✅ Whisper загружен", "#4ade80")
        else:
            self._set_status("❌ Ошибка загрузки Whisper", "#ef4444")
        return ok, msg

    def _save_settings(self):
        """Сохранить настройки."""
        device = self.settings_device.currentText()
        dtype = self.settings_dtype.currentText()

        dev_map = {"auto": "auto", "gpu (CUDA)": "gpu", "cpu": "cpu"}
        d_type_map = {
            "float16": "float16", "bfloat16": "bfloat16", "float32": "float32"
        }

        self.settings["device"] = dev_map.get(device, "auto")
        self.settings["dtype"] = d_type_map.get(dtype, "float16")
        self.settings["tts_model_path"] = self.tts_model_path_edit.text().strip()
        self.settings["whisper_model_size"] = self.settings_whisper_size.currentText()
        self.settings["default_language"] = self.settings_default_lang.currentText()

        # Перезагрузка моделей с новыми настройками
        if self.tts_engine.model is not None:
            self.tts_engine.unload_model()
        if self.whisper_engine.model is not None:
            self.whisper_engine.unload_model()

        QMessageBox.information(self, "Настройки", "Сохранено! Перезапустите модели через вкладку Настройки.")
        self._log("Настройки сохранены")

    # ======================== Утилиты ========================

    def _apply_theme(self):
        """Применить тёмную тему."""
        self.setStyleSheet(DARK_THEME)

    def _set_status(self, text: str, color: str = "#e0e0e0"):
        self.status_label.set_status(text, color)

    def _log(self, msg: str):
        """Добавить запись в лог (GUI + файл)."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_line = f"[{timestamp}] {msg}"
        try:
            self.log_text.appendPlainText(log_line)
            scrollbar = self.log_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
        except Exception:
            pass
        # Запись в файл
        try:
            log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app.log")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{log_line}\n")
        except Exception:
            pass

    def dropEvent(self, event):
        """Обработка перетаскивания файлов."""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            for url in urls:
                file_path = url.toLocalFile()
                if os.path.isfile(file_path):
                    self._handle_dropped_file(file_path)
            event.acceptProposedAction()

    def _handle_dropped_file(self, file_path: str):
        """Обработать перетащенный файл."""
        ext = Path(file_path).suffix.lower()
        
        # Определяем, на какое поле бросили файл
        widget = self.focusWidget()
        
        if widget == self.voice_vector_path:
            # Бросили на поле .pt файла
            if ext == '.pt':
                self.voice_vector_path.setText(file_path)
                self._log(f"Перетащен .pt файл: {Path(file_path).name}")
            else:
                QMessageBox.warning(self, "Неподдерживаемый формат", 
                    f"На поле вектора голоса можно перетаскивать только .pt файлы.\nПолучен: {ext}")
        elif widget == self.ref_audio_path:
            # Бросили на поле референсного аудио
            audio_exts = ['.wav', '.mp3', '.ogg', '.flac']
            video_exts = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm']
            
            if ext in audio_exts:
                # Очищаем старые временные файлы
                self._cleanup_temp_files()
                self.ref_audio_path.setText(file_path)
                self._log(f"Перетащен аудиофайл: {Path(file_path).name}")
            elif ext in video_exts:
                # Видео — нужно извлечь аудио через FFmpeg
                if not self.video_extractor.is_ffmpeg_available():
                    QMessageBox.warning(
                        self, "FFmpeg не найден",
                        "Для видеофайлов необходим FFmpeg.\nУстановите: winget install Gyan.FFmpeg"
                    )
                    return
                # Запускаем извлечение в фоне
                self._set_status("⏳ Извлечение аудио из видео...", "#fbbf24")
                self.btn_video_extract.setEnabled(False)
                self._extract_thread = ExtractAudioThread(self.video_extractor, file_path)
                self._extract_thread.progress.connect(self._on_extract_progress)
                self._extract_thread.finished_success.connect(self._on_extract_success)
                self._extract_thread.finished_error.connect(self._on_extract_error)
                self._extract_thread.start()
            else:
                QMessageBox.warning(self, "Неподдерживаемый формат",
                    f"На поле аудио можно перетаскивать:\n"
                    f"  Аудио: {', '.join(audio_exts)}\n"
                    f"  Видео: {', '.join(video_exts)}\nПолучен: {ext}")
        else:
            # Неизвестное поле — определяем по расширению
            audio_exts = ['.wav', '.mp3', '.ogg', '.flac']
            video_exts = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm']
            
            if ext in audio_exts:
                self._cleanup_temp_files()
                self.ref_audio_path.setText(file_path)
                self._log(f"Перетащен аудиофайл: {Path(file_path).name}")
            elif ext in video_exts:
                if not self.video_extractor.is_ffmpeg_available():
                    QMessageBox.warning(
                        self, "FFmpeg не найден",
                        "Для видеофайлов необходим FFmpeg.\nУстановите: winget install Gyan.FFmpeg"
                    )
                    return
                self._set_status("⏳ Извлечение аудио из видео...", "#fbbf24")
                self.btn_video_extract.setEnabled(False)
                self._extract_thread = ExtractAudioThread(self.video_extractor, file_path)
                self._extract_thread.progress.connect(self._on_extract_progress)
                self._extract_thread.finished_success.connect(self._on_extract_success)
                self._extract_thread.finished_error.connect(self._on_extract_error)
                self._extract_thread.start()
            elif ext == '.pt':
                self.voice_vector_path.setText(file_path)
                self._log(f"Перетащен .pt файл: {Path(file_path).name}")

    def _cleanup_temp_files(self):
        """Очистить временные файлы."""
        for tmp_name in ['_temp_recording_path', '_temp_extracted_audio_path']:
            tmp_path = getattr(self, tmp_name, None)
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
                setattr(self, tmp_name, None)

    def closeEvent(self, event):
        """Очистка при закрытии."""
        # Останавливаем потоки
        if self.tts_thread and self.tts_thread.isRunning():
            self.tts_thread.wait(3000)
        if self.transcribe_thread and self.transcribe_thread.isRunning():
            self.transcribe_thread.wait(3000)

        # Останавливаем плеер
        if self.audio_player:
            self.audio_player.stop()

        # Выгружаем модели
        self.tts_engine.unload_model()
        self.whisper_engine.unload_model()

        # Очищаем GPU кэш
        self.gpu_manager.clear_cache()

        # Удаляем временные файлы записей и извлечённого аудио
        for tmp_name in ['_temp_recording_path', '_temp_extracted_audio_path']:
            tmp_path = getattr(self, tmp_name, None)
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                    self._log(f"Удалён временный файл: {tmp_path}")
                except Exception:
                    pass
                setattr(self, tmp_name, None)

        # Удаляем временный файл воспроизведения
        self._cleanup_temp_audio()

        event.accept()


# ======================== Глобальный перехват ошибок ========================

def _global_exception_hook(exc_type, exc_value, exc_traceback):
    """Глобальный обработчик необработанных исключений."""
    if issubclass(exc_type, KeyboardInterrupt) or issubclass(exc_type, SystemExit):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    tb = traceback.format_exception(exc_type, exc_value, exc_traceback)
    error_msg = "".join(tb)

    # Пишем в файл лога
    log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app.log")
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n=== UNCAUGHT EXCEPTION ===\n{error_msg}\n")
    except Exception:
        pass

    print(f"[FATAL] Unhandled exception:\n{error_msg}", file=sys.stderr)

sys.excepthook = _global_exception_hook

# ======================== Точка входа ========================

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Базовый стиль для кастомизации QSS

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
