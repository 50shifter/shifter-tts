"""GPU Manager — мониторинг и очистка VRAM через nvidia-smi + PyTorch."""

import subprocess
import threading
from typing import Dict, Optional

import torch


class GPUManager:
    """Управление GPU: мониторинг, очистка кэша, логирование."""

    def __init__(self):
        self._lock = threading.Lock()
        self._auto_clean = False
        self._log_callback = None  # callable(msg: str) -> None

    @property
    def auto_clean(self) -> bool:
        return self._auto_clean

    @auto_clean.setter
    def auto_clean(self, value: bool):
        self._auto_clean = value

    def set_log_callback(self, callback):
        """Установить коллбэк для логирования: callback(msg: str)"""
        self._log_callback = callback

    def _log(self, msg: str):
        if self._log_callback:
            self._log_callback(msg)

    def is_available(self) -> bool:
        return torch.cuda.is_available()

    def get_gpu_stats(self) -> Optional[Dict]:
        """Получить статистику GPU через nvidia-smi."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total,temperature.gpu,utilization.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                return None
            parts = [x.strip() for x in result.stdout.split(",")]
            if len(parts) < 5:
                return None
            return {
                "used": int(parts[0]),
                "total": int(parts[1]),
                "temp": int(parts[2]),
                "util": int(parts[3]),
                "power": float(parts[4]),
            }
        except Exception:
            return None

    def get_torch_vram(self) -> Dict[str, int]:
        """Получить статистику VRAM из PyTorch."""
        if not torch.cuda.is_available():
            return {"allocated": 0, "reserved": 0}
        allocated = torch.cuda.memory_allocated() // (1024 * 1024)
        reserved = torch.cuda.memory_reserved() // (1024 * 1024)
        return {"allocated": allocated, "reserved": reserved}

    def clear_cache(self) -> str:
        """Очистить CUDA кэш через PyTorch API."""
        with self._lock:
            if torch.cuda.is_available():
                before = self.get_torch_vram()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                after = self.get_torch_vram()
                freed = before["reserved"] - after["reserved"]
                msg = f"Кэш очищен. Освобождено: {max(freed, 0)} MiB | VRAM: {after['allocated']}/{after['reserved']} MiB"
                self._log(msg)
                return msg
            return "CUDA не доступен"

    def force_release_vram(self) -> str:
        """Принудительно освободить всю VRAM."""
        with self._lock:
            if torch.cuda.is_available():
                before = self.get_torch_vram()
                # Сброс всех тензоров
                del_tensor_refs = []  # на всякий случай
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.synchronize()
                after = self.get_torch_vram()
                stats = self.get_gpu_stats()
                freed = before["reserved"] - after["reserved"]
                msg = (
                    f"VRAM освобождён. Освобождено: {max(freed, 0)} MiB | "
                    f"Статус: {after['allocated']}/{after['reserved']} MiB | "
                    f"Температура: {stats['temp']}°C" if stats else ""
                )
                self._log(msg)
                return msg
            return "GPU не обнаружен"

    def auto_clean_if_needed(self):
        """Очистить кэш если включён авто-режим."""
        if self._auto_clean:
            return self.clear_cache()
        return None
