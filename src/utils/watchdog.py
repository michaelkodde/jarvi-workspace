"""
Resource Watchdog: Monitors system health and triggers graceful shutdowns on critical thresholds.
"""
import logging
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Dict, Any

try:
    import psutil
except ImportError as e:
    raise ImportError("Resource Watchdog requires 'psutil'. Install via: pip install psutil") from e

# Optional VRAM monitoring via pynvml (NVIDIA only). Graceful degradation if unavailable.
try:
    import pynvml
    _PYNVML_AVAILABLE = True
except ImportError:
    _PYNVML_AVAILABLE = False

logger = logging.getLogger(__name__)

__all__ = ["ResourceWatchdog", "WatchdogError", "SystemMetrics"]


class WatchdogError(Exception):
    """Custom exception for Watchdog initialization or critical failures."""
    pass


@dataclass
class SystemMetrics:
    """Strongly typed snapshot of current system resource usage."""
    ram_percent: float
    cpu_percent: float
    vram_percent: Optional[float]


class ResourceWatchdog:
    """
    Background daemon thread that monitors system RAM, CPU, and VRAM.
    Triggers a graceful shutdown callback if critical thresholds are breached.

    Args:
        config_path: Path to config.json containing threshold definitions.
        shutdown_callback: Callable to execute when a critical threshold is breached.
        check_interval: Seconds between health checks. Default is 5.0.

    Raises:
        WatchdogError: If initial config is invalid.
    """

    DEFAULT_THRESHOLDS = {
        "ram_warning": 80.0,
        "ram_critical": 90.0,
        "cpu_critical": 95.0,
        "vram_critical": 90.0,
    }

    def __init__(
        self,
        config_path: str,
        shutdown_callback: Callable[[], None],
        check_interval: float = 5.0,
    ) -> None:
        self._config_path = Path(config_path)
        self._shutdown_callback = shutdown_callback
        self._check_interval = check_interval
        
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_known_config: Dict[str, Any] = self.DEFAULT_THRESHOLDS.copy()
        self._nvml_initialized = False
        self._config_warning_logged = False # Prevents log spam if config is missing

        self._last_known_config = self._load_config(fail_fast=True)
        self._init_nvml()
        
        # Prime the CPU metric pump so the first check is accurate
        psutil.cpu_percent(interval=None)

    def _init_nvml(self) -> None:
        """Safely initializes NVIDIA Management Library for VRAM monitoring."""
        if not _PYNVML_AVAILABLE:
            logger.info("pynvml not installed. VRAM monitoring disabled.")
            return
        
        try:
            pynvml.nvmlInit()
            self._nvml_initialized = True
            logger.info("NVML initialized successfully for VRAM monitoring.")
        except pynvml.NVMLError as e:
            logger.warning(f"NVML initialization failed: {e}. VRAM monitoring disabled.")
            self._nvml_initialized = False

    def _load_config(self, fail_fast: bool = False) -> Dict[str, Any]:
        """
        Loads threshold configuration from JSON.
        
        Args:
            fail_fast: If True, raises WatchdogError on missing file (used at startup).
            
        Returns:
            Dictionary of threshold values.
        """
        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                watchdog_config = config.get("watchdog", {})
                
                merged = self.DEFAULT_THRESHOLDS.copy()
                merged.update(watchdog_config)
                
                # Reset the warning flag if the file is successfully read again
                self._config_warning_logged = False 
                return merged
        except FileNotFoundError:
            if fail_fast:
                raise WatchdogError(f"Config file not found at startup: {self._config_path}")
            if not self._config_warning_logged:
                logger.warning("Config file missing during runtime check. Using last known config.")
                self._config_warning_logged = True
            return self._last_known_config
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Failed to parse config.json. Using last known config.")
            return self._last_known_config

    def _get_metrics(self) -> SystemMetrics:
        """Gathers current system metrics."""
        ram = psutil.virtual_memory().percent
        cpu = psutil.cpu_percent(interval=None)
        vram: Optional[float] = None

        if self._nvml_initialized:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                # Guard against division by zero in case of driver corruption
                if info.total > 0:
                    vram = (info.used / info.total) * 100.0
                else:
                    vram = 0.0
            except pynvml.NVMLError as e:
                logger.debug(f"NVML error reading VRAM: {e}.")
                vram = None

        return SystemMetrics(ram_percent=ram, cpu_percent=cpu, vram_percent=vram)

    def _execute_shutdown(self) -> None:
        """Safely executes the shutdown callback in isolation."""
        try:
            self._shutdown_callback()
        except Exception as e:
            logger.critical(f"Shutdown callback failed: {e}", exc_info=True)

    def start(self) -> None:
        """Starts the watchdog background daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Watchdog is already running.")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="JarviWatchdog")
        self._thread.start()
        logger.info("Resource Watchdog started.")

    def _run_loop(self) -> None:
        """Internal daemon loop. Monitors metrics and triggers shutdown on critical breach."""
        while not self._stop_event.is_set():
            if self._stop_event.wait(self._check_interval):
                break

            current_thresholds = self._load_config(fail_fast=False)
            self._last_known_config = current_thresholds

            try:
                metrics = self._get_metrics()
            except Exception as e:
                logger.error(f"Failed to gather system metrics: {e}")
                continue

            is_critical = False
            if metrics.ram_percent >= current_thresholds["ram_critical"]:
                logger.critical(f"CRITICAL: RAM usage at {metrics.ram_percent}%")
                is_critical = True
            if metrics.cpu_percent >= current_thresholds["cpu_critical"]:
                logger.critical(f"CRITICAL: CPU usage at {metrics.cpu_percent}%")
                is_critical = True
            if metrics.vram_percent is not None and metrics.vram_percent >= current_thresholds["vram_critical"]:
                logger.critical(f"CRITICAL: VRAM usage at {metrics.vram_percent:.1f}%")
                is_critical = True

            if not is_critical and metrics.ram_percent >= current_thresholds["ram_warning"]:
                logger.warning(f"WARNING: RAM usage at {metrics.ram_percent}%")

            if is_critical:
                logger.critical("Initiating graceful shutdown sequence due to resource exhaustion.")
                # Fire-and-forget to prevent blocking the watchdog thread
                threading.Thread(target=self._execute_shutdown, daemon=True).start()
                break

        logger.info("Resource Watchdog thread terminated.")

    def stop(self) -> None:
        """Signals the watchdog thread to stop gracefully and cleans up NVML."""
        logger.info("Stopping Resource Watchdog...")
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
            
        if self._nvml_initialized:
            try:
                pynvml.nvmlShutdown()
                self._nvml_initialized = False
            except pynvml.NVMLError:
                pass
                
        logger.info("Resource Watchdog stopped.")