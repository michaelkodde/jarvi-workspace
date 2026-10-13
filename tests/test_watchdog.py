"""
Unit tests for the Resource Watchdog.
"""
import pytest
import threading
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.utils.watchdog import ResourceWatchdog, WatchdogError


@pytest.fixture
def mock_env(tmp_path: Path):
    """Sets up a temporary config and mock callback for testing."""
    config_file = tmp_path / "config.json"
    config_data = {
        "watchdog": {
            "ram_warning": 80.0,
            "ram_critical": 90.0,
            "cpu_critical": 95.0,
            "vram_critical": 90.0,
        }
    }
    config_file.write_text(json.dumps(config_data), encoding="utf-8")
    
    # Use an Event to synchronize the test instead of time.sleep()
    callback_event = threading.Event()
    callback = MagicMock(side_effect=lambda: callback_event.set())
    
    return config_file, callback, callback_event, tmp_path


# Force NVML off during tests to prevent real GPU driver calls
@patch("src.utils.watchdog._PYNVML_AVAILABLE", False)
@patch("src.utils.watchdog.psutil")
def test_watchdog_triggers_shutdown_on_critical_ram(mock_psutil, mock_env):
    """Verify that breaching RAM critical threshold triggers the shutdown callback."""
    config_file, callback, callback_event, tmp_path = mock_env
    
    mock_memory = MagicMock(percent=95.0)
    mock_psutil.virtual_memory.return_value = mock_memory
    mock_psutil.cpu_percent.return_value = 10.0

    watchdog = ResourceWatchdog(
        config_path=str(config_file),
        shutdown_callback=callback,
        check_interval=0.1
    )
    
    watchdog.start()
    
    # Wait deterministically for the callback to fire (max 2 seconds)
    assert callback_event.wait(timeout=2.0), "Shutdown callback was not triggered in time."
    
    watchdog.stop()
    callback.assert_called_once()


@patch("src.utils.watchdog._PYNVML_AVAILABLE", False)
@patch("src.utils.watchdog.psutil")
def test_watchdog_does_not_trigger_on_warning_only(mock_psutil, mock_env):
    """Verify that warning thresholds log but do not trigger shutdown."""
    config_file, callback, callback_event, tmp_path = mock_env
    
    mock_memory = MagicMock(percent=85.0)
    mock_psutil.virtual_memory.return_value = mock_memory
    mock_psutil.cpu_percent.return_value = 10.0

    watchdog = ResourceWatchdog(
        config_path=str(config_file),
        shutdown_callback=callback,
        check_interval=0.1
    )
    
    watchdog.start()
    
    # Wait long enough for a check to occur, but not long enough to trigger a timeout
    assert not callback_event.wait(timeout=0.5)
    
    watchdog.stop()
    callback.assert_not_called()


@patch("src.utils.watchdog._PYNVML_AVAILABLE", False)
@patch("src.utils.watchdog.psutil")
def test_watchdog_handles_missing_config_gracefully(mock_psutil, mock_env):
    """Verify that if config is deleted mid-run, it falls back safely without crashing."""
    config_file, callback, callback_event, tmp_path = mock_env
    
    mock_memory = MagicMock(percent=50.0)
    mock_psutil.virtual_memory.return_value = mock_memory
    mock_psutil.cpu_percent.return_value = 10.0

    watchdog = ResourceWatchdog(
        config_path=str(config_file),
        shutdown_callback=callback,
        check_interval=0.1
    )
    
    watchdog.start()
    config_file.unlink()
    
    # Ensure it keeps running and doesn't trigger shutdown
    assert not callback_event.wait(timeout=0.5)
    assert watchdog._thread is not None
    assert watchdog._thread.is_alive()
    
    watchdog.stop()


@patch("src.utils.watchdog._PYNVML_AVAILABLE", False)
def test_watchdog_fails_fast_on_missing_config_at_startup(mock_env):
    """Verify that missing config at __init__ raises WatchdogError."""
    _, callback, _, tmp_path = mock_env
    missing_config = tmp_path / "nonexistent.json"
    
    with pytest.raises(WatchdogError, match="Config file not found at startup"):
        ResourceWatchdog(
            config_path=str(missing_config),
            shutdown_callback=callback
        )


@patch("src.utils.watchdog._PYNVML_AVAILABLE", False)
@patch("src.utils.watchdog.psutil")
def test_watchdog_stops_cleanly(mock_psutil, mock_env):
    """Verify that stop() cleanly terminates the background thread."""
    config_file, callback, _, tmp_path = mock_env
    
    mock_memory = MagicMock(percent=50.0)
    mock_psutil.virtual_memory.return_value = mock_memory
    mock_psutil.cpu_percent.return_value = 10.0

    watchdog = ResourceWatchdog(
        config_path=str(config_file),
        shutdown_callback=callback,
        check_interval=1.0
    )
    
    watchdog.start()
    assert watchdog._thread is not None
    assert watchdog._thread.is_alive()
    
    watchdog.stop()
    
    assert watchdog._thread is None or not watchdog._thread.is_alive()