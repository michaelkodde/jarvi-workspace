import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

from .exceptions import DaemonError, ErrorCode

logger = logging.getLogger(__name__)

# Lock file becomes "stale" if the process is dead OR if it's older than this (seconds)
MAX_STALE_AGE_SEC = 3600  # 1 hour

class LockManager:
    """Manages exclusive access to the Daemon via a PID-based lock file."""
    
    def __init__(self, lock_file_path: Path):
        self.lock_file_path = lock_file_path

    def acquire(self) -> None:
        """
        Attempts to acquire the lock. 
        Raises DaemonError if already held by a living process.
        Cleans up stale locks automatically.
        """
        if self._is_locked():
            lock_data = self._read_lock()
            if self._is_stale(lock_data):
                logger.warning("Found stale lock file. Cleaning up.")
                self._release()
            else:
                raise DaemonError(
                    f"Daemon is already running (PID: {lock_data.get('pid')})",
                    error_code=ErrorCode.DAEMON_ALREADY_RUNNING
                )
        
        # Write new lock
        lock_data = {
            "pid": os.getpid(),
            "timestamp": time.time()
        }
        try:
            # Atomic write: write to temp, then replace
            temp_path = self.lock_file_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(lock_data, f)
            temp_path.replace(self.lock_file_path)
            logger.info(f"Lock acquired (PID: {os.getpid()})")
        except Exception as e:
            raise DaemonError(
                f"Failed to acquire lock file: {e}",
                error_code=ErrorCode.LOCK_ACQUISITION_FAILED,
                original_exception=e
            )

    def release(self) -> None:
        """Releases the lock file."""
        self._release()

    def _is_locked(self) -> bool:
        return self.lock_file_path.exists()

    def _read_lock(self) -> dict:
        try:
            with open(self.lock_file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def _is_stale(self, lock_data: dict) -> bool:
        pid = lock_data.get("pid")
        timestamp = lock_data.get("timestamp", 0)
        
        # Check 1: Age
        if time.time() - timestamp > MAX_STALE_AGE_SEC:
            return True
            
        # Check 2: Process liveness
        if pid:
            return not self._is_pid_alive(pid)
            
        return True

    def _is_pid_alive(self, pid: int) -> bool:
        """Checks if a process with the given PID is currently running."""
        try:
            # os.kill with signal 0 checks existence without sending a signal
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False
        except Exception:
            # Fallback for edge cases
            return False

    def _release(self) -> None:
        if self._is_locked():
            try:
                self.lock_file_path.unlink()
                logger.info("Lock released.")
            except Exception as e:
                logger.warning(f"Failed to remove lock file: {e}")

    def is_locked(self) -> bool:
        """Public method to check if the lock is currently held."""
        return self._is_locked()