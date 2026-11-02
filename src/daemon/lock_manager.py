# src/daemon/lock_manager.py
"""
Gold Standard Lock Manager.
Uses OS-level file descriptors for atomic, crash-proof locking.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

from .exceptions import DaemonError, ErrorCode

logger = logging.getLogger(__name__)

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

class LockManager:
    """Manages an OS-level file lock to ensure only one Daemon instance runs."""
    
    def __init__(self, lock_path: Path):
        self.lock_path = Path(lock_path)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._fd: Optional[int] = None

    def acquire(self) -> None:
        """Acquires the OS-level lock. Raises DaemonError if already held."""
        try:
            self._fd = os.open(str(self.lock_path), os.O_CREAT | os.O_RDWR)
            
            if sys.platform == "win32":
                try:
                    msvcrt.locking(self._fd, msvcrt.LK_NBLCK, 1)
                except OSError:
                    self._cleanup_fd()
                    self._raise_already_running()
            else:
                try:
                    fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    self._cleanup_fd()
                    self._raise_already_running()

            # Write metadata atomically
            metadata = {"pid": os.getpid(), "timestamp": time.time(), "platform": sys.platform}
            data = json.dumps(metadata).encode('utf-8')
            
            os.lseek(self._fd, 0, os.SEEK_SET)
            os.write(self._fd, data)
            
            # CRITICAL FIX: Truncate to the length of the new data, NOT 0.
            # Truncating to 0 on Windows releases the lock on byte 0.
            os.ftruncate(self._fd, len(data))
            os.fsync(self._fd)  # Ensure written to disk
            
            logger.info("Lock acquired (PID: %d) via OS file descriptor.", os.getpid())

        except OSError as e:
            self._cleanup_fd()
            raise DaemonError(f"Failed to acquire OS lock: {e}", ErrorCode.IO_ERROR)

    def release(self) -> None:
        """Releases the OS-level lock and cleans up the file."""
        if self._fd is not None:
            try:
                if sys.platform == "win32":
                    msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(self._fd, fcntl.LOCK_UN)
                
                self._cleanup_fd()
                
                if self.lock_path.exists():
                    try:
                        self.lock_path.unlink()
                    except OSError as e:
                        logger.debug(f"Could not delete lock file (handle may still be held by OS): {e}")
                        
                logger.info("Lock released.")
            except OSError as e:
                logger.error(f"Error releasing lock: {e}")

    def is_locked(self) -> bool:
        """Checks if the lock file exists and is held by a living process."""
        # If this instance holds the file descriptor, we are locked.
        if self._fd is not None:
            return True
            
        if not self.lock_path.exists():
            return False
        
        try:
            with open(self.lock_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            pid = data.get("pid")
            if pid and pid != os.getpid():
                os.kill(pid, 0)  # Raises ProcessLookupError if dead
                return True
            # If it's the current process, it's locked by us
            if pid == os.getpid():
                return True
        except PermissionError:
            # On Windows, if the file is locked by another process, open() raises PermissionError.
            return True
        except (json.JSONDecodeError, ProcessLookupError, OSError):
            return False
        return False

    def _cleanup_fd(self):
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

    def _raise_already_running(self):
        raise DaemonError(
            "Another instance of the Daemon is already running (OS Lock held).",
            ErrorCode.DAEMON_ALREADY_RUNNING
        )