from enum import Enum
from typing import Optional

class ErrorCode(str, Enum):
    """Machine-readable error codes for Daemon operations."""
    DAEMON_ALREADY_RUNNING = "DAEMON_ALREADY_RUNNING"
    INVALID_TICKET = "INVALID_TICKET"
    TICKET_TOO_LARGE = "TICKET_TOO_LARGE"
    DUPLICATE_TICKET = "DUPLICATE_TICKET"
    INGEST_FAILED = "INGEST_FAILED"
    JOB_NOT_FOUND = "JOB_NOT_FOUND"
    JOB_NOT_PENDING = "JOB_NOT_PENDING"
    IO_ERROR = "IO_ERROR"
    LOCK_ACQUISITION_FAILED = "LOCK_ACQUISITION_FAILED"
    INTERNAL = "INTERNAL"

class DaemonError(Exception):
    """
    Custom exception for all Daemon module errors. 
    Hides underlying library exceptions and chains from the original cause.
    """
    def __init__(
        self, 
        message: str, 
        error_code: ErrorCode = ErrorCode.INTERNAL,
        original_exception: Optional[Exception] = None
    ):
        super().__init__(message)
        self.error_code = error_code
        self.__cause__ = original_exception  # Explicit exception chaining