import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

@dataclass(frozen=True)
class Ticket:
    """Strict schema for an inbox ticket."""
    ticket_id: str
    job_type: str
    parameters: Dict[str, Any]
    priority: int = 5
    description: Optional[str] = None

    def __post_init__(self):
        # Validate ticket_id format
        if not re.match(r"^[a-z0-9_]{8,32}$", self.ticket_id):
            raise ValueError(f"Invalid ticket_id format: {self.ticket_id}")
        
        # Validate priority range
        if not (1 <= self.priority <= 10):
            raise ValueError(f"Priority must be between 1 and 10, got {self.priority}")

@dataclass(frozen=True)
class DaemonStats:
    """Operational statistics for the Daemon."""
    is_running: bool
    tickets_ingested: int
    uptime_sec: int
    last_heartbeat: str  # ISO 8601 UTC
    queue_stats: Dict[str, int] = field(default_factory=dict)
    persona_state: Optional[Dict[str, Any]] = None  # Placeholder for Phase 3

@dataclass(frozen=True)
class ApprovalItem:
    """Represents a single job awaiting HITL approval."""
    job_id: str
    job_type: str
    current_step: str
    description: str
    created_at: str

@dataclass(frozen=True)
class ApprovalsResult:
    """Result of querying pending approvals."""
    status: str
    approvals: Tuple[ApprovalItem, ...]
    count: int
    error: Optional[str] = None
    error_code: Optional[str] = None

@dataclass(frozen=True)
class ApprovalResult:
    """Result of an approve/reject action."""
    status: str
    job_id: Optional[str] = None
    error: Optional[str] = None
    error_code: Optional[str] = None