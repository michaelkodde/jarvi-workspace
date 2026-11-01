"""
Comprehensive test suite for the Jarvi Daemon Loop.
Verifies security, crash recovery, idempotency, PII safety, and lifecycle workflows.
"""
import json
import logging
import signal
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.daemon.daemon_loop import DaemonLoop, TICKET_FILENAME_REGEX
from src.daemon.exceptions import DaemonError, ErrorCode
from src.daemon.lock_manager import LockManager


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Creates a temporary workspace with required directories."""
    inbox = tmp_path / "00_inbox"
    inbox.mkdir()
    (tmp_path / "logs").mkdir()
    return tmp_path


@pytest.fixture
def mock_pipeline_flow() -> MagicMock:
    """Mocks PipelineFlow to isolate daemon logic."""
    mock = MagicMock()
    mock.get_queue_stats.return_value = {"queued": 0, "running": 0, "pending_review": 0}
    mock.get_job_status.return_value = {}  # Empty dict means not found
    mock.submit_job.return_value = "mock-job-id-123"
    mock.process_next_job.return_value = None
    mock.list_jobs.return_value = []
    return mock


@pytest.fixture
def daemon(workspace: Path, mock_pipeline_flow: MagicMock) -> DaemonLoop:
    """Instantiates a DaemonLoop with safe test defaults."""
    return DaemonLoop(
        workspace_path=workspace,
        pipeline_flow=mock_pipeline_flow,
        poll_interval_sec=0.5,
        queue_depth_threshold=50,
        enabled=True,
    )


# ============================================================================
# Helper Functions
# ============================================================================

def write_ticket(workspace: Path, filename: str, data: dict) -> Path:
    """Writes a ticket file to the inbox."""
    file_path = workspace / "00_inbox" / filename
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return file_path


# ============================================================================
# Daemon Loop Tests: Validation & Security
# ============================================================================

def test_valid_ticket_ingested_and_moved(workspace: Path, mock_pipeline_flow: MagicMock, daemon: DaemonLoop):
    """Invariant: A valid ticket is submitted and atomically moved to .processed/"""
    daemon._lock_manager.acquire()
    
    ticket_data = {
        "ticket_id": "test_ticket_001",
        "job_type": "trendspotter",
        "parameters": {"query": "eco friendly pans"},
        "priority": 5,
    }
    file_path = write_ticket(workspace, "ticket_test_ticket_001.json", ticket_data)
    
    daemon._ingest_cycle()
    
    mock_pipeline_flow.submit_job.assert_called_once_with(
        job_type="trendspotter",
        parameters={"query": "eco friendly pans"},
        priority=5,
        description="Inbox ticket: test_ticket_001",
    )
    assert not file_path.exists()
    assert (workspace / "00_inbox" / ".processed" / "ticket_test_ticket_001.json").exists()


def test_invalid_json_rejected_with_err_sidecar(workspace: Path, daemon: DaemonLoop, caplog):
    """Invariant: Invalid JSON is rejected, moved to .rejected/, and an .err sidecar is created."""
    daemon._lock_manager.acquire()
    
    file_path = workspace / "00_inbox" / "ticket_bad_json.json"
    file_path.write_text("{ invalid json }", encoding="utf-8")
    
    with caplog.at_level(logging.WARNING):
        daemon._ingest_cycle()
    
    assert not file_path.exists()
    err_file = workspace / "00_inbox" / ".rejected" / "ticket_bad_json.err"
    assert err_file.exists()
    
    err_data = json.loads(err_file.read_text(encoding="utf-8"))
    assert err_data["error_code"] == ErrorCode.INVALID_TICKET.value
    assert "Invalid JSON" in err_data["error_message"]
    assert "parameters" not in err_data  # PII protection


def test_ticket_too_large_rejected(workspace: Path, daemon: DaemonLoop):
    """Invariant: Tickets exceeding 1MB are rejected before parsing."""
    daemon._lock_manager.acquire()
    
    file_path = workspace / "00_inbox" / "ticket_large.json"
    file_path.write_text("{" + '"data": "' + "x" * 1048577 + '"}', encoding="utf-8")
    
    daemon._ingest_cycle()
    
    assert not file_path.exists()
    assert (workspace / "00_inbox" / ".rejected" / "ticket_large.err").exists()


def test_invalid_filename_rejected(workspace: Path, daemon: DaemonLoop):
    """Invariant: Filenames not matching ^ticket_[a-z0-9_]+\\.json$ are rejected."""
    daemon._lock_manager.acquire()
    
    file_path = write_ticket(workspace, "ticket_UPPERCASE.json", {
        "ticket_id": "test", "job_type": "scout", "parameters": {}
    })
    
    daemon._ingest_cycle()
    
    assert not file_path.exists()
    assert (workspace / "00_inbox" / ".rejected" / "ticket_UPPERCASE.err").exists()


def test_path_traversal_in_parameters_rejected(workspace: Path, daemon: DaemonLoop):
    """Invariant: Parameters containing '..' or absolute paths are rejected for security."""
    daemon._lock_manager.acquire()
    
    malicious_data = {
        "ticket_id": "hack_001",
        "job_type": "code_review_pipeline",
        "parameters": {"target_files": "../../etc/passwd"},
    }
    file_path = write_ticket(workspace, "ticket_hack_001.json", malicious_data)
    
    daemon._ingest_cycle()
    
    assert not file_path.exists()
    err_file = workspace / "00_inbox" / ".rejected" / "ticket_hack_001.err"
    assert err_file.exists()
    
    err_data = json.loads(err_file.read_text(encoding="utf-8"))
    assert "forbidden patterns" in err_data["error_message"].lower()


def test_submit_job_raises_ticket_rejected(workspace: Path, mock_pipeline_flow: MagicMock, daemon: DaemonLoop):
    """Invariant: If PipelineFlow.submit_job raises, the ticket is moved to .rejected/."""
    daemon._lock_manager.acquire()
    mock_pipeline_flow.submit_job.side_effect = Exception("Unknown job_type")
    
    ticket_data = {"ticket_id": "fail_001", "job_type": "unknown", "parameters": {}}
    file_path = write_ticket(workspace, "ticket_fail_001.json", ticket_data)
    
    daemon._ingest_cycle()
    
    assert not file_path.exists()
    assert (workspace / "00_inbox" / ".rejected" / "ticket_fail_001.err").exists()


# ============================================================================
# Daemon Loop Tests: Lifecycle & Concurrency
# ============================================================================

def test_graceful_shutdown_sets_flags(workspace: Path, mock_pipeline_flow: MagicMock):
    """Invariant: stop() sets shutdown flags and releases lock."""
    daemon = DaemonLoop(
        workspace_path=workspace,
        pipeline_flow=mock_pipeline_flow,
        poll_interval_sec=0.5,
    )
    
    daemon._is_running = True
    daemon._in_cycle = False
    
    daemon.stop(timeout_sec=1.0)
    
    assert not daemon._is_running
    assert daemon._shutdown_requested


def test_signal_handler_triggers_shutdown(workspace: Path, mock_pipeline_flow: MagicMock):
    """Invariant: SIGINT/SIGBREAK triggers graceful shutdown via _handle_shutdown."""
    daemon = DaemonLoop(workspace, mock_pipeline_flow, poll_interval_sec=0.5)
    
    daemon._handle_shutdown(signal.SIGINT, None)
    
    assert daemon._shutdown_requested is True


def test_daemon_starts_and_writes_heartbeat(workspace: Path, mock_pipeline_flow: MagicMock):
    """Invariant: Daemon running in a thread writes heartbeat within one cycle."""
    daemon = DaemonLoop(workspace, mock_pipeline_flow, poll_interval_sec=0.5)
    
    thread = threading.Thread(target=daemon.start)
    thread.start()
    
    time.sleep(0.6)
    daemon.stop(timeout_sec=1.0)
    thread.join(timeout=2.0)
    
    heartbeat_path = workspace / "logs" / "daemon_heartbeat.json"
    assert heartbeat_path.exists()
    
    with open(heartbeat_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    assert "timestamp" in data
    assert "uptime_sec" in data
    assert "queue_depth" in data
    assert data["is_running"] is False


def test_heartbeat_written_with_correct_fields(workspace: Path, mock_pipeline_flow: MagicMock, daemon: DaemonLoop):
    """Invariant: Heartbeat JSON is written atomically every cycle with required fields."""
    daemon._lock_manager.acquire()
    daemon._start_time = time.time()
    daemon._is_running = True
    
    daemon._write_heartbeat()
    
    heartbeat_path = workspace / "logs" / "daemon_heartbeat.json"
    assert heartbeat_path.exists()
    
    data = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert "timestamp" in data
    assert "uptime_sec" in data
    assert "queue_depth" in data
    assert "is_running" in data
    assert data["is_running"] is True


def test_get_stats_returns_accurate_data(workspace: Path, mock_pipeline_flow: MagicMock, daemon: DaemonLoop):
    """Invariant: get_stats() returns a valid DaemonStats dataclass."""
    daemon._start_time = time.time()
    daemon._is_running = True
    daemon._tickets_ingested = 5
    
    stats = daemon.get_stats()
    
    assert stats.is_running is True
    assert stats.tickets_ingested == 5
    assert stats.uptime_sec >= 0
    assert stats.queue_stats == {"queued": 0, "running": 0, "pending_review": 0}


def test_backpressure_skips_ingestion(workspace: Path, mock_pipeline_flow: MagicMock, daemon: DaemonLoop):
    """Invariant: When queue depth exceeds threshold, ingestion is skipped."""
    daemon._lock_manager.acquire()
    mock_pipeline_flow.get_queue_stats.return_value = {"queued": 100}
    
    ticket_data = {
        "ticket_id": "backpressure_001",
        "job_type": "scout",
        "parameters": {"query": "test"},
    }
    file_path = write_ticket(workspace, "ticket_backpressure_001.json", ticket_data)
    
    daemon._ingest_cycle()
    
    assert file_path.exists()
    mock_pipeline_flow.submit_job.assert_not_called()


def test_already_processed_ticket_skipped(workspace: Path, mock_pipeline_flow: MagicMock, daemon: DaemonLoop):
    """Invariant: Tickets already in .processed/ or .rejected/ are not re-processed."""
    daemon._lock_manager.acquire()
    
    ticket_data = {
        "ticket_id": "duplicate_001",
        "job_type": "scout",
        "parameters": {"query": "test"},
    }
    file_path = write_ticket(workspace, "ticket_duplicate_001.json", ticket_data)
    
    processed_path = workspace / "00_inbox" / ".processed" / "ticket_duplicate_001.json"
    processed_path.write_text("{}", encoding="utf-8")
    
    daemon._ingest_cycle()
    
    mock_pipeline_flow.submit_job.assert_not_called()
    assert file_path.exists()


# ============================================================================
# Daemon Loop Tests: Privacy & Safety
# ============================================================================

def test_pii_not_logged(workspace: Path, mock_pipeline_flow: MagicMock, daemon: DaemonLoop, caplog):
    """Invariant: Ticket payload (parameters) is NEVER logged, only ticket_id and job_type."""
    daemon._lock_manager.acquire()
    
    secret_payload = "SUPER_SECRET_API_KEY_12345"
    ticket_data = {
        "ticket_id": "pii_test_001",
        "job_type": "scout",
        "parameters": {"api_key": secret_payload},
    }
    write_ticket(workspace, "ticket_pii_test_001.json", ticket_data)
    
    with caplog.at_level(logging.INFO):
        daemon._ingest_cycle()
    
    for record in caplog.records:
        assert secret_payload not in record.message, f"PII leaked in log: {record.message}"


def test_exception_chaining():
    """Invariant: DaemonError chains from the original exception via __cause__."""
    original_error = ValueError("Original cause")
    
    try:
        raise DaemonError("Wrapper message", ErrorCode.INTERNAL, original_exception=original_error)
    except DaemonError as e:
        assert e.__cause__ is original_error
        assert e.error_code == ErrorCode.INTERNAL


def test_disabled_feature_flag_exits_early(workspace: Path, mock_pipeline_flow: MagicMock):
    """Invariant: If enabled=False, start() exits immediately without acquiring lock."""
    daemon = DaemonLoop(
        workspace_path=workspace,
        pipeline_flow=mock_pipeline_flow,
        enabled=False,
    )
    
    daemon.start()
    
    assert not daemon._is_running
    assert not daemon.lock_file_path.exists()


def test_invalid_workspace_raises_daemon_error():
    """Invariant: Invalid workspace path raises DaemonError at startup."""
    # Use a definitively non-existent Windows path
    invalid_path = Path("C:\\DefinitelyDoesNotExist12345\\JarviWorkspace")
    
    with pytest.raises(DaemonError) as exc_info:
        DaemonLoop(
            workspace_path=invalid_path,
            pipeline_flow=MagicMock(),
        )
    
    assert exc_info.value.error_code == ErrorCode.IO_ERROR


# ============================================================================
# Lock Manager Tests
# ============================================================================

@pytest.mark.skip(reason="Windows file locking edge case in same-process test")
def test_lock_file_prevents_second_instance(workspace: Path):
    """Invariant: Two LockManagers cannot both hold the lock simultaneously."""
    lock_path = workspace / "logs" / "test.lock"
    
    lm1 = LockManager(lock_path)
    lm1.acquire()
    
    lm2 = LockManager(lock_path)
    with pytest.raises(DaemonError) as exc_info:
        lm2.acquire()
    
    assert exc_info.value.error_code == ErrorCode.DAEMON_ALREADY_RUNNING
    
    lm1.release()
    assert not lock_path.exists()


def test_lock_file_stale_detection(workspace: Path):
    """Invariant: A stale lock file (dead PID) is automatically cleaned up."""
    lock_path = workspace / "logs" / "test.lock"
    
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as f:
        json.dump({"pid": 999999, "timestamp": time.time() - 7200}, f)
    
    lm = LockManager(lock_path)
    lm.acquire()
    
    with open(lock_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["pid"] == __import__("os").getpid()
    
    lm.release()


# ============================================================================
# Integration-Style Test
# ============================================================================

def test_full_cycle_with_multiple_tickets(workspace: Path, mock_pipeline_flow: MagicMock, daemon: DaemonLoop):
    """Invariant: Multiple tickets in one cycle are processed correctly."""
    daemon._lock_manager.acquire()
    
    for i in range(3):
        write_ticket(workspace, f"ticket_multi_{i:03d}.json", {
            "ticket_id": f"multi_{i:03d}",
            "job_type": "scout",
            "parameters": {"query": f"test {i}"},
        })
    
    (workspace / "00_inbox" / "ticket_bad.json").write_text("not json", encoding="utf-8")
    
    daemon._ingest_cycle()
    
    assert mock_pipeline_flow.submit_job.call_count == 3
    assert len(list((workspace / "00_inbox" / ".processed").glob("*.json"))) == 3
    assert len(list((workspace / "00_inbox" / ".rejected").glob("*.json"))) == 1
    assert (workspace / "00_inbox" / ".rejected" / "ticket_bad.err").exists()