# tests/test_daemon.py
"""Tests for the Daemon Loop and Query API."""
from __future__ import annotations

import json
import os
import signal
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.companion.daemon import (
    ApprovalItem,
    ApprovalResult,
    ApprovalsResult,
    DaemonError,
    DaemonLoop,
    ErrorCode,
    QueryAPI,
)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def mock_pipeline_flow() -> MagicMock:
    mock = MagicMock()
    mock.get_queue_stats.return_value = {"queued": 0, "running": 0, "pending_review": 0}
    mock.get_job_status.return_value = {}  # Empty dict means not found
    mock.submit_job.return_value = "mock_job_id"
    mock.process_next_job.return_value = None
    mock.list_jobs.return_value = []
    mock.approve_job = MagicMock()
    mock.reject_job = MagicMock()
    return mock


def test_daemon_starts_and_writes_heartbeat(workspace: Path, mock_pipeline_flow: MagicMock) -> None:
    daemon = DaemonLoop(workspace, mock_pipeline_flow, poll_interval_sec=0.1, install_signals=False)
    
    thread = threading.Thread(target=daemon.start)
    thread.start()
    
    time.sleep(0.3)  # Allow at least one cycle
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


def test_valid_ticket_ingested_and_moved(workspace: Path, mock_pipeline_flow: MagicMock) -> None:
    inbox = workspace / "00_inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    ticket_path = inbox / "ticket_001.json"
    ticket_data = {
        "ticket_id": "t001",
        "job_type": "research",
        "parameters": {"query": "test"},
        "priority": 5,
    }
    with open(ticket_path, "w", encoding="utf-8") as f:
        json.dump(ticket_data, f)
    
    daemon = DaemonLoop(workspace, mock_pipeline_flow, poll_interval_sec=0.1, install_signals=False)
    daemon._ingest_tickets()
    
    mock_pipeline_flow.submit_job.assert_called_once_with(
        job_type="research", parameters={"query": "test"}, priority=5, description=""
    )
    assert not ticket_path.exists()
    assert (workspace / "00_inbox" / ".processed" / "ticket_001.json").exists()


def test_invalid_json_ticket_rejected(workspace: Path, mock_pipeline_flow: MagicMock) -> None:
    inbox = workspace / "00_inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    ticket_path = inbox / "ticket_002.json"
    with open(ticket_path, "w", encoding="utf-8") as f:
        f.write("{ invalid json }")
    
    daemon = DaemonLoop(workspace, mock_pipeline_flow, poll_interval_sec=0.1, install_signals=False)
    daemon._ingest_tickets()
    
    mock_pipeline_flow.submit_job.assert_not_called()
    assert not ticket_path.exists()
    assert (workspace / "00_inbox" / ".rejected" / "ticket_002.json").exists()
    
    err_path = workspace / "00_inbox" / ".rejected" / "ticket_002.err"
    assert err_path.exists()
    with open(err_path, "r", encoding="utf-8") as f:
        err_data = json.load(f)
    
    assert err_data["error_code"] == ErrorCode.INVALID_TICKET
    assert "parameters" not in err_data  # PII protection


def test_oversized_ticket_rejected(workspace: Path, mock_pipeline_flow: MagicMock) -> None:
    inbox = workspace / "00_inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    ticket_path = inbox / "ticket_003.json"
    with open(ticket_path, "w", encoding="utf-8") as f:
        json.dump({"ticket_id": "t003", "job_type": "test", "data": "x" * (1024 * 1024 + 1)}, f)
    
    daemon = DaemonLoop(workspace, mock_pipeline_flow, poll_interval_sec=0.1, install_signals=False)
    daemon._ingest_tickets()
    
    mock_pipeline_flow.submit_job.assert_not_called()
    assert (workspace / "00_inbox" / ".rejected" / "ticket_003.json").exists()


def test_duplicate_ticket_handled_idempotently(workspace: Path, mock_pipeline_flow: MagicMock) -> None:
    inbox = workspace / "00_inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    ticket_path = inbox / "ticket_004.json"
    with open(ticket_path, "w", encoding="utf-8") as f:
        json.dump({"ticket_id": "t004", "job_type": "test"}, f)
    
    mock_pipeline_flow.get_job_status.return_value = {"job_id": "t004", "status": "completed"}
    
    daemon = DaemonLoop(workspace, mock_pipeline_flow, poll_interval_sec=0.1, install_signals=False)
    daemon._ingest_tickets()
    
    mock_pipeline_flow.submit_job.assert_not_called()
    assert (workspace / "00_inbox" / ".processed" / "ticket_004.json").exists()


def test_lock_file_prevents_second_instance(workspace: Path, mock_pipeline_flow: MagicMock) -> None:
    daemon1 = DaemonLoop(workspace, mock_pipeline_flow, poll_interval_sec=0.5, install_signals=False)
    
    assert daemon1._acquire_lock() is True
    
    daemon2 = DaemonLoop(workspace, mock_pipeline_flow, poll_interval_sec=0.5, install_signals=False)
    assert daemon2._acquire_lock() is False  # Should fail to acquire
    
    daemon1._release_lock()


def test_backpressure_skips_ingestion(workspace: Path, mock_pipeline_flow: MagicMock) -> None:
    inbox = workspace / "00_inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    ticket_path = inbox / "ticket_005.json"
    with open(ticket_path, "w", encoding="utf-8") as f:
        json.dump({"ticket_id": "t005", "job_type": "test"}, f)
    
    mock_pipeline_flow.get_queue_stats.return_value = {"queued": 60}  # Exceeds default 50
    
    daemon = DaemonLoop(workspace, mock_pipeline_flow, poll_interval_sec=0.1, queue_depth_threshold=50, install_signals=False)
    daemon._ingest_tickets()
    
    mock_pipeline_flow.submit_job.assert_not_called()
    assert ticket_path.exists()  # Remains in inbox


def test_query_api_get_pending_approvals(workspace: Path, mock_pipeline_flow: MagicMock) -> None:
    mock_pipeline_flow.list_jobs.return_value = [
        {"job_id": "j1", "job_type": "research", "current_step": "hitl", "description": "test", "created_at": "2026-01-01T00:00:00Z"}
    ]
    
    api = QueryAPI(mock_pipeline_flow)
    result = api.get_pending_approvals()
    
    assert result.status == "success"
    assert result.count == 1
    assert isinstance(result.approvals[0], ApprovalItem)
    assert result.approvals[0].job_id == "j1"


def test_query_api_approve_and_reject(workspace: Path, mock_pipeline_flow: MagicMock) -> None:
    api = QueryAPI(mock_pipeline_flow)
    
    result = api.approve_job("j1", feedback="looks good")
    assert result.status == "success"
    assert result.job_id == "j1"
    mock_pipeline_flow.approve_job.assert_called_once_with("j1", feedback="looks good")
    
    result = api.reject_job("j2", reason="invalid data")
    assert result.status == "success"
    assert result.job_id == "j2"
    mock_pipeline_flow.reject_job.assert_called_once_with("j2", reason="invalid data")


def test_graceful_shutdown_handler(workspace: Path, mock_pipeline_flow: MagicMock) -> None:
    daemon = DaemonLoop(workspace, mock_pipeline_flow, poll_interval_sec=0.5, install_signals=False)
    
    # Manually trigger the shutdown handler (simulating SIGINT)
    daemon._handle_shutdown(signal.SIGINT, None)
    
    assert daemon._shutdown_requested is True