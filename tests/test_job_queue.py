"""Unit tests for the JobQueue."""
import json
from pathlib import Path

import pytest

from src.services.job_queue import JobQueue, JobRecord, QueueError


@pytest.fixture
def queue(tmp_path: Path):
    db_path = tmp_path / "test_queue.db"
    return JobQueue(db_path=str(db_path))


def test_enqueue_claim_priority_order(queue):
    queue.enqueue("j1", "test", {"a": 1}, priority=1)
    queue.enqueue("j2", "test", {"a": 2}, priority=5)
    queue.enqueue("j3", "test", {"a": 3}, priority=5)
    
    j = queue.claim_next()
    assert j.job_id == "j2"
    assert j.status == "running"
    
    j = queue.claim_next()
    assert j.job_id == "j3"
    
    j = queue.claim_next()
    assert j.job_id == "j1"
    
    assert queue.claim_next() is None


def test_complete_step_atomic(queue):
    queue.enqueue("j1", "test", {"a": 1})
    job = queue.claim_next()
    
    queue.complete_step(
        "j1", "step1",
        artifacts={"step1": "output1"},
        completed_steps=("step1",),
    )
    
    job = queue.get_job("j1")
    assert job.artifacts == {"step1": "output1"}
    assert job.completed_steps == ("step1",)


def test_recover_orphans(queue):
    queue.enqueue("j1", "test", {"a": 1})
    job = queue.claim_next()
    queue.complete_step("j1", "step1", {"s1": "out"}, ("step1",))
    
    recovered = queue.recover_orphans()
    assert recovered == 1
    
    job = queue.get_job("j1")
    assert job.status == "queued"
    assert job.completed_steps == ("step1",)
    assert job.artifacts == {"s1": "out"}


def test_atomic_claim_two_instances(tmp_path):
    db_path = tmp_path / "shared.db"
    q1 = JobQueue(db_path=str(db_path))
    q2 = JobQueue(db_path=str(db_path))
    
    q1.enqueue("j1", "test", {"a": 1})
    
    j1 = q1.claim_next()
    j2 = q2.claim_next()
    
    assert j1 is not None
    assert j1.job_id == "j1"
    assert j2 is None


def test_sqlite_failure_raises_queue_error(tmp_path):
    # Create a file where the directory should be to force a failure
    blocker = tmp_path / "blocker"
    blocker.write_text("I am a file, not a directory", encoding="utf-8")
    
    # Try to create a database inside the "file" (which will fail)
    invalid_path = blocker / "queue.db"
    with pytest.raises(QueueError):
        JobQueue(db_path=str(invalid_path))