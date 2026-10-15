"""Unit tests for the PublicationLedger utility."""
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from src.utils.publication_ledger import LedgerError, PublicationLedger, PublicationRecord


@pytest.fixture
def ledger(tmp_path: Path):
    db_path = tmp_path / "test_publications.db"
    return PublicationLedger(db_path=str(db_path))


def test_log_and_retrieve(ledger):
    record = PublicationRecord(
        publication_id="pub1",
        timestamp="2026-10-16T12:00:00Z",
        content_hash="hash1",
        destination="file",
        target="output/test.json",
        status="success",
        affiliate_links_inserted=2,
    )
    ledger.log_publication(record)
    records = ledger.get_publications()
    assert len(records) == 1
    assert records[0].publication_id == "pub1"
    assert records[0].content_hash == "hash1"


def test_check_duplicate(ledger):
    record = PublicationRecord(
        publication_id="pub1",
        timestamp="2026-10-16T12:00:00Z",
        content_hash="dup_hash",
        destination="file",
        target="test.json",
        status="success",
        affiliate_links_inserted=0,
    )
    ledger.log_publication(record)
    assert ledger.check_duplicate("dup_hash") is True
    assert ledger.check_duplicate("other_hash") is False


def test_ledger_connection_failure(tmp_path):
    # Mock sqlite3.connect to raise an error
    with patch("src.utils.publication_ledger.sqlite3.connect", side_effect=sqlite3.Error("Connection failed")):
        with pytest.raises(LedgerError):
            PublicationLedger(db_path=str(tmp_path / "test.db"))