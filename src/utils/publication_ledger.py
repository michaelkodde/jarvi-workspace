"""
Publication Ledger: SQLite-backed audit trail for all publication attempts.

Provides persistent tracking of content deliveries with deduplication support.
Uses WAL mode for concurrent access and short-lived connections.
"""
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

__all__ = ["PublicationLedger", "PublicationRecord", "LedgerError"]


class LedgerError(Exception):
    """Raised when ledger operations fail."""
    pass


@dataclass(frozen=True)
class PublicationRecord:
    """Immutable record of a publication attempt."""
    publication_id: str
    timestamp: str
    content_hash: str
    destination: str
    target: str
    status: str
    affiliate_links_inserted: int
    error_message: Optional[str] = None


class PublicationLedger:
    """
    SQLite-backed ledger for tracking publication attempts.
    
    Uses WAL mode for concurrent access and short-lived connections to prevent
    daemon locking issues.
    
    Args:
        db_path: Path to the SQLite database file.
    
    Raises:
        LedgerError: If database initialization fails.
    """

    def __init__(self, db_path: str = "dashboard/publications.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the publications table if it does not exist."""
        try:
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS publications (
                        publication_id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        content_hash TEXT NOT NULL,
                        destination TEXT NOT NULL,
                        target TEXT NOT NULL,
                        status TEXT NOT NULL,
                        affiliate_links_inserted INTEGER NOT NULL,
                        error_message TEXT
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_content_hash ON publications(content_hash)
                """)
                conn.commit()
        except sqlite3.Error as e:
            raise LedgerError(f"Failed to initialize ledger database: {e}") from e

    def log_publication(self, record: PublicationRecord) -> None:
        """
        Log a publication attempt to the ledger.
        
        Args:
            record: The publication record to log.
        
        Raises:
            LedgerError: If the insert fails.
        """
        try:
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("""
                    INSERT OR REPLACE INTO publications
                    (publication_id, timestamp, content_hash, destination, target, status, affiliate_links_inserted, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.publication_id,
                    record.timestamp,
                    record.content_hash,
                    record.destination,
                    record.target,
                    record.status,
                    record.affiliate_links_inserted,
                    record.error_message,
                ))
                conn.commit()
        except sqlite3.Error as e:
            raise LedgerError(f"Failed to log publication: {e}") from e

    def get_publications(self) -> List[PublicationRecord]:
        """
        Retrieve all publication records.
        
        Returns:
            List of PublicationRecord instances.
        
        Raises:
            LedgerError: If the query fails.
        """
        try:
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                cursor = conn.execute("""
                    SELECT publication_id, timestamp, content_hash, destination, target, status, affiliate_links_inserted, error_message
                    FROM publications
                    ORDER BY timestamp DESC
                """)
                return [
                    PublicationRecord(
                        publication_id=row[0],
                        timestamp=row[1],
                        content_hash=row[2],
                        destination=row[3],
                        target=row[4],
                        status=row[5],
                        affiliate_links_inserted=row[6],
                        error_message=row[7],
                    )
                    for row in cursor.fetchall()
                ]
        except sqlite3.Error as e:
            raise LedgerError(f"Failed to retrieve publications: {e}") from e

    def get_by_id(self, publication_id: str) -> Optional[PublicationRecord]:
        """
        Retrieve a publication record by ID.
        
        Args:
            publication_id: The publication ID to look up.
        
        Returns:
            PublicationRecord if found, None otherwise.
        
        Raises:
            LedgerError: If the query fails.
        """
        try:
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                cursor = conn.execute("""
                    SELECT publication_id, timestamp, content_hash, destination, target, status, affiliate_links_inserted, error_message
                    FROM publications
                    WHERE publication_id = ?
                """, (publication_id,))
                row = cursor.fetchone()
                if row:
                    return PublicationRecord(
                        publication_id=row[0],
                        timestamp=row[1],
                        content_hash=row[2],
                        destination=row[3],
                        target=row[4],
                        status=row[5],
                        affiliate_links_inserted=row[6],
                        error_message=row[7],
                    )
                return None
        except sqlite3.Error as e:
            raise LedgerError(f"Failed to retrieve publication by ID: {e}") from e

    def check_duplicate(self, content_hash: str) -> bool:
        """
        Check if a content hash has already been published.
        
        Args:
            content_hash: The content hash to check.
        
        Returns:
            True if duplicate found, False otherwise.
        
        Raises:
            LedgerError: If the query fails.
        """
        try:
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                cursor = conn.execute(
                    "SELECT 1 FROM publications WHERE content_hash = ? LIMIT 1",
                    (content_hash,)
                )
                return cursor.fetchone() is not None
        except sqlite3.Error as e:
            raise LedgerError(f"Failed to check duplicate: {e}") from e