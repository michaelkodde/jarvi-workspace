"""
Scout Checkpoint: SQLite-backed state persistence for the saturation loop.
"""
import json
import logging
import sqlite3
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = ["ScoutCheckpoint"]


class ScoutCheckpoint:
    """
    Manages SQLite-backed checkpointing for the Scout agent's saturation loop.
    Uses WAL mode and short-lived connections to prevent daemon locking issues.
    """

    def __init__(self, db_path: str = "dashboard/scout_checkpoints.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initializes the checkpoint table if it does not exist."""
        with sqlite3.connect(self.db_path, timeout=5.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scout_state (
                    run_id TEXT PRIMARY KEY,
                    iteration INTEGER,
                    current_query TEXT,
                    accumulated_products TEXT
                )
            """)
            conn.commit()

    def load_state(self, run_id: str) -> Optional[Tuple[int, str, List[dict]]]:
        """
        Loads the checkpoint state for a given run_id.
        
        Returns:
            Tuple of (iteration, current_query, accumulated_products) or None if not found.
        """
        with sqlite3.connect(self.db_path, timeout=5.0) as conn:
            cursor = conn.execute(
                "SELECT iteration, current_query, accumulated_products FROM scout_state WHERE run_id = ?",
                (run_id,)
            )
            row = cursor.fetchone()
            if row:
                products_json = row[2]
                products = json.loads(products_json) if products_json else []
                return row[0], row[1], products
        return None

    def save_state(self, run_id: str, iteration: int, current_query: str, accumulated_products: List[dict]) -> None:
        """
        Atomically saves or updates the checkpoint state with full product data.
        """
        products_json = json.dumps(accumulated_products)
        with sqlite3.connect(self.db_path, timeout=5.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("""
                INSERT OR REPLACE INTO scout_state (run_id, iteration, current_query, accumulated_products)
                VALUES (?, ?, ?, ?)
            """, (run_id, iteration, current_query, products_json))
            conn.commit()

    def clear_state(self, run_id: str) -> None:
        """
        Removes the checkpoint state upon successful completion.
        """
        with sqlite3.connect(self.db_path, timeout=5.0) as conn:
            conn.execute("DELETE FROM scout_state WHERE run_id = ?", (run_id,))
            conn.commit()