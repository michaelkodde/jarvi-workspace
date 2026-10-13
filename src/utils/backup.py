"""
Backup Utility: Creates timestamped, compressed archives of the workspace.
"""
import json
import logging
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = ["BackupError", "BackupResult", "create_backup"]

# Directories and files to exclude from the backup to save space and prevent leaks.
EXCLUDED_NAMES = frozenset({
    "venv", ".venv", "env", ".env", 
    ".git", "__pycache__", ".pytest_cache", 
    "logs", "backups", "node_modules"
})

DEFAULT_WORKSPACE = Path.cwd()
DEFAULT_BACKUP_DIR = Path.cwd() / "backups"


class BackupError(Exception):
    """Custom exception for backup initialization or critical failures."""
    pass


@dataclass
class BackupResult:
    """Strongly typed result of a backup operation."""
    archive_path: Path
    files_processed: int
    files_skipped: int


def _load_config(config_path: str) -> dict:
    """
    Loads backup configuration from config.json.
    
    Args:
        config_path: Path to the configuration file.
        
    Returns:
        Dictionary containing workspace and backup paths.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            return {
                "workspace_path": Path(config.get("workspace_path", DEFAULT_WORKSPACE)),
                "backup_path": Path(config.get("backup_path", DEFAULT_BACKUP_DIR)),
            }
    except (FileNotFoundError, json.JSONDecodeError, IOError) as e:
        logger.warning("Failed to load backup config. Using safe defaults.")
        return {
            "workspace_path": DEFAULT_WORKSPACE,
            "backup_path": DEFAULT_BACKUP_DIR,
        }


def create_backup(
    source_dir: Optional[str] = None,
    backup_dir: Optional[str] = None,
    config_path: str = "config.json"
) -> BackupResult:
    """
    Creates a timestamped .zip archive of the workspace, gracefully skipping locked files.
    
    Args:
        source_dir: Optional override for the source workspace directory.
        backup_dir: Optional override for the destination backup directory.
        config_path: Path to config.json for default paths.
        
    Returns:
        BackupResult containing the archive path and processing statistics.
        
    Raises:
        BackupError: If the source directory does not exist or a critical I/O error occurs.
    """
    config = _load_config(config_path)
    src = Path(source_dir) if source_dir else config["workspace_path"]
    dst = Path(backup_dir) if backup_dir else config["backup_path"]

    if not src.exists() or not src.is_dir():
        raise BackupError(f"Source directory does not exist: {src}")

    dst.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"jarvi_backup_{timestamp}.zip"
    archive_path = dst / archive_name

    files_processed = 0
    files_skipped = 0

    logger.info("Starting backup of %s to %s", src, archive_path)

    try:
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in src.rglob("*"):
                # Skip excluded directories and files
                if any(part in EXCLUDED_NAMES for part in file_path.parts):
                    continue
                
                # CRITICAL: Skip the backup directory itself if it's nested inside the source
                if file_path.is_relative_to(dst):
                    continue

                if file_path.is_file():
                    # CRITICAL: Skip symlinks to prevent Windows resolution errors or security risks
                    if file_path.is_symlink():
                        continue
                        
                    try:
                        # Store relative to the source directory to keep zip structure clean
                        arcname = file_path.relative_to(src)
                        zf.write(file_path, arcname)
                        files_processed += 1
                    except PermissionError:
                        logger.warning("Skipping locked file: %s", file_path)
                        files_skipped += 1
                    except OSError as e:
                        logger.warning("Skipping file due to OS error: %s (%s)", file_path, e)
                        files_skipped += 1

    except OSError as e:
        # Best-effort cleanup: delete incomplete zip without masking the original error
        try:
            archive_path.unlink(missing_ok=True)
        except OSError:
            pass  # Ignore cleanup failures; the original error is what matters
        raise BackupError(f"Failed to write backup archive: {e}") from e

    logger.info(
        "Backup complete: %s (Processed: %d, Skipped: %d)", 
        archive_path, files_processed, files_skipped
    )
    
    return BackupResult(
        archive_path=archive_path,
        files_processed=files_processed,
        files_skipped=files_skipped
    )