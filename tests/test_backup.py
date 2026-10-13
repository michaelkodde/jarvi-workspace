"""
Unit tests for the Backup Utility.
"""
import json
import pytest
import zipfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.utils.backup import create_backup, BackupError, BackupResult


@pytest.fixture
def mock_env(tmp_path: Path):
    """Sets up a temporary workspace, backup dir, and config for testing."""
    src_dir = tmp_path / "workspace"
    src_dir.mkdir()
    
    # Create some dummy files
    (src_dir / "file1.txt").write_text("data1")
    (src_dir / "file2.txt").write_text("data2")
    (src_dir / ".env").write_text("SECRET=123")
    
    dst_dir = tmp_path / "backups"
    
    config_file = tmp_path / "config.json"
    config_data = {
        "workspace_path": str(src_dir),
        "backup_path": str(dst_dir)
    }
    config_file.write_text(json.dumps(config_data), encoding="utf-8")
    
    return src_dir, dst_dir, config_file


def test_create_backup_success_and_exclusions(mock_env):
    """Verify that a standard backup creates a valid zip, returns correct stats, and excludes .env."""
    src_dir, dst_dir, config_file = mock_env
    
    result = create_backup(config_path=str(config_file))
    
    assert isinstance(result, BackupResult)
    assert result.files_processed == 2 # file1.txt and file2.txt
    assert result.files_skipped == 0
    assert result.archive_path.exists()
    
    # Gold Standard: Actually inspect the zip to prove .env was excluded
    with zipfile.ZipFile(result.archive_path, "r") as zf:
        names = zf.namelist()
        assert "file1.txt" in names
        assert ".env" not in names


def test_create_backup_skips_locked_files(mock_env):
    """Verify that PermissionError on a file is caught and skipped cleanly."""
    src_dir, dst_dir, config_file = mock_env
    
    # Mock rglob to yield a file that will raise PermissionError when written
    mock_file = MagicMock(spec=Path)
    mock_file.is_file.return_value = True
    mock_file.is_symlink.return_value = False
    mock_file.parts = ("workspace", "locked.txt")
    mock_file.is_relative_to.return_value = False
    
    with patch("src.utils.backup.Path.rglob", return_value=[mock_file]):
        with patch("zipfile.ZipFile.write", side_effect=PermissionError("Access denied")):
            result = create_backup(config_path=str(config_file))
            
            assert result.files_processed == 0
            assert result.files_skipped == 1


def test_create_backup_raises_on_missing_source(mock_env):
    """Verify that a missing source directory raises BackupError immediately."""
    _, _, config_file = mock_env
    
    with pytest.raises(BackupError, match="Source directory does not exist"):
        create_backup(source_dir="/nonexistent/path", config_path=str(config_file))


def test_create_backup_handles_missing_config(mock_env):
    """Verify that missing config.json falls back to safe defaults without crashing."""
    src_dir, dst_dir, _ = mock_env
    
    # Use a non-existent config path, but override source/dest so it doesn't fail on cwd
    result = create_backup(
        source_dir=str(src_dir), 
        backup_dir=str(dst_dir), 
        config_path="missing_config.json"
    )
    
    assert result.files_processed == 2
    assert result.archive_path.exists()


def test_create_backup_creates_destination_dir(mock_env):
    """Verify that the backup utility creates the destination directory if it doesn't exist."""
    src_dir, _, config_file = mock_env
    deep_dst = src_dir / "deep" / "nested" / "backups"
    
    result = create_backup(
        source_dir=str(src_dir), 
        backup_dir=str(deep_dst), 
        config_path=str(config_file)
    )
    
    assert deep_dst.exists()
    assert result.archive_path.exists()