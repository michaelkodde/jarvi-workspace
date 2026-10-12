"""
Unit tests for the Token Ledger.
"""
import pytest
from pathlib import Path
import threading

from src.utils.token_ledger import TokenLedger, TokenUsage


def test_ledger_auto_provisions_and_calculates_cost(tmp_path: Path):
    """Test that the ledger creates the directory, header, logs a row, and calculates cost."""
    log_dir = tmp_path / "dashboard"
    config_file = tmp_path / "config.json"

    config_file.write_text('{"token_costs": {"qwen2.5:7b": 0.001}}')

    ledger = TokenLedger(config_path=str(config_file), log_dir=str(log_dir))

    usage = TokenUsage(
        agent_name="router",
        model_name="qwen2.5:7b",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150
    )

    ledger.log_usage(usage)

    log_file = log_dir / "cost_log.csv"
    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert "router" in lines[1]
    assert "150" in lines[1]
    assert "$0.000150" in lines[1]


def test_ledger_validates_negative_tokens(tmp_path: Path):
    """Test that the ledger rejects negative token counts."""
    log_dir = tmp_path / "dashboard"
    config_file = tmp_path / "config.json"
    config_file.write_text('{"token_costs": {}}')

    ledger = TokenLedger(config_path=str(config_file), log_dir=str(log_dir))

    usage = TokenUsage("agent", "model", -10, 10, 0)

    with pytest.raises(ValueError):
        ledger.log_usage(usage)


def test_ledger_thread_safety(tmp_path: Path):
    """Test that concurrent writes do not corrupt the CSV."""
    log_dir = tmp_path / "dashboard"
    config_file = tmp_path / "config.json"
    config_file.write_text('{"token_costs": {}}')

    ledger = TokenLedger(config_path=str(config_file), log_dir=str(log_dir))

    def log_many():
        for _ in range(50):
            usage = TokenUsage("agent", "model", 10, 10, 20)
            ledger.log_usage(usage)

    threads = [threading.Thread(target=log_many) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    log_file = log_dir / "cost_log.csv"
    lines = log_file.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 251


def test_ledger_handles_missing_config_file(tmp_path: Path):
    """Test that the ledger gracefully handles a missing config.json."""
    log_dir = tmp_path / "dashboard"
    missing_config = tmp_path / "nonexistent.json"

    ledger = TokenLedger(config_path=str(missing_config), log_dir=str(log_dir))

    usage = TokenUsage("agent", "model", 10, 10, 20)
    ledger.log_usage(usage)

    log_file = log_dir / "cost_log.csv"
    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert "$0.000000" in lines[1]


def test_ledger_appends_multiple_rows(tmp_path: Path):
    """Test that multiple log_usage calls append correctly."""
    log_dir = tmp_path / "dashboard"
    config_file = tmp_path / "config.json"
    config_file.write_text('{"token_costs": {}}')

    ledger = TokenLedger(config_path=str(config_file), log_dir=str(log_dir))

    for i in range(5):
        usage = TokenUsage(f"agent{i}", "model", 10, 10, 20)
        ledger.log_usage(usage)

    log_file = log_dir / "cost_log.csv"
    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6  # 1 header + 5 data rows


def test_ledger_shutdown_is_noop(tmp_path: Path):
    """Test that shutdown method exists and doesn't raise."""
    log_dir = tmp_path / "dashboard"
    config_file = tmp_path / "config.json"
    config_file.write_text('{"token_costs": {}}')

    ledger = TokenLedger(config_path=str(config_file), log_dir=str(log_dir))
    ledger.shutdown()  # Should not raise