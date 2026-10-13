"""
Unit tests for the Automation Guard.
"""
import pytest
import requests
from unittest.mock import patch, MagicMock

from src.utils.automation_guard import (
    AutomationError,
    human_delay,
    retry_with_backoff,
)


@patch("src.utils.automation_guard.time.sleep")
@patch("src.utils.automation_guard.random.uniform", return_value=3.5)
def test_human_delay_calls_sleep_with_random_value(mock_uniform, mock_sleep):
    """Verify human_delay calculates a random uniform value and sleeps."""
    human_delay(2.0, 5.0)
    mock_uniform.assert_called_once_with(2.0, 5.0)
    mock_sleep.assert_called_once_with(3.5)


def test_human_delay_raises_on_invalid_range():
    """Verify human_delay raises ValueError if min > max."""
    with pytest.raises(ValueError):
        human_delay(5.0, 2.0)


def test_retry_succeeds_on_first_try():
    """Verify the decorator returns immediately on success."""
    @retry_with_backoff(max_retries=3, base_delay=0.0)
    def success_func():
        return "success"

    assert success_func() == "success"


@patch("src.utils.automation_guard.time.sleep")
@patch("src.utils.automation_guard.random.uniform", return_value=0.1)
def test_retry_recovers_from_http_429(mock_uniform, mock_sleep):
    """Verify the decorator retries on HTTP 429 and eventually succeeds."""
    call_count = 0

    @retry_with_backoff(max_retries=3, base_delay=0.0)
    def flaky_func():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            mock_resp = MagicMock(spec=requests.Response)
            mock_resp.status_code = 429
            return mock_resp
        return "recovered"

    assert flaky_func() == "recovered"
    assert call_count == 3


@patch("src.utils.automation_guard.time.sleep")
@patch("src.utils.automation_guard.random.uniform", return_value=0.1)
def test_retry_recovers_from_timeout_error(mock_uniform, mock_sleep):
    """Verify the decorator retries on TimeoutError and eventually succeeds."""
    call_count = 0

    @retry_with_backoff(max_retries=3, base_delay=0.0)
    def timeout_func():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise TimeoutError("Network hang")
        return "recovered"

    assert timeout_func() == "recovered"
    assert call_count == 2


@patch("src.utils.automation_guard.time.sleep")
@patch("src.utils.automation_guard.random.uniform", return_value=0.1)
def test_retry_exhausts_on_persistent_500_raises_automation_error(mock_uniform, mock_sleep):
    """Verify that exhausting retries on a bad HTTP status raises AutomationError."""
    @retry_with_backoff(max_retries=2, base_delay=0.0)
    def fail_func():
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.status_code = 500
        return mock_resp

    with pytest.raises(AutomationError, match="failed after 2 attempts"):
        fail_func()


@patch("src.utils.automation_guard.time.sleep")
@patch("src.utils.automation_guard.random.uniform", return_value=0.1)
def test_retry_exhausts_on_persistent_exception_raises_automation_error(mock_uniform, mock_sleep):
    """Verify that exhausting retries on an exception raises AutomationError with chaining."""
    @retry_with_backoff(max_retries=2, base_delay=0.0)
    def fail_func():
        raise requests.exceptions.ConnectionError("DNS failure")

    with pytest.raises(AutomationError) as exc_info:
        fail_func()
        
    # Verify exception chaining preserves the original error
    assert isinstance(exc_info.value.__cause__, requests.exceptions.ConnectionError)


def test_retry_ignores_non_response_results():
    """Verify the decorator does not retry if the result is not a requests.Response (e.g., Playwright Page)."""
    call_count = 0

    @retry_with_backoff(max_retries=3, base_delay=0.0)
    def playwright_func():
        nonlocal call_count
        call_count += 1
        return {"status": "ok"} # Simulating a dict return

    result = playwright_func()
    assert result == {"status": "ok"}
    assert call_count == 1 # Should not retry