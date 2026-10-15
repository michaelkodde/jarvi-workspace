"""
Automation Guard: Daemon-safe rate limiting and retry logic for web scraping.
"""
import logging
import random
import time
from typing import Any

import requests
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

__all__ = ["human_delay", "retry_with_backoff"]


def human_delay() -> None:
    """
    Introduces a randomized human-like delay between 2.0 and 5.0 seconds.
    Prevents bot-detection triggers during synchronous scraping operations.
    """
    delay = random.uniform(2.0, 5.0)
    logger.debug("Applying human delay of %.2f seconds", delay)
    time.sleep(delay)


def _is_retryable_exception(exception: Exception) -> bool:
    """
    Determines if an exception warrants a retry based on HTTP status codes.
    Only retries on 429 (Too Many Requests) and 5xx server errors.
    """
    if isinstance(exception, requests.exceptions.RequestException):
        if hasattr(exception, "response") and exception.response is not None:
            return exception.response.status_code in [429, 500, 502, 503, 504]
    return False


retry_with_backoff = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception(_is_retryable_exception),
    reraise=True,
)