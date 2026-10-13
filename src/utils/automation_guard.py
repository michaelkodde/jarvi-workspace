"""
Automation Guard: Thread-safe utilities for human-like pacing and resilient web automation.
"""
import json
import logging
import random
import time
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_result,
    stop_after_attempt,
    wait_random_exponential,
    before_sleep_log,
    RetryError,
)

logger = logging.getLogger(__name__)

__all__ = ["AutomationError", "human_delay", "retry_with_backoff"]


class AutomationError(Exception):
    """Custom exception for automation failures after exhausting retries."""
    pass


@dataclass
class _AutomationConfig:
    """Strongly typed configuration for automation retries and delays."""
    max_retries: int
    base_delay: float


def _load_config() -> _AutomationConfig:
    """
    Loads automation configuration from config.json.
    
    Falls back to safe hardcoded defaults if the file is missing, invalid, 
    or inaccessible to prevent startup crashes during module import.
    
    Returns:
        _AutomationConfig with resolved retry and delay parameters.
    """
    try:
        config_path = Path("config.json")
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                auto_config = data.get("automation", {})
                return _AutomationConfig(
                    max_retries=int(auto_config.get("max_retries", 3)),
                    base_delay=float(auto_config.get("base_delay", 2.0)),
                )
    except Exception as e:
        logger.debug("Failed to load automation config, using defaults: %s", e)
        
    return _AutomationConfig(max_retries=3, base_delay=2.0)


def human_delay(min_sec: float = 2.0, max_sec: float = 5.0) -> None:
    """
    Pauses execution for a random duration to simulate human-like pacing.
    
    Args:
        min_sec: Minimum delay in seconds.
        max_sec: Maximum delay in seconds.
    """
    if min_sec > max_sec:
        raise ValueError("min_sec cannot be greater than max_sec.")
        
    delay = random.uniform(min_sec, max_sec)
    logger.info("Pacing: sleeping for %.2f seconds.", delay)
    time.sleep(delay)


def retry_with_backoff(
    max_retries: Optional[int] = None, 
    base_delay: Optional[float] = None
) -> Callable:
    """
    Decorator factory that applies resilient, human-like exponential backoff to a function.
    
    Retries on HTTP 429, 500, 502, 503, 504 status codes (if returning requests.Response)
    and standard network/timeout exceptions.
    
    Args:
        max_retries: Maximum number of attempts. Falls back to config.json or default (3).
        base_delay: Base multiplier for exponential wait. Falls back to config.json or default (2.0).
        
    Returns:
        A decorator that wraps the target function with retry logic.
        
    Raises:
        AutomationError: If all retry attempts are exhausted.
    """
    config = _load_config()
    final_max_retries = max_retries if max_retries is not None else config.max_retries
    final_base_delay = base_delay if base_delay is not None else config.base_delay

    def _should_retry_result(result: Any) -> bool:
        """Predicate for tenacity: retries if result is a failing HTTP Response."""
        if isinstance(result, requests.Response):
            return result.status_code in {429, 500, 502, 503, 504}
        return False

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tenacity_retry = retry(
                retry=retry_if_result(_should_retry_result) | 
                      retry_if_exception_type((requests.exceptions.RequestException, TimeoutError)),
                wait=wait_random_exponential(multiplier=final_base_delay, max=60.0),
                stop=stop_after_attempt(final_max_retries),
                before_sleep=before_sleep_log(logger, logging.WARNING),
                reraise=False,
            )
            
            decorated_func = tenacity_retry(func)
            
            try:
                return decorated_func(*args, **kwargs)
            except RetryError as e:
                last_attempt = e.last_attempt
                underlying_exception = last_attempt.exception()
                
                if underlying_exception:
                    logger.error(
                        "Automation failed after %d attempts due to exception: %s", 
                        final_max_retries, underlying_exception
                    )
                    raise AutomationError(f"Automation failed after {final_max_retries} attempts.") from underlying_exception
                else:
                    logger.error(
                        "Automation failed after %d attempts due to persistent bad result (e.g., 500).", 
                        final_max_retries
                    )
                    raise AutomationError(f"Automation failed after {final_max_retries} attempts.") from e

        return wrapper
    return decorator