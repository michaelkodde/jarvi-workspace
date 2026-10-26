"""
LLM Client: Gold standard, daemon-safe communication with local Ollama instance.
"""
import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import requests.adapters
import urllib3.util.retry
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# --- Gold Standard: Module Constants ---
CONNECT_TIMEOUT = 60
READ_TIMEOUT = 180
MODEL_KEEP_ALIVE = "10m"
CONFIG_TTL = 60  # Cache config for 60 seconds to balance live updates vs disk I/O

# --- Gold Standard: Explicit Public API ---
__all__ = ["query_ollama", "shutdown", "verify_model_available", "OllamaAPIError", "OllamaResponse"]


class OllamaAPIError(Exception):
    """Custom exception for all Ollama client errors. Hides underlying library exceptions."""
    pass


@dataclass
class OllamaResponse:
    """Strongly typed response object for IDE autocomplete and type safety."""
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


# --- Gold Standard: Thread-Local Session with Zombie Connection Protection ---
_thread_local = threading.local()


def _get_session() -> requests.Session:
    """Lazily initializes a thread-safe persistent session with TCP retry protection."""
    if not hasattr(_thread_local, "session") or _thread_local.session is None:
        _thread_local.session = requests.Session()
        _thread_local.session.headers.update({"Content-Type": "application/json"})
        
        # Mount retry adapter to handle silent TCP drops (Zombie connections)
        retry_strategy = urllib3.util.retry.Retry(
            total=1,
            backoff_factor=1,
            status_forcelist=[502, 503, 504],
            allowed_methods=["POST"]  # Safe for local LLM inference
        )
        adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
        _thread_local.session.mount("http://", adapter)
        _thread_local.session.mount("https://", adapter)
        
    return _thread_local.session


# --- Gold Standard: TTL Config Cache ---
_config_cache = {"url": None, "timestamp": 0.0}


def _get_ollama_url() -> str:
    """Reads config.json with a 60-second TTL cache to prevent disk I/O bottlenecks."""
    now = time.time()
    if _config_cache["url"] and (now - _config_cache["timestamp"]) < CONFIG_TTL:
        return _config_cache["url"]
        
    try:
        current_dir = Path(__file__).resolve().parent
        for parent in current_dir.parents:
            if (parent / "config.json").exists():
                config_path = parent / "config.json"
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    url = config.get("ollama_url", "http://localhost:11434")
                    _config_cache["url"] = url
                    _config_cache["timestamp"] = now
                    return url
    except Exception as e:
        logger.warning("Failed to read config.json, using default Ollama URL: %s", e)
        
    return "http://localhost:11434"


def _should_retry(e: BaseException) -> bool:
    """Determines if an exception warrants a retry (Network drops, Timeouts, 5xx, 429)."""
    if isinstance(e, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
        return True
    if isinstance(e, requests.exceptions.HTTPError):
        if e.response is not None:
            return e.response.status_code in [429, 500, 502, 503, 504]
    return False


def verify_model_available(model: str) -> bool:
    """
    Dynamically queries Ollama to verify if a model is available.
    Used for service discovery and lazy-loaded routing to prevent VRAM thrashing.
    
    Args:
        model: The model name to check (e.g., "qwen2.5:7b").
        
    Returns:
        True if the model is available, False otherwise.
    """
    base_url = _get_ollama_url()
    try:
        # Use a short timeout for fast service discovery
        response = requests.get(f"{base_url}/api/tags", timeout=120)
        response.raise_for_status()
        models = [m["name"] for m in response.json().get("models", [])]
        # Check for exact match or if the requested model is a substring of an available model
        return any(model == m or model in m for m in models)
    except requests.exceptions.RequestException as e:
        logger.warning("Failed to verify model availability at %s: %s", base_url, e)
        return False


@retry(
    stop=stop_after_attempt(3), 
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception(_should_retry)
)
def query_ollama(
    model: str, 
    prompt: str, 
    system_prompt: str | None = None,
    images: list[str] | None = None,
    options: dict[str, Any] | None = None,
    timeout: tuple[int, int] | None = None
) -> OllamaResponse:
    """
    Gold Standard LLM Client.
    Returns a strongly typed OllamaResponse object.
    Swallows all library-specific exceptions and raises OllamaAPIError.
    
    Args:
        model: The model name (e.g., "qwen2.5:7b").
        prompt: The user prompt.
        system_prompt: The system prompt (playbook).
        images: Optional list of base64 encoded images.
        options: Optional model parameters (e.g., temperature, keep_alive).
        timeout: Tuple of (connect_timeout, read_timeout) in seconds.
        
    Returns:
        OllamaResponse containing the generated content and token usage.
        
    Raises:
        OllamaAPIError: If the API call fails or returns an error.
    """
    base_url = _get_ollama_url()
    url = f"{base_url}/api/chat"
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
        
    user_msg = {"role": "user", "content": prompt}
    if images:
        user_msg["images"] = images
    messages.append(user_msg)

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "keep_alive": MODEL_KEEP_ALIVE 
    }
    if options:
        payload["options"] = options
        
    logger.info("Querying Ollama [%s]: %d chars", model, len(prompt))
    
    actual_timeout = timeout or (CONNECT_TIMEOUT, READ_TIMEOUT)
    
    try:
        response = _get_session().post(url, json=payload, timeout=120)
        response.raise_for_status()
        
        try:
            data = response.json()
        except json.JSONDecodeError as e:
            raise OllamaAPIError(f"Ollama returned invalid JSON (Status {response.status_code})") from e
            
    except requests.exceptions.RequestException as e:
        raise OllamaAPIError(f"Network/HTTP error communicating with Ollama: {e}") from e

    if "error" in data:
        raise OllamaAPIError(f"Ollama API Error: {data['error']}")
        
    if "message" not in data or "content" not in data["message"]:
        raise OllamaAPIError("Malformed Ollama response structure")

    content = data["message"]["content"]
    
    # SECURITY FIX: Never log prompt content, even partially, to prevent PII leaks
    if not content:
        logger.warning("Ollama [%s] returned empty content.", model)

    prompt_tokens = data.get("prompt_eval_count", 0)
    completion_tokens = data.get("eval_count", 0)
    
    logger.info("Ollama [%s] response: %d chars | Tokens: %d", model, len(content), prompt_tokens + completion_tokens)
    
    return OllamaResponse(
        content=content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens
    )


def shutdown():
    """Gracefully closes all thread-local connection pools. Call this on application exit."""
    if hasattr(_thread_local, "session") and _thread_local.session is not None:
        _thread_local.session.close()
        _thread_local.session = None
        logger.info("Ollama client session closed.")