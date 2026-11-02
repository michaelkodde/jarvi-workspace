# src/utils/llm_client.py
"""
Gold Standard LLM Client.
Features: Native Sync/Async I/O, Connection Pooling, Thread-Safe Circuit Breaker, 
and Protocol-based abstraction for future model providers.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol

import httpx

logger = logging.getLogger(__name__)

# ============================================================================
# LEGACY EXCEPTIONS (For backward compatibility with BaseAgent)
# ============================================================================

class OllamaAPIError(Exception):
    """Legacy exception for backward compatibility with BaseAgent."""
    def __init__(self, message: str, status_code: int = 0):
        super().__init__(message)
        self.status_code = status_code

# ============================================================================
# PROTOCOLS
# ============================================================================

class LLMResponse(Protocol):
    content: str
    model: str
    usage: Dict[str, int]

class LLMProviderProtocol(Protocol):
    """Contract for any LLM provider (Ollama, vLLM, local API, etc.)."""
    def generate(
        self, 
        model: str, 
        prompt: str, 
        system_prompt: Optional[str] = None, 
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> LLMResponse:
        ...

# ============================================================================
# THREAD-SAFE CIRCUIT BREAKER
# ============================================================================

class CircuitState(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

@dataclass
class CircuitBreaker:
    failure_threshold: int = 3
    recovery_timeout: float = 10.0
    
    state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    failure_count: int = field(default=0, init=False)
    last_failure_time: float = field(default=0.0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def allow_request(self) -> bool:
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    return True
                return False
            return True  # HALF_OPEN

    def record_success(self):
        with self._lock:
            self.failure_count = 0
            self.state = CircuitState.CLOSED

    def record_failure(self):
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit Breaker OPENED after {self.failure_count} failures.")

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass(frozen=True)
class OllamaResponse:
    content: str
    model: str
    usage: Dict[str, int]

# ============================================================================
# NATIVE CLIENTS
# ============================================================================

class BaseOllamaClient:
    """Shared configuration and circuit breaker logic."""
    def __init__(self, base_url: str = "http://localhost:11434", timeout: float = 120.0):
        self.base_url = base_url
        self.timeout = timeout
        self.circuit_breaker = CircuitBreaker()

    def _build_payload(
        self, 
        model: str, 
        prompt: str, 
        system_prompt: Optional[str] = None, 
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": kwargs.get("options", {"temperature": 0.6})
        }
        if system_prompt:
            payload["system"] = system_prompt
        if tools:
            payload["tools"] = tools
        return payload

    def _handle_response(self, response: httpx.Response, model: str) -> OllamaResponse:
        response.raise_for_status()
        data = response.json()
        self.circuit_breaker.record_success()
        return OllamaResponse(
            content=data.get("response", ""),
            model=model,
            usage={"eval_count": data.get("eval_count", 0), "prompt_eval_count": data.get("prompt_eval_count", 0)}
        )

    def _handle_error(self, e: Exception):
        self.circuit_breaker.record_failure()
        logger.error(f"Ollama request failed: {e}")
        raise

class SyncOllamaClient(BaseOllamaClient):
    """Production-grade synchronous client with connection pooling."""
    def __init__(self, base_url: str = "http://localhost:11434", timeout: float = 120.0, max_connections: int = 20):
        super().__init__(base_url, timeout)
        self._client = httpx.Client(
            base_url=base_url, 
            timeout=timeout,
            limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=5)
        )

    def generate(self, model: str, prompt: str, system_prompt: Optional[str] = None, tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> OllamaResponse:
        if not self.circuit_breaker.allow_request():
            raise RuntimeError("Ollama service is currently unavailable (Circuit Breaker OPEN).")
        
        try:
            response = self._client.post("/api/generate", json=self._build_payload(model, prompt, system_prompt, tools, **kwargs))
            return self._handle_response(response, model)
        except httpx.HTTPStatusError as e:
            raise OllamaAPIError(f"Ollama returned status {e.response.status_code}", e.response.status_code) from e
        except Exception as e:
            self._handle_error(e)

    def close(self):
        self._client.close()

class AsyncOllamaClient(BaseOllamaClient):
    """Production-grade asynchronous client for FastAPI/GUI."""
    def __init__(self, base_url: str = "http://localhost:11434", timeout: float = 120.0):
        super().__init__(base_url, timeout)
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def generate(self, model: str, prompt: str, system_prompt: Optional[str] = None, tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> OllamaResponse:
        if not self.circuit_breaker.allow_request():
            raise RuntimeError("Ollama service is currently unavailable (Circuit Breaker OPEN).")
        
        try:
            response = await self._client.post("/api/generate", json=self._build_payload(model, prompt, system_prompt, tools, **kwargs))
            return self._handle_response(response, model)
        except httpx.HTTPStatusError as e:
            raise OllamaAPIError(f"Ollama returned status {e.response.status_code}", e.response.status_code) from e
        except Exception as e:
            self._handle_error(e)

    async def close(self):
        await self._client.aclose()

# ============================================================================
# ADAPTERS & GLOBAL INSTANCES
# ============================================================================

class AgentClientAdapter:
    """
    Bridges the new Gold Standard SyncOllamaClient with the legacy BaseAgent callable signature.
    BaseAgent expects: client(prompt="...", system_prompt="...")
    """
    def __init__(self, client: SyncOllamaClient, model_name: str):
        self._client = client
        self._model_name = model_name

    def __call__(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> OllamaResponse:
        kwargs.pop('model', None) # BaseAgent sometimes passes model, we override it with the bound model
        return self._client.generate(
            model=self._model_name, 
            prompt=prompt, 
            system_prompt=system_prompt, 
            **kwargs
        )

_sync_client: Optional[SyncOllamaClient] = None
_async_client: Optional[AsyncOllamaClient] = None

def get_sync_ollama_client() -> SyncOllamaClient:
    global _sync_client
    if _sync_client is None:
        _sync_client = SyncOllamaClient()
    return _sync_client

def get_async_ollama_client() -> AsyncOllamaClient:
    global _async_client
    if _async_client is None:
        _async_client = AsyncOllamaClient()
    return _async_client

def query_ollama(model: str, prompt: str, system_prompt: str = None, **kwargs) -> OllamaResponse:
    """Legacy synchronous entry point for Router compatibility."""
    return get_sync_ollama_client().generate(model=model, prompt=prompt, system_prompt=system_prompt, **kwargs)