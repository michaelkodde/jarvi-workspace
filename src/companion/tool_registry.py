# src/companion/tool_registry.py
"""MCP-compliant tool dispatcher for the Jarvi Companion.

Wraps BaseAgent instances, the Router, and the DiscoveryEngine into a
uniform, schema-validated callable interface. Stateless, thread-safe,
and daemon-safe.

This module NEVER logs, prints, or returns parameter contents (PII protection).
Exception messages from underlying agents are stripped — only the exception
type name is exposed to callers.
"""
from __future__ import annotations

import json
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import Any

from src.agents.base_agent import BaseAgent
from src.agents.router import Router
from src.companion.discovery import DiscoveryEngine

logger = logging.getLogger(__name__)

__all__ = [
    "ErrorCode",
    "ToolRegistry",
    "ToolRegistryError",
    "ToolResult",
    "ToolSchema",
]

# Compiled once at module load — zero per-call overhead.
_TOOL_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_MAX_TOOL_NAME_LEN = 64
_MAX_PARAMS_BYTES = 10 * 1024 * 1024  # 10 MB
_DEFAULT_TIMEOUT_SEC = 300.0
_POOL_MAX_WORKERS = 4


class ErrorCode:
    """Machine-readable error codes (MCP-aligned)."""

    UNKNOWN_TOOL = "UNKNOWN_TOOL"
    MISSING_PARAM = "MISSING_PARAM"
    INVALID_TOOL_NAME = "INVALID_TOOL_NAME"
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
    AGENT_FAILED = "AGENT_FAILED"
    TIMEOUT = "TIMEOUT"
    INTERNAL = "INTERNAL"


class ToolRegistryError(Exception):
    """Raised when a tool invocation fails. Always chains from the original exception."""

    def __init__(self, message: str, error_code: str = ErrorCode.INTERNAL) -> None:
        super().__init__(message)
        self.error_code = error_code


@dataclass(frozen=True)
class ToolResult:
    """Standardized result from any tool invocation.

    Attributes:
        status: "success" or "error".
        output: Tool-specific JSON-serializable payload on success, None on error.
            Typed as Any because MCP requires JSON-compatible dicts, not Python types.
        error: Human-readable error message on failure, None on success.
            Exception messages from underlying agents are stripped to type names only.
        error_code: Machine-readable error code on failure, None on success.
        execution_time_ms: Wall-clock time for the call, always populated.
    """

    status: str
    output: Any | None
    error: str | None
    error_code: str | None
    execution_time_ms: int


@dataclass(frozen=True)
class ToolSchema:
    """MCP-compliant tool schema.

    Attributes:
        name: Tool identifier (lowercase, matches call_tool input).
        description: Human-readable description for UI/LLM consumption.
        input_schema: JSON Schema describing valid params.
        output_schema: JSON Schema describing the tool's output.
        annotations: MCP metadata (title, readOnlyHint, etc.). Immutable by convention.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    annotations: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Per-tool output schemas (MCP-compliant JSON Schema)
# ---------------------------------------------------------------------------

_TOKEN_USAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "prompt_tokens": {"type": "integer"},
        "completion_tokens": {"type": "integer"},
        "total_tokens": {"type": "integer"},
    },
    "required": ["prompt_tokens", "completion_tokens", "total_tokens"],
}

_AGENT_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "agent": {"type": "string"},
        "artifact": {"type": "string"},
        "token_usage": _TOKEN_USAGE_SCHEMA,
    },
    "required": ["agent", "artifact", "token_usage"],
}

_ROUTER_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "target_agent": {"type": "string"},
        "target_model": {"type": "string"},
        "cleaned_prompt": {"type": "string"},
        "confidence": {"type": "number"},
        "token_usage": _TOKEN_USAGE_SCHEMA,
    },
    "required": ["target_agent", "target_model", "cleaned_prompt", "confidence", "token_usage"],
}

_DISCOVERY_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "agent_count": {"type": "integer"},
        "agents": {"type": "array", "items": {"type": "string"}},
        "utils_count": {"type": "integer"},
        "pipelines_count": {"type": "integer"},
        "missing_pipeline_agents": {"type": "array", "items": {"type": "string"}},
        "scan_errors": {"type": "array", "items": {"type": "string"}},
        "status": {"type": "string", "enum": ["healthy", "degraded"]},
    },
    "required": [
        "agent_count", "agents", "utils_count", "pipelines_count",
        "missing_pipeline_agents", "scan_errors", "status",
    ],
}


def _normalize_token_usage(token_usage: Any) -> dict[str, int]:
    """Convert token_usage to a dict, handling both dict and dataclass forms.

    Args:
        token_usage: Either a dict or an object with a to_dict() method.

    Returns:
        A dict with prompt_tokens, completion_tokens, total_tokens.

    Raises:
        ToolRegistryError: If the structure is invalid.
    """
    if hasattr(token_usage, "to_dict") and callable(token_usage.to_dict):
        token_usage = token_usage.to_dict()
    if not isinstance(token_usage, dict):
        raise ToolRegistryError(
            "token_usage must be a dict or have a to_dict() method",
            ErrorCode.AGENT_FAILED,
        )
    return token_usage


def _check_params_size(params: dict[str, Any]) -> int:
    """Fast-path size check for params.

    Uses sys.getsizeof for a shallow estimate. Falls back to full JSON
    serialization only when the estimate is within 2x of the limit.

    Args:
        params: The parameters dict to measure.

    Returns:
        The serialized size in bytes.

    Raises:
        ToolRegistryError: If params cannot be serialized or exceeds the limit.
    """
    if len(params) < 1000:
        shallow_estimate = sys.getsizeof(params) + sum(
            sys.getsizeof(k) + sys.getsizeof(v) for k, v in params.items()
        )
        if shallow_estimate < _MAX_PARAMS_BYTES // 4:
            return shallow_estimate

    try:
        serialized = json.dumps(params)
    except (TypeError, ValueError, OverflowError) as e:
        raise ToolRegistryError(
            f"params not JSON-serializable: {type(e).__name__}",
            ErrorCode.MISSING_PARAM,
        ) from e

    size = len(serialized.encode("utf-8"))
    if size > _MAX_PARAMS_BYTES:
        raise ToolRegistryError(
            f"params size {size} exceeds {_MAX_PARAMS_BYTES} bytes",
            ErrorCode.PAYLOAD_TOO_LARGE,
        )
    return size


class ToolRegistry:
    """MCP-compliant dispatcher for Jarvi tools.

    Wraps BaseAgent instances, the Router, and the DiscoveryEngine into a
    uniform, schema-validated callable interface. Stateless, thread-safe,
    and daemon-safe.

    Args:
        agents: Dict mapping agent names to BaseAgent instances.
        router: Router instance for intent classification.
        discovery_engine: DiscoveryEngine instance for codebase scanning.
        timeout_sec: Per-call timeout in seconds. Default 300. Calls exceeding
            this return ToolResult with error_code=TIMEOUT.

    Raises:
        ToolRegistryError: If any dependency is invalid or missing required methods.
    """

    def __init__(
        self,
        agents: dict[str, BaseAgent],
        router: Router,
        discovery_engine: DiscoveryEngine,
        timeout_sec: float = _DEFAULT_TIMEOUT_SEC,
    ) -> None:
        if not isinstance(agents, dict):
            raise ToolRegistryError("agents must be a dict", ErrorCode.INTERNAL)
        if not isinstance(router, Router):
            raise ToolRegistryError("router must be a Router instance", ErrorCode.INTERNAL)
        if not isinstance(discovery_engine, DiscoveryEngine):
            raise ToolRegistryError(
                "discovery_engine must be a DiscoveryEngine instance", ErrorCode.INTERNAL
            )
        if not isinstance(timeout_sec, (int, float)) or timeout_sec <= 0:
            raise ToolRegistryError("timeout_sec must be a positive number", ErrorCode.INTERNAL)

        for name, agent in agents.items():
            if not hasattr(agent, "execute") or not callable(agent.execute):
                raise ToolRegistryError(
                    f"Agent '{name}' missing execute() method", ErrorCode.INTERNAL
                )

        if not hasattr(router, "route") or not callable(router.route):
            raise ToolRegistryError("Router missing route() method", ErrorCode.INTERNAL)

        if not hasattr(discovery_engine, "scan") or not callable(discovery_engine.scan):
            raise ToolRegistryError("DiscoveryEngine missing scan() method", ErrorCode.INTERNAL)

        self._agents: dict[str, BaseAgent] = {
            name.lower(): agent for name, agent in agents.items()
        }
        self._router = router
        self._discovery = discovery_engine
        self._timeout_sec = float(timeout_sec)
        self._schemas: dict[str, ToolSchema] = self._build_schemas()
        # Shared daemon thread pool for timeout enforcement.
        # Daemon threads die with the process — no leak risk.
        self._pool = ThreadPoolExecutor(
            max_workers=_POOL_MAX_WORKERS,
            thread_name_prefix="tool-",
        )

    def list_tools(self) -> list[str]:
        """Return a sorted list of all registered tool names.

        Returns:
            Sorted list of lowercase tool name strings.
        """
        return sorted(self._schemas.keys())

    def get_tool_schema(self, tool_name: str) -> ToolSchema | None:
        """Return the MCP schema for a tool, or None if unknown.

        Args:
            tool_name: Tool name (case-insensitive).

        Returns:
            ToolSchema if found, None otherwise.
        """
        if not isinstance(tool_name, str):
            return None
        return self._schemas.get(tool_name.lower())

    def call_tool(self, tool_name: str, params: dict[str, Any]) -> ToolResult:
        """Invoke a tool with the given parameters.

        Args:
            tool_name: Tool name (case-insensitive, max 64 chars,
                must match ^[a-z][a-z0-9_]{0,63}$).
            params: Tool parameters (max 10MB when JSON-serialized).
                NEVER logged or persisted by this registry.

        Returns:
            ToolResult with status, output, error, error_code, execution_time_ms.
            On success: status="success", output populated, error=None.
            On failure: status="error", output=None, error populated.
        """
        start = time.perf_counter()

        if not isinstance(tool_name, str) or not _TOOL_NAME_RE.match(tool_name):
            return self._error_result(
                f"Invalid tool_name: must match {_TOOL_NAME_RE.pattern}",
                ErrorCode.INVALID_TOOL_NAME,
                start,
            )

        if not isinstance(params, dict):
            return self._error_result(
                "params must be a dict", ErrorCode.MISSING_PARAM, start
            )

        try:
            _check_params_size(params)
        except ToolRegistryError as e:
            return self._error_result(str(e), e.error_code, start)

        tool_key = tool_name.lower()
        if tool_key not in self._schemas:
            return self._error_result(
                f"Unknown tool: {tool_key}. Available: {self.list_tools()}",
                ErrorCode.UNKNOWN_TOOL,
                start,
            )

        try:
            future = self._pool.submit(self._dispatch, tool_key, params)
            output = future.result(timeout=self._timeout_sec)
        except FuturesTimeoutError:
            logger.warning(
                "Tool %s exceeded timeout of %.1fs", tool_key, self._timeout_sec
            )
            return self._error_result(
                f"Tool '{tool_key}' exceeded timeout of {self._timeout_sec}s",
                ErrorCode.TIMEOUT,
                start,
            )
        except ToolRegistryError as e:
            return self._error_result(str(e), e.error_code, start)
        except (SystemExit, KeyboardInterrupt):
            raise
        except Exception as e:
            logger.exception("Unexpected error in tool %s", tool_key)
            return self._error_result(
                f"Internal error: {type(e).__name__}", ErrorCode.INTERNAL, start
            )

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info("Tool %s completed in %dms", tool_key, elapsed_ms)
        return ToolResult(
            status="success",
            output=output,
            error=None,
            error_code=None,
            execution_time_ms=elapsed_ms,
        )

    def _dispatch(self, tool_key: str, params: dict[str, Any]) -> dict[str, Any]:
        """Internal dispatcher — runs in the thread pool.

        Args:
            tool_key: Lowercase tool name.
            params: Validated parameters dict.

        Returns:
            Tool-specific output dict.

        Raises:
            ToolRegistryError: On any dispatch failure.
        """
        if tool_key == "router":
            return self._call_router(params)
        if tool_key == "discovery":
            return self._call_discovery(params)
        return self._call_agent(tool_key, params)

    def _error_result(self, message: str, code: str, start: float) -> ToolResult:
        """Build an error ToolResult with timing.

        Args:
            message: Human-readable error message (must not contain PII).
            code: Machine-readable error code.
            start: perf_counter start time.

        Returns:
            ToolResult with status="error".
        """
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.warning("Tool call failed [%s]: %s", code, message)
        return ToolResult(
            status="error",
            output=None,
            error=message,
            error_code=code,
            execution_time_ms=elapsed_ms,
        )

    def _call_agent(self, agent_name: str, params: dict[str, Any]) -> dict[str, Any]:
        """Execute a BaseAgent and format the result.

        Args:
            agent_name: Lowercase agent name.
            params: Dict with required 'task_prompt' and optional 'context'.

        Returns:
            Dict with agent, artifact, token_usage.

        Raises:
            ToolRegistryError: On missing params or agent failure.
        """
        agent = self._agents[agent_name]
        task_prompt = params.get("task_prompt")
        if not task_prompt or not isinstance(task_prompt, str):
            raise ToolRegistryError(
                "Missing or invalid required parameter: 'task_prompt'",
                ErrorCode.MISSING_PARAM,
            )
        context = params.get("context")
        if context is not None and not isinstance(context, str):
            raise ToolRegistryError(
                "Parameter 'context' must be a string or null",
                ErrorCode.MISSING_PARAM,
            )

        try:
            result = agent.execute(task_prompt, context)
        except Exception as e:
            logger.exception("Agent '%s' raised %s", agent_name, type(e).__name__)
            raise ToolRegistryError(
                f"Agent '{agent_name}' failed: {type(e).__name__}",
                ErrorCode.AGENT_FAILED,
            ) from e

        if result is None:
            raise ToolRegistryError(
                f"Agent '{agent_name}' returned None", ErrorCode.AGENT_FAILED
            )

        try:
            token_usage = _normalize_token_usage(result.token_usage)
        except ToolRegistryError as e:
            raise ToolRegistryError(
                f"Agent '{agent_name}' returned invalid token_usage",
                ErrorCode.AGENT_FAILED,
            ) from e

        return {
            "agent": agent_name,
            "artifact": result.artifact,
            "token_usage": token_usage,
        }

    def _call_router(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute the Router and format the result.

        Args:
            params: Dict with required 'raw_prompt'.

        Returns:
            Dict with target_agent, target_model, cleaned_prompt, confidence, token_usage.

        Raises:
            ToolRegistryError: On missing params or router failure.
        """
        raw_prompt = params.get("raw_prompt")
        if not raw_prompt or not isinstance(raw_prompt, str):
            raise ToolRegistryError(
                "Missing or invalid required parameter: 'raw_prompt'",
                ErrorCode.MISSING_PARAM,
            )

        try:
            result = self._router.route(raw_prompt)
        except Exception as e:
            logger.exception("Router raised %s", type(e).__name__)
            raise ToolRegistryError(
                f"Router failed: {type(e).__name__}",
                ErrorCode.AGENT_FAILED,
            ) from e

        if result is None:
            raise ToolRegistryError(
                "Router returned None", ErrorCode.AGENT_FAILED
            )

        try:
            token_usage = _normalize_token_usage(result.token_usage)
        except ToolRegistryError as e:
            raise ToolRegistryError(
                "Router returned invalid token_usage",
                ErrorCode.AGENT_FAILED,
            ) from e

        return {
            "target_agent": result.target_agent,
            "target_model": result.target_model,
            "cleaned_prompt": result.cleaned_prompt,
            "confidence": result.confidence,
            "token_usage": token_usage,
        }

    def _call_discovery(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute the Discovery Engine and format the result.

        Args:
            params: Dict with optional 'force' (bool, default False).

        Returns:
            Dict with agent_count, agents, utils_count, pipelines_count,
            missing_pipeline_agents, scan_errors, status.

        Raises:
            ToolRegistryError: On invalid params or discovery failure.
        """
        force = params.get("force", False)
        if not isinstance(force, bool):
            raise ToolRegistryError(
                "Parameter 'force' must be a boolean", ErrorCode.MISSING_PARAM
            )

        try:
            registry = self._discovery.scan(force=force)
        except Exception as e:
            logger.exception("Discovery raised %s", type(e).__name__)
            raise ToolRegistryError(
                f"Discovery failed: {type(e).__name__}",
                ErrorCode.AGENT_FAILED,
            ) from e

        has_errors = any(e.severity == "error" for e in registry.scan_errors)
        return {
            "agent_count": len(registry.agents),
            "agents": sorted(registry.agents.keys()),
            "utils_count": len(registry.utils),
            "pipelines_count": len(registry.pipelines),
            "missing_pipeline_agents": list(registry.missing_pipeline_agents),
            "scan_errors": [e.file for e in registry.scan_errors if e.severity == "error"],
            "status": "degraded" if has_errors else "healthy",
        }

    def _build_schemas(self) -> dict[str, ToolSchema]:
        """Build MCP-compliant schemas for all registered tools.

        Returns:
            Dict mapping tool name to ToolSchema.
        """
        schemas: dict[str, ToolSchema] = {}

        for name in self._agents:
            schemas[name] = ToolSchema(
                name=name,
                description=f"Execute the {name.capitalize()} agent.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "task_prompt": {
                            "type": "string",
                            "description": "The task to perform.",
                        },
                        "context": {
                            "type": "string",
                            "description": "Optional context for the task.",
                        },
                    },
                    "required": ["task_prompt"],
                },
                output_schema=_AGENT_OUTPUT_SCHEMA,
                annotations={
                    "title": name.capitalize(),
                    "readOnlyHint": False,
                    "destructiveHint": False,
                    "idempotentHint": True,
                    "openWorldHint": name in ("scout", "publisher"),
                },
            )

        schemas["router"] = ToolSchema(
            name="router",
            description="Classify user intent and route to the appropriate agent.",
            input_schema={
                "type": "object",
                "properties": {
                    "raw_prompt": {
                        "type": "string",
                        "description": "The user's raw input.",
                    },
                },
                "required": ["raw_prompt"],
            },
            output_schema=_ROUTER_OUTPUT_SCHEMA,
            annotations={
                "title": "Router",
                "readOnlyHint": True,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": False,
            },
        )

        schemas["discovery"] = ToolSchema(
            name="discovery",
            description="Scan the codebase and report agent health, playbooks, and pipelines.",
            input_schema={
                "type": "object",
                "properties": {
                    "force": {
                        "type": "boolean",
                        "description": "Force a full re-scan, bypassing cache.",
                        "default": False,
                    },
                },
                "required": [],
            },
            output_schema=_DISCOVERY_OUTPUT_SCHEMA,
            annotations={
                "title": "Discovery Engine",
                "readOnlyHint": True,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": False,
            },
        )

        return schemas