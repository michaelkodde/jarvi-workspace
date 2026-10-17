# tests/test_tool_registry.py
"""Tests for the MCP-compliant Tool Registry."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.agents.base_agent import AgentResult, BaseAgent
from src.agents.router import Router, RoutingResult
from src.companion.discovery import DiscoveryEngine
from src.companion.tool_registry import (
    ErrorCode,
    ToolRegistry,
    ToolRegistryError,
    ToolResult,
    ToolSchema,
)


class MockAgent(BaseAgent):
    def __init__(
        self,
        name: str,
        raise_on_execute: Exception | None = None,
        return_none: bool = False,
        sleep_sec: float = 0.0,
    ) -> None:
        self.agent_name = name
        self.model_name = "test-model"
        self._raise = raise_on_execute
        self._return_none = return_none
        self._sleep_sec = sleep_sec

    def execute(self, task_prompt: str, context: str | None = None) -> AgentResult:
        if self._sleep_sec > 0:
            time.sleep(self._sleep_sec)
        if self._raise is not None:
            raise self._raise
        if self._return_none:
            return None  # type: ignore[return-value]
        return AgentResult(
            agent_name=self.agent_name,
            artifact=f"Mocked {self.agent_name} output for: {task_prompt}",
            token_usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )


@pytest.fixture
def mock_router() -> Router:
    router = MagicMock(spec=Router)
    router.route.return_value = RoutingResult(
        target_agent="scout",
        target_model="qwen2.5:14b",
        cleaned_prompt="Find trending niches",
        confidence=0.95,
        token_usage={"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
    )
    return router


@pytest.fixture
def mock_discovery(tmp_path: Path) -> DiscoveryEngine:
    (tmp_path / "src" / "agents").mkdir(parents=True)
    (tmp_path / "brain" / "skills").mkdir(parents=True)
    return DiscoveryEngine(tmp_path, force=True)


@pytest.fixture
def registry(mock_router: Router, mock_discovery: DiscoveryEngine) -> ToolRegistry:
    agents = {
        "scout": MockAgent("Scout"),
        "creator": MockAgent("Creator"),
    }
    return ToolRegistry(agents=agents, router=mock_router, discovery_engine=mock_discovery)


def test_call_agent_success(registry: ToolRegistry) -> None:
    result = registry.call_tool("scout", {"task_prompt": "Find products", "context": "Non-toxic"})
    assert isinstance(result, ToolResult)
    assert result.status == "success"
    assert result.error is None
    assert result.error_code is None
    assert result.execution_time_ms >= 0
    assert "Mocked Scout output" in result.output["artifact"]
    assert result.output["token_usage"]["total_tokens"] == 30


def test_call_router_success(registry: ToolRegistry) -> None:
    result = registry.call_tool("router", {"raw_prompt": "Find trending niches"})
    assert result.status == "success"
    assert result.output["target_agent"] == "scout"
    assert result.output["confidence"] == 0.95
    assert "cleaned_prompt" in result.output


def test_call_discovery_success(registry: ToolRegistry) -> None:
    result = registry.call_tool("discovery", {"force": True})
    assert result.status == "success"
    assert "agent_count" in result.output
    assert result.output["status"] in ("healthy", "degraded")


def test_list_tools_returns_sorted(registry: ToolRegistry) -> None:
    tools = registry.list_tools()
    assert tools == sorted(tools)
    assert {"scout", "creator", "router", "discovery"} <= set(tools)


def test_get_tool_schema_returns_mcp_schema(registry: ToolRegistry) -> None:
    schema = registry.get_tool_schema("scout")
    assert isinstance(schema, ToolSchema)
    assert schema.name == "scout"
    assert schema.input_schema["type"] == "object"
    assert schema.output_schema["type"] == "object"
    assert "task_prompt" in schema.input_schema["required"]
    assert "annotations" in schema.__dataclass_fields__
    assert "readOnlyHint" in schema.annotations


def test_get_tool_schema_unknown_returns_none(registry: ToolRegistry) -> None:
    assert registry.get_tool_schema("nonexistent") is None


def test_get_tool_schema_non_string_returns_none(registry: ToolRegistry) -> None:
    assert registry.get_tool_schema(123) is None  # type: ignore[arg-type]


def test_case_insensitive_tool_lookup(registry: ToolRegistry) -> None:
    # Uppercase should be normalized to lowercase and succeed
    result = registry.call_tool("SCOUT", {"task_prompt": "x"})
    assert result.status == "success"

    # Mixed case should also work
    result = registry.call_tool("Scout", {"task_prompt": "x"})
    assert result.status == "success"

    # Router in uppercase
    result = registry.call_tool("ROUTER", {"raw_prompt": "x"})
    assert result.status == "success"


def test_unknown_tool_returns_error(registry: ToolRegistry) -> None:
    result = registry.call_tool("nonexistent", {})
    assert result.status == "error"
    assert result.error_code == ErrorCode.UNKNOWN_TOOL
    assert "Unknown tool" in result.error


def test_invalid_tool_name_rejected(registry: ToolRegistry) -> None:
    # Invalid names: empty, starts with digit, contains dash/space, too long
    # Note: "UPPER" is now VALID (case-insensitive) and gets normalized
    for bad_name in ("", "1starts_with_digit", "has-dash", "has space", "a" * 65):
        result = registry.call_tool(bad_name, {})
        assert result.status == "error"
        assert result.error_code == ErrorCode.INVALID_TOOL_NAME


def test_missing_required_param_returns_error(registry: ToolRegistry) -> None:
    result = registry.call_tool("scout", {})
    assert result.status == "error"
    assert result.error_code == ErrorCode.MISSING_PARAM


def test_invalid_context_type_returns_error(registry: ToolRegistry) -> None:
    result = registry.call_tool("scout", {"task_prompt": "x", "context": 123})
    assert result.status == "error"
    assert result.error_code == ErrorCode.MISSING_PARAM


def test_non_dict_params_rejected(registry: ToolRegistry) -> None:
    result = registry.call_tool("scout", "not a dict")  # type: ignore[arg-type]
    assert result.status == "error"
    assert result.error_code == ErrorCode.MISSING_PARAM


def test_payload_too_large_rejected(registry: ToolRegistry) -> None:
    huge = {"data": "x" * (11 * 1024 * 1024)}
    result = registry.call_tool("scout", {"task_prompt": "x", "extra": huge})
    assert result.status == "error"
    assert result.error_code == ErrorCode.PAYLOAD_TOO_LARGE


def test_non_serializable_params_rejected(registry: ToolRegistry) -> None:
    result = registry.call_tool("scout", {"task_prompt": "x", "bad": object()})
    assert result.status == "error"
    assert result.error_code == ErrorCode.MISSING_PARAM


def test_agent_exception_chained_with_message_stripped(
    mock_router: Router, mock_discovery: DiscoveryEngine
) -> None:
    pii_secret = "super-secret-user-data-12345"
    failing_agent = MockAgent("Failing", raise_on_execute=ValueError(f"boom: {pii_secret}"))
    reg = ToolRegistry(
        agents={"failing": failing_agent},
        router=mock_router,
        discovery_engine=mock_discovery,
    )
    result = reg.call_tool("failing", {"task_prompt": "x"})
    assert result.status == "error"
    assert result.error_code == ErrorCode.AGENT_FAILED
    assert "ValueError" in result.error
    assert pii_secret not in result.error


def test_agent_exception_has_cause(
    mock_router: Router, mock_discovery: DiscoveryEngine
) -> None:
    original = ValueError("original")
    failing_agent = MockAgent("Failing", raise_on_execute=original)
    reg = ToolRegistry(
        agents={"failing": failing_agent},
        router=mock_router,
        discovery_engine=mock_discovery,
    )
    with pytest.raises(ToolRegistryError) as exc_info:
        reg._call_agent("failing", {"task_prompt": "x"})
    assert exc_info.value.__cause__ is original


def test_agent_returns_none(mock_router: Router, mock_discovery: DiscoveryEngine) -> None:
    none_agent = MockAgent("None", return_none=True)
    reg = ToolRegistry(
        agents={"none": none_agent},
        router=mock_router,
        discovery_engine=mock_discovery,
    )
    result = reg.call_tool("none", {"task_prompt": "x"})
    assert result.status == "error"
    assert result.error_code == ErrorCode.AGENT_FAILED


def test_keyboard_interrupt_not_swallowed(
    mock_router: Router, mock_discovery: DiscoveryEngine
) -> None:
    bad = MockAgent("Bad", raise_on_execute=KeyboardInterrupt())
    reg = ToolRegistry(
        agents={"bad": bad},
        router=mock_router,
        discovery_engine=mock_discovery,
    )
    with pytest.raises(KeyboardInterrupt):
        reg.call_tool("bad", {"task_prompt": "x"})


def test_timeout_enforced(
    mock_router: Router, mock_discovery: DiscoveryEngine
) -> None:
    slow = MockAgent("Slow", sleep_sec=2.0)
    reg = ToolRegistry(
        agents={"slow": slow},
        router=mock_router,
        discovery_engine=mock_discovery,
        timeout_sec=0.1,
    )
    start = time.perf_counter()
    result = reg.call_tool("slow", {"task_prompt": "x"})
    elapsed = time.perf_counter() - start

    assert result.status == "error"
    assert result.error_code == ErrorCode.TIMEOUT
    assert "timeout" in result.error.lower()
    assert elapsed < 1.5


def test_fast_agent_completes_before_timeout(registry: ToolRegistry) -> None:
    result = registry.call_tool("scout", {"task_prompt": "x"})
    assert result.status == "success"


def test_router_returns_none(mock_discovery: DiscoveryEngine) -> None:
    router = MagicMock(spec=Router)
    router.route.return_value = None
    reg = ToolRegistry(
        agents={},
        router=router,
        discovery_engine=mock_discovery,
    )
    result = reg.call_tool("router", {"raw_prompt": "x"})
    assert result.status == "error"
    assert result.error_code == ErrorCode.AGENT_FAILED


def test_router_raises_exception(mock_discovery: DiscoveryEngine) -> None:
    router = MagicMock(spec=Router)
    router.route.side_effect = RuntimeError("router boom")
    reg = ToolRegistry(
        agents={},
        router=router,
        discovery_engine=mock_discovery,
    )
    result = reg.call_tool("router", {"raw_prompt": "x"})
    assert result.status == "error"
    assert result.error_code == ErrorCode.AGENT_FAILED
    assert "RuntimeError" in result.error


def test_discovery_raises_exception(mock_router: Router) -> None:
    discovery = MagicMock(spec=DiscoveryEngine)
    discovery.scan.side_effect = OSError("disk missing")
    reg = ToolRegistry(
        agents={},
        router=mock_router,
        discovery_engine=discovery,
    )
    result = reg.call_tool("discovery", {})
    assert result.status == "error"
    assert result.error_code == ErrorCode.AGENT_FAILED
    assert "OSError" in result.error


def test_params_never_logged(registry: ToolRegistry, caplog: pytest.LogCaptureFixture) -> None:
    secret = "super-secret-user-data-12345"
    with caplog.at_level(logging.DEBUG):
        registry.call_tool("scout", {"task_prompt": secret})

    for record in caplog.records:
        assert secret not in record.getMessage()


def test_invalid_agents_type_raises() -> None:
    with pytest.raises(ToolRegistryError):
        ToolRegistry(
            agents="not a dict",  # type: ignore[arg-type]
            router=MagicMock(spec=Router),
            discovery_engine=MagicMock(spec=DiscoveryEngine),
        )


def test_agent_missing_execute_raises() -> None:
    class BadAgent:
        pass

    with pytest.raises(ToolRegistryError):
        ToolRegistry(
            agents={"bad": BadAgent()},  # type: ignore[dict-item]
            router=MagicMock(spec=Router),
            discovery_engine=MagicMock(spec=DiscoveryEngine),
        )


def test_invalid_router_type_raises(mock_discovery: DiscoveryEngine) -> None:
    with pytest.raises(ToolRegistryError):
        ToolRegistry(
            agents={},
            router="not a router",  # type: ignore[arg-type]
            discovery_engine=mock_discovery,
        )


def test_invalid_timeout_raises(
    mock_router: Router, mock_discovery: DiscoveryEngine
) -> None:
    with pytest.raises(ToolRegistryError):
        ToolRegistry(
            agents={},
            router=mock_router,
            discovery_engine=mock_discovery,
            timeout_sec=-1,
        )


def test_execution_time_tracked(registry: ToolRegistry) -> None:
    result = registry.call_tool("scout", {"task_prompt": "x"})
    assert result.execution_time_ms >= 0
    assert isinstance(result.execution_time_ms, int)


def test_execution_time_tracked_on_error(registry: ToolRegistry) -> None:
    result = registry.call_tool("nonexistent", {})
    assert result.execution_time_ms >= 0


def test_lazy_import_from_package() -> None:
    from src import companion
    assert companion.ToolRegistry is not None
    assert companion.ToolResult is not None
    assert companion.ToolSchema is not None
    assert companion.ToolRegistryError is not None
    assert companion.ErrorCode is not None
    assert companion.DiscoveryEngine is not None
    assert companion.SystemRegistry is not None


def test_lazy_import_unknown_attribute_raises() -> None:
    from src import companion
    with pytest.raises(AttributeError):
        _ = companion.NonexistentSymbol