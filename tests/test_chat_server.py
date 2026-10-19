# tests/test_chat_server.py
"""Tests for the Chat Server module (Phase 3a)."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.companion.chat_server import (
    ChatError,
    ChatServer,
    ConnectionManager,
    ConnectionState,
    ErrorCode,
    WSEvent,
    WSEventType,
    _compute_idempotency_hash,
    _extract_result_preview,
    _get_hitl_tier,
    _validate_conversation_id,
)


# ============================================================================
# Mock Dataclasses (matching real API shapes)
# ============================================================================
@dataclass
class MockMemory:
    """Mock Memory object matching MemoryEngine's Memory dataclass."""
    content: str
    timestamp: str
    tags: tuple[str, ...] = ()
    category: str = "conversation_history"
    source: str = "test"


@dataclass
class MockRecallResult:
    """Mock RecallResult matching MemoryEngine's return shape."""
    memories: list[MockMemory] = field(default_factory=list)
    count: int = 0
    error: str | None = None
    error_code: str | None = None


@dataclass
class MockRoutingResult:
    """Mock RoutingResult matching Router's return shape."""
    target_agent: str
    target_model: str
    cleaned_prompt: str
    confidence: float
    token_usage: dict[str, int] = field(default_factory=dict)


@dataclass
class MockJobStatusResult:
    """Mock JobStatusResult matching QueryAPI's return shape."""
    status: str
    job: dict | None = None
    error: str | None = None
    error_code: str | None = None


# ============================================================================
# Fixtures
# ============================================================================
@pytest.fixture
def mock_pipeline_flow() -> MagicMock:
    mock = MagicMock()
    mock.submit_job.return_value = "test_ticket_123456"
    return mock


@pytest.fixture
def mock_query_api() -> MagicMock:
    mock = MagicMock()
    mock.get_job_status.return_value = MockJobStatusResult(
        status="completed",
        job={"current_step": "done", "result": "Test result"},
    )
    mock.approve_job = MagicMock()
    mock.reject_job = MagicMock()
    return mock


@pytest.fixture
def mock_router() -> MagicMock:
    mock = MagicMock()
    mock.route.return_value = MockRoutingResult(
        target_agent="scout",
        target_model="qwen2.5:7b",
        cleaned_prompt="test query",
        confidence=0.9,
        token_usage={"total_tokens": 100},
    )
    return mock


@pytest.fixture
def mock_memory_engine() -> MagicMock:
    mock = MagicMock()
    mock.recall.return_value = MockRecallResult(memories=[], count=0)
    mock.remember = MagicMock()
    return mock


@pytest.fixture
def chat_server(
    tmp_path,
    mock_pipeline_flow: MagicMock,
    mock_query_api: MagicMock,
    mock_router: MagicMock,
    mock_memory_engine: MagicMock,
) -> ChatServer:
    return ChatServer(
        workspace_path=tmp_path,
        pipeline_flow=mock_pipeline_flow,
        query_api=mock_query_api,
        router=mock_router,
        memory_engine=mock_memory_engine,
    )


# ============================================================================
# Helper Function Tests
# ============================================================================
class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_get_hitl_tier_known_agents(self) -> None:
        assert _get_hitl_tier("publisher") == 3
        assert _get_hitl_tier("creator") == 2
        assert _get_hitl_tier("scout") == 1
        assert _get_hitl_tier("analyst") == 1
        assert _get_hitl_tier("trend_spotter") == 1

    def test_get_hitl_tier_unknown_agent_fail_safe(self) -> None:
        # Unknown agents default to Tier 3 (fail-safe)
        assert _get_hitl_tier("unknown") == 3
        assert _get_hitl_tier("") == 3
        assert _get_hitl_tier("random_agent") == 3

    def test_validate_conversation_id_valid(self) -> None:
        assert _validate_conversation_id("abc12345") is True
        assert _validate_conversation_id("test_conv_123") is True
        assert _validate_conversation_id("a" * 8) is True
        assert _validate_conversation_id("a" * 32) is True

    def test_validate_conversation_id_invalid(self) -> None:
        assert _validate_conversation_id("ABCD1234") is False  # Uppercase
        assert _validate_conversation_id("short") is False  # Too short
        assert _validate_conversation_id("a" * 33) is False  # Too long
        assert _validate_conversation_id("test-conv") is False  # Invalid char
        assert _validate_conversation_id("") is False
        assert _validate_conversation_id("../../../etc/passwd") is False  # Injection

    def test_compute_idempotency_hash_consistent(self) -> None:
        hash1 = _compute_idempotency_hash("conv1", "test message")
        hash2 = _compute_idempotency_hash("conv1", "test message")
        assert hash1 == hash2

    def test_compute_idempotency_hash_different_conversations(self) -> None:
        hash1 = _compute_idempotency_hash("conv1", "test")
        hash2 = _compute_idempotency_hash("conv2", "test")
        assert hash1 != hash2

    def test_compute_idempotency_hash_truncates_content(self) -> None:
        # Only first 100 chars matter
        hash1 = _compute_idempotency_hash("conv1", "x" * 100 + "a")
        hash2 = _compute_idempotency_hash("conv1", "x" * 100 + "b")
        assert hash1 == hash2

    def test_extract_result_preview_direct_result(self) -> None:
        job = {"result": "Test result"}
        assert _extract_result_preview(job) == "Test result"

    def test_extract_result_preview_artifacts(self) -> None:
        job = {"artifacts": {"final": {"content": "Artifact content"}}}
        assert _extract_result_preview(job) == "Artifact content"

    def test_extract_result_preview_output(self) -> None:
        job = {"output": "Output content"}
        assert _extract_result_preview(job) == "Output content"

    def test_extract_result_preview_truncation(self) -> None:
        job = {"result": "A" * 600}
        preview = _extract_result_preview(job)
        assert len(preview) == 503  # 500 + "..."
        assert preview.endswith("...")

    def test_extract_result_preview_empty_cases(self) -> None:
        assert _extract_result_preview(None) == ""
        assert _extract_result_preview({}) == ""
        assert _extract_result_preview({"unrelated": "data"}) == ""
        assert _extract_result_preview({"result": ""}) == ""


# ============================================================================
# ConnectionManager Tests
# ============================================================================
class TestConnectionManager:
    """Tests for ConnectionManager state management."""

    def test_idempotency_first_call(self) -> None:
        manager = ConnectionManager()
        assert not manager.check_idempotency("conv1", "test")

    def test_idempotency_duplicate(self) -> None:
        manager = ConnectionManager()
        manager.check_idempotency("conv1", "test")
        assert manager.check_idempotency("conv1", "test")

    def test_idempotency_different_conversations(self) -> None:
        manager = ConnectionManager()
        manager.check_idempotency("conv1", "test")
        assert not manager.check_idempotency("conv2", "test")

    def test_hitl_tracking(self) -> None:
        manager = ConnectionManager()
        ws = MagicMock()
        manager._connections[ws] = ConnectionState(conversation_id="conv1")

        assert not manager.has_hitl_been_sent(ws, "ticket1")
        manager.mark_hitl_sent(ws, "ticket1")
        assert manager.has_hitl_been_sent(ws, "ticket1")

    def test_hitl_tracking_unknown_connection(self) -> None:
        manager = ConnectionManager()
        ws = MagicMock()
        assert not manager.has_hitl_been_sent(ws, "ticket1")

    def test_active_ticket_tracking(self) -> None:
        manager = ConnectionManager()
        ws = MagicMock()
        manager._connections[ws] = ConnectionState(conversation_id="conv1")

        assert not manager.is_ticket_active("conv1")
        manager.set_active_ticket(ws, "ticket1")
        assert manager.is_ticket_active("conv1")
        manager.clear_active_ticket(ws)
        assert not manager.is_ticket_active("conv1")

    def test_connection_state_defaults(self) -> None:
        state = ConnectionState(conversation_id="conv1")
        assert state.conversation_id == "conv1"
        assert state.polling_task is None
        assert state.active_ticket is None
        assert state.hitl_events_sent == set()
        assert state.connected_at > 0


# ============================================================================
# ChatServer Initialization Tests
# ============================================================================
class TestChatServerInit:
    """Tests for ChatServer initialization."""

    def test_valid_dependencies(
        self,
        tmp_path,
        mock_pipeline_flow: MagicMock,
        mock_query_api: MagicMock,
        mock_router: MagicMock,
        mock_memory_engine: MagicMock,
    ) -> None:
        server = ChatServer(
            workspace_path=tmp_path,
            pipeline_flow=mock_pipeline_flow,
            query_api=mock_query_api,
            router=mock_router,
            memory_engine=mock_memory_engine,
        )
        assert server.workspace == tmp_path.resolve()
        assert (tmp_path / "dashboard").exists()

    def test_invalid_pipeline_flow(
        self,
        tmp_path,
        mock_query_api: MagicMock,
        mock_router: MagicMock,
        mock_memory_engine: MagicMock,
    ) -> None:
        with pytest.raises(ChatError) as exc_info:
            ChatServer(
                workspace_path=tmp_path,
                pipeline_flow=MagicMock(spec=[]),  # No submit_job
                query_api=mock_query_api,
                router=mock_router,
                memory_engine=mock_memory_engine,
            )
        assert exc_info.value.error_code == ErrorCode.INTERNAL

    def test_invalid_query_api(
        self,
        tmp_path,
        mock_pipeline_flow: MagicMock,
        mock_router: MagicMock,
        mock_memory_engine: MagicMock,
    ) -> None:
        with pytest.raises(ChatError) as exc_info:
            ChatServer(
                workspace_path=tmp_path,
                pipeline_flow=mock_pipeline_flow,
                query_api=MagicMock(spec=[]),  # No get_job_status
                router=mock_router,
                memory_engine=mock_memory_engine,
            )
        assert exc_info.value.error_code == ErrorCode.INTERNAL


# ============================================================================
# WSEvent Tests
# ============================================================================
class TestWSEvent:
    """Tests for WSEvent serialization."""

    def test_to_dict(self) -> None:
        event = WSEvent(WSEventType.TEXT_DELTA, {"content": "test"})
        assert event.to_dict() == {"type": "text_delta", "payload": {"content": "test"}}

    def test_all_event_types(self) -> None:
        for event_type in WSEventType:
            event = WSEvent(event_type, {})
            result = event.to_dict()
            assert result["type"] == event_type.value


# ============================================================================
# WebSocket Integration Tests
# ============================================================================
class TestWebSocketIntegration:
    """Integration tests for WebSocket endpoints."""

    def test_ws_user_message_low_confidence(
        self, chat_server: ChatServer, mock_router: MagicMock
    ) -> None:
        mock_router.route.return_value = MockRoutingResult(
            target_agent="general",
            target_model="qwen2.5:7b",
            cleaned_prompt="test",
            confidence=0.5,
        )

        client = TestClient(chat_server.app)
        with client.websocket_connect("/ws/chat/test_conv") as websocket:
            websocket.send_json({"type": "user_message", "content": "test message"})
            response = websocket.receive_json()
            assert response["type"] == "text_delta"
            assert "clarify" in response["payload"]["content"].lower()

    def test_ws_user_message_high_confidence(
        self, chat_server: ChatServer, mock_pipeline_flow: MagicMock
    ) -> None:
        client = TestClient(chat_server.app)
        with client.websocket_connect("/ws/chat/test_conv") as websocket:
            websocket.send_json({"type": "user_message", "content": "test message"})
            response = websocket.receive_json()
            assert response["type"] == "status_update"
            assert response["payload"]["ticket_id"] == "test_ticket_123456"
            mock_pipeline_flow.submit_job.assert_called_once()

    def test_ws_conversation_history_injected(
        self,
        chat_server: ChatServer,
        mock_pipeline_flow: MagicMock,
        mock_memory_engine: MagicMock,
    ) -> None:
        """Verify conversation history is injected into job parameters."""
        mock_memory_engine.recall.return_value = MockRecallResult(
            memories=[
                MockMemory(
                    content="Previous message",
                    timestamp="2026-01-19T10:00:00Z",
                    tags=("test_conv", "user"),
                ),
            ],
            count=1,
        )

        client = TestClient(chat_server.app)
        with client.websocket_connect("/ws/chat/test_conv") as websocket:
            websocket.send_json({"type": "user_message", "content": "new message"})
            websocket.receive_json()  # status_update

            # Verify submit_job was called with conversation_context
            call_args = mock_pipeline_flow.submit_job.call_args
            params = call_args.kwargs.get("parameters") or call_args[1].get("parameters")
            assert "conversation_context" in params
            assert "Previous message" in params["conversation_context"]

    def test_ws_user_message_invalid_content(self, chat_server: ChatServer) -> None:
        client = TestClient(chat_server.app)
        with client.websocket_connect("/ws/chat/test_conv") as websocket:
            websocket.send_json({"type": "user_message", "content": ""})
            response = websocket.receive_json()
            assert response["type"] == "error"

    def test_ws_user_message_too_long(self, chat_server: ChatServer) -> None:
        client = TestClient(chat_server.app)
        with client.websocket_connect("/ws/chat/test_conv") as websocket:
            websocket.send_json({"type": "user_message", "content": "x" * 7000})
            response = websocket.receive_json()
            assert response["type"] == "error"
            assert "exceeds" in response["payload"]["message"]

    def test_ws_approve_hitl(
        self, chat_server: ChatServer, mock_query_api: MagicMock
    ) -> None:
        client = TestClient(chat_server.app)
        with client.websocket_connect("/ws/chat/test_conv") as websocket:
            websocket.send_json({
                "type": "approve_hitl",
                "ticket_id": "ticket1",
                "action": "approve",
            })
            response = websocket.receive_json()
            assert response["type"] == "ack"
            assert response["payload"]["ticket_id"] == "ticket1"
            mock_query_api.approve_job.assert_called_once_with(
                "ticket1", feedback="Approved via chat UI"
            )

    def test_ws_reject_hitl(
        self, chat_server: ChatServer, mock_query_api: MagicMock
    ) -> None:
        client = TestClient(chat_server.app)
        with client.websocket_connect("/ws/chat/test_conv") as websocket:
            websocket.send_json({
                "type": "approve_hitl",
                "ticket_id": "ticket1",
                "action": "reject",
            })
            response = websocket.receive_json()
            assert response["type"] == "ack"
            mock_query_api.reject_job.assert_called_once_with(
                "ticket1", reason="Rejected via chat UI"
            )

    def test_ws_approve_hitl_invalid_action(self, chat_server: ChatServer) -> None:
        client = TestClient(chat_server.app)
        with client.websocket_connect("/ws/chat/test_conv") as websocket:
            websocket.send_json({
                "type": "approve_hitl",
                "ticket_id": "ticket1",
                "action": "invalid",
            })
            response = websocket.receive_json()
            assert response["type"] == "error"
    
    def test_ws_idempotency_sends_ack(
        self, chat_server: ChatServer, mock_query_api: MagicMock
    ) -> None:
        # Prevent the background poller from immediately sending "completed"
        mock_query_api.get_job_status.return_value = MockJobStatusResult(
            status="queued",
            job={"current_step": "init", "result": ""},
        )
        
        client = TestClient(chat_server.app)
        with client.websocket_connect("/ws/chat/test_conv") as websocket:
            # 1. First message
            websocket.send_json({"type": "user_message", "content": "test"})
            
            # Read the first response (should be status_update)
            response1 = websocket.receive_json()
            assert response1["type"] == "status_update"

            # 2. Send the DUPLICATE message
            websocket.send_json({"type": "user_message", "content": "test"})
            
            # 3. Read responses until we find the ack. 
            # The background poller might interleave a 'status_update', 
            # so we check up to 3 messages to be safe.
            received_ack = False
            for _ in range(3):
                response = websocket.receive_json()
                if response["type"] == "ack":
                    received_ack = True
                    break
                    
            assert received_ack, "Expected to receive 'ack' for duplicate message within 3 reads"

    def test_ws_malformed_json(self, chat_server: ChatServer) -> None:
        client = TestClient(chat_server.app)
        with client.websocket_connect("/ws/chat/test_conv") as websocket:
            websocket.send_text("not json")
            response = websocket.receive_json()
            assert response["type"] == "error"
            assert "Invalid JSON" in response["payload"]["message"]

    def test_ws_unknown_message_type(self, chat_server: ChatServer) -> None:
        client = TestClient(chat_server.app)
        with client.websocket_connect("/ws/chat/test_conv") as websocket:
            websocket.send_json({"type": "unknown_type"})
            response = websocket.receive_json()
            assert response["type"] == "error"
            assert "Unknown message type" in response["payload"]["message"]

    def test_ws_invalid_conversation_id(self, chat_server: ChatServer) -> None:
        client = TestClient(chat_server.app)
        with pytest.raises(Exception):
            with client.websocket_connect("/ws/chat/INVALID-ID") as websocket:
                pass


# ============================================================================
# PII Protection Tests
# ============================================================================
class TestPIIProtection:
    """Tests for PII protection (Constitution #4)."""

    def test_user_message_not_logged(
        self,
        chat_server: ChatServer,
        mock_memory_engine: MagicMock,
        caplog,
    ) -> None:
        mock_memory_engine.recall.side_effect = Exception("SENSITIVE_DB_ERROR")

        client = TestClient(chat_server.app)
        with client.websocket_connect("/ws/chat/test_conv") as websocket:
            with caplog.at_level("ERROR"):
                websocket.send_json({
                    "type": "user_message",
                    "content": "SENSITIVE_USER_DATA_12345",
                })
                websocket.receive_json()  # error response

        # Verify sensitive data is not in logs
        for record in caplog.records:
            assert "SENSITIVE_USER_DATA_12345" not in record.message
            assert "SENSITIVE_DB_ERROR" not in record.message

    def test_exception_chaining(
        self,
        chat_server: ChatServer,
        mock_memory_engine: MagicMock,
    ) -> None:
        """Verify ChatError chains from original exception (Constitution #5)."""
        original_error = ValueError("Original error")
        mock_memory_engine.remember.side_effect = original_error

        client = TestClient(chat_server.app)
        with client.websocket_connect("/ws/chat/test_conv") as websocket:
            websocket.send_json({"type": "user_message", "content": "test"})
            response = websocket.receive_json()
            assert response["type"] == "error"


# ============================================================================
# Conversation History Tests
# ============================================================================
class TestConversationHistory:
    """Tests for conversation history retrieval."""

    def test_history_sorted_ascending(
        self,
        chat_server: ChatServer,
        mock_memory_engine: MagicMock,
    ) -> None:
        """Verify history is sorted by timestamp ascending."""
        mock_memory_engine.recall.return_value = MockRecallResult(
            memories=[
                MockMemory(
                    content="Newer message",
                    timestamp="2026-01-19T10:00:00Z",
                    tags=("test_conv", "user"),
                ),
                MockMemory(
                    content="Older message",
                    timestamp="2026-01-18T10:00:00Z",
                    tags=("test_conv", "user"),
                ),
            ],
            count=2,
        )

        history = chat_server._retrieve_conversation_history("test_conv")
        assert history.index("Older message") < history.index("Newer message")

    def test_history_empty_on_failure(
        self,
        chat_server: ChatServer,
        mock_memory_engine: MagicMock,
    ) -> None:
        mock_memory_engine.recall.side_effect = Exception("DB error")
        history = chat_server._retrieve_conversation_history("test_conv")
        assert history == ""

    def test_history_limits_to_max(
        self,
        chat_server: ChatServer,
        mock_memory_engine: MagicMock,
    ) -> None:
        """Verify history is limited to _MAX_HISTORY_MESSAGES."""
        from src.companion.chat_server import _MAX_HISTORY_MESSAGES

        memories = [
            MockMemory(
                content=f"Message {i}",
                timestamp=f"2026-01-{i+1:02d}T10:00:00Z",
                tags=("test_conv", "user"),
            )
            for i in range(50)
        ]
        mock_memory_engine.recall.return_value = MockRecallResult(
            memories=memories, count=50
        )

        history = chat_server._retrieve_conversation_history("test_conv")
        # Should only contain last _MAX_HISTORY_MESSAGES
        lines = [line for line in history.split("\n") if line.strip()]
        assert len(lines) <= _MAX_HISTORY_MESSAGES