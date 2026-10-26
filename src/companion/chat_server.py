# src/companion/chat_server.py
"""
Lightweight chat interface for Jarvi Companion.

Provides a FastAPI-based WebSocket server for natural-language interaction,
real-time agent status updates, and HITL approvals.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

__all__ = [
    "ChatError",
    "ChatServer",
    "ConnectionManager",
    "ConnectionState",
    "ErrorCode",
    "WSEvent",
    "WSEventType",
]

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================
_MAX_MESSAGE_LENGTH = 6000
_MAX_HISTORY_MESSAGES = 20
_IDEMPOTENCY_TTL_SEC = 60
_RESULT_PREVIEW_MAX_CHARS = 500
_POLL_INTERVAL_SEC = 2.0
_MAX_POLLING_DURATION_SEC = 3600  # 1 hour hard timeout
_CONVERSATION_ID_RE = re.compile(r"^[a-z0-9_-]{8,32}$")
_DEFAULT_CLARIFYING_QUESTION = (
    "I'm not sure I understand. Could you clarify what you need?"
)

# HITL Tier Classification (Master Prompt Pillar 5.4)
# Tier 1: Auto-execute (internal research, drafts)
# Tier 2: Approval required (proposals, email drafts)
# Tier 3: Strict approval (financial, publishing, deletion)
HITL_TIER_MAP: dict[str, int] = {
    "publisher": 3,       # Publishing to live platforms
    "creator": 2,         # Content creation
    "scout": 1,           # Product research
    "analyst": 1,         # Market analysis
    "trend_spotter": 1,   # Trend research
}
_DEFAULT_HITL_TIER = 3  # Fail-safe: unknown agents require strict approval


# ============================================================================
# Exceptions & Error Codes
# ============================================================================
class ErrorCode:
    """Machine-readable error codes for chat operations."""

    ROUTER_FAILED = "ROUTER_FAILED"
    INVALID_MESSAGE = "INVALID_MESSAGE"
    CONVERSATION_NOT_FOUND = "CONVERSATION_NOT_FOUND"
    DB_ERROR = "DB_ERROR"
    PIPELINE_ERROR = "PIPELINE_ERROR"
    UNAUTHORIZED = "UNAUTHORIZED"
    INTERNAL = "INTERNAL"


class ChatError(Exception):
    """Raised when a chat operation fails. Chains from the original exception."""

    def __init__(self, message: str, error_code: str = ErrorCode.INTERNAL) -> None:
        super().__init__(message)
        self.error_code = error_code


# ============================================================================
# Data Models
# ============================================================================
class WSEventType(str, Enum):
    """WebSocket event types."""

    TEXT_DELTA = "text_delta"
    STATUS_UPDATE = "status_update"
    HITL_REQUIRED = "hitl_required"
    COMPLETED = "completed"
    ERROR = "error"
    ACK = "ack"


@dataclass(frozen=True)
class WSEvent:
    """WebSocket event payload (MCP-Ready Schema, Constitution #11)."""

    type: WSEventType
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {"type": self.type.value, "payload": self.payload}


@dataclass
class ConnectionState:
    """State for a single WebSocket connection.

    Consolidates all per-connection state into a single object to prevent
    fragmentation and make lifecycle management explicit.
    """

    conversation_id: str
    polling_task: asyncio.Task[None] | None = None
    active_ticket: str | None = None
    hitl_events_sent: set[str] = field(default_factory=set)
    connected_at: float = field(default_factory=time.time)


# ============================================================================
# Helper Functions
# ============================================================================
def _get_hitl_tier(target_agent: str) -> int:
    """Map target_agent to HITL tier.

    Args:
        target_agent: Agent name from Router.

    Returns:
        HITL tier (1, 2, or 3). Defaults to 3 for unknown agents (fail-safe).
    """
    return HITL_TIER_MAP.get(target_agent, _DEFAULT_HITL_TIER)


def _compute_idempotency_hash(conversation_id: str, content: str) -> str:
    """Compute idempotency hash for message deduplication.

    Args:
        conversation_id: Conversation identifier.
        content: Message content (first 100 chars).

    Returns:
        SHA256 hash string.
    """
    data = f"{conversation_id}:{content[:100]}"
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _validate_conversation_id(conversation_id: str) -> bool:
    """Validate conversation_id format to prevent injection.

    Args:
        conversation_id: Conversation identifier to validate.

    Returns:
        True if valid, False otherwise.
    """
    return bool(_CONVERSATION_ID_RE.match(conversation_id))


def _extract_result_preview(job: dict[str, Any] | None) -> str:
    """Extract result preview from job dict defensively.

    The exact job structure depends on PipelineFlow implementation. This
    function tries multiple common locations and falls back gracefully.

    Args:
        job: Job dict from QueryAPI.

    Returns:
        Truncated result preview string, or empty string if unavailable.
    """
    if not job or not isinstance(job, dict):
        return ""

    # Try common result locations
    content: str = ""

    # Option 1: Direct "result" field
    if "result" in job and isinstance(job["result"], str):
        content = job["result"]
    # Option 2: Nested "artifacts" dict
    elif "artifacts" in job and isinstance(job["artifacts"], dict):
        artifacts = job["artifacts"]
        if artifacts:
            last_artifact = list(artifacts.values())[-1]
            if isinstance(last_artifact, dict):
                content = last_artifact.get("content", "")
    # Option 3: "output" field (MCP-style)
    elif "output" in job and isinstance(job["output"], str):
        content = job["output"]
    # Option 4: "description" as fallback
    elif "description" in job and isinstance(job["description"], str):
        content = job["description"]

    if not content:
        return ""

    if len(content) > _RESULT_PREVIEW_MAX_CHARS:
        return content[:_RESULT_PREVIEW_MAX_CHARS] + "..."
    return content


# ============================================================================
# Connection Manager
# ============================================================================
class ConnectionManager:
    """Manages WebSocket connections and their state.

    Thread-safe via FastAPI's async event loop. All state is per-connection,
    preventing cross-connection contamination.
    """

    def __init__(self) -> None:
        self._connections: dict[WebSocket, ConnectionState] = {}
        self._idempotency_cache: dict[str, float] = {}

    async def connect(self, websocket: WebSocket, conversation_id: str) -> None:
        """Register a new WebSocket connection.

        Args:
            websocket: WebSocket connection.
            conversation_id: Conversation identifier.

        Raises:
            ChatError: If conversation_id is invalid.
        """
        if not _validate_conversation_id(conversation_id):
            raise ChatError(
                f"Invalid conversation_id format: {conversation_id}",
                ErrorCode.INVALID_MESSAGE,
            )
        await websocket.accept()
        self._connections[websocket] = ConnectionState(conversation_id=conversation_id)
        logger.info("WebSocket connected: conversation_id=%s", conversation_id)

    async def disconnect(self, websocket: WebSocket) -> None:
        """Unregister a WebSocket connection and cancel its polling task.

        Args:
            websocket: WebSocket connection.
        """
        state = self._connections.pop(websocket, None)
        if state and state.polling_task and not state.polling_task.done():
            state.polling_task.cancel()
            try:
                await state.polling_task
            except asyncio.CancelledError:
                pass
        if state:
            logger.info("WebSocket disconnected: conversation_id=%s", state.conversation_id)

    async def send_event(self, websocket: WebSocket, event: WSEvent) -> None:
        """Send a WebSocket event to the client.

        Args:
            websocket: WebSocket connection.
            event: Event to send.

        Raises:
            ChatError: If send fails.
        """
        try:
            await websocket.send_json(event.to_dict())
        except Exception as e:
            raise ChatError(
                f"Failed to send WebSocket event: {type(e).__name__}",
                ErrorCode.INTERNAL,
            ) from e

    def start_polling(
        self, websocket: WebSocket, ticket_id: str, poll_func: Callable[..., Any]
    ) -> None:
        """Start a background polling task for a ticket.

        Args:
            websocket: WebSocket connection.
            ticket_id: Ticket identifier.
            poll_func: Async function to poll job status.
        """
        state = self._connections.get(websocket)
        if not state:
            return
        # Cancel existing task if any
        if state.polling_task and not state.polling_task.done():
            state.polling_task.cancel()
        state.active_ticket = ticket_id
        state.polling_task = asyncio.create_task(poll_func(websocket, ticket_id))

    def get_state(self, websocket: WebSocket) -> ConnectionState | None:
        """Get connection state.

        Args:
            websocket: WebSocket connection.

        Returns:
            ConnectionState or None.
        """
        return self._connections.get(websocket)

    def is_ticket_active(self, conversation_id: str) -> bool:
        """Check if a conversation has an active ticket.

        Args:
            conversation_id: Conversation identifier.

        Returns:
            True if a ticket is active, False otherwise.
        """
        return any(
            state.active_ticket is not None
            for state in self._connections.values()
            if state.conversation_id == conversation_id
        )

    def set_active_ticket(self, websocket: WebSocket, ticket_id: str) -> None:
        """Mark a ticket as active for a connection.

        Args:
            websocket: WebSocket connection.
            ticket_id: Ticket identifier.
        """
        state = self._connections.get(websocket)
        if state:
            state.active_ticket = ticket_id

    def clear_active_ticket(self, websocket: WebSocket) -> None:
        """Clear the active ticket for a connection.

        Args:
            websocket: WebSocket connection.
        """
        state = self._connections.get(websocket)
        if state:
            state.active_ticket = None

    def check_idempotency(self, conversation_id: str, content: str) -> bool:
        """Check if a message has been processed recently (idempotency).

        Args:
            conversation_id: Conversation identifier.
            content: Message content.

        Returns:
            True if message is a duplicate, False otherwise.
        """
        # Prune expired entries
        now = time.time()
        self._idempotency_cache = {
            k: v
            for k, v in self._idempotency_cache.items()
            if now - v < _IDEMPOTENCY_TTL_SEC
        }

        msg_hash = _compute_idempotency_hash(conversation_id, content)
        if msg_hash in self._idempotency_cache:
            return True

        self._idempotency_cache[msg_hash] = now
        return False

    def mark_hitl_sent(self, websocket: WebSocket, ticket_id: str) -> None:
        """Mark that HITL event has been sent for a ticket.

        Args:
            websocket: WebSocket connection.
            ticket_id: Ticket identifier.
        """
        state = self._connections.get(websocket)
        if state:
            state.hitl_events_sent.add(ticket_id)

    def has_hitl_been_sent(self, websocket: WebSocket, ticket_id: str) -> bool:
        """Check if HITL event has been sent for a ticket.

        Args:
            websocket: WebSocket connection.
            ticket_id: Ticket identifier.

        Returns:
            True if sent, False otherwise.
        """
        state = self._connections.get(websocket)
        return ticket_id in state.hitl_events_sent if state else False


# ============================================================================
# Chat Server
# ============================================================================
class ChatServer:
    """FastAPI-based chat server with WebSocket support.

    Composes with existing Daemon infrastructure (Router, PipelineFlow,
    QueryAPI, MemoryEngine) via dependency injection. Never rebuilds or
    duplicates their functionality.

    Args:
        workspace_path: Root of jarvis-workspace.
        pipeline_flow: Existing PipelineFlow instance.
        query_api: Existing QueryAPI instance.
        router: Existing Router instance.
        memory_engine: Existing MemoryEngine instance.
        auth_token: Optional authentication token for WebSocket connections.

    Raises:
        ChatError: If any dependency is invalid.
    """

    def __init__(
        self,
        workspace_path: str | Path,
        pipeline_flow: Any,
        query_api: Any,
        router: Any,
        memory_engine: Any,
        auth_token: str | None = None,
        lifespan: Any = None,
    ) -> None:
        if not hasattr(pipeline_flow, "submit_job"):
            raise ChatError(
                "pipeline_flow must have submit_job method", ErrorCode.INTERNAL
            )
        if not hasattr(query_api, "get_job_status"):
            raise ChatError(
                "query_api must have get_job_status method", ErrorCode.INTERNAL
            )
        if not hasattr(router, "route"):
            raise ChatError("router must have route method", ErrorCode.INTERNAL)
        if not hasattr(memory_engine, "recall"):
            raise ChatError(
                "memory_engine must have recall method", ErrorCode.INTERNAL
            )

        self.workspace = Path(workspace_path).resolve()
        self.pipeline_flow = pipeline_flow
        self.query_api = query_api
        self.router = router
        self.memory_engine = memory_engine
        self.auth_token = auth_token
        self.manager = ConnectionManager()

        # Ensure dashboard directory exists
        dashboard_dir = self.workspace / "dashboard"
        dashboard_dir.mkdir(parents=True, exist_ok=True)

        self.app = FastAPI(title="Jarvi Chat Server", lifespan=lifespan)
        
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        # ------------------------------------------

        self._setup_routes()

    # ------------------------------------------------------------------------
    # Route Setup
    # ------------------------------------------------------------------------
    def _setup_routes(self) -> None:
        """Setup FastAPI routes."""

        @self.app.get("/")
        async def root() -> FileResponse:
            """Serve the chat HTML interface."""
            return FileResponse(self.workspace / "dashboard" / "chat.html")

        @self.app.get("/static/{filename}")
        async def static_files(filename: str) -> FileResponse:
            """Serve static files (CSS, JS)."""
            return FileResponse(self.workspace / "dashboard" / filename)

        @self.app.websocket("/ws/chat/{conversation_id}")
        async def websocket_endpoint(
            websocket: WebSocket,
            conversation_id: str | None = None,
            token: str | None = None,
        ) -> None:
            """WebSocket endpoint for chat communication.

            Args:
                websocket: WebSocket connection.
                conversation_id: Optional conversation identifier.
                token: Optional authentication token.
            """
            # Authentication check
            if self.auth_token and token != self.auth_token:
                await websocket.close(code=4001, reason="Unauthorized")
                return

            if conversation_id is None:
                conversation_id = uuid.uuid4().hex[:12]

            if not _validate_conversation_id(conversation_id):
                await websocket.close(code=4002, reason="Invalid conversation_id")
                return

            async with self._websocket_session(websocket, conversation_id):
                try:
                    while True:
                        try:
                            data = await websocket.receive_json()
                        except json.JSONDecodeError:
                            await self._send_error(websocket, "Invalid JSON")
                            continue
                        except WebSocketDisconnect:
                            break

                        msg_type = data.get("type") if isinstance(data, dict) else None

                        if msg_type == "user_message":
                            await self._handle_user_message(
                                websocket, data, conversation_id
                            )
                        elif msg_type == "approve_hitl":
                            await self._handle_approve_hitl(websocket, data)
                        else:
                            await self._send_error(
                                websocket, f"Unknown message type: {msg_type}"
                            )
                except WebSocketDisconnect:
                    pass
                except ChatError as e:
                    logger.error("ChatError: %s (code=%s)", e, e.error_code)
                    try:
                        await self._send_error(websocket, str(e))
                    except Exception:
                        pass
                except Exception as e:
                    logger.exception("Unexpected WebSocket error")
                    try:
                        await self._send_error(websocket, "Internal server error")
                    except Exception:
                        pass

    @asynccontextmanager
    async def _websocket_session(
        self, websocket: WebSocket, conversation_id: str
    ) -> Any:
        """Async context manager for WebSocket session lifecycle.

        Ensures cleanup happens even on unexpected exceptions.

        Args:
            websocket: WebSocket connection.
            conversation_id: Conversation identifier.
        """
        await self.manager.connect(websocket, conversation_id)
        try:
            yield
        finally:
            await self.manager.disconnect(websocket)

    # ------------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------------
    async def _send_error(self, websocket: WebSocket, message: str) -> None:
        """Send an error event to the client.

        Args:
            websocket: WebSocket connection.
            message: Error message.
        """
        try:
            await self.manager.send_event(
                websocket, WSEvent(WSEventType.ERROR, {"message": message})
            )
        except Exception as e:
            logger.error("Failed to send error event: %s", type(e).__name__)

    async def _safe_execute(
        self,
        operation: Callable[..., Any],
        error_message: str,
        error_code: str = ErrorCode.INTERNAL,
    ) -> Any:
        """Execute an operation with proper exception chaining.

        Satisfies Constitution #5: "raise a single, custom module exception
        (with exception chaining from e)".

        Args:
            operation: Callable to execute (sync or async).
            error_message: Error message for the ChatError.
            error_code: Error code for the ChatError.

        Returns:
            Operation result.

        Raises:
            ChatError: If operation fails, chained from original exception.
        """
        try:
            if asyncio.iscoroutinefunction(operation):
                return await operation()
            return operation()
        except ChatError:
            raise  # Already wrapped, don't double-wrap
        except Exception as e:
            raise ChatError(error_message, error_code) from e

    def _retrieve_conversation_history(self, conversation_id: str) -> str:
        """Retrieve and format conversation history.

        Note: This is synchronous because MemoryEngine.recall() is synchronous.
        The call is fast (<20ms) and doesn't block the event loop meaningfully.

        Args:
            conversation_id: Conversation identifier.

        Returns:
            Formatted history string, or empty string on failure.
        """
        try:
            recall_result = self.memory_engine.recall(
                category="conversation_history",
                tags=[conversation_id],
                limit=_MAX_HISTORY_MESSAGES,
            )
            memories = getattr(recall_result, "memories", None)
            if not isinstance(memories, list):
                return ""
            # Sort by timestamp ascending (chronological order)
            sorted_memories = sorted(
                memories, key=lambda m: getattr(m, "timestamp", "")
            )[-_MAX_HISTORY_MESSAGES:]
            return "\n".join(
                f"{getattr(m, 'tags', [''])[0]}: {getattr(m, 'content', '')}"
                for m in sorted_memories
            )
        except Exception as e:
            logger.warning(
                "Failed to retrieve conversation history: %s", type(e).__name__
            )
            return ""

    def _save_memory(
        self, content: str, conversation_id: str, role: str
    ) -> None:
        """Save a message to MemoryEngine.

        Args:
            content: Message content.
            conversation_id: Conversation identifier.
            role: 'user' or 'assistant'.

        Raises:
            ChatError: If save fails, chained from original exception.
        """
        try:
            self.memory_engine.remember(
                content=content,
                category="conversation_history",
                tags=[conversation_id, role],
            )
        except Exception as e:
            raise ChatError(
                f"Failed to save {role} message: {type(e).__name__}",
                ErrorCode.DB_ERROR,
            ) from e

    # ------------------------------------------------------------------------
    # Message Handlers
    # ------------------------------------------------------------------------
    async def _handle_user_message(
        self, websocket: WebSocket, data: dict[str, Any], conversation_id: str
    ) -> None:
        """Handle a user_message event.

        Orchestrates validation, memory save, routing, and job submission.

        Args:
            websocket: WebSocket connection.
            data: Message data.
            conversation_id: Conversation identifier.
        """
        content = data.get("content", "")

        # Validate content
        if not content or not isinstance(content, str):
            await self._send_error(websocket, "Invalid message: content is required")
            return

        if len(content) > _MAX_MESSAGE_LENGTH:
            await self._send_error(
                websocket, f"Message exceeds {_MAX_MESSAGE_LENGTH} characters"
            )
            return

        # Check idempotency
        if self.manager.check_idempotency(conversation_id, content):
            logger.warning("Duplicate message detected, sending acknowledgment")
            await self.manager.send_event(
                websocket, WSEvent(WSEventType.ACK, {"message": "Message already received"})
            )
            return

        # Check if a ticket is already active
        if self.manager.is_ticket_active(conversation_id):
            await self._send_error(
                websocket, "Please wait for current task to complete"
            )
            return

        # Save user message to MemoryEngine
        try:
            await self._safe_execute(
                lambda: self._save_memory(content, conversation_id, "user"),
                "Failed to save user message",
                ErrorCode.DB_ERROR,
            )
        except ChatError as e:
            logger.error("Memory save failed: %s", e)
            await self._send_error(websocket, "Memory unavailable")
            return

        # Retrieve conversation history for context
        history_context = self._retrieve_conversation_history(conversation_id)

        # Route the message
        try:
            routing_result = await self._safe_execute(
                lambda: self.router.route(content),
                "Router failed",
                ErrorCode.ROUTER_FAILED,
            )
        except ChatError as e:
            logger.error("Routing failed: %s", e)
            await self._send_error(websocket, "Router unavailable")
            return

        # Handle low confidence
        if routing_result.confidence < 0.7:
            await self._handle_low_confidence(websocket, conversation_id)
            return

        # Submit job and start polling
        try:
            await self._submit_job_and_start_polling(
                websocket, conversation_id, content, routing_result, history_context
            )
        except ChatError as e:
            logger.error("Job submission failed: %s", e)
            await self._send_error(websocket, f"Failed to start task: {e.error_code}")

    async def _handle_low_confidence(
        self, websocket: WebSocket, conversation_id: str
    ) -> None:
        """Handle low confidence routing result.

        Args:
            websocket: WebSocket connection.
            conversation_id: Conversation identifier.
        """
        await self.manager.send_event(
            websocket,
            WSEvent(WSEventType.TEXT_DELTA, {"content": _DEFAULT_CLARIFYING_QUESTION}),
        )
        try:
            await self._safe_execute(
                lambda: self._save_memory(
                    _DEFAULT_CLARIFYING_QUESTION, conversation_id, "assistant"
                ),
                "Failed to save assistant message",
                ErrorCode.DB_ERROR,
            )
        except ChatError as e:
            # Non-fatal: log but don't fail the flow
            logger.warning("Failed to save clarifying question: %s", e)

    async def _submit_job_and_start_polling(
        self,
        websocket: WebSocket,
        conversation_id: str,
        content: str,
        routing_result: Any,
        history_context: str,
    ) -> None:
        """Submit job to PipelineFlow and start background polling.

        Args:
            websocket: WebSocket connection.
            conversation_id: Conversation identifier.
            content: User message content.
            routing_result: Routing result from Router.
            history_context: Formatted conversation history.

        Raises:
            ChatError: If job submission fails, chained from original exception.
        """
        target_agent = routing_result.target_agent
        hitl_tier = _get_hitl_tier(target_agent)

        # Build job parameters with conversation context
        job_params: dict[str, Any] = {
            "query": routing_result.cleaned_prompt,
            "hitl_tier": hitl_tier,
            "conversation_id": conversation_id,
        }
        # Inject conversation history for agent context (not router)
        if history_context:
            job_params["conversation_context"] = history_context

        try:
            ticket_id = await self._safe_execute(
                lambda: self.pipeline_flow.submit_job(
                    job_type=target_agent,
                    parameters=job_params,
                    priority=5,
                    description=f"Chat request: {content[:50]}...",
                ),
                "Failed to submit job",
                ErrorCode.PIPELINE_ERROR,
            )
        except ChatError:
            raise

        # Save acknowledgment
        acknowledgment = f"I've started working on that. Ticket: {ticket_id}"
        try:
            await self._safe_execute(
                lambda: self._save_memory(
                    acknowledgment, conversation_id, "assistant"
                ),
                "Failed to save acknowledgment",
                ErrorCode.DB_ERROR,
            )
        except ChatError as e:
            logger.warning("Failed to save acknowledgment: %s", e)

        # Mark ticket as active
        self.manager.set_active_ticket(websocket, ticket_id)

        # Stream status update
        await self.manager.send_event(
            websocket,
            WSEvent(
                WSEventType.STATUS_UPDATE,
                {"ticket_id": ticket_id, "step": "queued", "status": "queued"},
            ),
        )

        # Start background polling
        self.manager.start_polling(websocket, ticket_id, self._poll_job_status)

    async def _handle_approve_hitl(
        self, websocket: WebSocket, data: dict[str, Any]
    ) -> None:
        """Handle an approve_hitl event.

        Args:
            websocket: WebSocket connection.
            data: Approval data.
        """
        ticket_id = data.get("ticket_id")
        action = data.get("action")

        if not ticket_id or action not in ("approve", "reject"):
            await self._send_error(websocket, "Invalid approval data")
            return

        try:
            if action == "approve":
                await self._safe_execute(
                    lambda: self.query_api.approve_job(
                        ticket_id, feedback="Approved via chat UI"
                    ),
                    "Failed to approve job",
                    ErrorCode.INTERNAL,
                )
            else:
                await self._safe_execute(
                    lambda: self.query_api.reject_job(
                        ticket_id, reason="Rejected via chat UI"
                    ),
                    "Failed to reject job",
                    ErrorCode.INTERNAL,
                )
            logger.info("HITL action: ticket_id=%s, action=%s", ticket_id, action)
            # Send acknowledgment back to client
            await self.manager.send_event(
                websocket,
                WSEvent(
                    WSEventType.ACK,
                    {"message": f"Job {action}d successfully", "ticket_id": ticket_id},
                ),
            )
        except ChatError as e:
            logger.error("HITL action failed: %s", e)
            await self._send_error(websocket, "Failed to process approval")

    # ------------------------------------------------------------------------
    # Background Polling
    # ------------------------------------------------------------------------
    async def _poll_job_status(self, websocket: WebSocket, ticket_id: str) -> None:
        """Background task to poll job status and stream updates."""
        start_time = time.time()
        last_status = None  # Track last sent status to avoid spam
        last_step = None    # Track last sent step to avoid spam
        
        try:
            while True:
                if time.time() - start_time > _MAX_POLLING_DURATION_SEC:
                    logger.warning("Polling timeout reached for ticket_id=%s", ticket_id)
                    await self._send_error(websocket, "Job timed out")
                    self.manager.clear_active_ticket(websocket)
                    break

                try:
                    status_result = self.query_api.get_job_status(ticket_id)
                    job_status = getattr(status_result, "status", None)
                    if job_status is None and isinstance(status_result, dict):
                        job_status = status_result.get("status", "unknown")

                    job = getattr(status_result, "job", None)
                    if job is None and isinstance(status_result, dict):
                        job = status_result.get("job")

                    current_step = (
                        job.get("current_step", "") if isinstance(job, dict) else ""
                    )

                    # --- ONLY SEND UPDATE IF STATUS CHANGED ---
                    status_changed = (job_status != last_status or current_step != last_step)
                    
                    if job_status == "pending_review":
                        if not self.manager.has_hitl_been_sent(websocket, ticket_id):
                            await self.manager.send_event(
                                websocket,
                                WSEvent(
                                    WSEventType.HITL_REQUIRED,
                                    {
                                        "ticket_id": ticket_id,
                                        "tier": 3,
                                        "preview": "Job requires approval",
                                    },
                                ),
                            )
                            self.manager.mark_hitl_sent(websocket, ticket_id)
                    elif job_status == "completed":
                        result_preview = _extract_result_preview(job)
                        await self.manager.send_event(
                            websocket,
                            WSEvent(
                                WSEventType.COMPLETED,
                                {
                                    "ticket_id": ticket_id,
                                    "result_preview": result_preview,
                                },
                            ),
                        )
                        self.manager.clear_active_ticket(websocket)
                        break
                    elif job_status == "failed":
                        await self._send_error(websocket, "Job failed")
                        self.manager.clear_active_ticket(websocket)
                        break
                    elif status_changed:  # Only send status update if it changed
                        await self.manager.send_event(
                            websocket,
                            WSEvent(
                                WSEventType.STATUS_UPDATE,
                                {
                                    "ticket_id": ticket_id,
                                    "step": current_step,
                                    "status": job_status,
                                },
                            ),
                        )
                        last_status = job_status
                        last_step = current_step
                    # -----------------------------------------

                except Exception as e:
                    logger.error("Polling error: %s", type(e).__name__)

                await asyncio.sleep(_POLL_INTERVAL_SEC)
        except asyncio.CancelledError:
            logger.info("Polling task cancelled: ticket_id=%s", ticket_id)
            self.manager.clear_active_ticket(websocket)
            raise