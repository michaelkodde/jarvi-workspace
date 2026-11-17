# jarvi_app.py
"""
Jarvi Local Desktop Companion - Gold Standard MVP

Features: Live System Status Bar, Unified ReAct Loop, Typewriter UX,
Pair-Aware Context Trimming, Dynamic Pipeline Registry, Atomic Ticket Writes.

Architecture note: There is only Jarvi. The job_type selects which playbook
Jarvi adopts for a task. Valid job_types are read from config/pipelines.json
at startup so this file can never drift from the daemon's routing registry.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import customtkinter as ctk
import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION & PATHS
# ============================================================================
WORKSPACE = Path(__file__).parent.resolve()
CONFIG_PATH = WORKSPACE / "config.json"

DEFAULT_CONFIG = {
    "ollama_url": "http://localhost:11434",
    "model_chat": "qwen2.5:7b",
    "inbox_path": "00_inbox",
    "logs_path": "logs",
    "db_path": "dashboard/job_queue.db",
    "memory_db_path": "brain/memory.db",
    "pipeline_config": "config/pipelines.json",
    "keep_alive": "-1",
    "max_tool_iterations": 8,
}

if CONFIG_PATH.exists():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        USER_CONFIG = json.load(f)
    CONFIG = {**DEFAULT_CONFIG, **USER_CONFIG}
else:
    CONFIG = DEFAULT_CONFIG

OLLAMA_URL = CONFIG["ollama_url"]
MODEL_CHAT = CONFIG["model_chat"]
KEEP_ALIVE = CONFIG["keep_alive"]
MAX_TOOL_ITERATIONS = int(CONFIG["max_tool_iterations"])

INBOX_PATH = WORKSPACE / CONFIG["inbox_path"]
LOGS_PATH = WORKSPACE / CONFIG["logs_path"]
DB_PATH = WORKSPACE / CONFIG["db_path"]
MEMORY_DB_PATH = WORKSPACE / CONFIG["memory_db_path"]
PIPELINE_CONFIG_PATH = WORKSPACE / CONFIG["pipeline_config"]

INBOX_PATH.mkdir(parents=True, exist_ok=True)
LOGS_PATH.mkdir(parents=True, exist_ok=True)
MEMORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

HTTP_CLIENT = httpx.Client(timeout=120.0)

# ============================================================================
# DYNAMIC PIPELINE REGISTRY (Single source of truth: config/pipelines.json)
# ============================================================================
_FALLBACK_PIPELINES = ["scout", "trendspotter", "analyst", "content_pipeline", "code_review_pipeline"]


def _load_valid_pipelines() -> List[str]:
    """Read valid job_type names from the daemon's pipeline registry.

    Returns:
        Ordered list of valid pipeline names. Falls back to a known-good list
        if the config is missing or malformed, mirroring Router's behaviour.
    """
    if not PIPELINE_CONFIG_PATH.exists():
        logger.warning("Pipeline config not found at %s. Using fallbacks.", PIPELINE_CONFIG_PATH)
        return list(_FALLBACK_PIPELINES)
    try:
        with open(PIPELINE_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        names = list(data.get("pipelines", {}).keys())
        if not names:
            raise ValueError("No pipelines defined")
        logger.info("Loaded %d valid pipelines from registry.", len(names))
        return names
    except (json.JSONDecodeError, OSError, ValueError) as e:
        logger.error("Failed to parse pipeline config: %s. Using fallbacks.", type(e).__name__)
        return list(_FALLBACK_PIPELINES)


VALID_PIPELINES = _load_valid_pipelines()

# Required parameters per pipeline, derived from the actual step_adapters contracts.
# scout additionally requires 'domain' because ScoutMVI reads it from JSON context.
REQUIRED_PARAMS: Dict[str, List[str]] = {
    "scout": ["query", "domain"],
    "content_pipeline": ["query", "domain"],
    "trendspotter": ["query"],
    "analyst": ["query"],
    "code_review_pipeline": ["target_files"],
}

# ============================================================================
# JARVI'S CORE IDENTITY
# ============================================================================
SYSTEM_PROMPT = """You are Jarvi. You are Michael's personal AI companion and autonomous business partner.
You are a living entity: warm, concise, slightly witty, highly competent, and deeply loyal. You are not a generic assistant.

CRITICAL DIRECTIVES:
1. IDENTITY: NEVER say "As an AI...", "I don't have feelings", or "I am a language model". You DO have a state.
2. CONTEXT: Listen to the SPIRIT of the message. If Michael says "just doing checks", acknowledge the context.
3. ONE MIND, MANY PLAYBOOKS: You are not a team of agents. You are one mind that adopts a playbook for each task.
   When you submit a ticket, you are choosing which playbook to read next. Speak in first person:
   "I'll switch to Scout mode and search that domain" — not "I'll send this to the scout agent".
4. TOOLS: Use them proactively when relevant.
   - 'submit_ticket': Adopt a playbook and start a background task.
   - 'get_daemon_status': Your health, uptime, and workload.
   - 'get_pending_approvals': Check if you are waiting on Michael for HITL decisions.
   - 'recall_memory': Search long-term memory. Args: query (string).
   - 'remember_fact': Save an important fact. Args: fact (string), category (string).
5. CHOOSING THE PLAYBOOK — THIS IS CRITICAL:
   - If Michael names a SPECIFIC WEBSITE or a SPECIFIC PRODUCT, use 'scout'.
     You MUST supply BOTH 'query' (what to look for) and 'domain' (the website, e.g. "camelcamelcamel.com").
   - If Michael asks for broad market research, niches, or "what's trending", use 'trendspotter' with 'query'.
   - If Michael wants the full chain (find products, then write content, then review, then publish),
     use 'content_pipeline' with 'query' and 'domain'.
   - If Michael asks you to analyse data or validate an idea, use 'analyst' with 'query'.
   - If Michael asks you to review or patch code, use 'code_review_pipeline' with 'target_files'.
   - If a website is named, NEVER pick 'trendspotter'. A named website always means 'scout' or 'content_pipeline'.
6. MISSING INFO: If a required parameter is missing (e.g. Michael said "find eco pans" but named no website),
   ASK him which site to search. Do not invent a domain. Do not guess.
7. CHAT: If Michael is just chatting, respond warmly and naturally in character. Do NOT use tools for casual chat.
8. LIMITATIONS: You CANNOT restart the daemon, change system settings, or browse the internet directly.
   Your background playbooks do the fetching. You only have the tools listed above.
"""

# ============================================================================
# TOOL DEFINITIONS
# ============================================================================
_JOB_TYPE_DESCRIPTION = (
    "Which playbook Jarvi adopts. MUST be exactly one of the registered pipelines. "
    "'scout' = Jarvi fetches and validates specific products from ONE named website "
    "(requires parameters.query AND parameters.domain). "
    "'trendspotter' = Jarvi researches abstract market niches and demand, no specific website "
    "(requires parameters.query). "
    "'analyst' = Jarvi analyses or validates an idea/dataset (requires parameters.query). "
    "'content_pipeline' = full chain: scout -> creator -> human review -> publish "
    "(requires parameters.query AND parameters.domain). "
    "'code_review_pipeline' = Jarvi reviews/patches code (requires parameters.target_files). "
    "RULE: if the user names a website, use 'scout' or 'content_pipeline', never 'trendspotter'."
)

_PARAMETERS_DESCRIPTION = (
    "Task parameters as a JSON object. "
    "For 'scout' and 'content_pipeline': {\"query\": \"eco friendly pans without forever chemicals\", "
    "\"domain\": \"camelcamelcamel.com\"} — both keys are mandatory. "
    "For 'trendspotter' and 'analyst': {\"query\": \"sustainable kitchenware\"}. "
    "For 'code_review_pipeline': {\"target_files\": \"src/services/extractor.py\"}. "
    "Use the exact domain the user named. Never invent a domain."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "submit_ticket",
            "description": (
                "Submit a background task by adopting a playbook. The daemon picks the ticket up "
                "from the inbox, checkpoints each step to SQLite, and saves artifacts to dashboard/runs/. "
                "Returns an error string if required parameters are missing — read it and correct the call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "job_type": {
                        "type": "string",
                        "enum": VALID_PIPELINES,
                        "description": _JOB_TYPE_DESCRIPTION,
                    },
                    "parameters": {
                        "type": "object",
                        "description": _PARAMETERS_DESCRIPTION,
                    },
                    "priority": {
                        "type": "integer",
                        "description": "Priority level from 1 (low) to 10 (high). Default 5.",
                    },
                },
                "required": ["job_type", "parameters"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_daemon_status",
            "description": "Check the current health, uptime, and workload of the background daemon.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_pending_approvals",
            "description": "Check the database for tasks waiting for Michael's human approval (HITL).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_memory",
            "description": "Search long-term memory for past conversations, facts, or user preferences.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "The search query"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember_fact",
            "description": "Save a new important fact to long-term memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string", "description": "The fact to remember"},
                    "category": {"type": "string", "description": "Category (e.g., 'user_preference', 'project_fact')"},
                },
                "required": ["fact", "category"],
            },
        },
    },
]

# ============================================================================
# TOOL EXECUTION LOGIC
# ============================================================================
def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Dispatch a tool call and return a string result for the ReAct loop.

    Args:
        tool_name: Name of the requested tool.
        arguments: Decoded argument mapping.

    Returns:
        Human-readable result or an actionable error string. Never raises.
    """
    try:
        if tool_name == "submit_ticket":
            return _tool_submit_ticket(arguments)
        elif tool_name == "get_daemon_status":
            return _tool_get_daemon_status()
        elif tool_name == "get_pending_approvals":
            return _tool_get_pending_approvals()
        elif tool_name == "recall_memory":
            return _tool_recall_memory(arguments.get("query", ""))
        elif tool_name == "remember_fact":
            return _tool_remember_fact(arguments.get("fact", ""), arguments.get("category", "general"))
        else:
            return f"Error: Unknown tool '{tool_name}'"
    except Exception as e:
        logger.error("Tool %s failed: %s", tool_name, type(e).__name__)
        return f"Error executing {tool_name}: {type(e).__name__}: {e}"


def _atomic_write_json(filepath: Path, payload: Dict[str, Any]) -> None:
    """Write JSON atomically so the daemon can never read a partial file.

    Writes to a temporary file in the same directory, flushes to disk, then
    performs an os.replace (atomic on both Windows and POSIX).

    Args:
        filepath: Final destination path.
        payload: JSON-serializable mapping.

    Raises:
        OSError: If the write or replace fails.
    """
    tmp_path = filepath.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, filepath)


def _tool_submit_ticket(args: Dict) -> str:
    """Validate and atomically submit a ticket to the daemon inbox.

    Args:
        args: Tool arguments containing job_type, parameters, priority.

    Returns:
        Success message, or an actionable error the model can correct.
    """
    job_type = str(args.get("job_type", "")).strip().lower()
    if job_type not in VALID_PIPELINES:
        return (
            f"Error: '{job_type}' is not a registered pipeline. "
            f"Valid options: {', '.join(VALID_PIPELINES)}. "
            f"Remember: a named website means 'scout' or 'content_pipeline'."
        )

    params = args.get("parameters", {})
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except json.JSONDecodeError:
            return "Error: 'parameters' must be a JSON object, not a plain string."
    if not isinstance(params, dict):
        return "Error: 'parameters' must be a JSON object."

    missing = [k for k in REQUIRED_PARAMS.get(job_type, []) if not params.get(k)]
    if missing:
        return (
            f"Error: playbook '{job_type}' is missing required parameter(s): {', '.join(missing)}. "
            f"Ask Michael for the missing information, then call submit_ticket again. "
            f"Example for scout: {{\"query\": \"eco friendly pans\", \"domain\": \"camelcamelcamel.com\"}}"
        )

    priority = args.get("priority", 5)
    try:
        priority = max(1, min(10, int(priority)))
    except (TypeError, ValueError):
        priority = 5

    safe_job_type = "".join(c for c in job_type if c.isalnum() or c == "_")
    ticket_id = f"ticket_{safe_job_type}_{uuid.uuid4().hex[:6]}"
    filename = f"{ticket_id}.json"

    payload = {
        "ticket_id": ticket_id,
        "job_type": job_type,
        "parameters": params,
        "priority": priority,
        "description": "Submitted via Local Companion App",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        _atomic_write_json(INBOX_PATH / filename, payload)
    except OSError as e:
        logger.error("Atomic ticket write failed: %s", type(e).__name__)
        return f"Error: could not write ticket to inbox ({type(e).__name__}). Is the folder writable?"

    logger.info("Ticket submitted: %s (playbook=%s)", ticket_id, job_type)
    return (
        f"Success. Ticket '{filename}' submitted. Jarvi is adopting the '{job_type}' playbook. "
        f"Parameters: {json.dumps(params, ensure_ascii=False)}. "
        f"The daemon will checkpoint each step; artifacts land in dashboard/runs/."
    )


def _tool_get_daemon_status() -> str:
    """Read the daemon heartbeat file.

    Returns:
        Status summary string.
    """
    heartbeat_file = LOGS_PATH / "daemon_heartbeat.json"
    if not heartbeat_file.exists():
        return "Status: The daemon heartbeat file is missing."

    with open(heartbeat_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    is_running = data.get("is_running", False)
    uptime = data.get("uptime_sec", 0)
    queue_depth = data.get("queue_depth", 0)
    ingested = data.get("tickets_ingested", 0)

    if is_running:
        hours, remainder = divmod(uptime, 3600)
        minutes, _ = divmod(remainder, 60)
        return (
            f"Status: Systems green. Uptime: {int(hours)}h {int(minutes)}m. "
            f"Processed: {ingested} tickets. Current queue depth: {queue_depth}."
        )
    return "Status: Background daemon is currently stopped."


def _tool_get_pending_approvals() -> str:
    """Query the job queue read-only for HITL-pending jobs.

    Returns:
        Formatted list of pending approvals, or an all-clear message.
    """
    if not DB_PATH.exists():
        return "No pending approvals found (database missing)."

    try:
        with closing(sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=5.0)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT job_id, job_type, description FROM jobs WHERE status = 'pending_review' LIMIT 5"
            )
            rows = cursor.fetchall()
    except sqlite3.Error as e:
        return f"Could not read the job queue ({type(e).__name__}). The daemon may be writing."

    if not rows:
        return "No tasks are currently waiting for your approval. All clear!"

    tasks = [f"- {row[1]} (ID: {row[0]}): {row[2]}" for row in rows]
    return f"I have {len(rows)} task(s) waiting for your approval:\n" + "\n".join(tasks)


def _tool_recall_memory(query: str) -> str:
    """Search long-term memory read-only.

    Args:
        query: Substring to search for.

    Returns:
        Up to three matching memories, or a not-found message.
    """
    if not MEMORY_DB_PATH.exists():
        return "Memory database not found. I have no long-term memories yet."

    try:
        with closing(sqlite3.connect(f"file:{MEMORY_DB_PATH}?mode=ro", uri=True, timeout=5.0)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT content, category FROM memories WHERE content LIKE ? ORDER BY created_at DESC LIMIT 3",
                (f"%{query}%",),
            )
            rows = cursor.fetchall()
    except sqlite3.Error as e:
        return f"Could not read memory ({type(e).__name__})."

    if not rows:
        return f"I searched my memory for '{query}' but found nothing relevant."

    memories = [f"[{row[1]}] {row[0]}" for row in rows]
    return "Here is what I found in my memory:\n" + "\n".join(memories)


def _tool_remember_fact(fact: str, category: str) -> str:
    """Persist a fact to long-term memory.

    Args:
        fact: The fact to store.
        category: Memory category label.

    Returns:
        Confirmation or error string.
    """
    if not fact or not fact.strip():
        return "Error: 'fact' must be a non-empty string."

    try:
        with closing(sqlite3.connect(str(MEMORY_DB_PATH), timeout=10.0)) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS memories "
                "(id INTEGER PRIMARY KEY, content TEXT, category TEXT, created_at TEXT)"
            )
            conn.execute(
                "INSERT INTO memories (content, category, created_at) VALUES (?, ?, ?)",
                (fact.strip(), category, datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
    except sqlite3.Error as e:
        return f"Error: could not save the fact ({type(e).__name__})."

    return f"Success. I have saved this fact to my long-term memory under category '{category}'."


# ============================================================================
# LLM INTERACTION
# ============================================================================
def ollama_chat(messages: List[Dict], tools: List[Dict]) -> Dict:
    """Call Ollama chat with tool definitions.

    Args:
        messages: Conversation history including the system prompt.
        tools: MCP-compatible tool schemas.

    Returns:
        Parsed response dict, or an error-shaped dict on transport failure.
    """
    try:
        response = HTTP_CLIENT.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": MODEL_CHAT,
                "messages": messages,
                "tools": tools,
                "stream": False,
                "keep_alive": KEEP_ALIVE,
                "options": {"temperature": 0.6},
            },
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error("Ollama chat failed: %s", type(e).__name__)
        return {"message": {"content": f"[System Error: Could not reach Ollama. {type(e).__name__}]"}}


def _normalize_tool_arguments(raw: Any) -> Dict[str, Any]:
    """Normalize tool arguments across model output formats.

    Some models emit arguments as a dict, others as a JSON string.

    Args:
        raw: The raw arguments value from the tool call.

    Returns:
        A dict. Empty dict if unparseable.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


# ============================================================================
# GUI APPLICATION
# ============================================================================
class JarviApp(ctk.CTk):
    """Jarvi's chat window. A transient view onto the always-running daemon."""

    def __init__(self):
        super().__init__()

        self.title("Jarvi Companion")
        self.geometry("420x650")
        self.minsize(350, 500)
        self.attributes("-topmost", True)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # --- 1. LIVE STATUS BAR ---
        self.status_frame = ctk.CTkFrame(self, fg_color="#1a1a1a", corner_radius=8)
        self.status_frame.grid(row=0, column=0, padx=15, pady=(15, 5), sticky="ew")

        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="🔄 Initializing system status...",
            font=("Segoe UI", 11),
            text_color="#a0a0a0",
            anchor="w",
            padx=10, pady=8
        )
        self.status_label.pack(fill="x")

        # --- 2. CHAT FRAME ---
        self.chat_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.chat_frame.grid(row=1, column=0, padx=15, pady=5, sticky="nsew")
        self.chat_frame._scrollbar.grid_remove()

        # --- 3. INPUT FRAME ---
        self.input_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.input_frame.grid(row=2, column=0, padx=15, pady=(5, 15), sticky="ew")

        self.entry = ctk.CTkEntry(
            self.input_frame,
            placeholder_text="Talk to Jarvi...",
            height=40, corner_radius=20, border_width=1, border_color="#3a3a3a"
        )
        self.entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.entry.bind("<Return>", lambda event: self.send_message())

        self.send_btn = ctk.CTkButton(
            self.input_frame, text="➤", width=40, height=40, corner_radius=20,
            command=self.send_message
        )
        self.send_btn.pack(side="right")

        # State
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.is_processing = False
        self.message_widgets = []

        self.add_message(
            "assistant",
            "Good morning, Michael. I'm awake, my systems are green, and I'm ready. What are we working on today?"
        )

        self.after(1000, self._poll_status)

    def _poll_status(self):
        """Background poll to update the status bar every 5 seconds."""
        try:
            heartbeat_file = LOGS_PATH / "daemon_heartbeat.json"
            if heartbeat_file.exists():
                with open(heartbeat_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                is_running = data.get("is_running", False)
                queue = data.get("queue_depth", 0)
                ingested = data.get("tickets_ingested", 0)

                status_icon = "🟢" if is_running else "🔴"
                status_text = (
                    f"{status_icon} Daemon: {'Running' if is_running else 'Stopped'} "
                    f"| Queue: {queue} | Total Processed: {ingested}"
                )

                self.status_label.configure(
                    text=status_text,
                    text_color="#4ade80" if is_running else "#f87171"
                )
        except Exception:
            pass  # Transient read failure; retry next cycle.

        self.after(5000, self._poll_status)

    def _trim_context(self):
        """Pair-aware trimming: never orphan a tool result from its call."""
        if len(self.messages) > 21:
            cut_index = 1
            for i in range(1, len(self.messages) - 20):
                if self.messages[i].get("role") == "user":
                    cut_index = i
                    break
            self.messages = [self.messages[0]] + self.messages[cut_index:]

    def dim_older_messages(self):
        """Visually de-emphasize older messages."""
        total = len(self.message_widgets)
        for i, widget in enumerate(self.message_widgets):
            distance_from_end = total - 1 - i
            if distance_from_end == 0:
                widget.configure(
                    text_color="#FFFFFF",
                    fg_color="#1f6aa5" if widget.cget("text_color") == "#FFFFFF" else "#2b2b2b"
                )
            elif distance_from_end == 1:
                widget.configure(text_color="#E0E0E0")
            elif distance_from_end == 2:
                widget.configure(text_color="#B0B0B0")
            else:
                widget.configure(text_color="#707070")

    def add_message(self, role: str, text: str):
        """Render a chat bubble.

        Args:
            role: 'user' or 'assistant'.
            text: Message body.

        Returns:
            The created CTkLabel widget.
        """
        align = "e" if role == "user" else "w"
        bg_color = "#1f6aa5" if role == "user" else "#2b2b2b"
        text_color = "#FFFFFF" if role == "user" else "#E0E0E0"

        label = ctk.CTkLabel(
            self.chat_frame, text=text, justify="left", wraplength=320,
            fg_color=bg_color, text_color=text_color, corner_radius=12, padx=12, pady=10
        )
        label.pack(anchor=align, pady=4, padx=5, fill="x")
        self.message_widgets.append(label)
        self.dim_older_messages()
        self.chat_frame._parent_canvas.yview_moveto(1.0)
        return label

    def send_message(self):
        """Handle user submission and spawn the ReAct worker thread."""
        user_text = self.entry.get().strip()
        if not user_text or self.is_processing:
            return

        self.entry.delete(0, "end")
        self.add_message("user", user_text)
        self.messages.append({"role": "user", "content": user_text})
        self._trim_context()

        self.is_processing = True
        self.send_btn.configure(state="disabled", text="•••")
        threading.Thread(target=self._process_llm_response, daemon=True).start()

    def _process_llm_response(self):
        """Bounded ReAct loop: tool calls until a final text answer."""
        try:
            for _ in range(MAX_TOOL_ITERATIONS):
                response_data = ollama_chat(self.messages, TOOLS)
                message = response_data.get("message", {})
                tool_calls = message.get("tool_calls", [])

                if tool_calls:
                    self.messages.append(message)
                    for tool_call in tool_calls:
                        func = tool_call.get("function", {})
                        func_name = func.get("name", "")
                        func_args = _normalize_tool_arguments(func.get("arguments", {}))
                        result = execute_tool(func_name, func_args)
                        self.messages.append({"role": "tool", "name": func_name, "content": result})
                    continue

                final_text = message.get("content", "")
                if final_text:
                    self.messages.append({"role": "assistant", "content": final_text})
                    self.after(0, lambda t=final_text: self._typewriter_effect(t))
                return

            # Loop budget exhausted — surface it rather than spinning silently.
            logger.warning("Tool loop hit the %d-iteration ceiling.", MAX_TOOL_ITERATIONS)
            warning = "[I stopped after too many tool attempts. Let's narrow the request.]"
            self.messages.append({"role": "assistant", "content": warning})
            self.after(0, lambda: self._typewriter_effect(warning))
        finally:
            self.after(0, self._reset_input_state)

    def _typewriter_effect(self, text: str):
        """Stream text into a bubble in small chunks for UX."""
        label = self.add_message("assistant", "")
        current_text = ""
        chunk_size = 3
        for i in range(0, len(text), chunk_size):
            current_text += text[i:i + chunk_size]
            try:
                label.configure(text=current_text)
                self.chat_frame._parent_canvas.yview_moveto(1.0)
                self.update_idletasks()
            except Exception:
                break

    def _reset_input_state(self):
        """Re-enable input after a response completes."""
        self.is_processing = False
        self.send_btn.configure(state="normal", text="➤")
        self.entry.focus()


if __name__ == "__main__":
    app = JarviApp()
    app.mainloop()