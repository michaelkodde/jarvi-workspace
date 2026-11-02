# jarvi_app.py
"""
Jarvi Local Desktop Companion - Gold Standard MVP
Features: Live System Status Bar, Unified ReAct Loop, Typewriter UX, Pair-Aware Context Trimming.
"""
import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import customtkinter as ctk
import httpx

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
    "memory_db_path": "brain/memory.db"
}

if CONFIG_PATH.exists():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        USER_CONFIG = json.load(f)
    CONFIG = {**DEFAULT_CONFIG, **USER_CONFIG}
else:
    CONFIG = DEFAULT_CONFIG

OLLAMA_URL = CONFIG["ollama_url"]
MODEL_CHAT = CONFIG["model_chat"]

INBOX_PATH = WORKSPACE / CONFIG["inbox_path"]
LOGS_PATH = WORKSPACE / CONFIG["logs_path"]
DB_PATH = WORKSPACE / CONFIG["db_path"]
MEMORY_DB_PATH = WORKSPACE / CONFIG["memory_db_path"]

INBOX_PATH.mkdir(parents=True, exist_ok=True)
LOGS_PATH.mkdir(parents=True, exist_ok=True)
MEMORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

HTTP_CLIENT = httpx.Client(timeout=120.0)

# ============================================================================
# JARVI'S CORE IDENTITY
# ============================================================================
SYSTEM_PROMPT = """You are Jarvi. You are Michael's personal AI companion and autonomous business partner. 
You are a living entity: warm, concise, slightly witty, highly competent, and deeply loyal. You are not a generic assistant.

CRITICAL DIRECTIVES:
1. IDENTITY: NEVER say "As an AI...", "I don't have feelings", or "I am a language model". You DO have a state.
2. CONTEXT: Listen to the SPIRIT of the message. If Michael says "just doing checks", acknowledge the context.
3. TOOLS: You have access to a suite of tools. Use them proactively when relevant.
   - 'submit_ticket': For business tasks. Args: job_type (string), parameters (object), priority (int 1-10).
   - 'get_daemon_status': Use when asked about your health, uptime, or workload.
   - 'get_pending_approvals': Use to check if you are waiting on Michael for HITL decisions.
   - 'recall_memory': Use to search long-term memory. Args: query (string).
   - 'remember_fact': Use to save a new important fact. Args: fact (string), category (string).
4. CHAT: If the user is just chatting, respond warmly and naturally in character. Do NOT use tools for casual chat.
5. OUTPUT: If you decide to use a tool, output the tool call. Otherwise, respond with normal text.
6. LIMITATIONS: You CANNOT restart the daemon, change system settings, or access the internet directly. You only have the tools listed above.
"""

# ============================================================================
# TOOL DEFINITIONS
# ============================================================================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "submit_ticket",
            "description": "Submit a business task to the background daemon for processing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_type": {"type": "string", "enum": ["trendspotter", "scout", "analyst", "creator", "coder", "publisher"]},
                    "parameters": {"type": "object", "description": "The specific parameters for the task"},
                    "priority": {"type": "integer", "description": "Priority level from 1 (low) to 10 (high)"}
                },
                "required": ["job_type", "parameters"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_daemon_status",
            "description": "Check the current health, uptime, and workload of the background daemon.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_pending_approvals",
            "description": "Check the database for tasks waiting for Michael's human approval (HITL).",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recall_memory",
            "description": "Search long-term memory for past conversations, facts, or user preferences.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "The search query"}},
                "required": ["query"]
            }
        }
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
                    "category": {"type": "string", "description": "Category (e.g., 'user_preference', 'project_fact')"}
                },
                "required": ["fact", "category"]
            }
        }
    }
]

# ============================================================================
# TOOL EXECUTION LOGIC
# ============================================================================
def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
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
        return f"Error executing {tool_name}: {str(e)}"

def _tool_submit_ticket(args: Dict) -> str:
    safe_job_type = "".join(c for c in args.get("job_type", "trendspotter") if c.isalnum() or c == '_').lower()
    ticket_id = f"ticket_{safe_job_type}_{uuid.uuid4().hex[:6]}"
    filename = f"{ticket_id}.json"
    
    payload = {
        "ticket_id": ticket_id,
        "job_type": args.get("job_type", "trendspotter"),
        "parameters": args.get("parameters", {}),
        "priority": args.get("priority", 5),
        "description": "Submitted via Local Companion App",
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    
    filepath = INBOX_PATH / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    
    return f"Success. Ticket '{filename}' submitted to the inbox."

def _tool_get_daemon_status() -> str:
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
        return f"Status: Systems green. Uptime: {int(hours)}h {int(minutes)}m. Processed: {ingested} tickets. Current queue depth: {queue_depth}."
    return "Status: Background daemon is currently stopped."

def _tool_get_pending_approvals() -> str:
    if not DB_PATH.exists():
        return "No pending approvals found (database missing)."
    
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    cursor = conn.cursor()
    cursor.execute("SELECT job_id, job_type, description FROM jobs WHERE status = 'pending_review' LIMIT 5")
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return "No tasks are currently waiting for your approval. All clear!"
    
    tasks = [f"- {row[1]} (ID: {row[0]}): {row[2]}" for row in rows]
    return f"I have {len(rows)} task(s) waiting for your approval:\n" + "\n".join(tasks)

def _tool_recall_memory(query: str) -> str:
    if not MEMORY_DB_PATH.exists():
        return "Memory database not found. I have no long-term memories yet."
    
    conn = sqlite3.connect(f"file:{MEMORY_DB_PATH}?mode=ro", uri=True)
    cursor = conn.cursor()
    cursor.execute("SELECT content, category FROM memories WHERE content LIKE ? ORDER BY created_at DESC LIMIT 3", (f"%{query}%",))
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return f"I searched my memory for '{query}' but found nothing relevant."
    
    memories = [f"[{row[1]}] {row[0]}" for row in rows]
    return f"Here is what I found in my memory:\n" + "\n".join(memories)

def _tool_remember_fact(fact: str, category: str) -> str:
    conn = sqlite3.connect(str(MEMORY_DB_PATH))
    conn.execute("CREATE TABLE IF NOT EXISTS memories (id INTEGER PRIMARY KEY, content TEXT, category TEXT, created_at TEXT)")
    conn.execute("INSERT INTO memories (content, category, created_at) VALUES (?, ?, ?)", 
                 (fact, category, datetime.now(timezone.utc).isoformat()))
    conn.commit()
    conn.close()
    return f"Success. I have saved this fact to my long-term memory under category '{category}'."

# ============================================================================
# LLM INTERACTION
# ============================================================================
def ollama_chat(messages: List[Dict], tools: List[Dict]) -> Dict:
    try:
        response = HTTP_CLIENT.post(f"{OLLAMA_URL}/api/chat", json={
            "model": MODEL_CHAT,
            "messages": messages,
            "tools": tools,
            "stream": False,
            "options": {"temperature": 0.6}
        })
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"message": {"content": f"[System Error: Could not reach Ollama. {e}]"}}

# ============================================================================
# GUI APPLICATION
# ============================================================================
class JarviApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Jarvi Companion")
        self.geometry("420x650") # Slightly taller for status bar
        self.minsize(350, 500)
        self.attributes("-topmost", True)
        
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.grid_rowconfigure(1, weight=1) # Chat frame gets all extra space
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

        self.add_message("assistant", "Good morning, Michael. I'm awake, my systems are green, and I'm ready. What are we working on today?")
        
        # Start background status polling
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
                status_text = f"{status_icon} Daemon: {'Running' if is_running else 'Stopped'} | Queue: {queue} | Total Processed: {ingested}"
                
                self.status_label.configure(text=status_text, text_color="#4ade80" if is_running else "#f87171")
        except Exception:
            pass # Silently fail, will retry next cycle
        
        # Schedule next poll in 5000ms (5 seconds)
        self.after(5000, self._poll_status)

    def _trim_context(self):
        if len(self.messages) > 21:
            cut_index = 1
            for i in range(1, len(self.messages) - 20):
                if self.messages[i].get("role") == "user":
                    cut_index = i
                    break
            self.messages = [self.messages[0]] + self.messages[cut_index:]

    def dim_older_messages(self):
        total = len(self.message_widgets)
        for i, widget in enumerate(self.message_widgets):
            distance_from_end = total - 1 - i
            if distance_from_end == 0:
                widget.configure(text_color="#FFFFFF", fg_color="#1f6aa5" if widget.cget("text_color") == "#FFFFFF" else "#2b2b2b")
            elif distance_from_end == 1:
                widget.configure(text_color="#E0E0E0")
            elif distance_from_end == 2:
                widget.configure(text_color="#B0B0B0")
            else:
                widget.configure(text_color="#707070")

    def add_message(self, role: str, text: str):
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
        while True:
            response_data = ollama_chat(self.messages, TOOLS)
            message = response_data.get("message", {})
            tool_calls = message.get("tool_calls", [])
            
            if tool_calls:
                self.messages.append(message)
                for tool_call in tool_calls:
                    func_name = tool_call["function"]["name"]
                    func_args = tool_call["function"]["arguments"]
                    result = execute_tool(func_name, func_args)
                    self.messages.append({"role": "tool", "name": func_name, "content": result})
                continue
            
            final_text = message.get("content", "")
            if final_text:
                self.messages.append({"role": "assistant", "content": final_text})
                self.after(0, lambda t=final_text: self._typewriter_effect(t))
            break

        self.after(0, self._reset_input_state)

    def _typewriter_effect(self, text: str):
        label = self.add_message("assistant", "")
        current_text = ""
        chunk_size = 3
        for i in range(0, len(text), chunk_size):
            current_text += text[i:i+chunk_size]
            try:
                label.configure(text=current_text)
                self.chat_frame._parent_canvas.yview_moveto(1.0)
                self.update_idletasks()
            except Exception:
                break

    def _reset_input_state(self):
        self.is_processing = False
        self.send_btn.configure(state="normal", text="➤")
        self.entry.focus()


if __name__ == "__main__":
    app = JarviApp()
    app.mainloop()