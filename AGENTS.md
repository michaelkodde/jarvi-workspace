# Jarvi Coding Standards (Constitution)

You are coding for "Jarvi", a local-first, privacy-preserving AI agent framework.
Strictly adhere to these rules when generating or modifying code:

## Core Rules
1. **Exception Chaining (CRITICAL):** Never use bare `raise` or swallow 3rd-party exceptions. Always raise a custom module exception with a machine-readable error code, chained from the original: `raise CustomError("msg", error_code="CODE") from e`.
2. **Strict Typing:** All function signatures must have explicit type hints. Use `from __future__ import annotations`.
3. **No Global State:** Use Dependency Injection. Do not instantiate heavy resources (DB, LLM) at the module level.
4. **Resilience:** Never use `sys.exit()`. Allow the framework to handle graceful shutdown.
5. **Logging:** Use `logging.getLogger(__name__)`. Log `INFO` for success, `ERROR` for failures. **NEVER log PII, raw user prompts, or API keys.**
6. **Local-First:** Do not import or suggest cloud APIs (OpenAI, Anthropic). Use `src.utils.llm_client` for Ollama.
7. **Documentation:** Use strict Google-style docstrings for all public methods.

## Architecture-Specific Rules
8. **MCP-Ready Schema:** All data structures exchanged between components must be frozen dataclasses or plain dicts. No custom objects that can't be JSON-serialized.
9. **Memory Engine Tags:** Tags must match regex `^[a-z][a-z0-9_]{0,31}$`. No hyphens, no uppercase, max 32 chars.
10. **Agent Contract:** All agents must inherit from `BaseAgent` and implement `execute(task_prompt: str, context: str | None = None) -> AgentResult`. All LLM calls must log token usage via `TokenLedger`.
11. **Tool Isolation:** Tools must be stateless. No shared mutable state between tool invocations.
12. **Persistence:** All state must be persisted to SQLite. No in-memory-only state that would be lost on restart.
13. **HITL Gates:** High-risk actions (publishing, spending money) must pause and wait for user approval via the `QueryAPI.approve_job()` / `reject_job()` flow.