"""
Base Agent: Abstract base class for all specialist agents in the Jarvi pipeline.
"""
import logging
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from src.utils.automation_guard import retry_with_backoff
from src.utils.llm_client import OllamaResponse
from src.utils.token_ledger import TokenLedger, TokenUsage

logger = logging.getLogger(__name__)

__all__ = ["BaseAgent", "AgentResult", "AgentExecutionError"]

MAX_INPUT_CHARS = 6000
TRUNCATION_SUFFIX = "... [TRUNCATED]"


class AgentExecutionError(Exception):
    """Custom exception for agent execution failures."""
    
    def __init__(self, agent_name: str, message: str) -> None:
        self.agent_name = agent_name
        super().__init__(f"Agent {agent_name} failed: {message}")


@dataclass
class AgentResult:
    """Strongly typed output for a specialist agent."""
    agent_name: str
    artifact: str
    token_usage: dict[str, int]


class BaseAgent(ABC):
    """
    Concrete base class for all specialist agents.
    
    Centralizes prompt engineering, LLM execution, and token accounting.
    Inherits from ABC to prevent direct instantiation, but provides a fully
    implemented execute() method to ensure DRY compliance across all agents.
    """

    def __init__(
        self,
        llm_client: Callable[..., OllamaResponse],
        token_ledger: TokenLedger,
        playbook_path: str,
        agent_name: str,
        model_name: str,
    ) -> None:
        # Prevent direct instantiation of the base class
        if type(self) is BaseAgent:
            raise TypeError("BaseAgent cannot be instantiated directly. Use a specialist subclass.")

        self._llm_client = llm_client
        self._token_ledger = token_ledger
        self._agent_name = agent_name
        self._model_name = model_name
        
        self._playbook_path = Path(playbook_path)
        self._system_prompt = self._load_playbook()

    def _load_playbook(self) -> str:
        """Loads and validates the agent's playbook."""
        if not self._playbook_path.exists():
            raise FileNotFoundError(f"Playbook not found: {self._playbook_path}")
        
        content = self._playbook_path.read_text(encoding="utf-8")
        if not content.strip():
            raise ValueError(f"Playbook is empty: {self._playbook_path}")
            
        return content

    @retry_with_backoff()
    def _call_llm(self, prompt: str, system_prompt: str) -> OllamaResponse:
        """Internal LLM invocation with automated retries."""
        return self._llm_client(
            model=self._model_name,
            prompt=prompt,
            system_prompt=system_prompt,
            options={"keep_alive": "5m", "temperature": 0.1},
            timeout=(5.0, 60.0)
        )

    def _truncate_inputs(self, task_prompt: str, context: Optional[str]) -> tuple[str, Optional[str]]:
        """Enforces the strict character limit on combined inputs, calculating exact XML overhead."""
        # Base overhead for the task block
        task_overhead = len("\n\n<task>\n\n</task>")
        max_task_len = MAX_INPUT_CHARS - task_overhead
        
        if len(task_prompt) >= max_task_len:
            task_prompt = task_prompt[:max_task_len - len(TRUNCATION_SUFFIX)] + TRUNCATION_SUFFIX
            context = None
        else:
            if context:
                # Additional overhead if context is present
                context_overhead = len("<context>\n\n</context>")
                max_context_len = MAX_INPUT_CHARS - task_overhead - len(task_prompt) - context_overhead - len(TRUNCATION_SUFFIX)
                
                if len(context) > max_context_len:
                    context = context[:max_context_len] + TRUNCATION_SUFFIX
                    
        return task_prompt, context

    def _strip_markdown(self, content: str) -> str:
        """Robustly strips the first markdown code block, ignoring filler text or trailing whitespace."""
        start_idx = content.find("```")
        if start_idx != -1:
            end_idx = content.find("```", start_idx + 3)
            if end_idx != -1:
                # Extract content between the first and second code fence
                block = content[start_idx + 3 : end_idx].strip()
                
                # Remove language tag if present (e.g., "json", "python")
                if block and not block.startswith(("{", "[", "<")):
                    block = block.split("\n", 1)[-1].strip()
                    
                return block
                
        return content.strip()

    def _log_tokens(self, usage: dict[str, int]) -> None:
        """Safely logs token usage to the ledger."""
        try:
            token_usage = TokenUsage(
                agent_name=self._agent_name,
                model_name=self._model_name,
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
                total_tokens=usage["total_tokens"]
            )
            self._token_ledger.log_usage(token_usage)
        except Exception as e:
            logger.warning("Failed to log token usage for %s: %s", self._agent_name, e)

    def execute(self, task_prompt: str, context: Optional[str] = None) -> AgentResult:
        """Executes the agent's core logic."""
        task_prompt, context = self._truncate_inputs(task_prompt, context)
        
        user_prompt_parts = []
        if context:
            user_prompt_parts.append(f"<context>\n{context}\n</context>")
        user_prompt_parts.append(f"<task>\n{task_prompt}\n</task>")
        user_prompt = "\n\n".join(user_prompt_parts)

        try:
            response = self._call_llm(user_prompt, self._system_prompt)
            
            usage = {
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "total_tokens": response.total_tokens
            }
            
            artifact = self._strip_markdown(response.content)
            self._log_tokens(usage)
            
            logger.info("%s executed successfully. Tokens: %d", self._agent_name, usage["total_tokens"])
            
            return AgentResult(
                agent_name=self._agent_name,
                artifact=artifact,
                token_usage=usage
            )
            
        except Exception as e:
            self._log_tokens({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
            logger.error("%s execution failed: %s", self._agent_name, e)
            # Use type name to prevent massive tracebacks in the exception message itself
            raise AgentExecutionError(self._agent_name, type(e).__name__) from e