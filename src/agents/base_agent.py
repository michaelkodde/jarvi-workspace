"""
Base Agent: Abstract base class for all Jarvi agents.
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.utils.automation_guard import retry_with_backoff
from src.utils.llm_client import OllamaAPIError, OllamaResponse
from src.utils.token_ledger import TokenLedger, TokenUsage

logger = logging.getLogger(__name__)

__all__ = ["AgentResult", "BaseAgent"]


@dataclass(frozen=True)
class AgentResult:
    """Strongly typed result from agent execution."""
    agent_name: str
    artifact: str
    token_usage: dict[str, int]


class BaseAgent(ABC):
    """
    Abstract base class for all Jarvi agents.
    
    Provides common functionality: playbook loading, LLM communication,
    token logging, and markdown stripping.
    """

    def __init__(
        self,
        llm_client: Callable[..., OllamaResponse],
        token_ledger: TokenLedger,
        playbook_path: str,
        agent_name: str,
        model_name: str,
    ) -> None:
        self.llm_client = llm_client
        self.token_ledger = token_ledger
        self.playbook_path = Path(playbook_path)
        self.agent_name = agent_name
        self.model_name = model_name
        self.playbook = self._load_playbook()

    def _load_playbook(self) -> str:
        """Loads the agent's playbook from disk."""
        if not self.playbook_path.exists():
            raise FileNotFoundError(f"Playbook not found: {self.playbook_path}")
        
        with open(self.playbook_path, "r", encoding="utf-8") as f:
            return f.read()

    @retry_with_backoff
    def _call_llm(
        self,
        prompt: str,
        system_prompt: str | None = None,
        context: str | None = None,
    ) -> OllamaResponse:
        """
        Calls the LLM with retry logic.
        
        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt override.
            context: Optional context to append to the prompt.
            
        Returns:
            OllamaResponse from the LLM.
        """
        full_prompt = prompt
        if context:
            full_prompt = f"<context>\n{context}\n</context>\n\n{prompt}"
        
        response = self.llm_client(
            model=self.model_name,
            prompt=full_prompt,
            system_prompt=system_prompt or self.playbook,
            options={"keep_alive": "-1"},
        )
        
        self.token_ledger.log_usage(
            TokenUsage(
                agent_name=self.agent_name,
                model_name=self.model_name,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                total_tokens=response.total_tokens,
            )
        )
        
        return response

    def _strip_markdown_fences(self, content: str) -> str:
        """Removes markdown code fences from LLM output."""
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        
        if content.endswith("```"):
            content = content[:-3]
        
        return content.strip()

    @abstractmethod
    def execute(self, task_prompt: str, context: str | None = None) -> AgentResult:
        """
        Executes the agent's primary task.
        
        Args:
            task_prompt: The task to perform.
            context: Optional context for the task.
            
        Returns:
            AgentResult containing the artifact and token usage.
        """
        pass