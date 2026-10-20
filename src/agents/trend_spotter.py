"""
Trend Spotter Agent: Identifies high-demand, low-competition market niches.
"""
from typing import Callable

from src.agents.base_agent import BaseAgent
from src.utils.llm_client import OllamaResponse
from src.utils.token_ledger import TokenLedger

__all__ = ["TrendSpotter"]


class TrendSpotter(BaseAgent):
    """
    Concrete specialist agent for market trend analysis.
    
    Inherits all execution, truncation, and logging logic from BaseAgent.
    Outputs a strictly formatted JSON array of niche ideas based on a seed keyword.
    """

    def __init__(
        self,
        llm_client: Callable[..., OllamaResponse],
        token_ledger: TokenLedger,
        model_name: str,
        playbook_path: str = "src/agents/playbooks/trend_spotter.md",
    ) -> None:
        super().__init__(
            llm_client=llm_client,
            token_ledger=token_ledger,
            playbook_path=playbook_path,
            agent_name="TrendSpotter",
            model_name=model_name,
        )

        def execute(self, task_prompt: str, context: str | None = None) -> "AgentResult":
        """
        Executes the agent's primary task using inherited helpers.
        """
        from src.agents.base_agent import AgentResult
        
        # 1. Call the LLM using the inherited helper (playbook is auto-applied as system prompt)
        response = self._call_llm(prompt=task_prompt, context=context)
        
        # 2. Clean the output using the inherited helper
        cleaned_content = self._strip_markdown_fences(response.content)
        
        # 3. Return the standardized result
        return AgentResult(
            artifact=cleaned_content,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
        )