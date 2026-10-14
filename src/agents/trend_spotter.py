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