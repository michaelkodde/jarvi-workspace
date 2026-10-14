"""
Analyst Agent: Validates niches against ethical and supply-chain rules.
"""
from typing import Callable

from src.agents.base_agent import BaseAgent
from src.utils.llm_client import OllamaResponse
from src.utils.token_ledger import TokenLedger

__all__ = ["Analyst"]


class Analyst(BaseAgent):
    """
    Concrete specialist agent for niche validation and PRD generation.
    
    Inherits all execution, truncation, and logging logic from BaseAgent.
    Takes a single raw niche JSON (including confidence score) and outputs 
    a strictly formatted Markdown PRD with an immediate Final Verdict.
    """

    def __init__(
        self,
        llm_client: Callable[..., OllamaResponse],
        token_ledger: TokenLedger,
        model_name: str,
        playbook_path: str = "src/agents/playbooks/analyst.md",
    ) -> None:
        super().__init__(
            llm_client=llm_client,
            token_ledger=token_ledger,
            playbook_path=playbook_path,
            agent_name="Analyst",
            model_name=model_name,
        )