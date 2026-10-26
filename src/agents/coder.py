# src/agents/coder.py
from __future__ import annotations
import subprocess
import logging
from pathlib import Path
from typing import Callable

from src.agents.base_agent import BaseAgent, AgentResult
from src.utils.llm_client import OllamaResponse
from src.utils.token_ledger import TokenLedger

logger = logging.getLogger(__name__)

class CoderError(Exception):
    """Custom exception for Coder agent failures, compliant with Jarvi error standards."""
    def __init__(self, message: str, error_code: str = "CODER_EXECUTION_FAILED"):
        super().__init__(message)
        self.error_code = error_code


class Coder(BaseAgent):
    """
    Autonomous code review and patching agent.
    Integrates with PipelineFlow for queued, HITL-gated code hardening.
    """

    def __init__(
        self,
        llm_client: Callable[..., OllamaResponse],
        token_ledger: TokenLedger,
        model_name: str,
        playbook_path: str = "src/agents/playbooks/coder.md",
    ) -> None:
        super().__init__(
            llm_client=llm_client,
            token_ledger=token_ledger,
            playbook_path=playbook_path,
            agent_name="Coder",
            model_name=model_name,
        )

    def execute(self, task_prompt: str, context: str | None = None) -> AgentResult:
        try:
            target_files = [p.strip() for p in (context or "").split(",") if p.strip()]
            if not target_files:
                raise CoderError("No target files provided in context", error_code="NO_TARGET_FILES")

            code_context = ""
            for file_path in target_files:
                path = Path(file_path)
                if path.exists():
                    code_context += f"\n=== {file_path} ===\n{path.read_text(encoding='utf-8')}\n"
                else:
                    code_context += f"\n=== {file_path} ===\n[FILE NOT FOUND]\n"

            response = self._call_llm(prompt=task_prompt, context=code_context)
            cleaned_artifact = self._strip_markdown_fences(response.content)

            try:
                test_result = subprocess.run(
                    ["python", "-m", "pytest", "tests/", "-v", "--tb=short"],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=Path.cwd(),
                )
                test_status = test_result.stdout if test_result.returncode == 0 else f"FAILURES:\n{test_result.stdout}\n{test_result.stderr}"
            except subprocess.TimeoutExpired:
                test_status = "Pytest execution timed out after 60 seconds."
            except Exception as test_err:
                test_status = f"Pytest execution failed: {test_err}"

            final_artifact = f"## Code Review & Patch Proposal\n\n{cleaned_artifact}\n\n## Current Test Status\n```\n{test_status}\n```"

            return AgentResult(
                agent_name=self.agent_name,
                artifact=final_artifact,
                token_usage={
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                    "total_tokens": response.total_tokens,
                }
            )

        except CoderError:
            raise
        except Exception as e:
            logger.error(f"💥 CODER AGENT CRASHED: {type(e).__name__}: {str(e)}")
            raise CoderError(
                f"Coder execution failed: {type(e).__name__}: {str(e)}",
                error_code="UNEXPECTED_EXECUTION_FAILURE"
            ) from e