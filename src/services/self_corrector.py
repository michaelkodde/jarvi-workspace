# src/services/self_corrector.py
"""
Self Corrector (Module 3).
Post-execution self-correction module that intercepts validation failures, 
constructs Reflexion Prompt, and re-invokes the LLM with Chain-of-Thought.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol

__all__ = [
    "SelfCorrector",
    "SelfCorrectorError",
    "CorrectionResult",
    "ErrorCode",
]

logger = logging.getLogger(__name__)

# Constants
MAX_RETRIES = 2
BASE_BACKOFF_SEC = 2.0  # Exponential backoff: 2s, 4s

class ErrorCode(Enum):
    AGENT_NOT_FOUND = "AGENT_NOT_FOUND"
    MAX_RETRIES_EXHAUSTED = "MAX_RETRIES_EXHAUSTED"
    REFLEXION_FAILED = "REFLEXION_FAILED"
    INVALID_AGENT_RESPONSE = "INVALID_AGENT_RESPONSE"
    INTERNAL = "INTERNAL"

class SelfCorrectorError(Exception):
    """Raised when self-correction fails."""
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.INTERNAL, original_error: Optional[Exception] = None) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.original_error = original_error

class LLMClientProtocol(Protocol):
    """Protocol for the LLM client."""
    def generate(self, model: str, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Any:
        ...

@dataclass(frozen=True)
class CorrectionResult:
    """Result of a self-correction operation."""
    status: str  # "success", "escalated_to_hitl", "failed"
    job_id: str
    retry_count: int
    corrected_output: Optional[str]
    thinking_log: List[str]
    error: Optional[str]
    error_code: Optional[ErrorCode]

class SelfCorrector:
    """
    Post-execution self-correction module that implements Reflexion-style retry.
    """

    def __init__(
        self,
        llm_client: LLMClientProtocol,
        model_name: str,
        max_retries: int = MAX_RETRIES,
        base_backoff: float = BASE_BACKOFF_SEC,
    ) -> None:
        self.llm_client = llm_client
        self.model_name = model_name
        self.max_retries = max_retries
        self.base_backoff = base_backoff

    def _build_reflexion_prompt(
        self,
        agent_name: str,
        original_prompt: str,
        error_message: str,
        failed_output: str,
        attempt_number: int,
    ) -> str:
        """Constructs a Reflexion Prompt with Chain-of-Thought scratchpad."""
        return f"""You are performing a self-correction cycle for the agent: {agent_name}

ORIGINAL TASK:
{original_prompt}

YOUR PREVIOUS ATTEMPT (FAILED):
{failed_output}

ERROR MESSAGE:
{error_message}

ATTEMPT NUMBER: {attempt_number} of {self.max_retries + 1}

INSTRUCTIONS:
1. First, analyze WHY your previous attempt failed. What went wrong?
2. Think step-by-step about how to fix this issue.
3. Generate a corrected output that addresses the error.

Use the following XML structure for your response:

<thinking>
Analyze the error. What specific constraint or validation rule did you violate? 
How will you fix it in this attempt?
</thinking>

<corrected_output>
Your corrected response goes here. Must be valid and pass validation.
</corrected_output>

CRITICAL: Your corrected output MUST be different from your previous attempt. 
Address the specific error mentioned above."""

    def _extract_xml_tag(self, text: str, tag_name: str) -> Optional[str]:
        """Robustly extracts content from an XML tag, ignoring markdown wrappers."""
        # Remove markdown code blocks if present
        clean_text = re.sub(r'^```[a-zA-Z]*\n?', '', text)
        clean_text = re.sub(r'\n?```$', '', clean_text)
        
        pattern = rf"<{tag_name}>(.*?)</{tag_name}>"
        match = re.search(pattern, clean_text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def correct(
        self,
        job_id: str,
        agent_name: str,
        original_prompt: str,
        error: Exception,
        failed_output: str,
    ) -> CorrectionResult:
        """
        Attempts to self-correct a failed agent execution using Reflexion.
        """
        thinking_log: List[str] = []
        last_error: Exception = error
        current_failed_output = failed_output

        for attempt in range(1, self.max_retries + 1):
            logger.info(
                "Self-correction attempt %d/%d for job %s (agent: %s). Error: %s",
                attempt, self.max_retries, job_id, agent_name, str(last_error)
            )

            # Exponential backoff before retry
            if attempt > 1:
                backoff_sec = self.base_backoff ** (attempt - 1)
                logger.info("Backing off for %.1f seconds before retry...", backoff_sec)
                time.sleep(backoff_sec)

            # Build reflexion prompt
            reflexion_prompt = self._build_reflexion_prompt(
                agent_name=agent_name,
                original_prompt=original_prompt,
                error_message=str(last_error),
                failed_output=current_failed_output,
                attempt_number=attempt,
            )

            # Invoke LLM with reflexion prompt
            try:
                llm_response = self.llm_client.generate(
                    model=self.model_name,
                    prompt=reflexion_prompt,
                    options={"temperature": 0.3}  # Lower temp for deterministic correction
                )
                
                response_content = getattr(llm_response, "content", str(llm_response))
                
                thinking_text = self._extract_xml_tag(response_content, "thinking") or "Thinking log not captured."
                thinking_log.append(f"Attempt {attempt}:\n{thinking_text}")
                
                corrected_output = self._extract_xml_tag(response_content, "corrected_output")
                
                if not corrected_output:
                    logger.warning("Reflexion failed to produce <corrected_output> on attempt %d", attempt)
                    last_error = SelfCorrectorError("Reflexion did not produce valid corrected output", ErrorCode.REFLEXION_FAILED)
                    current_failed_output = response_content
                    continue

                # Verify the output actually changed
                if corrected_output.strip() == current_failed_output.strip():
                    logger.warning("Corrected output is identical to failed output. Retrying...")
                    last_error = SelfCorrectorError("Corrected output did not change", ErrorCode.INVALID_AGENT_RESPONSE)
                    current_failed_output = corrected_output
                    continue

                # Success!
                logger.info("Self-correction successful for job %s on attempt %d", job_id, attempt)
                return CorrectionResult(
                    status="success",
                    job_id=job_id,
                    retry_count=attempt,
                    corrected_output=corrected_output,
                    thinking_log=thinking_log,
                    error=None,
                    error_code=None,
                )

            except Exception as e:
                logger.error("Reflexion LLM call failed on attempt %d: %s", attempt, type(e).__name__)
                last_error = e
                current_failed_output = ""
                thinking_log.append(f"Attempt {attempt}: LLM call failed - {str(e)}")

        # Max retries exhausted
        logger.error("Self-correction failed after %d attempts for job %s. Escalating to HITL.", self.max_retries, job_id)
        return CorrectionResult(
            status="escalated_to_hitl",
            job_id=job_id,
            retry_count=self.max_retries,
            corrected_output=None,
            thinking_log=thinking_log,
            error=f"Max retries exhausted. Last error: {str(last_error)}",
            error_code=ErrorCode.MAX_RETRIES_EXHAUSTED,
        )