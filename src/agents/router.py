"""Intent routing for the Jarvi pipeline.

Local-model design
------------------
Small local models classify well and rewrite poorly. The Router therefore
asks the model for only three things: an agent, a model, and a confidence
score. The user's prompt is carried forward by Python, unchanged apart from
the length guard.

This keeps generation short, prevents the model from silently altering user
intent, and removes a prompt-injection surface.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.utils.llm_client import OllamaAPIError, OllamaResponse, query_ollama
from src.utils.token_ledger import TokenLedger, TokenUsage

logger = logging.getLogger(__name__)

__all__ = ["Router", "RoutingResult", "RoutingError"]

MAX_INPUT_CHARS = 6000
MIN_CONFIDENCE = 0.7
DEFAULT_AGENT = "general"
DEFAULT_MODEL = "qwen2.5:7b"

_OUTPUT_CONTRACT = (
    "Classify the request inside the <request> tags.\n"
    "Follow the routing policy in your system instructions.\n"
    "Output ONE JSON object. No markdown, no prose, no code fences.\n"
    "Required keys: target_agent (string), target_model (string), "
    "confidence (number between 0 and 1).\n"
    'Example: {"target_agent":"general","target_model":"qwen2.5:7b","confidence":0.9}\n'
)


class RoutingError(Exception):
    """Raised when a prompt cannot be classified into a valid route."""


@dataclass
class RoutingResult:
    """A routing decision.

    Attributes:
        target_agent: Specialist that should handle the request.
        target_model: Model the pipeline should load next.
        cleaned_prompt: Original request, length-bounded but never rewritten.
        confidence: Model's stated confidence, between 0.0 and 1.0.
        token_usage: Tokens consumed by the ROUTER call, not the specialist.
    """

    target_agent: str
    target_model: str
    cleaned_prompt: str
    confidence: float
    token_usage: dict[str, int]


class Router:
    """Classify requests and degrade safely when the local model is unavailable.

    Args:
        ollama_model: Lightweight model used for routing (e.g., qwen2.5:3b).
        playbook_path: Markdown policy injected as the system message.
        llm_callable: Injected inference callable. Defaults to query_ollama.
        token_ledger: Injected ledger. A workspace-default ledger is used if
            omitted; tests must always inject one to stay isolated.

    Raises:
        FileNotFoundError: If the playbook does not exist.
        ValueError: If the playbook is empty or whitespace-only.
    """

    def __init__(
        self,
        ollama_model: str,
        playbook_path: str,
        llm_callable: Callable[..., OllamaResponse] = query_ollama,
        token_ledger: TokenLedger | None = None,
    ) -> None:
        self._model = ollama_model
        self._playbook_path = Path(playbook_path)
        self._llm = llm_callable
        self._ledger = token_ledger if token_ledger is not None else TokenLedger()
        self._system_prompt = self._load_playbook()

    def _load_playbook(self) -> str:
        """Load and validate the routing policy.

        Returns:
            The playbook text, cached for the lifetime of the instance.

        Raises:
            FileNotFoundError: If the playbook is missing.
            ValueError: If the playbook contains no policy.
        """
        if not self._playbook_path.exists():
            logger.critical("Router playbook missing: %s", self._playbook_path)
            raise FileNotFoundError(f"Router playbook not found: {self._playbook_path}")

        content = self._playbook_path.read_text(encoding="utf-8")
        if not content.strip():
            logger.critical("Router playbook is empty: %s", self._playbook_path)
            raise ValueError(f"Router playbook is empty: {self._playbook_path}")

        return content

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        """Extract the first complete JSON object from model output.

        ``raw_decode`` stops at the end of the first object, so markdown
        fences, trailing commentary, and nested braces are handled without
        regular expressions.

        Args:
            text: Raw model output.

        Returns:
            The decoded JSON object.

        Raises:
            ValueError: If no decodable JSON object is present.
        """
        start = text.find("{")
        if start == -1:
            raise ValueError("No JSON object in router output.")

        try:
            data, _ = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError as exc:
            raise ValueError("Router output is not valid JSON.") from exc

        if not isinstance(data, dict):
            raise ValueError("Router output is not a JSON object.")

        return data

    @staticmethod
    def _validate_routing_data(data: dict[str, Any]) -> tuple[str, str, float]:
        """Validate and extract required fields from the model's JSON output.

        Local 3B/7B models frequently omit fields or return wrong types.
        This method enforces strict validation.

        Args:
            data: Decoded JSON object from the model.

        Returns:
            Tuple of (target_agent, target_model, confidence).

        Raises:
            RoutingError: If any required field is missing, empty, or invalid.
        """
        # Validate target_agent
        target_agent = data.get("target_agent")
        if not target_agent:
            raise RoutingError("Missing or empty 'target_agent' field.")
        if not isinstance(target_agent, str):
            raise RoutingError(f"'target_agent' must be a string, got {type(target_agent).__name__}.")
        target_agent = target_agent.strip()
        if not target_agent:
            raise RoutingError("'target_agent' is empty after stripping whitespace.")

        # Validate target_model
        target_model = data.get("target_model")
        if not target_model:
            raise RoutingError("Missing or empty 'target_model' field.")
        if not isinstance(target_model, str):
            raise RoutingError(f"'target_model' must be a string, got {type(target_model).__name__}.")
        target_model = target_model.strip()
        if not target_model:
            raise RoutingError("'target_model' is empty after stripping whitespace.")

        # Validate confidence with defensive type coercion
        raw_confidence = data.get("confidence")
        if raw_confidence is None:
            raise RoutingError("Missing 'confidence' field.")
        
        try:
            confidence = float(raw_confidence)
        except (ValueError, TypeError) as exc:
            raise RoutingError(
                f"'confidence' must be a number, got {type(raw_confidence).__name__}: {raw_confidence}"
            ) from exc

        # Validate range
        if not 0.0 <= confidence <= 1.0:
            raise RoutingError(f"Confidence {confidence} is outside valid range [0.0, 1.0].")

        return target_agent, target_model, confidence

    def _record_usage(self, usage: dict[str, int]) -> None:
        """Bill the model that actually ran: the router, not the target.

        Accounting is best-effort. A full disk or a bad price map must never
        take down a routing decision that already succeeded.

        Args:
            usage: Token counts from a completed router call.
        """
        try:
            self._ledger.log_usage(
                TokenUsage(
                    agent_name="router",
                    model_name=self._model,
                    prompt_tokens=usage["prompt_tokens"],
                    completion_tokens=usage["completion_tokens"],
                    total_tokens=usage["total_tokens"],
                )
            )
        except Exception as exc:
            logger.error("Token accounting failed; routing continues: %s", exc)

    def route(self, raw_prompt: str) -> RoutingResult:
        """Classify a request into a specialist route.

        Every terminal path records token usage, including malformed
        responses and low-confidence fallbacks. Only a transport failure
        before any response is received records zero usage.

        Args:
            raw_prompt: Untrusted user request.

        Returns:
            A routing decision with the prompt carried forward unchanged.

        Raises:
            RoutingError: If the model's response cannot be parsed or its
                confidence is outside the valid range.
        """
        started = time.perf_counter()

        if len(raw_prompt) > MAX_INPUT_CHARS:
            logger.warning("Router input truncated to %d characters.", MAX_INPUT_CHARS)
            prompt = raw_prompt[:MAX_INPUT_CHARS] + "... [TRUNCATED]"
        else:
            prompt = raw_prompt

        message = f"{_OUTPUT_CONTRACT}\n<request>\n{prompt}\n</request>"

        try:
            response = self._llm(
                model=self._model,
                prompt=message,
                system_prompt=self._system_prompt,
                options={"temperature": 0.0, "num_predict": 128},
                timeout=(0.5, 1.0),
            )
        except OllamaAPIError as exc:
            logger.critical("Router unavailable; degrading to %s: %s", DEFAULT_AGENT, exc)
            fallback_usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            self._record_usage(fallback_usage)
            return RoutingResult(
                DEFAULT_AGENT, DEFAULT_MODEL, prompt, 0.0, fallback_usage
            )
        except Exception as exc:
            logger.critical("Unexpected router transport failure: %s", exc)
            raise RoutingError("Unexpected router transport failure.") from exc

        usage = {
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "total_tokens": response.total_tokens,
        }

        try:
            data = self._extract_json(response.content)
            target_agent, target_model, confidence = self._validate_routing_data(data)
        except RoutingError:
            # Validation failures are already RoutingErrors; just record usage and re-raise
            self._record_usage(usage)
            raise
        except (ValueError, TypeError) as exc:
            self._record_usage(usage)
            raise RoutingError(
                f"Malformed router response: {response.content[:100]}"
            ) from exc

        if confidence < MIN_CONFIDENCE:
            logger.warning(
                "Low confidence (%.2f); degrading to %s.", confidence, DEFAULT_AGENT
            )
            result = RoutingResult(
                DEFAULT_AGENT, DEFAULT_MODEL, prompt, 0.0, usage
            )
        else:
            result = RoutingResult(
                target_agent=target_agent,
                target_model=target_model,
                cleaned_prompt=prompt,
                confidence=confidence,
                token_usage=usage,
            )

        self._record_usage(usage)
        logger.info(
            "route=%s next_model=%s elapsed_ms=%d tokens=%d",
            result.target_agent,
            result.target_model,
            int((time.perf_counter() - started) * 1000),
            usage["total_tokens"],
        )
        return result