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
# Changed from "general" to "analyst" as "analyst" is a valid fallback pipeline
DEFAULT_AGENT = "analyst"
DEFAULT_MODEL = "qwen2.5:7b"

_OUTPUT_CONTRACT = (
    "Classify the request inside the <request> tags.\n"
    "Follow the routing policy in your system instructions.\n"
    "Output ONE JSON object. No markdown, no prose, no code fences.\n"
    "Required keys: target_agent (string), target_model (string), "
    "confidence (number between 0 and 1).\n"
    'Example: {"target_agent":"analyst","target_model":"qwen2.5:7b","confidence":0.9}\n'
)


class RoutingError(Exception):
    """Raised when a prompt cannot be classified into a valid route."""


@dataclass
class RoutingResult:
    """A routing decision.

    Attributes:
        target_agent: Specialist pipeline that should handle the request.
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
        pipeline_config_path: Path to pipelines.json for dynamic registry.
        llm_callable: Injected inference callable. Defaults to query_ollama.
        token_ledger: Injected ledger.

    Raises:
        FileNotFoundError: If the playbook does not exist.
        ValueError: If the playbook is empty or whitespace-only.
    """

    def __init__(
        self,
        ollama_model: str,
        playbook_path: str,
        pipeline_config_path: str = "config/pipelines.json",
        llm_callable: Callable[..., OllamaResponse] = query_ollama,
        token_ledger: TokenLedger | None = None,
    ) -> None:
        self._model = ollama_model
        self._playbook_path = Path(playbook_path)
        self._pipeline_config_path = Path(pipeline_config_path)
        self._llm = llm_callable
        self._ledger = token_ledger if token_ledger is not None else TokenLedger()
        
        # 1. Load valid pipelines from the single source of truth
        self._valid_pipelines = self._load_valid_pipelines()
        
        # 2. Build the system prompt with dynamic registry injection
        self._system_prompt = self._load_playbook() + self._build_registry_injection()

    def _load_valid_pipelines(self) -> list[str]:
        """Load valid pipeline names from config/pipelines.json."""
        if not self._pipeline_config_path.exists():
            logger.warning(
                "Pipeline config not found at %s. Using hardcoded fallbacks.", 
                self._pipeline_config_path
            )
            return ["analyst", "scout", "trendspotter", "content_pipeline", "code_review_pipeline"]
        
        try:
            with open(self._pipeline_config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return list(data.get("pipelines", {}).keys())
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to parse pipeline config: %s. Using fallbacks.", e)
            return ["analyst", "scout", "trendspotter", "content_pipeline", "code_review_pipeline"]

    def _build_registry_injection(self) -> str:
        """Dynamically inject the valid pipeline list into the system prompt."""
        pipelines_str = ", ".join(f'"{p}"' for p in self._valid_pipelines)
        fallback = "analyst" if "analyst" in self._valid_pipelines else self._valid_pipelines[0]
        
        return (
            f"\n\n## STRICT ROUTING REGISTRY (DO NOT DEVIATE)\n"
            f"You MUST set 'target_agent' to EXACTLY one of these valid pipeline names: {pipelines_str}.\n"
            f"Do not invent new names. Do not use variations (e.g., use 'trendspotter', not 'trend_spotter').\n"
            f"If the request does not clearly match one, default to '{fallback}'."
        )

    def _load_playbook(self) -> str:
        """Load and validate the routing policy."""
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
        """Extract the first complete JSON object from model output."""
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
        """Validate and extract required fields from the model's JSON output."""
        target_agent = data.get("target_agent")
        if not target_agent or not isinstance(target_agent, str) or not target_agent.strip():
            raise RoutingError("Missing or invalid 'target_agent' field.")
        target_agent = target_agent.strip()

        target_model = data.get("target_model")
        if not target_model or not isinstance(target_model, str) or not target_model.strip():
            raise RoutingError("Missing or invalid 'target_model' field.")
        target_model = target_model.strip()

        raw_confidence = data.get("confidence")
        if raw_confidence is None:
            raise RoutingError("Missing 'confidence' field.")
        
        try:
            confidence = float(raw_confidence)
        except (ValueError, TypeError) as exc:
            raise RoutingError(f"'confidence' must be a number, got {type(raw_confidence).__name__}") from exc

        if not 0.0 <= confidence <= 1.0:
            raise RoutingError(f"Confidence {confidence} is outside valid range [0.0, 1.0].")

        return target_agent, target_model, confidence

    def _record_usage(self, usage: dict[str, int]) -> None:
        """Bill the model that actually ran: the router, not the target."""
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
        """Classify a request into a specialist route."""
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
                timeout=(1.0, 10),
            )
        except OllamaAPIError as exc:
            logger.critical("Router unavailable; degrading to %s: %s", DEFAULT_AGENT, exc)
            fallback_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            self._record_usage(fallback_usage)
            return RoutingResult(DEFAULT_AGENT, DEFAULT_MODEL, prompt, 0.0, fallback_usage)
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
            
            # --- DYNAMIC REGISTRY ENFORCEMENT ---
            if target_agent not in self._valid_pipelines:
                fallback_agent = "analyst" if "analyst" in self._valid_pipelines else self._valid_pipelines[0]
                logger.warning(
                    "Router hallucinated invalid target_agent '%s'. Falling back to '%s'. Valid: %s",
                    target_agent, fallback_agent, self._valid_pipelines
                )
                target_agent = fallback_agent
            # ------------------------------------
            
        except RoutingError:
            self._record_usage(usage)
            raise
        except (ValueError, TypeError) as exc:
            self._record_usage(usage)
            raise RoutingError(f"Malformed router response: {response.content[:100]}") from exc

        if confidence < MIN_CONFIDENCE:
            logger.warning("Low confidence (%.2f); degrading to %s.", confidence, DEFAULT_AGENT)
            result = RoutingResult(DEFAULT_AGENT, DEFAULT_MODEL, prompt, 0.0, usage)
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