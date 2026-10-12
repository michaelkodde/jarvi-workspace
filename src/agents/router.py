"""
Router Agent: Classifies incoming prompts and routes them to specialist agents.
"""
import logging
import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any

from src.utils.llm_client import query_ollama, OllamaAPIError, OllamaResponse

logger = logging.getLogger(__name__)

__all__ = ["Router", "RoutingResult", "RoutingError"]


class RoutingError(Exception):
    """Custom exception for routing failures."""
    pass


@dataclass
class RoutingResult:
    """
    Strongly typed output for the Router agent.
    """
    target_agent: str
    target_model: str
    cleaned_prompt: str
    confidence: float
    token_usage: dict[str, int]


class Router:
    """
    Routes user prompts to the appropriate specialist agent based on a playbook.
    """
    
    MAX_INPUT_CHARS = 6000 

    def __init__(
        self, 
        ollama_model: str, 
        playbook_path: str, 
        llm_callable: Callable[..., OllamaResponse] = query_ollama
    ):
        self.ollama_model = ollama_model
        self.playbook_path = Path(playbook_path)
        self.llm_callable = llm_callable
        self.system_prompt = self._load_and_validate_playbook()

    def _load_and_validate_playbook(self) -> str:
        """Fails fast if the routing playbook is missing or empty."""
        if not self.playbook_path.exists():
            logger.critical(f"Router playbook not found: {self.playbook_path}")
            raise FileNotFoundError(f"Router playbook not found: {self.playbook_path}")
        
        content = self.playbook_path.read_text(encoding="utf-8")
        if not content.strip():
            logger.critical(f"Router playbook is empty: {self.playbook_path}")
            raise ValueError(f"Router playbook is empty: {self.playbook_path}")
            
        return content

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        start_idx = text.find('{')
        if start_idx == -1:
            raise ValueError("No JSON object found in text")
        
        decoder = json.JSONDecoder()
        try:
            data, _ = decoder.raw_decode(text, start_idx)
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode JSON: {e}")

    def route(self, raw_prompt: str) -> RoutingResult:
        start_time = time.time()
        
        if len(raw_prompt) > self.MAX_INPUT_CHARS:
            logger.warning("Prompt exceeded character limit; truncating for routing.")
            safe_prompt = raw_prompt[:self.MAX_INPUT_CHARS] + "... [TRUNCATED]"
        else:
            safe_prompt = raw_prompt

        prompt = (
            "You are a routing engine. Analyze the user request and output a single JSON object.\n"
            "Example: {\"target_agent\": \"business\", \"target_model\": \"qwen2.5:7b\", \"cleaned_prompt\": \"find pans\", \"confidence\": 0.9}\n"
            f"Request: {safe_prompt}"
        )

        try:
            response = self.llm_callable(
                model=self.ollama_model,
                prompt=prompt,
                system_prompt=self.system_prompt,
                options={"temperature": 0.1}, 
                timeout=(0.5, 1.0) 
            )
            
            try:
                data = self._extract_json(response.content)
                
                result = RoutingResult(
                    target_agent=str(data.get("target_agent", "general")),
                    target_model=str(data.get("target_model", "qwen2.5:7b")),
                    cleaned_prompt=str(data.get("cleaned_prompt", safe_prompt)),
                    confidence=float(data.get("confidence", 0.5)),
                    token_usage={
                        "prompt_tokens": response.prompt_tokens,
                        "completion_tokens": response.completion_tokens,
                        "total_tokens": response.total_tokens
                    }
                )
                
                if result.confidence < 0.7:
                    logger.warning(f"Low routing confidence ({result.confidence}). Falling back to default.")
                    return RoutingResult(
                        target_agent="general",
                        target_model="qwen2.5:7b",
                        cleaned_prompt=safe_prompt,
                        confidence=0.0,
                        token_usage=result.token_usage 
                    )

            except (ValueError, TypeError) as e:
                raise RoutingError(f"LLM returned invalid routing schema: {response.content[:100]}") from e

            exec_time_ms = int((time.time() - start_time) * 1000)
            logger.info(
                f"Routed to '{result.target_agent}' ({result.target_model}) "
                f"in {exec_time_ms}ms | Tokens: {result.token_usage['total_tokens']}"
            )
            return result

        except OllamaAPIError as e:
            logger.critical(f"Ollama routing failed. Falling back to default. Error: {e}")
            return RoutingResult(
                target_agent="general",
                target_model="qwen2.5:7b",
                cleaned_prompt=safe_prompt,
                confidence=0.0,
                token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            )
        except RoutingError:
            raise
        except Exception as e:
            logger.critical(f"Unexpected routing failure: {e}")
            raise RoutingError("Unexpected routing failure") from e