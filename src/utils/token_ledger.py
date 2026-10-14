"""
Token Ledger: Thread-safe logging of LLM token usage and estimated costs.
"""
import csv
import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["TokenLedger", "TokenUsage"]


@dataclass
class TokenUsage:
    """Strongly typed token usage record.

    Attributes:
        agent_name: Name of the agent that consumed tokens (e.g., 'router').
        model_name: Model that was used (e.g., 'qwen2.5:3b').
        prompt_tokens: Number of tokens in the input prompt.
        completion_tokens: Number of tokens in the model's response.
        total_tokens: Sum of prompt and completion tokens.
    """
    agent_name: str
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class TokenLedger:
    """Thread-safe ledger for logging LLM token usage to a CSV file.

    Automatically calculates costs based on the token_costs map in config.json.
    All file operations are protected by a threading.Lock to prevent corruption
    when multiple agents log concurrently.

    Args:
        config_path: Path to config.json containing token_costs map.
        log_dir: Directory where cost_log.csv will be created.

    Raises:
        IOError: If the ledger cannot write to the log directory.
    """

    def __init__(self, config_path: str = "config.json", log_dir: str = "dashboard") -> None:
        self.log_dir = Path(log_dir)
        self.log_file = self.log_dir / "cost_log.csv"
        self.config_path = Path(config_path)

        self._lock = threading.Lock()

        self._ensure_directory()
        self._ensure_csv_header()
        self.cost_map = self._load_cost_map()

    def _ensure_directory(self) -> None:
        """Creates the log directory if it does not exist."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_csv_header(self) -> None:
        """Writes the CSV header if the file is new or empty."""
        if not self.log_file.exists() or self.log_file.stat().st_size == 0:
            with open(self.log_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "agent_name", "model_name",
                    "prompt_tokens", "completion_tokens", "total_tokens", "estimated_cost"
                ])

    def _load_cost_map(self) -> dict[str, float]:
        """Loads cost per 1k tokens from config.json.

        Returns:
            Dictionary mapping model names to cost per 1k tokens.
            Returns empty dict if config.json is missing or malformed.
        """
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                return config.get("token_costs", {})
        except FileNotFoundError:
            logger.warning("Config file not found: %s. Defaulting to 0.0 costs.", self.config_path)
            return {}
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Failed to load token costs from config: %s. Defaulting to 0.0", e)
            return {}

    def _calculate_cost(self, model_name: str, total_tokens: int) -> float:
        """Calculates estimated cost based on model and token count.

        Args:
            model_name: The model that was used.
            total_tokens: Total tokens consumed.

        Returns:
            Estimated cost in dollars. Returns 0.0 if model not in cost map.
        """
        cost_per_1k = self.cost_map.get(model_name, 0.0)
        return (total_tokens / 1000.0) * cost_per_1k

    def log_usage(self, usage: TokenUsage) -> None:
        """Validates, calculates cost, and appends a token usage record to the CSV file.

        This method is thread-safe. Multiple agents can call it concurrently
        without corrupting the CSV file.

        Args:
            usage: The TokenUsage dataclass instance to log.

        Raises:
            ValueError: If any token count is negative.
            IOError: If the ledger cannot write to the file (after logging the error).
        """
        if usage.prompt_tokens < 0 or usage.completion_tokens < 0 or usage.total_tokens < 0:
            raise ValueError(f"Token counts cannot be negative: {usage}")

        estimated_cost = self._calculate_cost(usage.model_name, usage.total_tokens)

        with self._lock:
            try:
                with open(self.log_file, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.now().isoformat(),
                        usage.agent_name,
                        usage.model_name,
                        usage.prompt_tokens,
                        usage.completion_tokens,
                        usage.total_tokens,
                        f"${estimated_cost:.6f}"
                    ])
                logger.debug("Logged %d tokens for %s to %s", usage.total_tokens, usage.agent_name, self.log_file)
            except (IOError, PermissionError) as e:
                logger.error("Failed to write to token ledger: %s", e)
                raise

    def shutdown(self) -> None:
        """Graceful shutdown method for daemon compatibility.

        Currently a no-op since files are opened/closed per-write, but exists
        for API consistency with other daemon-safe modules.
        """
        logger.info("Token Ledger shutting down.")