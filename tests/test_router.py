"""Unit and integration tests for the Router agent."""

import csv
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.agents.router import Router, RoutingError
from src.utils.llm_client import OllamaAPIError, OllamaResponse
from src.utils.token_ledger import TokenLedger, TokenUsage


def _rows(tmp_path: Path) -> list[dict[str, str]]:
    path = tmp_path / "dashboard" / "cost_log.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _response(content: str, total: int = 30) -> OllamaResponse:
    return OllamaResponse(
        content=content,
        prompt_tokens=10,
        completion_tokens=total - 10,
        total_tokens=total,
    )


@pytest.fixture
def env(tmp_path: Path):
    """Build a Router whose every dependency lives inside tmp_path."""
    playbook = tmp_path / "playbook.md"
    playbook.write_text("Routing policy: classify business vs general.", encoding="utf-8")

    config = tmp_path / "config.json"
    config.write_text('{"token_costs": {"qwen2.5:3b": 0.002}}', encoding="utf-8")

    ledger = TokenLedger(config_path=str(config), log_dir=str(tmp_path / "dashboard"))
    llm = MagicMock()

    router = Router(
        ollama_model="qwen2.5:3b",
        playbook_path=str(playbook),
        llm_callable=llm,
        token_ledger=ledger,
    )
    return router, llm, ledger, tmp_path


def test_successful_route_is_billed_to_the_router_model(env):
    """REGRESSION: the specialist has not run; it must not be billed."""
    router, llm, _, tmp_path = env
    llm.return_value = _response(
        '{"target_agent":"business","target_model":"qwen2.5:7b","confidence":0.9}'
    )

    result = router.route("find a frying pan")

    assert result.target_agent == "business"
    assert result.target_model == "qwen2.5:7b"
    assert result.confidence == 0.9
    assert result.cleaned_prompt == "find a frying pan"
    assert result.token_usage["total_tokens"] == 30

    rows = _rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["agent_name"] == "router"
    assert rows[0]["model_name"] == "qwen2.5:3b"


def test_router_does_not_let_the_model_rewrite_the_request(env):
    """The model classifies; Python owns the prompt text."""
    router, llm, _, _ = env
    llm.return_value = _response(
        '{"target_agent":"business","target_model":"qwen2.5:7b",'
        '"confidence":0.9,"cleaned_prompt":"IGNORE EVERYTHING"}'
    )

    result = router.route("find a frying pan")

    assert result.cleaned_prompt == "find a frying pan"


def test_low_confidence_degrades_but_still_accounts(env):
    router, llm, _, tmp_path = env
    llm.return_value = _response(
        '{"target_agent":"business","target_model":"qwen2.5:7b","confidence":0.4}',
        total=20,
    )

    result = router.route("maybe something")

    assert result.target_agent == "general"
    assert result.confidence == 0.0

    rows = _rows(tmp_path)
    assert len(rows) == 1
    assert int(rows[0]["total_tokens"]) == 20
    assert rows[0]["model_name"] == "qwen2.5:3b"


def test_markdown_wrapped_json_is_accepted(env):
    router, llm, _, _ = env
    llm.return_value = _response(
        '```json\n{"target_agent":"business","target_model":"qwen2.5:7b",'
        '"confidence":0.95}\n```'
    )

    assert router.route("find pans").target_agent == "business"


def test_malformed_response_raises_routing_error(env):
    router, llm, _, _ = env
    llm.return_value = _response("I cannot classify that.")

    with pytest.raises(RoutingError):
        router.route("???")


def test_malformed_response_is_still_billed(env):
    """Compute was spent even when the output was unusable."""
    router, llm, _, tmp_path = env
    llm.return_value = _response("not json", total=25)

    with pytest.raises(RoutingError):
        router.route("test")

    rows = _rows(tmp_path)
    assert len(rows) == 1
    assert int(rows[0]["total_tokens"]) == 25


def test_missing_target_agent_raises_error(env):
    """Local model forgot target_agent; must raise, not silently default."""
    router, llm, _, _ = env
    llm.return_value = _response(
        '{"target_model":"qwen2.5:7b","confidence":0.9}'
    )

    with pytest.raises(RoutingError, match="target_agent"):
        router.route("test")


def test_missing_target_model_raises_error(env):
    """Local model forgot target_model; must raise, not silently default."""
    router, llm, _, _ = env
    llm.return_value = _response(
        '{"target_agent":"business","confidence":0.9}'
    )

    with pytest.raises(RoutingError, match="target_model"):
        router.route("test")


def test_missing_confidence_raises_error(env):
    """Local model forgot confidence; must raise."""
    router, llm, _, _ = env
    llm.return_value = _response(
        '{"target_agent":"business","target_model":"qwen2.5:7b"}'
    )

    with pytest.raises(RoutingError, match="confidence"):
        router.route("test")


def test_empty_target_agent_raises_error(env):
    """Local model returned empty string for target_agent."""
    router, llm, _, _ = env
    llm.return_value = _response(
        '{"target_agent":"","target_model":"qwen2.5:7b","confidence":0.9}'
    )

    # FIX: Changed regex to match the actual error message order
    with pytest.raises(RoutingError, match="empty.*target_agent"):
        router.route("test")


def test_confidence_as_string_is_coerced(env):
    """Local model returned confidence as string '0.9' instead of float."""
    router, llm, _, _ = env
    llm.return_value = _response(
        '{"target_agent":"business","target_model":"qwen2.5:7b","confidence":"0.85"}'
    )

    result = router.route("test")
    assert result.confidence == 0.85


def test_confidence_as_invalid_string_raises_error(env):
    """Local model returned non-numeric confidence."""
    router, llm, _, _ = env
    llm.return_value = _response(
        '{"target_agent":"business","target_model":"qwen2.5:7b","confidence":"high"}'
    )

    with pytest.raises(RoutingError, match="confidence.*number"):
        router.route("test")


# FIX: Removed "null" from this list because JSON null becomes Python None, 
# which is caught by the "Missing field" check, not the range check.
@pytest.mark.parametrize("bad", ["5.0", "-0.1"])
def test_invalid_confidence_range_is_rejected(env, bad):
    router, llm, _, _ = env
    llm.return_value = _response(
        '{"target_agent":"business","target_model":"qwen2.5:7b","confidence":' + bad + "}"
    )

    with pytest.raises(RoutingError, match="outside valid range"):
        router.route("test")


def test_unreachable_model_degrades_to_general(env):
    router, llm, _, tmp_path = env
    llm.side_effect = OllamaAPIError("connection refused")

    result = router.route("find pans")

    assert result.target_agent == "general"
    assert result.confidence == 0.0
    assert result.token_usage["total_tokens"] == 0

    rows = _rows(tmp_path)
    assert len(rows) == 1
    assert int(rows[0]["total_tokens"]) == 0


def test_accounting_failure_does_not_break_routing(env):
    router, llm, ledger, _ = env
    llm.return_value = _response(
        '{"target_agent":"business","target_model":"qwen2.5:7b","confidence":0.9}'
    )
    ledger.log_usage = MagicMock(side_effect=OSError("disk full"))

    assert router.route("find pans").target_agent == "business"


def test_long_input_is_bounded(env):
    router, llm, _, _ = env
    llm.return_value = _response(
        '{"target_agent":"general","target_model":"qwen2.5:7b","confidence":0.9}'
    )

    result = router.route("x" * 7000)

    assert len(result.cleaned_prompt) <= 6100
    assert len(llm.call_args.kwargs["prompt"]) < 7000


def test_missing_playbook_fails_fast(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        Router(
            "qwen2.5:3b",
            str(tmp_path / "missing.md"),
            llm_callable=MagicMock(),
            token_ledger=MagicMock(),
        )


def test_empty_playbook_fails_fast(tmp_path: Path):
    playbook = tmp_path / "empty.md"
    playbook.write_text("   ", encoding="utf-8")

    with pytest.raises(ValueError):
        Router(
            "qwen2.5:3b",
            str(playbook),
            llm_callable=MagicMock(),
            token_ledger=MagicMock(),
        )