"""
Unit tests for the Router agent.
"""
import pytest
from unittest.mock import MagicMock
from pathlib import Path

from src.agents.router import Router, RoutingResult, RoutingError
from src.utils.llm_client import OllamaAPIError, OllamaResponse


def test_successful_routing(tmp_path: Path):
    """Test that a valid LLM response is parsed correctly with token accounting."""
    playbook = tmp_path / "mock_playbook.md"
    playbook.write_text("You are a router.")
    
    mock_response = OllamaResponse(
        content='{"target_agent": "business", "target_model": "qwen2.5:7b", "cleaned_prompt": "find pans", "confidence": 0.9}',
        prompt_tokens=10, completion_tokens=20, total_tokens=30
    )
    
    mock_llm = MagicMock(return_value=mock_response)
    router = Router(ollama_model="qwen2.5:3b", playbook_path=str(playbook), llm_callable=mock_llm)
    
    result = router.route("find me a good frying pan")
    
    assert result.target_agent == "business"
    assert result.target_model == "qwen2.5:7b"
    assert result.confidence == 0.9
    assert result.token_usage["total_tokens"] == 30
    mock_llm.assert_called_once()


def test_low_confidence_fallback(tmp_path: Path):
    """CRITICAL TEST: Verify that confidence < 0.7 triggers the default fallback."""
    playbook = tmp_path / "mock_playbook.md"
    playbook.write_text("You are a router.")
    
    mock_response = OllamaResponse(
        content='{"target_agent": "business", "target_model": "qwen2.5:7b", "cleaned_prompt": "test", "confidence": 0.4}',
        prompt_tokens=10, completion_tokens=10, total_tokens=20
    )
    
    mock_llm = MagicMock(return_value=mock_response)
    router = Router(ollama_model="qwen2.5:3b", playbook_path=str(playbook), llm_callable=mock_llm)
    
    result = router.route("ambiguous prompt")
    
    assert result.target_agent == "general"
    assert result.confidence == 0.0
    assert result.token_usage["total_tokens"] == 20 


def test_invalid_json_fallback(tmp_path: Path):
    """Test that invalid JSON from the LLM raises a RoutingError."""
    playbook = tmp_path / "mock_playbook.md"
    playbook.write_text("You are a router.")
    
    mock_response = OllamaResponse(
        content="This is not JSON at all.",
        prompt_tokens=10, completion_tokens=10, total_tokens=20
    )
    mock_llm = MagicMock(return_value=mock_response)
    
    router = Router(ollama_model="qwen2.5:3b", playbook_path=str(playbook), llm_callable=mock_llm)
    
    with pytest.raises(RoutingError) as exc_info:
        router.route("test prompt")
    
    assert "invalid routing schema" in str(exc_info.value)


def test_ollama_unreachable_fallback(tmp_path: Path):
    """Test that a complete Ollama failure triggers graceful degradation."""
    playbook = tmp_path / "mock_playbook.md"
    playbook.write_text("You are a router.")
    
    mock_llm = MagicMock(side_effect=OllamaAPIError("Connection refused"))
    
    router = Router(ollama_model="qwen2.5:3b", playbook_path=str(playbook), llm_callable=mock_llm)
    result = router.route("test prompt")
    
    assert result.target_agent == "general"
    assert result.confidence == 0.0
    assert result.token_usage["total_tokens"] == 0


def test_missing_playbook_fails_fast(tmp_path: Path):
    """SPEC COMPLIANCE: Verify startup validation fails if playbook is missing."""
    missing_path = tmp_path / "non_existent.md"
    mock_llm = MagicMock()
    
    with pytest.raises(FileNotFoundError):
        Router(ollama_model="qwen2.5:3b", playbook_path=str(missing_path), llm_callable=mock_llm)


def test_empty_playbook_fails_fast(tmp_path: Path):
    """SPEC COMPLIANCE: Verify startup validation fails if playbook is empty."""
    playbook = tmp_path / "empty_playbook.md"
    playbook.write_text("") # Empty file
    mock_llm = MagicMock()
    
    with pytest.raises(ValueError):
        Router(ollama_model="qwen2.5:3b", playbook_path=str(playbook), llm_callable=mock_llm)