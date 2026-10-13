"""
Unit tests for the Base Agent.
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from src.agents.base_agent import BaseAgent, AgentResult, AgentExecutionError
from src.utils.llm_client import OllamaResponse
from src.utils.token_ledger import TokenLedger, TokenUsage


class MockAgent(BaseAgent):
    """Concrete subclass for testing BaseAgent."""
    pass


@pytest.fixture
def mock_env(tmp_path: Path):
    """Sets up a temporary playbook, mock client, and mock ledger."""
    playbook = tmp_path / "playbook.md"
    playbook.write_text("You are a test agent.", encoding="utf-8")
    
    llm_client = MagicMock()
    ledger = MagicMock(spec=TokenLedger)
    
    agent = MockAgent(
        llm_client=llm_client,
        token_ledger=ledger,
        playbook_path=str(playbook),
        agent_name="TestAgent",
        model_name="qwen2.5:7b"
    )
    return agent, llm_client, ledger


def test_execute_success(mock_env):
    """Verify successful execution returns AgentResult with stripped markdown."""
    agent, llm_client, ledger = mock_env
    
    agent._call_llm = MagicMock(return_value=OllamaResponse(
        content="```json\n{\"result\": \"success\"}\n```",
        prompt_tokens=10, completion_tokens=10, total_tokens=20
    ))
    
    result = agent.execute("Do the task", "Some context")
    
    assert isinstance(result, AgentResult)
    assert result.artifact == "{\"result\": \"success\"}"
    assert result.token_usage["total_tokens"] == 20
    
    ledger.log_usage.assert_called_once()
    logged_usage = ledger.log_usage.call_args[0][0]
    assert isinstance(logged_usage, TokenUsage)
    assert logged_usage.agent_name == "TestAgent"


def test_execute_handles_conversational_filler_and_trailing_space(mock_env):
    """Verify markdown stripping works with filler text and trailing whitespace."""
    agent, _, _ = mock_env
    
    agent._call_llm = MagicMock(return_value=OllamaResponse(
        content="Sure! Here is your data:\n```json\n{\"key\": \"value\"}\n``` \nHope this helps!",
        prompt_tokens=10, completion_tokens=10, total_tokens=20
    ))
    
    result = agent.execute("Task")
    assert result.artifact == "{\"key\": \"value\"}"


def test_execute_handles_no_markdown(mock_env):
    """Verify execution works if the model returns plain text without markdown."""
    agent, _, _ = mock_env
    
    agent._call_llm = MagicMock(return_value=OllamaResponse(
        content="Just plain text response.",
        prompt_tokens=10, completion_tokens=10, total_tokens=20
    ))
    
    result = agent.execute("Task")
    assert result.artifact == "Just plain text response."


def test_execute_truncates_context_preserving_task(mock_env):
    """Verify context is truncated to preserve the full task prompt."""
    agent, _, _ = mock_env
    
    agent._call_llm = MagicMock(return_value=OllamaResponse(
        content="done", prompt_tokens=1, completion_tokens=1, total_tokens=2
    ))
    
    long_context = "x" * 7000
    task = "Short task"
    
    agent.execute(task, long_context)
    
    # FIX: Use positional argument access
    called_prompt = agent._call_llm.call_args.args[0]
    assert "<task>\nShort task\n</task>" in called_prompt
    assert "... [TRUNCATED]" in called_prompt


def test_execute_truncates_task_if_too_long(mock_env):
    """Verify task is truncated and context dropped if task exceeds limit."""
    agent, _, _ = mock_env
    
    agent._call_llm = MagicMock(return_value=OllamaResponse(
        content="done", prompt_tokens=1, completion_tokens=1, total_tokens=2
    ))
    
    long_task = "y" * 7000
    
    agent.execute(long_task, "Some context")
    
    # FIX: Use positional argument access
    called_prompt = agent._call_llm.call_args.args[0]
    assert "<context>" not in called_prompt
    assert "... [TRUNCATED]" in called_prompt


def test_execute_conditional_xml_tags(mock_env):
    """Verify <context> tags are omitted if context is None."""
    agent, _, _ = mock_env
    
    agent._call_llm = MagicMock(return_value=OllamaResponse(
        content="done", prompt_tokens=1, completion_tokens=1, total_tokens=2
    ))
    
    agent.execute("Just a task", None)
    
    # FIX: Use positional argument access
    called_prompt = agent._call_llm.call_args.args[0]
    assert "<context>" not in called_prompt
    assert "<task>\nJust a task\n</task>" in called_prompt


def test_execute_logs_zero_tokens_on_failure(mock_env):
    """Verify 0 tokens are logged if the LLM call fails."""
    agent, _, ledger = mock_env
    
    agent._call_llm = MagicMock(side_effect=TimeoutError("Connection timed out"))
    
    with pytest.raises(AgentExecutionError) as exc_info:
        agent.execute("Fail task")
        
    assert exc_info.value.agent_name == "TestAgent"
    assert exc_info.value.__cause__ is not None
    assert "TimeoutError" in str(exc_info.value)
    
    ledger.log_usage.assert_called_once()
    logged_usage = ledger.log_usage.call_args[0][0]
    assert logged_usage.total_tokens == 0


def test_execute_continues_if_ledger_fails(mock_env):
    """Verify agent returns result even if token logging fails."""
    agent, _, ledger = mock_env
    
    agent._call_llm = MagicMock(return_value=OllamaResponse(
        content="done", prompt_tokens=1, completion_tokens=1, total_tokens=2
    ))
    ledger.log_usage.side_effect = OSError("Disk full")
    
    result = agent.execute("Task")
    
    assert result.artifact == "done"


def test_base_agent_prevents_direct_instantiation(tmp_path: Path):
    """Verify BaseAgent cannot be instantiated directly."""
    playbook = tmp_path / "playbook.md"
    playbook.write_text("test")
    
    with pytest.raises(TypeError, match="cannot be instantiated directly"):
        BaseAgent(
            llm_client=MagicMock(),
            token_ledger=MagicMock(),
            playbook_path=str(playbook),
            agent_name="Base",
            model_name="qwen2.5:7b"
        )


def test_missing_playbook_fails_fast(tmp_path: Path):
    """Verify missing playbook raises FileNotFoundError at init."""
    with pytest.raises(FileNotFoundError):
        MockAgent(
            llm_client=MagicMock(),
            token_ledger=MagicMock(),
            playbook_path=str(tmp_path / "missing.md"),
            agent_name="Test",
            model_name="qwen2.5:7b"
        )