"""
Unit tests for the Trend Spotter Agent.
"""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from src.agents.trend_spotter import TrendSpotter
from src.utils.llm_client import OllamaResponse
from src.utils.token_ledger import TokenLedger

PLAYBOOK_CONTENT = """# Role
You are an Expert Market Researcher.

# Task
Analyze the provided seed keyword and identify high-demand, low-competition market niches.

# Output Limit
Output 3 to 5 niches.

# Output Schema
You must output a raw JSON array. Each object in the array must have EXACTLY these keys:
- `niche_name` (string)
- `target_audience` (string)
- `estimated_demand` (string: MUST be "High", "Med", or "Low")
- `competition_level` (string: MUST be "High", "Med", or "Low")

# Negative Constraints
- Output MUST be a JSON array `[...]`, even if identifying only one niche.
- NEVER output a single root object (e.g., `{"niches": [...]}`).
- Output ONLY the raw JSON array.
- NO markdown code fences (e.g., do not use ```json).
- NO conversational text (e.g., "Here are the niches:").
- NO trailing commas after the last object in the array.
- NO inline comments (e.g., `// comment`).
- Use ONLY double quotes for keys and string values. NEVER use single quotes.
"""


@pytest.fixture
def mock_env(tmp_path: Path):
    """Sets up a temporary playbook, mock client, and mock ledger for test isolation."""
    playbook_path = tmp_path / "trend_spotter.md"
    playbook_path.write_text(PLAYBOOK_CONTENT, encoding="utf-8")
    
    llm_client = MagicMock()
    token_ledger = MagicMock(spec=TokenLedger)
    
    agent = TrendSpotter(
        llm_client=llm_client,
        token_ledger=token_ledger,
        model_name="qwen2.5:7b",
        playbook_path=str(playbook_path)
    )
    return agent, llm_client, token_ledger


def test_trend_spotter_execution_and_schema(mock_env):
    """Verify TrendSpotter correctly instantiates, calls LLM, and returns valid JSON array."""
    agent, llm_client, token_ledger = mock_env
    
    # Mock _call_llm directly to bypass the retry decorator and return the expected dataclass
    mock_response = OllamaResponse(
        content='[{"niche_name": "Ergo Chairs", "target_audience": "Remote Workers", "estimated_demand": "High", "competition_level": "Low"}]',
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150
    )
    agent._call_llm = MagicMock(return_value=mock_response)
    
    result = agent.execute("Home Office Ergonomics")
    
    # Verify AgentResult structure
    assert result.agent_name == "TrendSpotter"
    assert isinstance(result.artifact, str)
    assert result.token_usage["total_tokens"] == 150
    
    # Verify JSON parsing and strict schema compliance
    data = json.loads(result.artifact)
    assert isinstance(data, list)
    assert len(data) >= 1
    
    expected_keys = {"niche_name", "target_audience", "estimated_demand", "competition_level"}
    valid_demands = {"High", "Med", "Low"}
    valid_competition = {"High", "Med", "Low"}
    
    # CRITICAL: Validate EVERY item in the array, not just the first one
    for item in data:
        assert set(item.keys()) == expected_keys, f"Invalid keys in item: {item.keys()}"
        assert item["estimated_demand"] in valid_demands, f"Invalid demand: {item['estimated_demand']}"
        assert item["competition_level"] in valid_competition, f"Invalid competition: {item['competition_level']}"
    
    # Verify token logging occurred
    token_ledger.log_usage.assert_called_once()