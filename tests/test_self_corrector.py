# tests/test_self_corrector.py
"""Unit tests for Self Corrector module with synthetic failures."""
import pytest
from unittest.mock import MagicMock
from src.services.self_corrector import SelfCorrector, CorrectionResult, ErrorCode

def test_self_corrector_fixes_broken_json():
    """Verify that SelfCorrector retries and captures thinking log when given broken JSON."""
    
    # Mock the LLM client to return a fixed JSON after the first failure
    mock_llm = MagicMock()
    mock_llm.generate.return_value = MagicMock(
        content="""<thinking>The previous JSON was missing a closing brace. I will fix the syntax.</thinking>
<corrected_output>{"status": "success", "data": "fixed"}</corrected_output>"""
    )
    
    corrector = SelfCorrector(
        llm_client=mock_llm,
        model_name="qwen2.5:7b",
        max_retries=2,
        base_backoff=0.1  # Fast backoff for testing
    )
    
    # Simulate a synthetic failure (broken JSON)
    result = corrector.correct(
        job_id="synthetic_001",
        agent_name="scout",
        original_prompt="Find me eco-friendly pans",
        error=Exception("JSONDecodeError: Expecting ',' delimiter"),
        failed_output='{"status": "success", "data": "broken"' # Missing closing brace
    )
    
    assert result.status == "success"
    assert result.retry_count == 1
    assert result.corrected_output == '{"status": "success", "data": "fixed"}'
    assert len(result.thinking_log) > 0
    assert "missing a closing brace" in result.thinking_log[0].lower()

def test_self_corrector_escalates_on_max_retries():
    """Verify that SelfCorrector escalates to HITL if it cannot fix the error."""
    
    mock_llm = MagicMock()
    # Return the exact same broken output every time
    mock_llm.generate.return_value = MagicMock(
        content="<thinking>I don't know how to fix this.</thinking><corrected_output>{broken}</corrected_output>"
    )
    
    corrector = SelfCorrector(
        llm_client=mock_llm,
        model_name="qwen2.5:7b",
        max_retries=2,
        base_backoff=0.1
    )
    
    result = corrector.correct(
        job_id="synthetic_002",
        agent_name="scout",
        original_prompt="Find me pans",
        error=Exception("Validation failed"),
        failed_output="{broken}"
    )
    
    assert result.status == "escalated_to_hitl"
    assert result.error_code == ErrorCode.MAX_RETRIES_EXHAUSTED
    assert result.retry_count == 2