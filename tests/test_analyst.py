"""
Unit tests for the Analyst Agent.
"""
import re
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from src.agents.analyst import Analyst
from src.utils.llm_client import OllamaResponse
from src.utils.token_ledger import TokenLedger

PLAYBOOK_CONTENT = """# Role
You are a ruthless Product Manager and Ethical Compliance Officer. Your job is to validate market niches and reject anything that does not meet strict ethical, supply-chain, and confidence standards.

# Task
You will receive an **Assigned Ticket** (a JSON object) representing a market niche, including a `confidence_score` from the Trend Spotter. Analyze the ticket based on the provided context and your internal knowledge.

# Input Schema (The Ticket)
The JSON will contain:
- `niche_name` (string)
- `target_audience` (string)
- `confidence_score` (string: "High", "Med", or "Low")

# Clean-Context Rule (CRITICAL)
You may receive additional research context. IGNORE any context that does not directly pertain to the ethical or supply-chain analysis of the specific niche provided in the ticket. Do not let irrelevant tangents influence your verdict.

# Proof Criteria (CRITICAL - Pillar 2)
To APPROVE a ticket, the context MUST provide verifiable proof of the following:
1. **Brand Proof:** The context must explicitly name at least one major, verified brand (e.g., Logitech, Herman Miller) operating in this niche.
2. **Ethical Proof:** There must be no evidence of greenwashing, opaque supply chains, or third-party dropshipping.
*Rule:* Even if `confidence_score` is High, if these Proof Criteria are not met in the context, you MUST REJECT.

# Ethical Guardrail (CRITICAL - Pillar 3)
- ONLY approve products from major, verified, and reputable brands.
- NEVER approve unknown third-party sellers, dropshippers, or brands with supply chain opacity.
- If `confidence_score` is "Low", OR if the `confidence_score` key is missing from the ticket, you MUST automatically REJECT the niche to save compute.

# Output Format
You MUST start your response immediately with the `## Final Verdict` header. Do not write any introductory text, and do not repeat the input ticket.

## Final Verdict
[APPROVED or REJECTED]
[One brief sentence explaining the reason]

# Formatting Rules (CRITICAL)
- The word APPROVED or REJECTED MUST be in ALL CAPS.
- Do NOT wrap the word APPROVED or REJECTED in code ticks (e.g., `REJECTED`) or bold markdown (e.g., **REJECTED**). Output the raw word only.
- After the reason, output NOTHING ELSE. No concluding remarks, no polite sign-offs.

# Conditional Sections
- IF the verdict is **REJECTED**: Your output MUST end immediately after the reason. Do NOT generate ANY markdown headers (H2, H3, H4, etc.) for the remaining sections.
- IF the verdict is **APPROVED**: You MUST generate the following 4 H2 headers in this exact order:

## Ethical Validation
[Detailed analysis of brand reputation and ethical compliance]

## Supply Chain Viability
[Analysis of manufacturing, shipping, and reliability]

## Target Demographics
[Detailed breakdown of the target audience]

## Recommended Product Types
[Specific product recommendations that fit the niche]
"""


@pytest.fixture
def mock_env(tmp_path: Path):
    """Sets up a temporary playbook, mock client, and mock ledger for test isolation."""
    playbook_path = tmp_path / "analyst.md"
    playbook_path.write_text(PLAYBOOK_CONTENT, encoding="utf-8")
    
    llm_client = MagicMock()
    token_ledger = MagicMock(spec=TokenLedger)
    
    agent = Analyst(
        llm_client=llm_client,
        token_ledger=token_ledger,
        model_name="qwen2.5:7b",
        playbook_path=str(playbook_path)
    )
    return agent, llm_client, token_ledger


def test_analyst_rejected_verdict_short_circuit(mock_env):
    """Verify that a REJECTED verdict short-circuits and omits all other sections, even with trailing filler."""
    agent, _, _ = mock_env
    
    # Simulate leading whitespace and trailing conversational filler (common 7B quirk)
    mock_response = OllamaResponse(
        content="\n\n## Final Verdict\nREJECTED\nReason: Relies on unverified third-party dropshippers.\n\nHope this helps!",
        prompt_tokens=100, completion_tokens=20, total_tokens=120
    )
    agent._call_llm = MagicMock(return_value=mock_response)
    
    result = agent.execute('{"niche_name": "Cheap Gadgets", "target_audience": "Budget Shoppers", "confidence_score": "High"}')
    
    artifact = result.artifact
    
    # Verify Verdict is present and in ALL CAPS
    assert re.search(r'\bREJECTED\b', artifact)
    
    # Verify Final Verdict is the very first thing (ignoring leading whitespace)
    assert artifact.strip().startswith("## Final Verdict"), "Final Verdict must be the first header"
    
    # Verify Short-Circuit: The strings MUST NOT be present anywhere (catches H3 bypasses)
    assert "Ethical Validation" not in artifact
    assert "Supply Chain" not in artifact
    assert "Target Demographics" not in artifact
    assert "Recommended Product Types" not in artifact


def test_analyst_missing_confidence_auto_reject(mock_env):
    """Verify that a missing confidence score triggers an automatic rejection."""
    agent, _, _ = mock_env
    
    mock_response = OllamaResponse(
        content="## Final Verdict\nREJECTED\nReason: Confidence score is missing.",
        prompt_tokens=100, completion_tokens=15, total_tokens=115
    )
    agent._call_llm = MagicMock(return_value=mock_response)
    
    # Note: confidence_score is intentionally omitted from the ticket
    result = agent.execute('{"niche_name": "Obscure Niche", "target_audience": "Nobody"}')
    
    assert re.search(r'\bREJECTED\b', result.artifact)
    assert "Ethical Validation" not in result.artifact


def test_analyst_fails_proof_criteria(mock_env):
    """Verify that High confidence is overridden if Proof Criteria (major brands) are missing."""
    agent, _, _ = mock_env
    
    # Mocking a scenario where Trend Spotter was confident, but context lacks brand proof
    mock_response = OllamaResponse(
        content="## Final Verdict\nREJECTED\nReason: Context lacks proof of major verified brands.",
        prompt_tokens=100, completion_tokens=15, total_tokens=115
    )
    agent._call_llm = MagicMock(return_value=mock_response)
    
    # Context explicitly mentions dropshippers, failing Proof Criteria
    context = "Found many cheap suppliers on AliExpress and unknown dropshippers."
    result = agent.execute('{"niche_name": "Generic Widgets", "target_audience": "Everyone", "confidence_score": "High"}', context=context)
    
    assert re.search(r'\bREJECTED\b', result.artifact)
    assert "Ethical Validation" not in result.artifact


def test_analyst_approved_verdict_full_prd(mock_env):
    """Verify that an APPROVED verdict includes all required headers in the correct order."""
    agent, _, _ = mock_env
    
    mock_response = OllamaResponse(
        content="""## Final Verdict
APPROVED
Reason: High demand from verified major brands like Herman Miller.

## Ethical Validation
Brands are fully verified.

## Supply Chain Viability
Excellent logistics.

## Target Demographics
Remote workers.

## Recommended Product Types
Ergonomic chairs.""",
        prompt_tokens=100, completion_tokens=150, total_tokens=250
    )
    agent._call_llm = MagicMock(return_value=mock_response)
    
    result = agent.execute('{"niche_name": "Ergo Chairs", "target_audience": "Remote Workers", "confidence_score": "High"}')
    
    artifact = result.artifact
    
    # Verify Verdict is present and in ALL CAPS
    assert re.search(r'\bAPPROVED\b', artifact)
    
    # Verify all 5 headers are present
    required_headers = [
        "## Final Verdict",
        "## Ethical Validation",
        "## Supply Chain Viability",
        "## Target Demographics",
        "## Recommended Product Types"
    ]
    
    for header in required_headers:
        assert header in artifact, f"Missing header: {header}"
        
    # Verify Final Verdict is the FIRST header
    indices = [artifact.find(h) for h in required_headers]
    assert indices[0] == min(indices), "Final Verdict must be the first header"
    assert indices == sorted(indices), "Headers must be in the correct order"