"""
Unit tests for the Scout Agent.
"""
import json
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agents.scout import Scout
from src.utils.llm_client import OllamaResponse
from src.utils.product_scraper import ProductData
from src.utils.token_ledger import TokenLedger

PLAYBOOK_CONTENT = """# Role
You are an Expert Product Validator and Saturation Assessor.

# Task
You will receive a JSON list of ALREADY SCRAPED products and a specific context containing Ethical Validation and proof criteria. 

# Rules
1. Use ONLY the provided context to define what constitutes a "major, verified brand".
2. Filter the scraped products to ONLY include products from major, verified brands.
3. Discard any unknown, dropshipped, or unverified brands.
4. Format the remaining verified products into a strict JSON array.
5. NEVER invent, guess, or modify URLs, ASINs, or prices.
6. For saturation assessment, output ONLY strict JSON: {"saturated": bool, "next_query": str}.
7. If nothing survives filtering, output the exact N/A fallback schema.
8. Output ONLY raw JSON. No markdown fences, no commentary.
"""


@pytest.fixture
def mock_env(tmp_path: Path):
    """Sets up a temporary playbook, mock client, mock ledger, and mock checkpoint for test isolation."""
    playbook_path = tmp_path / "scout.md"
    playbook_path.write_text(PLAYBOOK_CONTENT, encoding="utf-8")
    
    # Create banned brands file
    banned_path = tmp_path / "banned_brands.json"
    banned_path.write_text(json.dumps({
        "version": "1.0",
        "banned": [
            {
                "name": "GenericDropship",
                "aliases": ["generic dropship"],
                "reason": "Failed ethical validation",
                "banned_date": "2026-10-16"
            }
        ]
    }), encoding="utf-8")
    
    llm_client = MagicMock()
    token_ledger = MagicMock(spec=TokenLedger)
    
    agent = Scout(
        llm_client=llm_client,
        token_ledger=token_ledger,
        model_name="qwen2.5:7b",
        playbook_path=str(playbook_path),
        banned_brands_path=str(banned_path),
    )
    # Override checkpoint for test isolation
    agent.checkpoint.db_path = str(tmp_path / "test_checkpoint.db")
    agent.checkpoint._init_db()
    
    return agent, llm_client, token_ledger


def test_scout_banned_brands_filtered(mock_env):
    """Verify Python-level banned brand filtering works."""
    agent, llm_client, token_ledger = mock_env
    
    products = [
        ProductData("Chair 1", "Logitech", 100.0, "USD", "url1", "ASIN1"),
        ProductData("Chair 2", "GenericDropship", 50.0, "USD", "url2", "ASIN2"),
        ProductData("Chair 3", "Herman Miller", 1500.0, "USD", "url3", "ASIN3"),
    ]
    
    with patch("src.agents.scout.search_products") as mock_search:
        mock_search.return_value = products
        
        llm_client.side_effect = [
            OllamaResponse(
                content='{"saturated": true, "next_query": ""}',
                prompt_tokens=10, completion_tokens=10, total_tokens=20,
            ),
            OllamaResponse(
                content='[{"product_name": "Chair 1", "brand": "Logitech", "price": 100.0, "currency": "USD", "url": "url1", "asin": "ASIN1"}, {"product_name": "Chair 3", "brand": "Herman Miller", "price": 1500.0, "currency": "USD", "url": "url3", "asin": "ASIN3"}]',
                prompt_tokens=20, completion_tokens=50, total_tokens=70,
            ),
        ]
        
        result = agent.execute("Ergonomic Chair", "Context: Only Logitech and Herman Miller are verified.")
        
        data = json.loads(result.artifact)
        assert len(data) == 2
        brands = {p["brand"] for p in data}
        assert "GenericDropship" not in brands
        assert "Logitech" in brands
        assert "Herman Miller" in brands


def test_scout_circuit_breaker(mock_env):
    """Verify circuit breaker triggers after MAX_SAFETY_CEILING iterations."""
    agent, llm_client, token_ledger = mock_env
    
    products = [
        ProductData("Chair 1", "Logitech", 100.0, "USD", "url1", "ASIN1"),
    ]
    
    with patch("src.agents.scout.search_products") as mock_search:
        mock_search.return_value = products
        
        # LLM never says saturated, and provides a DIFFERENT query each time 
        # to avoid triggering the stagnation break.
        llm_client.side_effect = [
            OllamaResponse(
                content='{"saturated": false, "next_query": "query 1"}',
                prompt_tokens=10, completion_tokens=10, total_tokens=20,
            ),
            OllamaResponse(
                content='{"saturated": false, "next_query": "query 2"}',
                prompt_tokens=10, completion_tokens=10, total_tokens=20,
            ),
            OllamaResponse(
                content='{"saturated": false, "next_query": "query 3"}',
                prompt_tokens=10, completion_tokens=10, total_tokens=20,
            ),
            # Final formatting call
            OllamaResponse(
                content='[{"product_name": "Chair 1", "brand": "Logitech", "price": 100.0, "currency": "USD", "url": "url1", "asin": "ASIN1"}]',
                prompt_tokens=20, completion_tokens=50, total_tokens=70,
            )
        ]
        
        # Temporarily reduce ceiling for test
        import src.agents.scout as scout_module
        original_ceiling = scout_module.MAX_SAFETY_CEILING
        scout_module.MAX_SAFETY_CEILING = 3
        
        try:
            result = agent.execute("Ergonomic Chair", "Context: Logitech is verified.")
            
            # Should have called search_products exactly 3 times (circuit breaker)
            assert mock_search.call_count == 3
            
            # Should have called LLM 4 times (3 assessments + 1 final formatting)
            assert llm_client.call_count == 4
            
            data = json.loads(result.artifact)
            assert len(data) == 1
            assert data[0]["brand"] == "Logitech"
        finally:
            scout_module.MAX_SAFETY_CEILING = original_ceiling