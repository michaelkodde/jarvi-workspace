"""Unit tests for the Creator agent."""
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.agents.creator import ContentRequestError, Creator
from src.utils.llm_client import OllamaResponse
from src.utils.token_ledger import TokenLedger

PLAYBOOK_CONTENT = """# Role
You are a TEXT-ONLY content strategist. Output ONLY a raw JSON object.
"""

BRAND_VOICE = {
    "version": "1.0",
    "brand_name": "TestBrand",
    "tones": {
        "professional": {"description": "Formal", "forbidden_phrases": [], "preferred_phrases": []},
        "casual": {"description": "Friendly", "forbidden_phrases": [], "preferred_phrases": []},
    },
    "templates": {
        "five_reasons": {"structure": ["hook", "r1", "r2", "r3", "r4", "r5", "cta"]},
    },
    "conversion_frameworks": {"AIDA": "A-I-D-A"},
    "platform_guidelines": {"twitter": {"max_chars": 280}},
    "affiliate_disclosure_default": "Default disclosure for tests.",
    "image_suggestion_defaults": {"product_hero": "Default hero image"},
}

SAMPLE_PRODUCTS = [
    {
        "product_name": "ErgoChair Pro",
        "brand": "Logitech",
        "price": 299.99,
        "currency": "USD",
        "url": "https://amazon.com/dp/B08ABC1234",
        "asin": "B08ABC1234",
    },
    {
        "product_name": "DeskMate 3000",
        "brand": "Herman Miller",
        "price": 1499.0,
        "currency": "USD",
        "url": "https://amazon.com/dp/B09XYZ5678",
        "asin": "B09XYZ5678",
    },
]


def _make_context(
    content_type: str = "social_post",
    platform: str = "twitter",
    tone: str = "professional",
    product_index: int = 0,
    variant_count: int = 1,
    template: str | None = None,
    destination: str = "orchestrator",
    format_for: str | None = None,
    **routing_extras,
) -> str:
    ctx = {
        "niche_name": "Ergonomic Chairs",
        "target_demographics": "Remote workers 25-45",
        "recommended_product_types": "Ergonomic chairs",
        "ethical_validation": "Only verified brands",
        "supply_chain_viability": "Strong",
        "target_keywords": ["ergonomic chair", "office chair"],
        "social_proof": {"avg_rating": 4.5, "review_count": 1247},
        "content_request": {
            "request_id": "test-req-001",
            "product_index": product_index,
            "content_type": content_type,
            "platform": platform,
            "tone": tone,
            "template": template,
            "variant_count": variant_count,
            "length_constraints": {},
            "conversion_framework": "AIDA",
            "output_routing": {
                "destination": destination,
                "format_for": format_for,
                **routing_extras,
            },
        },
    }
    return json.dumps(ctx)


def _twitter_response() -> str:
    return json.dumps({
        "platform": "twitter",
        "content": "Check out the ErgoChair Pro! https://amazon.com/dp/B08ABC1234",
        "hashtags": ["#ergonomic", "#office"],
        "affiliate_disclosure": "Affiliate link.",
        "cta": "Buy now",
        "image_suggestions": [{"description": "Chair photo", "alt_text": "ErgoChair", "aspect_ratio": "1:1"}],
        "design_metadata": {"layout": "hero", "visual_hierarchy": ["hero"], "suggested_palette": ["#fff"]},
    })


@pytest.fixture
def mock_env(tmp_path: Path):
    playbook = tmp_path / "creator.md"
    playbook.write_text(PLAYBOOK_CONTENT, encoding="utf-8")
    bv = tmp_path / "brand_voice.json"
    bv.write_text(json.dumps(BRAND_VOICE), encoding="utf-8")

    llm_client = MagicMock()
    token_ledger = MagicMock(spec=TokenLedger)

    agent = Creator(
        llm_client=llm_client,
        token_ledger=token_ledger,
        model_name="qwen2.5:7b",
        playbook_path=str(playbook),
        brand_voice_path=str(bv),
    )
    return agent, llm_client, token_ledger


def test_valid_twitter_post(mock_env):
    agent, llm_client, _ = mock_env
    llm_client.return_value = OllamaResponse(
        content=_twitter_response(), prompt_tokens=10, completion_tokens=20, total_tokens=30,
    )
    result = agent.execute(json.dumps(SAMPLE_PRODUCTS), _make_context())
    envelope = json.loads(result.artifact)
    assert envelope["variant_count"] == 1
    variant = envelope["variants"][0]
    assert variant["content_type"] == "social_post"
    assert "https://amazon.com/dp/B08ABC1234" in variant["content"]
    assert variant["affiliate_disclosure"] == "Affiliate link."
    assert len(variant["image_suggestions"]) == 1
    assert variant["output_metadata"]["content_hash"]
    assert variant["output_metadata"]["routing"]["destination"] == "orchestrator"


def test_multi_product_product_index(mock_env):
    agent, llm_client, _ = mock_env
    llm_client.return_value = OllamaResponse(
        content=json.dumps({
            "platform": "twitter",
            "content": "DeskMate 3000 https://amazon.com/dp/B09XYZ5678",
            "hashtags": [],
            "affiliate_disclosure": "Affiliate.",
            "cta": "Buy",
            "image_suggestions": [{"description": "x", "alt_text": "x", "aspect_ratio": "1:1"}],
            "design_metadata": {"layout": "", "visual_hierarchy": [], "suggested_palette": []},
        }),
        prompt_tokens=10, completion_tokens=20, total_tokens=30,
    )
    result = agent.execute(json.dumps(SAMPLE_PRODUCTS), _make_context(product_index=1))
    envelope = json.loads(result.artifact)
    variant = envelope["variants"][0]
    assert "B09XYZ5678" in variant["content"]
    assert variant["output_metadata"]["product_index"] == 1


def test_missing_url_raises_before_llm(mock_env):
    agent, llm_client, _ = mock_env
    bad_products = [{"product_name": "X", "brand": "Y", "price": 1.0, "currency": "USD", "asin": "A"}]
    with pytest.raises(ContentRequestError, match="missing required fields"):
        agent.execute(json.dumps(bad_products), _make_context())
    llm_client.assert_not_called()


def test_invalid_content_type_raises(mock_env):
    agent, llm_client, _ = mock_env
    with pytest.raises(ContentRequestError, match="content_type"):
        agent.execute(json.dumps(SAMPLE_PRODUCTS), _make_context(content_type="invalid"))
    llm_client.assert_not_called()


def test_invalid_tone_raises(mock_env):
    agent, llm_client, _ = mock_env
    with pytest.raises(ContentRequestError, match="tone"):
        agent.execute(json.dumps(SAMPLE_PRODUCTS), _make_context(tone="nonexistent"))
    llm_client.assert_not_called()


def test_llm_omits_affiliate_disclosure_injected(mock_env):
    agent, llm_client, _ = mock_env
    llm_client.return_value = OllamaResponse(
        content=json.dumps({
            "platform": "twitter",
            "content": "Post content https://amazon.com/dp/B08ABC1234",
            "hashtags": [],
            "cta": "Buy",
            "image_suggestions": [{"description": "x", "alt_text": "x", "aspect_ratio": "1:1"}],
            "design_metadata": {"layout": "", "visual_hierarchy": [], "suggested_palette": []},
        }),
        prompt_tokens=10, completion_tokens=20, total_tokens=30,
    )
    result = agent.execute(json.dumps(SAMPLE_PRODUCTS), _make_context())
    envelope = json.loads(result.artifact)
    variant = envelope["variants"][0]
    assert variant["affiliate_disclosure"] == "Default disclosure for tests."


def test_blog_truncated_at_word_limit(mock_env):
    agent, llm_client, _ = mock_env
    long_content = "word " * 3000
    llm_client.return_value = OllamaResponse(
        content=json.dumps({
            "title": "Long Blog",
            "content": long_content,
            "affiliate_disclosure": "Disclosure.",
            "cta": "Read more",
            "image_suggestions": [{"description": "x", "alt_text": "x", "aspect_ratio": "16:9"}],
            "design_metadata": {"layout": "", "visual_hierarchy": [], "suggested_palette": []},
        }),
        prompt_tokens=10, completion_tokens=20, total_tokens=30,
    )
    result = agent.execute(json.dumps(SAMPLE_PRODUCTS), _make_context(content_type="blog_post", platform="blog"))
    envelope = json.loads(result.artifact)
    variant = envelope["variants"][0]
    word_count = len(variant["content"].split())
    assert word_count <= 2000


def test_variant_count_three(mock_env):
    agent, llm_client, _ = mock_env
    llm_client.side_effect = [
        OllamaResponse(content=_twitter_response(), prompt_tokens=10, completion_tokens=20, total_tokens=30),
        OllamaResponse(content=_twitter_response(), prompt_tokens=10, completion_tokens=20, total_tokens=30),
        OllamaResponse(content=_twitter_response(), prompt_tokens=10, completion_tokens=20, total_tokens=30),
    ]
    result = agent.execute(json.dumps(SAMPLE_PRODUCTS), _make_context(variant_count=3))
    envelope = json.loads(result.artifact)
    assert envelope["variant_count"] == 3
    assert len(envelope["variants"]) == 3
    assert envelope["variants"][1]["output_metadata"]["parent_hash"] == envelope["variants"][0]["output_metadata"]["content_hash"]


def test_template_five_reasons(mock_env):
    agent, llm_client, _ = mock_env
    llm_client.return_value = OllamaResponse(
        content=json.dumps({
            "title": "5 Reasons",
            "content": "# Hook\nIntro\n## Reason 1\n...\n## Reason 2\n...\n## Reason 3\n...\n## Reason 4\n...\n## Reason 5\n...\n## CTA\nBuy now",
            "affiliate_disclosure": "Disclosure.",
            "cta": "Buy",
            "image_suggestions": [{"description": "x", "alt_text": "x", "aspect_ratio": "16:9"}],
            "design_metadata": {"layout": "", "visual_hierarchy": [], "suggested_palette": []},
        }),
        prompt_tokens=10, completion_tokens=20, total_tokens=30,
    )
    result = agent.execute(json.dumps(SAMPLE_PRODUCTS), _make_context(content_type="blog_post", platform="blog", template="five_reasons"))
    envelope = json.loads(result.artifact)
    assert "Reason 1" in envelope["variants"][0]["content"]


def test_social_proof_in_context(mock_env):
    agent, llm_client, _ = mock_env
    llm_client.return_value = OllamaResponse(
        content=_twitter_response(), prompt_tokens=10, completion_tokens=20, total_tokens=30,
    )
    agent.execute(json.dumps(SAMPLE_PRODUCTS), _make_context())
    call_args = llm_client.call_args
    prompt = call_args.kwargs.get("prompt")
    assert prompt is not None
    assert "4.5" in prompt or "1247" in prompt


def test_format_for_tts_agent(mock_env):
    agent, llm_client, _ = mock_env
    llm_client.return_value = OllamaResponse(
        content=json.dumps({
            "script": "Hello world, this is a test script with enough words.",
            "tone_notes": "Friendly",
            "affiliate_disclosure": "Disclosure.",
        }),
        prompt_tokens=10, completion_tokens=20, total_tokens=30,
    )
    result = agent.execute(
        json.dumps(SAMPLE_PRODUCTS),
        _make_context(content_type="voiceover_script", platform=None, destination="agent", format_for="tts_agent", next_agent="tts_agent"),
    )
    envelope = json.loads(result.artifact)
    routing = envelope["variants"][0]["output_metadata"]["routing"]
    assert routing["format_for"] == "tts_agent"
    assert routing["downstream_contract"]["format_for"] == "tts_agent"
    assert "script" in routing["downstream_contract"]["parameters"]


def test_malformed_json_returns_fallback(mock_env):
    agent, llm_client, _ = mock_env
    llm_client.return_value = OllamaResponse(
        content="not valid json at all", prompt_tokens=10, completion_tokens=20, total_tokens=30,
    )
    result = agent.execute(json.dumps(SAMPLE_PRODUCTS), _make_context())
    envelope = json.loads(result.artifact)
    assert envelope["variants"][0]["title"] == "Content Generation Failed"
    assert envelope["variants"][0]["output_metadata"]["routing"]["validation_passed"] is False


def test_revision_feedback_increments_version(mock_env):
    agent, llm_client, _ = mock_env
    llm_client.return_value = OllamaResponse(
        content=_twitter_response(), prompt_tokens=10, completion_tokens=20, total_tokens=30,
    )
    result = agent.execute(json.dumps(SAMPLE_PRODUCTS), _make_context(), revision_feedback="Make it punchier")
    envelope = json.loads(result.artifact)
    assert envelope["variants"][0]["output_metadata"]["version"] == 1
    call_args = llm_client.call_args
    prompt = call_args.kwargs.get("prompt")
    assert prompt is not None
    assert "Make it punchier" in prompt


def test_content_hash_deterministic(mock_env):
    agent, llm_client, _ = mock_env
    llm_client.side_effect = [
        OllamaResponse(content=_twitter_response(), prompt_tokens=10, completion_tokens=20, total_tokens=30),
        OllamaResponse(content=_twitter_response(), prompt_tokens=10, completion_tokens=20, total_tokens=30),
    ]
    r1 = agent.execute(json.dumps(SAMPLE_PRODUCTS), _make_context())
    r2 = agent.execute(json.dumps(SAMPLE_PRODUCTS), _make_context())
    e1 = json.loads(r1.artifact)
    e2 = json.loads(r2.artifact)
    assert e1["variants"][0]["output_metadata"]["content_hash"] == e2["variants"][0]["output_metadata"]["content_hash"]