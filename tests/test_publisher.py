"""Unit tests for the Publisher agent."""
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agents.publisher import PublicationError, Publisher
from src.utils.llm_client import OllamaResponse
from src.utils.publication_ledger import LedgerError
from src.utils.token_ledger import TokenLedger


def _make_envelope(
    destination: str = "file",
    file_path: str = "output/test.json",
    validation_passed: bool = True,
    content_hash: str = "abc123",
) -> str:
    return json.dumps({
        "request_id": "test-req-001",
        "variant_count": 1,
        "variants": [
            {
                "content_type": "social_post",
                "platform": "twitter",
                "content": "Check out https://amazon.com/dp/B08ABC1234",
                "output_metadata": {
                    "content_hash": content_hash,
                    "routing": {
                        "destination": destination,
                        "file_path": file_path,
                        "validation_passed": validation_passed,
                    }
                }
            }
        ],
        "output_metadata": {
            "content_hash": content_hash,
            "routing": {
                "destination": destination,
                "file_path": file_path,
                "validation_passed": validation_passed,
            }
        }
    })


def _make_context(
    affiliate_config: dict | None = None,
    product_urls: list | None = None,
) -> str:
    ctx = {}
    if affiliate_config is not None:
        ctx["affiliate_link_config"] = affiliate_config
    if product_urls is not None:
        ctx["product_urls"] = product_urls
    return json.dumps(ctx)


@pytest.fixture
def mock_env(tmp_path: Path):
    playbook = tmp_path / "publisher.md"
    playbook.write_text("# Publisher\nTest playbook", encoding="utf-8")
    ledger_db = tmp_path / "publications.db"

    llm_client = MagicMock()
    token_ledger = MagicMock(spec=TokenLedger)

    agent = Publisher(
        llm_client=llm_client,
        token_ledger=token_ledger,
        model_name="qwen2.5:7b",
        playbook_path=str(playbook),
        publication_ledger_path=str(ledger_db),
    )
    return agent, tmp_path


def test_valid_file_delivery(mock_env):
    agent, tmp_path = mock_env
    file_path = tmp_path / "output" / "test.json"
    envelope = _make_envelope(destination="file", file_path=str(file_path))
    result = agent.execute(envelope)
    report = json.loads(result.artifact)
    assert report["status"] == "delivered"
    assert file_path.exists()
    assert len(report["deliveries"]) == 1
    assert report["deliveries"][0]["status"] == "success"


def test_agent_routing(mock_env):
    agent, tmp_path = mock_env
    envelope = json.dumps({
        "variants": [{
            "content_type": "voiceover_script",
            "output_metadata": {
                "content_hash": "hash1",
                "routing": {
                    "destination": "agent",
                    "next_agent": "tts_agent",
                    "validation_passed": True,
                    "downstream_contract": {"script": "Hello", "tone": "casual"}
                }
            }
        }]
    })
    result = agent.execute(envelope)
    report = json.loads(result.artifact)
    assert report["deliveries"][0]["status"] == "routed"
    assert report["deliveries"][0]["formatted_content"] is not None
    contract = json.loads(report["deliveries"][0]["formatted_content"])
    assert contract["script"] == "Hello"


def test_affiliate_link_insertion(mock_env):
    agent, tmp_path = mock_env
    file_path = tmp_path / "output" / "test.json"
    envelope = _make_envelope(destination="file", file_path=str(file_path))
    context = _make_context(
        affiliate_config={"tag": "jarvi-20", "utm_source": "jarvi"},
        product_urls=["https://amazon.com/dp/B08ABC1234"]
    )
    result = agent.execute(envelope, context)
    report = json.loads(result.artifact)
    assert report["deliveries"][0]["affiliate_links_inserted"] == 1
    content = file_path.read_text(encoding="utf-8")
    assert "tag=jarvi-20" in content
    assert "utm_source=jarvi" in content


def test_invalid_envelope_raises(mock_env):
    agent, tmp_path = mock_env
    envelope = json.dumps({"variants": []})
    with pytest.raises(PublicationError, match="variants"):
        agent.execute(envelope)


def test_file_write_failure(mock_env):
    agent, tmp_path = mock_env
    # Mock _write_file to raise OSError
    with patch.object(agent, "_write_file", side_effect=OSError("Permission denied")):
        envelope = _make_envelope(destination="file", file_path="test.json")
        result = agent.execute(envelope)
    report = json.loads(result.artifact)
    assert report["status"] == "failed"
    assert report["deliveries"][0]["status"] == "failed"


def test_multiple_variants(mock_env):
    agent, tmp_path = mock_env
    file1 = tmp_path / "out1.json"
    file2 = tmp_path / "out2.json"
    envelope = json.dumps({
        "variants": [
            {
                "content_type": "social_post",
                "output_metadata": {
                    "content_hash": "hash1",
                    "routing": {"destination": "file", "file_path": str(file1), "validation_passed": True}
                }
            },
            {
                "content_type": "blog_post",
                "output_metadata": {
                    "content_hash": "hash2",
                    "routing": {"destination": "file", "file_path": str(file2), "validation_passed": True}
                }
            }
        ]
    })
    result = agent.execute(envelope)
    report = json.loads(result.artifact)
    assert len(report["deliveries"]) == 2
    assert file1.exists()
    assert file2.exists()


def test_deferred_destination(mock_env):
    agent, tmp_path = mock_env
    envelope = _make_envelope(destination="publisher")
    result = agent.execute(envelope)
    report = json.loads(result.artifact)
    assert report["deliveries"][0]["status"] == "deferred"
    assert "Phase 2+" in report["deliveries"][0]["message"]


def test_orchestrator_return(mock_env):
    agent, tmp_path = mock_env
    envelope = _make_envelope(destination="orchestrator")
    result = agent.execute(envelope)
    report = json.loads(result.artifact)
    assert report["deliveries"][0]["status"] == "routed"
    assert report["deliveries"][0]["formatted_content"] is not None
    assert "Returned to orchestrator" in report["deliveries"][0]["message"]


def test_validation_failed_skips(mock_env):
    agent, tmp_path = mock_env
    envelope = _make_envelope(validation_passed=False)
    result = agent.execute(envelope)
    report = json.loads(result.artifact)
    assert report["deliveries"][0]["status"] == "skipped"
    assert "Creator validation failed" in report["deliveries"][0]["message"]
    assert "notify_hitl_for_review" in report["next_actions"]


def test_duplicate_skips(mock_env):
    agent, tmp_path = mock_env
    file_path = tmp_path / "test.json"
    envelope = _make_envelope(destination="file", file_path=str(file_path), content_hash="dup123")
    agent.execute(envelope)
    result = agent.execute(envelope)
    report = json.loads(result.artifact)
    assert report["deliveries"][0]["status"] == "skipped"
    assert "Duplicate" in report["deliveries"][0]["message"]


def test_malformed_affiliate_config(mock_env, caplog):
    agent, tmp_path = mock_env
    file_path = tmp_path / "test.json"
    envelope = _make_envelope(destination="file", file_path=str(file_path))
    context = json.dumps({"affiliate_link_config": "not a dict"})
    with caplog.at_level("WARNING"):
        result = agent.execute(envelope, context)
    report = json.loads(result.artifact)
    assert report["deliveries"][0]["affiliate_links_inserted"] == 0
    assert "affiliate_link_config" in caplog.text


def test_ledger_error_continues(mock_env, caplog):
    agent, tmp_path = mock_env
    file_path = tmp_path / "test.json"
    envelope = _make_envelope(destination="file", file_path=str(file_path))
    with patch.object(agent._ledger, "log_publication", side_effect=LedgerError("DB error")):
        with caplog.at_level("ERROR"):
            result = agent.execute(envelope)
    report = json.loads(result.artifact)
    assert report["deliveries"][0]["status"] == "success"
    assert "Failed to log publication" in caplog.text


def test_all_failed_report_status(mock_env):
    agent, tmp_path = mock_env
    # Mock _write_file to raise OSError
    with patch.object(agent, "_write_file", side_effect=OSError("Permission denied")):
        envelope = json.dumps({
            "variants": [{
                "output_metadata": {
                    "content_hash": "h1",
                    "routing": {"destination": "file", "file_path": "test.json", "validation_passed": True}
                }
            }]
        })
        result = agent.execute(envelope)
    report = json.loads(result.artifact)
    assert report["status"] == "failed"
    assert "retry_failed_deliveries" in report["next_actions"]


def test_partial_report_status(mock_env):
    agent, tmp_path = mock_env
    file1 = tmp_path / "good.json"
    envelope = json.dumps({
        "variants": [
            {
                "output_metadata": {
                    "content_hash": "h1",
                    "routing": {"destination": "file", "file_path": str(file1), "validation_passed": True}
                }
            },
            {
                "output_metadata": {
                    "content_hash": "h2",
                    "routing": {"destination": "file", "file_path": "bad.json", "validation_passed": True}
                }
            }
        ]
    })
    # Mock _write_file to fail only for the second variant
    original_write = agent._write_file
    def mock_write(path, content):
        if "bad.json" in path:
            raise OSError("Permission denied")
        return original_write(path, content)
    
    with patch.object(agent, "_write_file", side_effect=mock_write):
        result = agent.execute(envelope)
    report = json.loads(result.artifact)
    assert report["status"] == "partial"


def test_file_already_exists_warning(mock_env, caplog):
    agent, tmp_path = mock_env
    file_path = tmp_path / "existing.json"
    file_path.write_text("old content", encoding="utf-8")
    envelope = _make_envelope(destination="file", file_path=str(file_path))
    with caplog.at_level("WARNING"):
        agent.execute(envelope)
    assert "already exists" in caplog.text
    assert file_path.read_text(encoding="utf-8") != "old content"


def test_empty_product_urls(mock_env):
    agent, tmp_path = mock_env
    file_path = tmp_path / "test.json"
    envelope = _make_envelope(destination="file", file_path=str(file_path))
    context = _make_context(affiliate_config={"tag": "test"}, product_urls=[])
    result = agent.execute(envelope, context)
    report = json.loads(result.artifact)
    assert report["deliveries"][0]["affiliate_links_inserted"] == 0


def test_non_product_urls_not_modified(mock_env):
    agent, tmp_path = mock_env
    file_path = tmp_path / "test.json"
    envelope = json.dumps({
        "variants": [{
            "content": "Visit https://example.com and https://amazon.com/dp/B08ABC1234",
            "output_metadata": {
                "content_hash": "h1",
                "routing": {"destination": "file", "file_path": str(file_path), "validation_passed": True}
            }
        }]
    })
    context = _make_context(
        affiliate_config={"tag": "jarvi"},
        product_urls=["https://amazon.com/dp/B08ABC1234"]
    )
    agent.execute(envelope, context)
    content = file_path.read_text(encoding="utf-8")
    # Verify example.com URL is unchanged
    assert "https://example.com" in content
    assert "https://example.com?" not in content
    # Verify amazon.com URL has affiliate tag
    assert "https://amazon.com/dp/B08ABC1234?tag=jarvi" in content