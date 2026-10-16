"""Unit tests for the PipelineFlow orchestrator."""
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.agents.base_agent import AgentResult
from src.services.pipeline_flow import PipelineFlow, PipelineError


def _make_agent_result(artifact: str) -> AgentResult:
    return AgentResult(
        agent_name="mock",
        artifact=artifact,
        token_usage={"total_tokens": 0},
    )


def _scout_artifact(products=None):
    if products is None:
        products = [
            {"product_name": "P1", "brand": "B1", "price": 10.0, "currency": "USD",
             "url": "https://example.com/p1", "asin": "A1"}
        ]
    return json.dumps(products)


def _creator_artifact(validation_passed=True):
    return json.dumps({
        "request_id": "req1",
        "variant_count": 1,
        "variants": [{
            "content_type": "social_post",
            "content": "test content",
            "output_metadata": {
                "content_hash": "hash1",
                "routing": {
                    "destination": "orchestrator",
                    "validation_passed": validation_passed,
                }
            }
        }],
        "output_metadata": {
            "content_hash": "hash1",
            "routing": {"destination": "orchestrator", "validation_passed": validation_passed}
        }
    })


def _publisher_artifact(status="delivered"):
    return json.dumps({
        "publication_id": "pub1",
        "timestamp": "2026-10-17T00:00:00Z",
        "status": status,
        "deliveries": [],
        "total_affiliate_links_inserted": 1,
        "next_actions": [],
    })


@pytest.fixture
def pipelines_config(tmp_path: Path):
    config_path = tmp_path / "pipelines.json"
    config_path.write_text(json.dumps({
        "defaults": {
            "content_request": {
                "content_type": "social_post",
                "platform": "twitter",
                "tone": "professional",
                "variant_count": 1,
            },
            "affiliate_link_config": {"tag": "test"},
        },
        "pipelines": {
            "content_pipeline": {
                "steps": [
                    {"name": "scout", "type": "agent", "agent": "scout"},
                    {"name": "creator", "type": "agent", "agent": "creator"},
                    {"name": "review_gate", "type": "hitl"},
                    {"name": "publisher", "type": "agent", "agent": "publisher"},
                ]
            }
        }
    }), encoding="utf-8")
    return str(config_path)


@pytest.fixture
def mock_agents():
    scout = MagicMock()
    creator = MagicMock()
    publisher = MagicMock()
    return {"scout": scout, "creator": creator, "publisher": publisher}


@pytest.fixture
def pipeline(tmp_path: Path, pipelines_config, mock_agents):
    db_path = tmp_path / "job_queue.db"
    return PipelineFlow(
        agents=mock_agents,
        job_queue_path=str(db_path),
        pipeline_definitions_path=pipelines_config,
    )


def test_submit_job(pipeline):
    job_id = pipeline.submit_job("content_pipeline", {"product_query": "chairs"})
    assert job_id
    status = pipeline.get_job_status(job_id)
    assert status["status"] == "queued"


def test_happy_path(pipeline, mock_agents):
    mock_agents["scout"].execute.return_value = _make_agent_result(_scout_artifact())
    mock_agents["creator"].execute.return_value = _make_agent_result(_creator_artifact())
    mock_agents["publisher"].execute.return_value = _make_agent_result(_publisher_artifact("delivered"))
    
    job_id = pipeline.submit_job("content_pipeline", {
        "product_query": "chairs",
        "ethical_validation": "verified brands only",
    })
    
    pipeline.process_next_job()
    status = pipeline.get_job_status(job_id)
    assert status["status"] == "pending_review"
    
    assert pipeline.approve_job(job_id) is True
    
    pipeline.process_next_job()
    status = pipeline.get_job_status(job_id)
    assert status["status"] == "published"
    assert "scout" in status["artifacts"]
    assert "creator" in status["artifacts"]
    assert "publisher" in status["artifacts"]


def test_hitl_gate_stops_before_publisher(pipeline, mock_agents):
    mock_agents["scout"].execute.return_value = _make_agent_result(_scout_artifact())
    mock_agents["creator"].execute.return_value = _make_agent_result(_creator_artifact())
    
    job_id = pipeline.submit_job("content_pipeline", {
        "product_query": "chairs",
        "ethical_validation": "verified",
    })
    
    pipeline.process_next_job()
    assert mock_agents["publisher"].execute.call_count == 0
    
    status = pipeline.get_job_status(job_id)
    assert status["status"] == "pending_review"


def test_approval_does_not_rerun_scout_creator(pipeline, mock_agents):
    mock_agents["scout"].execute.return_value = _make_agent_result(_scout_artifact())
    mock_agents["creator"].execute.return_value = _make_agent_result(_creator_artifact())
    mock_agents["publisher"].execute.return_value = _make_agent_result(_publisher_artifact())
    
    job_id = pipeline.submit_job("content_pipeline", {
        "product_query": "chairs",
        "ethical_validation": "verified",
    })
    
    pipeline.process_next_job()
    assert mock_agents["scout"].execute.call_count == 1
    assert mock_agents["creator"].execute.call_count == 1
    
    pipeline.approve_job(job_id)
    pipeline.process_next_job()
    
    assert mock_agents["scout"].execute.call_count == 1
    assert mock_agents["creator"].execute.call_count == 1
    assert mock_agents["publisher"].execute.call_count == 1


def test_reject_job(pipeline, mock_agents):
    mock_agents["scout"].execute.return_value = _make_agent_result(_scout_artifact())
    mock_agents["creator"].execute.return_value = _make_agent_result(_creator_artifact())
    
    job_id = pipeline.submit_job("content_pipeline", {
        "product_query": "chairs",
        "ethical_validation": "verified",
    })
    
    pipeline.process_next_job()
    assert pipeline.reject_job(job_id, "not good") is True
    
    status = pipeline.get_job_status(job_id)
    assert status["status"] == "rejected"
    assert mock_agents["publisher"].execute.call_count == 0


def test_cancel_job(pipeline):
    job_id = pipeline.submit_job("content_pipeline", {"product_query": "chairs"})
    assert pipeline.cancel_job(job_id) is True
    
    status = pipeline.get_job_status(job_id)
    assert status["status"] == "cancelled"


def test_scout_na_sentinel(pipeline, mock_agents):
    na_products = [
        {"product_name": "N/A", "brand": "N/A", "price": 0.0, "currency": "USD",
         "url": "N/A", "asin": "N/A"}
    ]
    mock_agents["scout"].execute.return_value = _make_agent_result(_scout_artifact(na_products))
    
    job_id = pipeline.submit_job("content_pipeline", {
        "product_query": "chairs",
        "ethical_validation": "verified",
    })
    
    pipeline.process_next_job()
    status = pipeline.get_job_status(job_id)
    assert status["status"] == "failed"
    assert status["error_message"] == "no_verified_products"
    assert mock_agents["creator"].execute.call_count == 0


def test_creator_validation_failed(pipeline, mock_agents):
    mock_agents["scout"].execute.return_value = _make_agent_result(_scout_artifact())
    mock_agents["creator"].execute.return_value = _make_agent_result(_creator_artifact(validation_passed=False))
    
    job_id = pipeline.submit_job("content_pipeline", {
        "product_query": "chairs",
        "ethical_validation": "verified",
    })
    
    pipeline.process_next_job()
    status = pipeline.get_job_status(job_id)
    assert status["status"] == "failed"
    assert status["error_message"] == "creator_validation_failed"


def test_permanent_error_no_retry(pipeline, mock_agents):
    from src.agents.creator import ContentRequestError
    mock_agents["scout"].execute.side_effect = ContentRequestError("bad input")
    
    job_id = pipeline.submit_job("content_pipeline", {
        "product_query": "chairs",
        "ethical_validation": "verified",
    })
    
    pipeline.process_next_job()
    status = pipeline.get_job_status(job_id)
    assert status["status"] == "failed"
    assert mock_agents["scout"].execute.call_count == 1


def test_transient_error_retry(pipeline, mock_agents):
    mock_agents["scout"].execute.side_effect = ConnectionError("network down")
    
    job_id = pipeline.submit_job("content_pipeline", {
        "product_query": "chairs",
        "ethical_validation": "verified",
    })
    
    pipeline.process_next_job()
    status = pipeline.get_job_status(job_id)
    assert status["status"] == "queued"
    assert status["retry_count"] == 1
    assert status["available_at"] > status["updated_at"]


def test_crash_recovery(pipeline, mock_agents, tmp_path):
    mock_agents["scout"].execute.return_value = _make_agent_result(_scout_artifact())
    mock_agents["creator"].execute.return_value = _make_agent_result(_creator_artifact())
    
    job_id = pipeline.submit_job("content_pipeline", {
        "product_query": "chairs",
        "ethical_validation": "verified",
    })
    
    pipeline.process_next_job()
    status = pipeline.get_job_status(job_id)
    assert "scout" in status["artifacts"]
    
    pipeline._queue.update_status(job_id, "queued")
    
    new_pipeline = PipelineFlow(
        agents=mock_agents,
        job_queue_path=str(tmp_path / "job_queue.db"),
        pipeline_definitions_path=pipeline._definitions_path,
    )
    
    new_pipeline.process_next_job()
    assert mock_agents["scout"].execute.call_count == 1
    assert mock_agents["creator"].execute.call_count == 1


def test_publisher_partial_fails_job(pipeline, mock_agents):
    mock_agents["scout"].execute.return_value = _make_agent_result(_scout_artifact())
    mock_agents["creator"].execute.return_value = _make_agent_result(_creator_artifact())
    mock_agents["publisher"].execute.return_value = _make_agent_result(_publisher_artifact("partial"))
    
    job_id = pipeline.submit_job("content_pipeline", {
        "product_query": "chairs",
        "ethical_validation": "verified",
    })
    
    pipeline.process_next_job()
    pipeline.approve_job(job_id)
    pipeline.process_next_job()
    
    status = pipeline.get_job_status(job_id)
    assert status["status"] == "failed"


def test_publisher_adapter_extracts_product_urls(pipeline, mock_agents):
    products = [
        {"product_name": "P1", "brand": "B1", "price": 10.0, "currency": "USD",
         "url": "https://example.com/p1", "asin": "A1"},
        {"product_name": "P2", "brand": "B2", "price": 20.0, "currency": "USD",
         "url": "https://example.com/p2", "asin": "A2"},
    ]
    mock_agents["scout"].execute.return_value = _make_agent_result(_scout_artifact(products))
    mock_agents["creator"].execute.return_value = _make_agent_result(_creator_artifact())
    mock_agents["publisher"].execute.return_value = _make_agent_result(_publisher_artifact())
    
    job_id = pipeline.submit_job("content_pipeline", {
        "product_query": "chairs",
        "ethical_validation": "verified",
    })
    
    pipeline.process_next_job()
    pipeline.approve_job(job_id)
    pipeline.process_next_job()
    
    call_args = mock_agents["publisher"].execute.call_args
    context = json.loads(call_args.args[1])
    assert "https://example.com/p1" in context["product_urls"]
    assert "https://example.com/p2" in context["product_urls"]


def test_unknown_agent_raises_at_construction(tmp_path):
    config_path = tmp_path / "pipelines.json"
    config_path.write_text(json.dumps({
        "defaults": {},
        "pipelines": {
            "bad_pipeline": {
                "steps": [{"name": "x", "type": "agent", "agent": "nonexistent"}]
            }
        }
    }), encoding="utf-8")
    
    with pytest.raises(PipelineError, match="nonexistent"):
        PipelineFlow(
            agents={},
            job_queue_path=str(tmp_path / "queue.db"),
            pipeline_definitions_path=str(config_path),
        )


def test_scout_malformed_json(pipeline, mock_agents):
    mock_agents["scout"].execute.return_value = _make_agent_result("not valid json")
    
    job_id = pipeline.submit_job("content_pipeline", {
        "product_query": "chairs",
        "ethical_validation": "verified",
    })
    
    pipeline.process_next_job()
    status = pipeline.get_job_status(job_id)
    assert status["status"] == "failed"
    assert "scout_artifact_malformed" in status["error_message"]


def test_creator_malformed_json(pipeline, mock_agents):
    mock_agents["scout"].execute.return_value = _make_agent_result(_scout_artifact())
    mock_agents["creator"].execute.return_value = _make_agent_result("not valid json")
    
    job_id = pipeline.submit_job("content_pipeline", {
        "product_query": "chairs",
        "ethical_validation": "verified",
    })
    
    pipeline.process_next_job()
    status = pipeline.get_job_status(job_id)
    assert status["status"] == "failed"
    assert "creator_artifact_malformed" in status["error_message"]


def test_approve_non_hitl_step_returns_false(pipeline, mock_agents):
    mock_agents["scout"].execute.return_value = _make_agent_result(_scout_artifact())
    
    job_id = pipeline.submit_job("content_pipeline", {
        "product_query": "chairs",
        "ethical_validation": "verified",
    })
    
    pipeline._queue.update_status(job_id, "pending_review", current_step="scout")
    
    result = pipeline.approve_job(job_id)
    assert result is False


def test_start_worker_idempotent(pipeline, mock_agents):
    t1 = pipeline.start_worker()
    t2 = pipeline.start_worker()
    assert t1 is t2
    pipeline.stop_worker()