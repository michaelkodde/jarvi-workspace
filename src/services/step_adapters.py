"""
Step Adapters: Input builders and guards for pipeline steps.

Adapters ONLY build (task_prompt, context) tuples. They do NOT call agents.
Guards validate artifacts after agent execution.
"""
import json
import logging
from typing import Any, Callable, Dict, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "AdapterError",
    "get_adapter",
    "scout_adapter",
    "creator_adapter",
    "publisher_adapter",
    "coder_adapter",
    "guard_scout_artifact",
    "guard_creator_artifact",
    "map_publisher_status",
    "trend_spotter_adapter",
    "analyst_adapter",
]


class AdapterError(Exception):
    """Raised when adapter or guard validation fails."""
    pass


def scout_adapter(
    artifacts: Dict[str, str],
    parameters: Dict[str, Any],
    defaults: Dict[str, Any],
) -> Tuple[str, str]:
    """
    Build inputs for the Scout agent.
    
    Args:
        artifacts: Previous artifacts (unused for Scout).
        parameters: Job parameters.
        defaults: Default configuration.
    
    Returns:
        Tuple of (task_prompt, context).
    
    Raises:
        AdapterError: If required parameters are missing.
    """
    try:
        task_prompt = parameters["product_query"]
    except KeyError as e:
        raise AdapterError(f"Missing required parameter: product_query") from e
    
    if not isinstance(task_prompt, str) or not task_prompt:
        raise AdapterError("product_query must be a non-empty string")
    
    context = parameters.get("ethical_validation", "")
    if not isinstance(context, str):
        context = str(context)
    
    return task_prompt, context


def creator_adapter(
    artifacts: Dict[str, str],
    parameters: Dict[str, Any],
    defaults: Dict[str, Any],
) -> Tuple[str, str]:
    """
    Build inputs for the Creator agent.
    
    Args:
        artifacts: Previous artifacts (must include "scout").
        parameters: Job parameters.
        defaults: Default configuration.
    
    Returns:
        Tuple of (task_prompt, context).
    
    Raises:
        AdapterError: If scout artifact is missing.
    """
    if "scout" not in artifacts:
        raise AdapterError("Missing scout artifact")
    
    task_prompt = artifacts["scout"]
    if not isinstance(task_prompt, str):
        raise AdapterError("scout artifact must be a string")
    
    default_cr = defaults.get("content_request", {})
    param_cr = parameters.get("content_request", {})
    
    if not isinstance(default_cr, dict):
        default_cr = {}
    if not isinstance(param_cr, dict):
        param_cr = {}
    
    content_request = {**default_cr, **param_cr}
    
    # Force destination to orchestrator (no file writes before approval)
    output_routing = content_request.get("output_routing", {})
    if not isinstance(output_routing, dict):
        output_routing = {}
    output_routing["destination"] = "orchestrator"
    content_request["output_routing"] = output_routing
    
    content_request.setdefault("request_id", parameters.get("job_id", ""))
    content_request.setdefault("product_index", 0)
    content_request.setdefault("content_type", "social_post")
    content_request.setdefault("platform", "twitter")
    content_request.setdefault("tone", "professional")
    content_request.setdefault("variant_count", 1)
    content_request.setdefault("length_constraints", {})
    
    context_dict = {
        "niche_name": parameters.get("niche_name", ""),
        "target_demographics": parameters.get("target_demographics", ""),
        "recommended_product_types": parameters.get("recommended_product_types", ""),
        "ethical_validation": parameters.get("ethical_validation", ""),
        "supply_chain_viability": parameters.get("supply_chain_viability", ""),
        "target_keywords": parameters.get("target_keywords", []),
        "social_proof": parameters.get("social_proof", {}),
        "content_request": content_request,
    }
    
    return task_prompt, json.dumps(context_dict)


def publisher_adapter(
    artifacts: Dict[str, str],
    parameters: Dict[str, Any],
    defaults: Dict[str, Any],
) -> Tuple[str, str]:
    """
    Build inputs for the Publisher agent.
    
    Extracts product_urls from the Scout artifact.
    
    Args:
        artifacts: Previous artifacts (must include "scout" and "creator").
        parameters: Job parameters.
        defaults: Default configuration.
    
    Returns:
        Tuple of (task_prompt, context).
    
    Raises:
        AdapterError: If required artifacts are missing or malformed.
    """
    if "creator" not in artifacts:
        raise AdapterError("Missing creator artifact")
    if "scout" not in artifacts:
        raise AdapterError("Missing scout artifact")
    
    task_prompt = artifacts["creator"]
    if not isinstance(task_prompt, str):
        raise AdapterError("creator artifact must be a string")
    
    try:
        scout_products = json.loads(artifacts["scout"])
    except json.JSONDecodeError as e:
        raise AdapterError(f"scout_artifact_malformed: {e.__class__.__name__}") from e
    
    product_urls = []
    if isinstance(scout_products, list):
        for p in scout_products:
            if isinstance(p, dict) and "url" in p:
                url = p["url"]
                if url and url != "N/A":
                    product_urls.append(url)
    
    affiliate_config = defaults.get("affiliate_link_config", {})
    if not isinstance(affiliate_config, dict):
        affiliate_config = {}
    
    context_dict = {
        "affiliate_link_config": affiliate_config,
        "product_urls": product_urls,
        "publish_immediately": True,
    }
    
    return task_prompt, json.dumps(context_dict)


def coder_adapter(
    artifacts: Dict[str, str],
    parameters: Dict[str, Any],
    defaults: Dict[str, Any],
) -> Tuple[str, str]:
    """
    Build inputs for the Coder agent.
    Translates job parameters into the (task_prompt, context) format.
    
    Args:
        artifacts: Previous artifacts (unused for Coder, as it reads directly from disk).
        parameters: Job parameters (must include 'target_files').
        defaults: Default configuration.
        
    Returns:
        Tuple of (task_prompt, context).
        
    Raises:
        AdapterError: If required parameters are missing or invalid.
    """
    # 1. STRICT VALIDATION: Ensure target_files exists and is a valid string
    target_files = parameters.get("target_files")
    if not target_files or not isinstance(target_files, str):
        raise AdapterError("Missing or invalid required parameter: 'target_files' (must be a comma-separated string of file paths)")
    
    # Clean the string to prevent parsing errors in the agent
    target_files = target_files.strip()
    if not target_files:
        raise AdapterError("'target_files' cannot be an empty string")

    # 2. Build task prompt with safe, typed defaults
    focus = parameters.get("focus", "Review the code for bugs, anti-patterns, and Constitution violations.")
    if not isinstance(focus, str):
        focus = str(focus)
        
    task_prompt = focus
    context = target_files
    
    return task_prompt, context

def trend_spotter_adapter(
    artifacts: Dict[str, str],
    parameters: Dict[str, Any],
    defaults: Dict[str, Any],
) -> Tuple[str, str]:
    """Build inputs for the Trend Spotter agent."""
    query = parameters.get("query", "")
    if not query or not isinstance(query, str):
        raise AdapterError("Missing or invalid required parameter: 'query'")
    
    # Trend spotter just needs the user's prompt as the task
    return query, ""


def analyst_adapter(
    artifacts: Dict[str, str],
    parameters: Dict[str, Any],
    defaults: Dict[str, Any],
) -> Tuple[str, str]:
    """Build inputs for the Analyst agent."""
    query = parameters.get("query", "")
    if not query or not isinstance(query, str):
        raise AdapterError("Missing or invalid required parameter: 'query'")
    
    return query, ""

ADAPTER_REGISTRY: Dict[str, Callable] = {
    "scout": scout_adapter,
    "creator": creator_adapter,
    "publisher": publisher_adapter,
    "coder": coder_adapter,
    "trend_spotter": trend_spotter_adapter,
    "analyst": analyst_adapter,
}


def get_adapter(agent_name: str) -> Callable:
    """
    Get the adapter for an agent.
    
    Args:
        agent_name: Name of the agent.
    
    Returns:
        Adapter callable.
    
    Raises:
        AdapterError: If no adapter exists for the agent.
    """
    if agent_name not in ADAPTER_REGISTRY:
        raise AdapterError(f"No adapter for agent: {agent_name}")
    return ADAPTER_REGISTRY[agent_name]


def guard_scout_artifact(artifact: str) -> None:
    """
    Validate Scout artifact.
    
    Args:
        artifact: Scout's output artifact (JSON string).
    
    Raises:
        AdapterError: If artifact is malformed or contains only N/A products.
    """
    if not isinstance(artifact, str):
        raise AdapterError("scout_artifact_malformed: not a string")
    
    try:
        products = json.loads(artifact)
    except json.JSONDecodeError as e:
        raise AdapterError(f"scout_artifact_malformed: {e.__class__.__name__}") from e
    
    if not isinstance(products, list):
        raise AdapterError("scout_artifact_malformed: not a list")
    
    if len(products) == 0:
        raise AdapterError("no_verified_products")
    
    all_na = all(
        isinstance(p, dict) and (p.get("asin") == "N/A" or p.get("url") == "N/A")
        for p in products
    )
    if all_na:
        raise AdapterError("no_verified_products")


def guard_creator_artifact(artifact: str) -> None:
    """
    Validate Creator artifact.
    
    Args:
        artifact: Creator's output artifact (JSON string).
    
    Raises:
        AdapterError: If artifact is malformed or validation failed.
    """
    if not isinstance(artifact, str):
        raise AdapterError("creator_artifact_malformed: not a string")
    
    try:
        envelope = json.loads(artifact)
    except json.JSONDecodeError as e:
        raise AdapterError(f"creator_artifact_malformed: {e.__class__.__name__}") from e
    
    if not isinstance(envelope, dict):
        raise AdapterError("creator_artifact_malformed: not a dict")
    
    variants = envelope.get("variants", [])
    if not isinstance(variants, list):
        raise AdapterError("creator_artifact_malformed: variants not a list")
    
    for variant in variants:
        if not isinstance(variant, dict):
            continue
        output_metadata = variant.get("output_metadata", {})
        if not isinstance(output_metadata, dict):
            continue
        routing = output_metadata.get("routing", {})
        if isinstance(routing, dict) and routing.get("validation_passed") is False:
            raise AdapterError("creator_validation_failed")


def map_publisher_status(artifact: str) -> str:
    """
    Map Publisher report status to job status.
    
    Args:
        artifact: Publisher's output artifact (JSON string).
    
    Returns:
        Job status string.
    
    Raises:
        AdapterError: If artifact is malformed.
    """
    if not isinstance(artifact, str):
        raise AdapterError("publisher_artifact_malformed: not a string")
    
    try:
        report = json.loads(artifact)
    except json.JSONDecodeError as e:
        raise AdapterError(f"publisher_artifact_malformed: {e.__class__.__name__}") from e
    
    if not isinstance(report, dict):
        raise AdapterError("publisher_artifact_malformed: not a dict")
    
    status = report.get("status", "failed")
    if status == "delivered":
        return "published"
    elif status in ("partial", "failed"):
        return "failed"
    elif status == "deferred":
        return "deferred"
    else:
        return "failed"