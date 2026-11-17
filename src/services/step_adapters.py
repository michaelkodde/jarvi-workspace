# src/services/step_adapters.py
"""
Step Adapters: Input builders and guards for pipeline steps.

Adapters ONLY build (task_prompt, context) tuples. They do NOT call agents.
Guards validate artifacts after agent execution.

Contract notes:
- scout_adapter emits a JSON context carrying 'query' and 'domain', which is
  what ScoutMVI.execute() parses. Domain is normalized to a bare hostname so it
  matches DomainRuleRegistry keys.
- Adapters are pure: they never mutate the caller's parameters dict.
- The user's query is PII-adjacent and is NEVER logged.
"""
import json
import logging
import re
from typing import Any, Callable, Dict, Tuple
from urllib.parse import urlsplit

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

_HOSTNAME_RE = re.compile(r"[a-z0-9.-]+")


class AdapterError(Exception):
    """Raised when adapter or guard validation fails."""
    pass


def _normalize_domain(raw: str) -> str:
    """Reduce a user-supplied domain or URL to a bare registry hostname.

    Strips scheme, credentials, port, path, query, fragment, trailing dot and a
    leading 'www.'. Meaningful subdomains (e.g. 'uk.') are PRESERVED because they
    address genuinely different sites.

    Args:
        raw: User-supplied domain, possibly a full URL.

    Returns:
        Lowercase bare hostname, or an empty string if it cannot be parsed.
    """
    value = raw.strip()
    if not value:
        return ""

    # urlsplit only populates netloc when a scheme or leading '//' is present.
    if "://" not in value:
        value = "//" + value

    try:
        host = urlsplit(value).hostname or ""
    except ValueError:
        return ""

    host = host.strip().strip(".").lower()
    if host.startswith("www."):
        host = host[4:]

    if not host or "." not in host:
        return ""
    if not _HOSTNAME_RE.fullmatch(host):
        return ""
    return host


def _require_text(parameters: Dict[str, Any], key: str, label: str) -> str:
    """Fetch a mandatory non-empty string parameter.

    Args:
        parameters: Job parameters mapping.
        key: Parameter name.
        label: Adapter name used in the error message.

    Returns:
        The stripped string value.

    Raises:
        AdapterError: If the key is missing, not a string, or blank. The message
            names the exact parameter so SelfCorrector can repair the call.
    """
    value = parameters.get(key)
    if not isinstance(value, str) or not value.strip():
        raise AdapterError(
            f"{label}: missing or invalid required parameter '{key}' (must be a non-empty string)"
        )
    return value.strip()


def _require_parameters(parameters: Any, label: str) -> Dict[str, Any]:
    """Validate that job parameters arrived as a JSON object.

    Args:
        parameters: Raw parameters from the job record.
        label: Adapter name used in the error message.

    Returns:
        The parameters mapping.

    Raises:
        AdapterError: If parameters is not a dict.
    """
    if not isinstance(parameters, dict):
        raise AdapterError(
            f"{label}: 'parameters' must be a JSON object, got {type(parameters).__name__}"
        )
    return parameters


def scout_adapter(
    artifacts: Dict[str, str],
    parameters: Dict[str, Any],
    defaults: Dict[str, Any],
) -> Tuple[str, str]:
    """Build inputs for ScoutMVI.

    ScoutMVI.execute() parses `context` as JSON and reads 'query' and 'domain'.
    Both are mandatory; no domain is ever invented or defaulted.

    Args:
        artifacts: Previous artifacts (unused; Scout is a pipeline entry point).
        parameters: Job parameters. Requires 'query' and 'domain'.
            Optional 'ethical_validation' is forwarded to the playbook context.
        defaults: Default configuration (unused; no silent domain fallback).

    Returns:
        Tuple of (task_prompt, context) where task_prompt is the query string and
        context is a JSON object string with keys query, domain, ethical_validation.

    Raises:
        AdapterError: If parameters is not a dict, a required field is missing or
            blank, or the domain cannot be parsed into a hostname.
    """
    params = _require_parameters(parameters, "scout")

    query = _require_text(params, "query", "scout")
    raw_domain = _require_text(params, "domain", "scout")

    domain = _normalize_domain(raw_domain)
    if not domain:
        raise AdapterError(
            "scout: 'domain' could not be parsed into a hostname. "
            "Provide a bare domain such as 'camelcamelcamel.com'."
        )

    ethical_validation = params.get("ethical_validation", "")
    if not isinstance(ethical_validation, str):
        ethical_validation = str(ethical_validation)

    context = json.dumps(
        {
            "query": query,
            "domain": domain,
            "ethical_validation": ethical_validation,
        },
        ensure_ascii=False,
    )

    # Domain only. The query is user input and must never reach the log.
    logger.info("scout_adapter: domain=%s", domain)

    return query, context


def creator_adapter(
    artifacts: Dict[str, str],
    parameters: Dict[str, Any],
    defaults: Dict[str, Any],
) -> Tuple[str, str]:
    """Build inputs for the Creator agent.

    Args:
        artifacts: Previous artifacts (must include "scout").
        parameters: Job parameters.
        defaults: Default configuration, may supply 'content_request'.

    Returns:
        Tuple of (task_prompt, context). task_prompt is the Scout product array
        as a JSON string; context is a JSON object string containing
        'content_request' plus niche metadata.

    Raises:
        AdapterError: If the scout artifact is missing or not a string.
    """
    if "scout" not in artifacts:
        raise AdapterError("Missing scout artifact")

    task_prompt = artifacts["scout"]
    if not isinstance(task_prompt, str):
        raise AdapterError("scout artifact must be a string")

    params = parameters if isinstance(parameters, dict) else {}
    default_cr = defaults.get("content_request", {}) if isinstance(defaults, dict) else {}
    param_cr = params.get("content_request", {})

    if not isinstance(default_cr, dict):
        default_cr = {}
    if not isinstance(param_cr, dict):
        param_cr = {}

    content_request = {**default_cr, **param_cr}

    # Copy before mutating: the merged dict is shallow, so the nested routing
    # object may still alias job.parameters. Adapters must be pure.
    output_routing = content_request.get("output_routing")
    output_routing = dict(output_routing) if isinstance(output_routing, dict) else {}

    # Force destination to orchestrator (no file writes before approval).
    output_routing["destination"] = "orchestrator"
    content_request["output_routing"] = output_routing

    content_request.setdefault("request_id", params.get("job_id", ""))
    content_request.setdefault("product_index", 0)
    content_request.setdefault("content_type", "social_post")
    content_request.setdefault("platform", "twitter")
    content_request.setdefault("tone", "professional")
    content_request.setdefault("variant_count", 1)
    content_request.setdefault("length_constraints", {})

    context_dict = {
        "niche_name": params.get("niche_name", ""),
        "target_demographics": params.get("target_demographics", ""),
        "recommended_product_types": params.get("recommended_product_types", ""),
        "ethical_validation": params.get("ethical_validation", ""),
        "supply_chain_viability": params.get("supply_chain_viability", ""),
        "target_keywords": params.get("target_keywords", []),
        "social_proof": params.get("social_proof", {}),
        "content_request": content_request,
    }

    return task_prompt, json.dumps(context_dict)


def publisher_adapter(
    artifacts: Dict[str, str],
    parameters: Dict[str, Any],
    defaults: Dict[str, Any],
) -> Tuple[str, str]:
    """Build inputs for the Publisher agent.

    Extracts product_urls from the Scout artifact so affiliate links can be
    inserted into the Creator envelope.

    Args:
        artifacts: Previous artifacts (must include "scout" and "creator").
        parameters: Job parameters (unused).
        defaults: Default configuration, may supply 'affiliate_link_config'.

    Returns:
        Tuple of (task_prompt, context). task_prompt is the Creator envelope JSON
        string; context is a JSON object string with affiliate_link_config,
        product_urls and publish_immediately.

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

    affiliate_config = defaults.get("affiliate_link_config", {}) if isinstance(defaults, dict) else {}
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
    """Build inputs for the Coder agent.

    Args:
        artifacts: Previous artifacts (unused; Coder reads from disk).
        parameters: Job parameters. Requires 'target_files' as a comma-separated
            string of paths. Optional 'focus' overrides the review instruction.
        defaults: Default configuration (unused).

    Returns:
        Tuple of (task_prompt, context) where task_prompt is the focus string and
        context is the target_files string.

    Raises:
        AdapterError: If 'target_files' is missing, not a string, or blank.
    """
    params = _require_parameters(parameters, "coder")
    target_files = _require_text(params, "target_files", "coder")

    focus = params.get(
        "focus",
        "Review the code for bugs, anti-patterns, and Constitution violations.",
    )
    if not isinstance(focus, str):
        focus = str(focus)

    return focus, target_files


def trend_spotter_adapter(
    artifacts: Dict[str, str],
    parameters: Dict[str, Any],
    defaults: Dict[str, Any],
) -> Tuple[str, str]:
    """Build inputs for the TrendSpotter playbook.

    Args:
        artifacts: Previous artifacts (unused).
        parameters: Job parameters. Requires 'query' (the seed topic).
        defaults: Default configuration (unused).

    Returns:
        Tuple of (query, "").

    Raises:
        AdapterError: If 'query' is missing or blank.
    """
    params = _require_parameters(parameters, "trend_spotter")
    query = _require_text(params, "query", "trend_spotter")
    return query, ""


def analyst_adapter(
    artifacts: Dict[str, str],
    parameters: Dict[str, Any],
    defaults: Dict[str, Any],
) -> Tuple[str, str]:
    """Build inputs for the Analyst playbook.

    Args:
        artifacts: Previous artifacts (unused).
        parameters: Job parameters. Requires 'query'.
        defaults: Default configuration (unused).

    Returns:
        Tuple of (query, "").

    Raises:
        AdapterError: If 'query' is missing or blank.
    """
    params = _require_parameters(parameters, "analyst")
    query = _require_text(params, "query", "analyst")
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
    """Get the adapter for an agent.

    Args:
        agent_name: Name of the agent as declared in pipelines.json.

    Returns:
        Adapter callable.

    Raises:
        AdapterError: If no adapter exists for the agent.
    """
    if agent_name not in ADAPTER_REGISTRY:
        raise AdapterError(f"No adapter for agent: {agent_name}")
    return ADAPTER_REGISTRY[agent_name]


def guard_scout_artifact(artifact: str) -> None:
    """Validate a Scout artifact.

    Args:
        artifact: Scout's output artifact (JSON string of a product array).

    Raises:
        AdapterError: If the artifact is malformed, empty, or contains only
            N/A placeholder products.
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
    """Validate a Creator artifact.

    Args:
        artifact: Creator's output artifact (JSON string of a ContentEnvelope).

    Raises:
        AdapterError: If the artifact is malformed or any variant reports
            validation_passed == False.
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
    """Map a Publisher report status to a job status.

    Args:
        artifact: Publisher's output artifact (JSON string of a PublicationReport).

    Returns:
        Job status string: 'published', 'failed', or 'deferred'.

    Raises:
        AdapterError: If the artifact is malformed.
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