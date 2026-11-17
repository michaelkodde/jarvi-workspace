# tests/test_step_adapters.py
"""
Unit tests for Step Adapters and guards.

Dependency-free: imports only src.services.step_adapters, so these run without
curl-cffi, selectolax, or Ollama present.
"""
import copy
import json

import pytest

from src.services.step_adapters import (
    AdapterError,
    analyst_adapter,
    coder_adapter,
    creator_adapter,
    get_adapter,
    guard_creator_artifact,
    guard_scout_artifact,
    map_publisher_status,
    publisher_adapter,
    scout_adapter,
    trend_spotter_adapter,
)


# ---------------------------------------------------------------------------
# scout_adapter — the ScoutMVI contract
# ---------------------------------------------------------------------------

def test_scout_adapter_emits_scoutmvi_compatible_context():
    """Context must be JSON carrying exactly the keys ScoutMVI reads."""
    params = {"query": "eco friendly pans", "domain": "camelcamelcamel.com"}
    task_prompt, context = scout_adapter({}, params, {})

    assert task_prompt == "eco friendly pans"
    parsed = json.loads(context)
    assert parsed["query"] == "eco friendly pans"
    assert parsed["domain"] == "camelcamelcamel.com"
    assert "ethical_validation" in parsed


def test_scout_adapter_forwards_ethical_validation():
    """Optional ethical criteria reach the playbook context."""
    params = {
        "query": "pans",
        "domain": "example.com",
        "ethical_validation": "no PFAS, no glued handles",
    }
    _, context = scout_adapter({}, params, {})
    assert json.loads(context)["ethical_validation"] == "no PFAS, no glued handles"


def test_scout_adapter_coerces_non_string_ethical_validation():
    """Non-string ethical_validation must not crash the adapter."""
    params = {"query": "pans", "domain": "example.com", "ethical_validation": ["a", "b"]}
    _, context = scout_adapter({}, params, {})
    assert isinstance(json.loads(context)["ethical_validation"], str)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("camelcamelcamel.com", "camelcamelcamel.com"),
        ("www.camelcamelcamel.com", "camelcamelcamel.com"),
        ("https://camelcamelcamel.com", "camelcamelcamel.com"),
        ("https://www.CamelCamelCamel.com/search?q=pans", "camelcamelcamel.com"),
        ("http://camelcamelcamel.com/x/y?z=1#frag", "camelcamelcamel.com"),
        ("camelcamelcamel.com:443", "camelcamelcamel.com"),
        ("  camelcamelcamel.com.  ", "camelcamelcamel.com"),
        ("//camelcamelcamel.com/deals", "camelcamelcamel.com"),
    ],
)
def _normalize_domain(raw: str) -> str:
    """Reduce a user-supplied domain or URL to a bare registry hostname.

    Strips scheme, credentials, port, path, query, fragment, trailing dot and a
    leading 'www.'. Meaningful subdomains (e.g. 'uk.') are PRESERVED because they
    address genuinely different sites.

    Args:
        raw: User-supplied domain, possibly a full URL or a scheme-relative URL.

    Returns:
        Lowercase bare hostname, or an empty string if it cannot be parsed.
    """
    value = raw.strip()
    if not value:
        return ""

    # urlsplit only populates netloc when a scheme or a leading '//' is present.
    # Do NOT prepend '//' if the value already starts with it — '////host' yields
    # an empty authority and silently loses the hostname.
    if "://" not in value and not value.startswith("//"):
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


def test_scout_adapter_preserves_meaningful_subdomain():
    """uk.camelcamelcamel.com is a different site; it must NOT be collapsed."""
    _, context = scout_adapter({}, {"query": "pans", "domain": "uk.camelcamelcamel.com"}, {})
    assert json.loads(context)["domain"] == "uk.camelcamelcamel.com"


def test_scout_adapter_missing_query_names_the_real_parameter():
    """Error text drives SelfCorrector; it must name 'query', not 'product_query'."""
    with pytest.raises(AdapterError) as exc:
        scout_adapter({}, {"domain": "example.com"}, {})
    assert "'query'" in str(exc.value)
    assert "product_query" not in str(exc.value)


def test_scout_adapter_missing_domain():
    """A domain is mandatory and must never be defaulted."""
    with pytest.raises(AdapterError) as exc:
        scout_adapter({}, {"query": "pans"}, {})
    assert "'domain'" in str(exc.value)


@pytest.mark.parametrize("bad", ["", "   ", "localhost", "not a domain", "http://", "???"])
def test_scout_adapter_rejects_unusable_domain(bad):
    """Garbage domains fail fast with an actionable message."""
    with pytest.raises(AdapterError):
        scout_adapter({}, {"query": "pans", "domain": bad}, {})


def test_scout_adapter_rejects_blank_query():
    with pytest.raises(AdapterError):
        scout_adapter({}, {"query": "   ", "domain": "example.com"}, {})


def test_scout_adapter_rejects_non_dict_parameters():
    """A malformed ticket must raise AdapterError, not TypeError."""
    with pytest.raises(AdapterError) as exc:
        scout_adapter({}, "just a string", {})
    assert "JSON object" in str(exc.value)


def test_scout_adapter_does_not_mutate_parameters():
    """Adapters are pure input builders."""
    params = {"query": "pans", "domain": "https://www.example.com/x"}
    snapshot = copy.deepcopy(params)
    scout_adapter({}, params, {})
    assert params == snapshot


# ---------------------------------------------------------------------------
# creator_adapter
# ---------------------------------------------------------------------------

def test_creator_adapter_requires_scout_artifact():
    with pytest.raises(AdapterError):
        creator_adapter({}, {}, {})


def test_creator_adapter_builds_valid_content_request():
    """Defaults must satisfy Creator's allowlists."""
    products = json.dumps(
        [{"product_name": "Pan", "brand": "B", "price": 1.0,
          "currency": "USD", "url": "https://x/1", "asin": "B1"}]
    )
    task_prompt, context = creator_adapter({"scout": products}, {}, {})
    parsed = json.loads(context)
    cr = parsed["content_request"]

    assert task_prompt == products
    assert cr["content_type"] == "social_post"
    assert cr["platform"] == "twitter"
    assert cr["tone"] == "professional"
    assert cr["variant_count"] == 1
    assert cr["output_routing"]["destination"] == "orchestrator"


def test_creator_adapter_forces_orchestrator_destination():
    """No file writes before HITL approval."""
    products = json.dumps([{"product_name": "P"}])
    params = {"content_request": {"output_routing": {"destination": "file", "file_path": "x.md"}}}
    _, context = creator_adapter({"scout": products}, params, {})
    assert json.loads(context)["content_request"]["output_routing"]["destination"] == "orchestrator"


def test_creator_adapter_does_not_mutate_caller_routing():
    """Regression: shallow merge used to alias and overwrite job.parameters."""
    products = json.dumps([{"product_name": "P"}])
    params = {"content_request": {"output_routing": {"destination": "file", "file_path": "x.md"}}}
    snapshot = copy.deepcopy(params)

    creator_adapter({"scout": products}, params, {})
    creator_adapter({"scout": products}, params, {})

    assert params == snapshot
    assert params["content_request"]["output_routing"]["destination"] == "file"


def test_creator_adapter_merges_defaults_then_params():
    products = json.dumps([{"product_name": "P"}])
    defaults = {"content_request": {"tone": "casual", "content_type": "blog_post"}}
    params = {"content_request": {"tone": "persuasive"}}
    _, context = creator_adapter({"scout": products}, params, defaults)
    cr = json.loads(context)["content_request"]
    assert cr["tone"] == "persuasive"
    assert cr["content_type"] == "blog_post"


# ---------------------------------------------------------------------------
# publisher_adapter
# ---------------------------------------------------------------------------

def test_publisher_adapter_extracts_urls_and_skips_placeholders():
    scout = json.dumps([{"url": "https://x/1"}, {"url": "N/A"}, {"url": ""}, {"no_url": 1}])
    creator = json.dumps({"variants": []})
    task_prompt, context = publisher_adapter({"scout": scout, "creator": creator}, {}, {})
    assert task_prompt == creator
    assert json.loads(context)["product_urls"] == ["https://x/1"]


def test_publisher_adapter_rejects_malformed_scout_artifact():
    with pytest.raises(AdapterError):
        publisher_adapter({"scout": "not json", "creator": "{}"}, {}, {})


def test_publisher_adapter_requires_both_artifacts():
    with pytest.raises(AdapterError):
        publisher_adapter({"scout": "[]"}, {}, {})
    with pytest.raises(AdapterError):
        publisher_adapter({"creator": "{}"}, {}, {})


# ---------------------------------------------------------------------------
# coder / trend_spotter / analyst
# ---------------------------------------------------------------------------

def test_coder_adapter_requires_target_files():
    with pytest.raises(AdapterError) as exc:
        coder_adapter({}, {}, {})
    assert "'target_files'" in str(exc.value)


def test_coder_adapter_success():
    focus, targets = coder_adapter({}, {"target_files": " a.py,b.py ", "focus": "check locks"}, {})
    assert focus == "check locks"
    assert targets == "a.py,b.py"


def test_trend_spotter_adapter_uses_query_not_seed_keyword():
    """Regression: the real parameter name is 'query'."""
    query, context = trend_spotter_adapter({}, {"query": "sustainable kitchenware"}, {})
    assert query == "sustainable kitchenware"
    assert context == ""


def test_trend_spotter_adapter_rejects_seed_keyword_only():
    with pytest.raises(AdapterError) as exc:
        trend_spotter_adapter({}, {"seed_keyword": "x"}, {})
    assert "'query'" in str(exc.value)


def test_analyst_adapter_requires_query():
    with pytest.raises(AdapterError):
        analyst_adapter({}, {}, {})
    query, _ = analyst_adapter({}, {"query": "validate this niche"}, {})
    assert query == "validate this niche"


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

def test_get_adapter_covers_every_pipeline_agent():
    for name in ("scout", "creator", "publisher", "coder", "trend_spotter", "analyst"):
        assert callable(get_adapter(name))


def test_get_adapter_rejects_unknown_agent():
    with pytest.raises(AdapterError):
        get_adapter("nonexistent")


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------

def test_guard_scout_accepts_scoutmvi_output_shape():
    """asdict(ProductRecord) emits exactly these six fields."""
    artifact = json.dumps(
        [{"product_name": "Pan", "brand": "B", "price": 29.99,
          "currency": "GBP", "url": "https://x/1", "asin": ""}]
    )
    guard_scout_artifact(artifact)  # must not raise


def test_guard_scout_rejects_empty_array():
    with pytest.raises(AdapterError) as exc:
        guard_scout_artifact("[]")
    assert "no_verified_products" in str(exc.value)


def test_guard_scout_rejects_all_na():
    artifact = json.dumps([{"asin": "N/A", "url": "N/A"}])
    with pytest.raises(AdapterError):
        guard_scout_artifact(artifact)


@pytest.mark.parametrize("bad", ["not json", "{}", "null", 123])
def test_guard_scout_rejects_malformed(bad):
    with pytest.raises(AdapterError):
        guard_scout_artifact(bad)


def test_guard_creator_detects_failed_validation():
    envelope = json.dumps(
        {"variants": [{"output_metadata": {"routing": {"validation_passed": False}}}]}
    )
    with pytest.raises(AdapterError) as exc:
        guard_creator_artifact(envelope)
    assert "creator_validation_failed" in str(exc.value)


def test_guard_creator_accepts_valid_envelope():
    envelope = json.dumps(
        {"variants": [{"output_metadata": {"routing": {"validation_passed": True}}}]}
    )
    guard_creator_artifact(envelope)


@pytest.mark.parametrize(
    "status,expected",
    [("delivered", "published"), ("partial", "failed"),
     ("failed", "failed"), ("deferred", "deferred"), ("weird", "failed")],
)
def test_map_publisher_status(status, expected):
    assert map_publisher_status(json.dumps({"status": status})) == expected


def test_map_publisher_status_rejects_malformed():
    with pytest.raises(AdapterError):
        map_publisher_status("not json")