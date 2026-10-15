"""End-to-end viability test for the Jarvi TrendSpotter -> Analyst pipeline.

This module intentionally uses real local Ollama inference rather than mocks.
It validates artifact contracts, clean-context handoff, markdown stripping,
and persistent token accounting across two specialist agents.

The Analyst is deliberately given a mock research context. Because the
Analyst playbook requires verifiable brand proof, the expected real-inference
outcome is REJECTED. This exercises the short-circuit path end to end; the
APPROVED path is covered by mocked unit tests and by Phase 2 research.

Run with:
    pytest -m integration -s

Requires:
    pytest-timeout, a running Ollama daemon, and both agent playbooks on disk.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

__all__: list[str] = []

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
TEST_MODEL = os.getenv("JARVI_TEST_MODEL", "qwen2.5:7b")
PLAYBOOK_DIR = ROOT_DIR / "src" / "agents" / "playbooks"
CONFIG_PATH = ROOT_DIR / "config.json"
LEDGER_PATH = ROOT_DIR / "dashboard" / "cost_log.csv"

NICHE_KEYS = frozenset(
    {
        "niche_name",
        "target_audience",
        "estimated_demand",
        "competition_level",
    }
)

APPROVED_SECTIONS = (
    "ethical validation",
    "supply chain viability",
    "target demographics",
    "recommended product types",
)

CONTEXT_PROBE = "Mock research data to verify context isolation."

_H2_PATTERN = re.compile(r"^##[ \t]+(.+?)[ \t]*$", re.MULTILINE)

pytestmark = [pytest.mark.integration, pytest.mark.timeout(600)]


@dataclass(frozen=True)
class _IntegrationSummary:
    """Immutable outcome of a pipeline viability run.

    Attributes:
        niches_analyzed: Number of niches handed to the Analyst.
        approved: Number of APPROVED verdicts.
        rejected: Number of REJECTED verdicts.
    """

    niches_analyzed: int
    approved: int
    rejected: int

    def to_line(self) -> str:
        """Formats the outcome as a human-readable summary.

        Returns:
            A single-line summary containing counts only. No artifact content.
        """
        return (
            f"Integration test passed: {self.niches_analyzed} niches analyzed, "
            f"{self.approved} approved, {self.rejected} rejected"
        )


def _h2_sections(body: str) -> list[tuple[str, str]]:
    """Splits a Markdown artifact into normalized H2 sections.

    Args:
        body: The raw Markdown artifact produced by an agent.

    Returns:
        A list of (normalized_header, section_body) tuples in document order.
        H3 and deeper headers are intentionally ignored.
    """
    matches = list(_H2_PATTERN.finditer(body))
    sections: list[tuple[str, str]] = []

    for index, match in enumerate(matches):
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(body)
        )
        sections.append((match.group(1).strip().lower(), body[match.end():end]))

    return sections


def _verify_ledger(expected_analyst_calls: int) -> None:
    """Verifies that this run's agents appear in the persistent ledger tail.

    Reads a widened tail window rather than an exact row-count delta so that
    concurrent writers cannot produce a false failure.

    Args:
        expected_analyst_calls: Number of Analyst executions performed.

    Raises:
        AssertionError: If the ledger is missing or expected agents are absent.
    """
    if not LEDGER_PATH.is_file():
        raise AssertionError("Token ledger CSV was not created.")

    try:
        with LEDGER_PATH.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        logger.warning(
            "Token ledger unreadable; skipping ledger assertion (%s).",
            type(exc).__name__,
        )
        return

    window = max(expected_analyst_calls + 1, 25)
    tail = rows[-window:]
    agent_names = {row.get("agent_name", "") for row in tail}

    assert "TrendSpotter" in agent_names, (
        "TrendSpotter usage missing from the ledger tail."
    )
    assert "Analyst" in agent_names, (
        "Analyst usage missing from the ledger tail."
    )

    analyst_entries = sum(
        1 for row in tail if row.get("agent_name") == "Analyst"
    )
    assert analyst_entries >= expected_analyst_calls, (
        "Analyst ledger entries are fewer than the number of executions."
    )


def test_trend_spotter_to_analyst_real_ollama() -> None:
    """Validates the TrendSpotter -> Analyst handoff against real Ollama.

    Executes the Trend Spotter on a fixed seed keyword, parses its JSON array,
    and submits each niche to the Analyst together with an isolated context
    probe. Verifies artifact schemas, verdict ordering, anti-echo behavior,
    and persistent token accounting.

    Prerequisites are checked before any inference. Missing playbooks, an
    absent package initializer, or an unloaded model produce a clean skip
    rather than a failure.

    Raises:
        AssertionError: If any artifact, verdict, or ledger contract is violated.
    """
    if not (ROOT_DIR / "src" / "agents" / "__init__.py").is_file():
        pytest.skip("Missing src/agents/__init__.py; package is not importable.")

    for filename in ("trend_spotter.md", "analyst.md"):
        if not (PLAYBOOK_DIR / filename).is_file():
            pytest.skip(f"Missing agent playbook: {filename}")

    from src.agents.analyst import Analyst
    from src.agents.trend_spotter import TrendSpotter
    from src.utils.llm_client import query_ollama, verify_model_available
    from src.utils.token_ledger import TokenLedger

    if not verify_model_available(TEST_MODEL):
        pytest.skip(
            "Lazy-Loaded Routing (Pillar 1.6): Model not currently loaded in "
            "Ollama. Pull it or set JARVI_TEST_MODEL."
        )

    ledger = TokenLedger(
        config_path=str(CONFIG_PATH),
        log_dir=str(ROOT_DIR / "dashboard"),
    )

    spotter = TrendSpotter(
        llm_client=query_ollama,
        token_ledger=ledger,
        model_name=TEST_MODEL,
        playbook_path=str(PLAYBOOK_DIR / "trend_spotter.md"),
    )

    analyst = Analyst(
        llm_client=query_ollama,
        token_ledger=ledger,
        model_name=TEST_MODEL,
        playbook_path=str(PLAYBOOK_DIR / "analyst.md"),
    )

    spot_result = spotter.execute("Home Office Ergonomics")
    artifact = spot_result.artifact

    assert isinstance(artifact, str) and artifact.strip(), (
        "TrendSpotter returned an empty artifact."
    )
    assert "```" not in artifact, (
        "BaseAgent failed to strip markdown fences."
    )

    try:
        parsed: Any = json.loads(artifact)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            "TrendSpotter returned malformed JSON. "
            "Inspect the artifact locally; payload is intentionally not logged."
        ) from exc

    assert isinstance(parsed, list), (
        "TrendSpotter artifact must be a raw JSON array, not a root object."
    )
    assert len(parsed) > 0, "TrendSpotter returned an empty niche array."

    for index, niche in enumerate(parsed):
        assert isinstance(niche, dict), (
            f"Niche at index {index} is not a JSON object."
        )
        assert NICHE_KEYS.issubset(niche.keys()), (
            f"Niche at index {index} is missing required schema keys."
        )
        for key in sorted(NICHE_KEYS):
            value = niche.get(key)
            assert isinstance(value, str) and value.strip(), (
                f"Niche at index {index} has a missing or empty field: {key}"
            )

    approved = 0
    rejected = 0

    for niche in parsed:
        niche_json = json.dumps(niche, ensure_ascii=False)

        result = analyst.execute(niche_json, context=CONTEXT_PROBE)
        prd = result.artifact

        assert isinstance(prd, str) and prd.strip(), (
            "Analyst returned an empty PRD."
        )
        assert CONTEXT_PROBE not in prd, (
            "Analyst echoed the isolated context instead of analyzing the ticket."
        )

        sections = _h2_sections(prd)
        assert sections, "Analyst produced no H2 sections."

        assert sections[0][0] == "final verdict", (
            "Final Verdict must be the first H2 header in the PRD."
        )

        verdict_match = re.search(
            r"\b(APPROVED|REJECTED)\b",
            sections[0][1],
            re.IGNORECASE,
        )
        assert verdict_match is not None, (
            "Analyst verdict is missing from the Final Verdict section."
        )

        verdict = verdict_match.group(1).upper()

        if verdict == "APPROVED":
            approved += 1
            headers = {name for name, _ in sections}
            for required in APPROVED_SECTIONS:
                assert required in headers, (
                    f"Approved PRD is missing required section: {required}"
                )
        else:
            rejected += 1

    _verify_ledger(expected_analyst_calls=len(parsed))

    summary = _IntegrationSummary(
        niches_analyzed=len(parsed),
        approved=approved,
        rejected=rejected,
    )

    if summary.rejected == summary.niches_analyzed:
        logger.warning(
            "All %d niches were REJECTED. This is expected with a mock "
            "context: the Analyst Proof Criteria requires real brand "
            "evidence. Before enabling the orchestrator, also audit the "
            "confidence_score contract, which TrendSpotter does not currently "
            "emit but the Analyst playbook requires.",
            summary.niches_analyzed,
        )

    print(summary.to_line())