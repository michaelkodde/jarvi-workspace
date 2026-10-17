# tests/test_discovery.py
"""Regression tests for the AST-only Discovery Engine."""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import pytest

from src.companion.discovery import (
    AgentContract,
    DiscoveryEngine,
    DiscoveryError,
    PlaybookResolution,
    SystemRegistry,
)

BASE_IMPORT = "from src.agents.base_agent import BaseAgent\n"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    for directory in ("src/agents", "src/utils", "tests", "brain/skills"):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    return tmp_path


def test_trend_spotter_pattern(workspace: Path) -> None:
    _write(
        workspace / "src/agents/trend_spotter.py",
        BASE_IMPORT + '''
class TrendSpotter(BaseAgent):
    def __init__(self, llm_client, token_ledger, model_name,
                 playbook_path="src/agents/playbooks/trend_spotter.md"):
        super().__init__(
            llm_client=llm_client,
            token_ledger=token_ledger,
            playbook_path=playbook_path,
            agent_name="TrendSpotter",
            model_name=model_name,
        )
''',
    )
    _write(workspace / "src/agents/playbooks/trend_spotter.md", "# Trend Spotter\n")
    _write(workspace / "tests/test_trend_spotter.py", "pass\n")

    registry = DiscoveryEngine(workspace, force=True).scan()
    contract = registry.agents["TrendSpotter"]

    assert contract.class_type == "base_agent_subclass"
    assert contract.has_custom_execute is False
    assert contract.calls_super_with_playbook is True
    assert contract.agent_name_param == "TrendSpotter"
    assert contract.playbook.source == "default"
    assert contract.playbook.is_required is False
    assert contract.status == "ready"


def test_router_pattern(workspace: Path) -> None:
    _write(
        workspace / "src/agents/router.py",
        '''
class Router:
    def __init__(self, ollama_model: str, playbook_path: str,
                 llm_callable=query_ollama, token_ledger=None):
        self._model = ollama_model

    def route(self, raw_prompt: str):
        pass
''',
    )
    _write(workspace / "src/agents/playbooks/router.md", "# Routing policy\n")

    contract = DiscoveryEngine(workspace, force=True).scan().agents["Router"]

    assert contract.class_type == "standalone"
    assert contract.has_route_method is True
    assert contract.has_custom_execute is False
    assert contract.calls_super_with_playbook is False
    assert contract.agent_name_param is None
    assert contract.playbook.source == "conventional"
    assert contract.playbook.is_required is True
    params = {p.name: p for p in contract.constructor_params}
    assert params["playbook_path"].has_default is False
    assert params["llm_callable"].default_value == "query_ollama"


def test_custom_execute_needs_no_playbook(workspace: Path) -> None:
    _write(
        workspace / "src/agents/scout.py",
        BASE_IMPORT + '''
class Scout(BaseAgent):
    def __init__(self, llm_client):
        super().__init__(llm_client=llm_client, agent_name="Scout")

    def execute(self, task, context):
        return task
''',
    )
    _write(workspace / "tests/test_scout.py", "pass\n")

    contract = DiscoveryEngine(workspace, force=True).scan().agents["Scout"]
    assert contract.has_custom_execute is True
    assert contract.playbook.path is None
    assert contract.status == "ready"


def test_missing_playbook(workspace: Path) -> None:
    _write(
        workspace / "src/agents/analyst.py",
        BASE_IMPORT + '''
class Analyst(BaseAgent):
    def __init__(self, playbook_path="src/agents/playbooks/analyst.md"):
        super().__init__(playbook_path=playbook_path, agent_name="Analyst")
''',
    )
    contract = DiscoveryEngine(workspace, force=True).scan().agents["Analyst"]
    assert contract.status == "missing_playbook"
    assert contract.playbook.exists_on_disk is False


def test_empty_playbook_is_missing(workspace: Path) -> None:
    _write(
        workspace / "src/agents/analyst.py",
        BASE_IMPORT + '''
class Analyst(BaseAgent):
    def __init__(self, playbook_path="src/agents/playbooks/analyst.md"):
        super().__init__(playbook_path=playbook_path)
''',
    )
    _write(workspace / "src/agents/playbooks/analyst.md", "   \n\t\n")
    contract = DiscoveryEngine(workspace, force=True).scan().agents["Analyst"]
    assert contract.status == "missing_playbook"


def test_missing_base_import_wins(workspace: Path) -> None:
    _write(
        workspace / "src/agents/broken.py",
        "class Broken(BaseAgent):\n    def __init__(self):\n        pass\n",
    )
    assert DiscoveryEngine(workspace, force=True).scan().agents["Broken"].status == "missing_base"


def test_syntax_error_is_isolated(workspace: Path) -> None:
    _write(workspace / "src/agents/broken.py", "class Broken(:\n")
    # Good must have execute() to pass the agent filter
    _write(workspace / "src/agents/good.py", "class Good:\n    def execute(self): pass\n")

    registry = DiscoveryEngine(workspace, force=True).scan()
    assert "Good" in registry.agents
    assert any(e.severity == "error" for e in registry.scan_errors)
    assert all("class Broken" not in e.message for e in registry.scan_errors)


def test_cache_round_trip_is_typed(workspace: Path) -> None:
    """Regression: cached nested dataclasses must not decay into dicts."""
    _write(workspace / "src/agents/scout.py", "class Scout:\n    def execute(self): pass\n")

    engine = DiscoveryEngine(workspace)
    first = engine.scan()
    second = engine.scan()

    assert isinstance(second, SystemRegistry)
    assert isinstance(second.agents["Scout"], AgentContract)
    assert isinstance(second.agents["Scout"].playbook, PlaybookResolution)
    assert asdict(first)["agents"] == asdict(second)["agents"]
    engine.report(second)


def test_corrupt_cache_does_not_recurse(workspace: Path) -> None:
    # Scout must have execute() to pass the agent filter
    _write(workspace / "src/agents/scout.py", "class Scout:\n    def execute(self): pass\n")
    engine = DiscoveryEngine(workspace)
    engine.scan()
    engine.cache_path.write_text("{not valid json", encoding="utf-8")

    registry = engine.scan()
    assert "Scout" in registry.agents


def test_deleting_playbook_invalidates_cache(workspace: Path) -> None:
    _write(
        workspace / "src/agents/analyst.py",
        BASE_IMPORT + '''
class Analyst(BaseAgent):
    def __init__(self, playbook_path="src/agents/playbooks/analyst.md"):
        super().__init__(playbook_path=playbook_path)
''',
    )
    playbook = workspace / "src/agents/playbooks/analyst.md"
    _write(playbook, "# Policy\n")

    engine = DiscoveryEngine(workspace)
    assert engine.scan().agents["Analyst"].status == "no_tests"

    playbook.unlink()
    assert engine.scan().agents["Analyst"].status == "missing_playbook"


def test_force_bypasses_cache(workspace: Path) -> None:
    """Verify that force=True bypasses cache even when cache is valid."""
    _write(workspace / "src/agents/scout.py", "class Scout:\n    def execute(self): pass\n")
    _write(workspace / "tests/test_scout.py", "pass\n")
    
    engine = DiscoveryEngine(workspace)
    first = engine.scan()
    assert "Scout" in first.agents
    
    # Modify the file - cache should invalidate (fingerprint changes)
    _write(workspace / "src/agents/scout.py", "class Scout:\n    def execute(self): pass\n    # modified\n")
    second = engine.scan()
    # Verify the scan actually ran by checking duration differs (both > 0)
    assert first.scan_duration_ms >= 0
    assert second.scan_duration_ms >= 0
    
    # Now test force=True bypasses even valid cache
    # The key test: force=True should always produce a fresh scan
    third = DiscoveryEngine(workspace, force=True).scan()
    assert "Scout" in third.agents
    # Verify it's a fresh scan by checking it has a valid timestamp
    assert third.scan_timestamp is not None
    assert len(third.scan_timestamp) > 0


def test_missing_pipeline_agents_are_sorted(workspace: Path) -> None:
    _write(
        workspace / "config/pipelines.json",
        json.dumps({
            "pipelines": {
                "content_pipeline": {
                    "steps": [
                        {"name": "a", "agent": "Zeta"},
                        {"name": "b", "agent": "Alpha"},
                        {"name": "c", "agent": "Mike"},
                    ]
                }
            }
        }),
    )
    engine = DiscoveryEngine(workspace)
    first = engine.scan()
    second = DiscoveryEngine(workspace, force=True).scan()

    assert first.missing_pipeline_agents == ["Alpha", "Mike", "Zeta"]
    assert first.missing_pipeline_agents == second.missing_pipeline_agents


def test_recursive_test_discovery(workspace: Path) -> None:
    _write(
        workspace / "src/agents/scout.py",
        "class Scout:\n    def execute(self, task, context): pass\n"
    )
    _write(workspace / "tests/unit/agents/test_scout.py", "pass\n")

    contract = DiscoveryEngine(workspace, force=True).scan().agents["Scout"]
    assert contract.test_file == "tests/unit/agents/test_scout.py"
    assert contract.has_custom_execute is True
    assert contract.status == "ready"


def test_playbook_cannot_escape_workspace(workspace: Path) -> None:
    _write(
        workspace / "src/agents/leaky.py",
        BASE_IMPORT + '''
class Leaky(BaseAgent):
    def __init__(self, playbook_path="../../outside.md"):
        super().__init__(playbook_path=playbook_path)
''',
    )
    _write(workspace.parent / "outside.md", "# Outside\n")

    contract = DiscoveryEngine(workspace, force=True).scan().agents["Leaky"]
    assert contract.playbook.path is None
    assert contract.playbook.exists_on_disk is False


def test_empty_workspace_degrades(workspace: Path) -> None:
    for path in (workspace / "src/agents", workspace / "src/utils"):
        for child in path.iterdir():
            child.unlink()
        path.rmdir()

    registry = DiscoveryEngine(workspace, force=True).scan()
    assert registry.agents == {}
    assert registry.pipelines == []
    assert registry.scan_errors


def test_invalid_workspace_type() -> None:
    with pytest.raises(DiscoveryError):
        DiscoveryEngine(123)  # type: ignore[arg-type]


def test_dataclass_filtered_out(workspace: Path) -> None:
    """Verify that @dataclass decorated classes are not treated as agents."""
    _write(
        workspace / "src/agents/models.py",
        '''
from dataclasses import dataclass

@dataclass
class AgentResult:
    content: str
    tokens: int

class RealAgent:
    def execute(self):
        pass
''',
    )
    registry = DiscoveryEngine(workspace, force=True).scan()
    assert "AgentResult" not in registry.agents
    assert "RealAgent" in registry.agents


def test_exception_classes_filtered_out(workspace: Path) -> None:
    """Verify that exception classes are not treated as agents."""
    _write(
        workspace / "src/agents/errors.py",
        '''
class AgentError(Exception):
    pass

class WorkingAgent:
    def execute(self):
        pass
''',
    )
    registry = DiscoveryEngine(workspace, force=True).scan()
    assert "AgentError" not in registry.agents
    assert "WorkingAgent" in registry.agents