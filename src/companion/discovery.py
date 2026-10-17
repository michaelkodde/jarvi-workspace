# src/companion/discovery.py
"""AST-only workspace discovery for the Jarvi Companion.

Never imports, instantiates, or executes agent code. Reads source as text,
parses with :mod:`ast`, and returns a typed :class:`SystemRegistry`.
"""
from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "AgentContract",
    "DiscoveryEngine",
    "DiscoveryError",
    "ParamSpec",
    "PipelineContract",
    "PlaybookResolution",
    "ScanError",
    "SystemRegistry",
    "UtilContract",
]

logger = logging.getLogger(__name__)

_CACHE_VERSION = 1
_SNAKE_1 = re.compile(r"(.)([A-Z][a-z]+)")
_SNAKE_2 = re.compile(r"([a-z0-9])([A-Z])")


class DiscoveryError(Exception):
    """Raised for unrecoverable engine configuration errors."""


@dataclass(frozen=True)
class ParamSpec:
    """One constructor parameter."""

    name: str
    has_default: bool
    default_value: str | None


@dataclass(frozen=True)
class PlaybookResolution:
    """Resolved playbook location and on-disk state."""

    path: str | None
    source: str | None
    is_required: bool
    exists_on_disk: bool


@dataclass(frozen=True)
class AgentContract:
    """Full contract of a discovered agent class."""

    name: str
    module_path: str
    class_type: str
    base_classes: list[str]
    playbook: PlaybookResolution
    has_custom_execute: bool
    has_route_method: bool
    calls_super_with_playbook: bool
    agent_name_param: str | None
    constructor_params: list[ParamSpec]
    test_file: str | None
    status: str


@dataclass(frozen=True)
class UtilContract:
    """Public surface of a utility module."""

    name: str
    module_path: str
    exports: list[str]


@dataclass(frozen=True)
class PipelineContract:
    """Ordered steps and agent references of a pipeline."""

    name: str
    steps: list[str]
    agent_refs: list[str]


@dataclass(frozen=True)
class ScanError:
    """A recoverable problem encountered during scanning."""

    file: str
    message: str
    severity: str


@dataclass(frozen=True)
class SystemRegistry:
    """Complete, serializable inventory of the workspace."""

    agents: dict[str, AgentContract]
    utils: list[UtilContract]
    pipelines: list[PipelineContract]
    missing_pipeline_agents: list[str]
    scan_errors: list[ScanError]
    scan_timestamp: str
    scan_duration_ms: int


def _snake(name: str) -> str:
    step = _SNAKE_1.sub(r"\1_\2", name)
    return _SNAKE_2.sub(r"\1_\2", step).lower()


def _base_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return "Unknown"


def _imports_base_agent(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and "base_agent" in node.module:
                if any(alias.name == "BaseAgent" for alias in node.names):
                    return True
        elif isinstance(node, ast.Import):
            if any("base_agent" in alias.name for alias in node.names):
                return True
    return False


def _constructor_params(init: ast.FunctionDef) -> list[ParamSpec]:
    specs: list[ParamSpec] = []
    args = init.args
    positional = [a for a in args.args if a.arg != "self"]
    default_offset = len(positional) - len(args.defaults)

    for index, arg in enumerate(positional):
        has_default = index >= default_offset
        value: str | None = None
        if has_default:
            node = args.defaults[index - default_offset]
            if isinstance(node, ast.Constant):
                value = str(node.value)
            else:
                try:
                    value = ast.unparse(node)
                except (ValueError, TypeError):
                    value = "<complex>"
        specs.append(ParamSpec(arg.arg, has_default, value))

    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        value = None
        if default is not None:
            if isinstance(default, ast.Constant):
                value = str(default.value)
            else:
                try:
                    value = ast.unparse(default)
                except (ValueError, TypeError):
                    value = "<complex>"
        specs.append(ParamSpec(arg.arg, default is not None, value))

    if args.vararg is not None:
        specs.append(ParamSpec(f"*{args.vararg.arg}", False, None))
    if args.kwarg is not None:
        specs.append(ParamSpec(f"**{args.kwarg.arg}", False, None))

    return specs


def _super_contract(init: ast.FunctionDef) -> tuple[bool, str | None]:
    forwarded = False
    agent_name: str | None = None

    for stmt in init.body:
        if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
            continue
        call = stmt.value
        if not isinstance(call.func, ast.Attribute) or call.func.attr != "__init__":
            continue
        if not isinstance(call.func.value, ast.Call):
            continue
        inner = call.func.value.func
        if not isinstance(inner, ast.Name) or inner.id != "super":
            continue
        for keyword in call.keywords:
            if keyword.arg == "playbook_path":
                forwarded = True
            if keyword.arg == "agent_name" and isinstance(keyword.value, ast.Constant):
                agent_name = str(keyword.value.value)

    return forwarded, agent_name


def _status(
    is_subclass: bool,
    imports_base: bool,
    has_custom_execute: bool,
    playbook: PlaybookResolution,
    test_file: str | None,
) -> str:
    if is_subclass and not imports_base:
        return "missing_base"
    if not has_custom_execute and playbook.path is None:
        return "missing_playbook"
    if playbook.path is not None and not playbook.exists_on_disk:
        return "missing_playbook"
    if test_file is None:
        return "no_tests"
    return "ready"


def _hydrate(payload: dict[str, Any]) -> SystemRegistry:
    agents: dict[str, AgentContract] = {}
    for name, raw in payload["agents"].items():
        record = dict(raw)
        record["playbook"] = PlaybookResolution(**record["playbook"])
        record["constructor_params"] = [
            ParamSpec(**param) for param in record["constructor_params"]
        ]
        agents[name] = AgentContract(**record)

    return SystemRegistry(
        agents=dict(sorted(agents.items())),
        utils=[UtilContract(**item) for item in payload["utils"]],
        pipelines=[PipelineContract(**item) for item in payload["pipelines"]],
        missing_pipeline_agents=list(payload["missing_pipeline_agents"]),
        scan_errors=[ScanError(**item) for item in payload["scan_errors"]],
        scan_timestamp=payload["scan_timestamp"],
        scan_duration_ms=payload["scan_duration_ms"],
    )


class DiscoveryEngine:
    """Build a typed workspace registry using AST parsing only.

    Args:
        workspace_path: Root of jarvis-workspace.
        force: Bypass the cache when True.

    Raises:
        DiscoveryError: If workspace_path is not a path-like value.
    """

    def __init__(self, workspace_path: str | Path, force: bool = False) -> None:
        if not isinstance(workspace_path, (str, Path)):
            raise DiscoveryError("workspace_path must be a str or Path")
        self.workspace = Path(workspace_path).resolve()
        self.force = force
        self.cache_path = self.workspace / "brain" / "skills" / "codebase_registry.json"

    def scan(self) -> SystemRegistry:
        if not self.force:
            cached = self._read_cache()
            if cached is not None:
                logger.info("Discovery cache hit.")
                return cached

        started = time.perf_counter()
        errors: list[ScanError] = []
        fingerprint = self._fingerprint()

        agents = self._scan_agents(errors)
        utils = self._scan_utils(errors)
        pipelines, missing = self._scan_pipelines(agents, errors)

        registry = SystemRegistry(
            agents=agents,
            utils=utils,
            pipelines=pipelines,
            missing_pipeline_agents=missing,
            scan_errors=errors,
            scan_timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            scan_duration_ms=int((time.perf_counter() - started) * 1000),
        )
        self._write_cache(registry, fingerprint)
        logger.info(
            "Discovery complete: agents=%d utils=%d pipelines=%d errors=%d",
            len(agents), len(utils), len(pipelines), len(errors),
        )
        return registry

    def report(self, registry: SystemRegistry) -> str:
        lines = [f"[DISCOVERY] Scanning {self.workspace.name}/ ..."]
        for name, contract in sorted(registry.agents.items()):
            if contract.has_custom_execute:
                kind = "custom execute()"
            elif contract.has_route_method:
                kind = "route()"
            else:
                kind = "playbook-driven"
            playbook = f"playbook:{contract.playbook.source or 'none'}"
            marker = "OK" if contract.status == "ready" else "WARN"
            lines.append(
                f"[DISCOVERY]   {name:<14} {contract.class_type:<20} "
                f"{kind:<17} {playbook:<22} {contract.status:<17} {marker}"
            )

        lines.append(
            f"[DISCOVERY] {len(registry.agents)} agents | "
            f"{len(registry.utils)} utils | {len(registry.pipelines)} pipelines"
        )
        missing = ", ".join(registry.missing_pipeline_agents) or "none"
        lines.append(f"[DISCOVERY] missing_pipeline_agents: {missing}")

        failures = sum(e.severity == "error" for e in registry.scan_errors)
        lines.append(f"[DISCOVERY] scan_errors: {failures}")
        for error in registry.scan_errors:
            lines.append(f"[DISCOVERY]   {error.severity}: {error.file} ({error.message})")

        lines.append(
            f"[DISCOVERY] completed in {registry.scan_duration_ms}ms "
            f"-> brain/skills/codebase_registry.json"
        )
        return "\n".join(lines)

    def _safe_mtime(self, path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    def _fingerprint(self) -> str:
        entries: list[str] = []
        watched = [
            ("src/agents", "*.py"),
            ("src/agents/playbooks", "*.md"),
            ("src/utils", "*.py"),
            ("tests", "*.py"),
        ]
        for relative, pattern in watched:
            base = self.workspace / relative
            if not base.is_dir():
                continue
            entries.append(f"{relative}/:{self._safe_mtime(base):.6f}")
            for path in sorted(base.rglob(pattern)):
                if path.is_file():
                    key = path.relative_to(self.workspace).as_posix()
                    entries.append(f"{key}:{self._safe_mtime(path):.6f}")

        config = self.workspace / "config" / "pipelines.json"
        if config.is_file():
            entries.append(f"config/pipelines.json:{self._safe_mtime(config):.6f}")

        return hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()

    def _read_cache(self) -> SystemRegistry | None:
        if not self.cache_path.is_file():
            return None
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if payload.get("version") != _CACHE_VERSION:
                return None
            if payload.get("fingerprint") != self._fingerprint():
                return None
            return _hydrate(payload["registry"])
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            logger.warning("Discovery cache rejected: %s", type(exc).__name__)
            return None

    def _write_cache(self, registry: SystemRegistry, fingerprint: str) -> None:
        payload = {
            "version": _CACHE_VERSION,
            "fingerprint": fingerprint,
            "registry": asdict(registry),
        }
        temporary: str | None = None
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(
                dir=str(self.cache_path.parent), suffix=".tmp"
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.cache_path)
            temporary = None
        except OSError as exc:
            logger.error("Cache write failed: %s", type(exc).__name__)
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass

    def _is_agent_class(self, node: ast.ClassDef, tree: ast.Module) -> bool:
        """Determine if a class is an actual agent (not a dataclass/exception/base).
        
        An agent is:
        - A BaseAgent subclass, OR
        - Has execute() or route() method, OR
        - Is explicitly named "Router"
        
        Excludes:
        - Classes decorated with @dataclass
        - Exception classes (inherit from Exception/BaseException)
        - The BaseAgent class itself
        """
        # Exclude BaseAgent itself
        if node.name == "BaseAgent":
            return False
        
        # Exclude @dataclass decorated classes
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == "dataclass":
                return False
            if isinstance(decorator, ast.Attribute) and decorator.attr == "dataclass":
                return False
            if isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name) and decorator.func.id == "dataclass":
                    return False
        
        # Exclude exception classes
        base_names = [_base_name(base) for base in node.bases]
        exception_bases = {"Exception", "BaseException", "Error", "Warning"}
        if any(base in exception_bases for base in base_names):
            return False
        
        # Include if it's a BaseAgent subclass
        if "BaseAgent" in base_names:
            return True
        
        # Include if it has execute() or route() method
        has_execute = any(
            isinstance(n, ast.FunctionDef) and n.name == "execute" for n in node.body
        )
        has_route = any(
            isinstance(n, ast.FunctionDef) and n.name == "route" for n in node.body
        )
        if has_execute or has_route:
            return True
        
        # Special case: Router
        if node.name == "Router":
            return True
        
        return False

    def _scan_agents(self, errors: list[ScanError]) -> dict[str, AgentContract]:
        """Discover every agent class under src/agents/."""
        root = self.workspace / "src" / "agents"
        if not root.is_dir():
            errors.append(ScanError("src/agents", "DirectoryMissing", "warning"))
            return {}

        contracts: dict[str, AgentContract] = {}
        for path in sorted(root.rglob("*.py")):
            if path.name == "__init__.py":
                continue
            relative = path.relative_to(self.workspace).as_posix()
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
            except SyntaxError as exc:
                errors.append(ScanError(relative, f"SyntaxError@{exc.lineno}", "error"))
                logger.error("Unparseable agent module: %s", relative)
                continue
            except (OSError, UnicodeError) as exc:
                errors.append(ScanError(relative, type(exc).__name__, "error"))
                continue

            imports_base = _imports_base_agent(tree)
            classes = sorted(
                (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)),
                key=lambda n: n.name,
            )
            for node in classes:
                # Filter: only include actual agents
                if not self._is_agent_class(node, tree):
                    continue
                
                contract = self._contract_for(node, relative, imports_base)
                contracts[contract.name] = contract

        return dict(sorted(contracts.items()))

    def _contract_for(
        self, node: ast.ClassDef, relative: str, imports_base: bool
    ) -> AgentContract:
        bases = [_base_name(base) for base in node.bases]
        is_subclass = "BaseAgent" in bases
        init = next(
            (
                n for n in node.body
                if isinstance(n, ast.FunctionDef) and n.name == "__init__"
            ),
            None,
        )
        params = _constructor_params(init) if init else []
        has_execute = any(
            isinstance(n, ast.FunctionDef) and n.name == "execute" for n in node.body
        )
        has_route = any(
            isinstance(n, ast.FunctionDef) and n.name == "route" for n in node.body
        )
        forwarded, agent_name = (
            _super_contract(init) if init is not None and is_subclass else (False, None)
        )
        playbook = self._resolve_playbook(node.name, params)
        test_file = self._find_test(node.name)

        return AgentContract(
            name=node.name,
            module_path=relative,
            class_type="base_agent_subclass" if is_subclass else "standalone",
            base_classes=bases,
            playbook=playbook,
            has_custom_execute=has_execute,
            has_route_method=has_route,
            calls_super_with_playbook=forwarded,
            agent_name_param=agent_name,
            constructor_params=params,
            test_file=test_file,
            status=_status(is_subclass, imports_base, has_execute, playbook, test_file),
        )

    def _resolve_playbook(
        self, class_name: str, params: list[ParamSpec]
    ) -> PlaybookResolution:
        spec = next((p for p in params if p.name == "playbook_path"), None)
        path: str | None = None
        source: str | None = None

        if spec is not None and spec.has_default and spec.default_value not in (None, "None"):
            path, source = spec.default_value, "default"

        if path is None:
            conventional = f"src/agents/playbooks/{_snake(class_name)}.md"
            if (self.workspace / conventional).is_file():
                path, source = conventional, "conventional"

        is_required = spec is not None and not spec.has_default
        exists = False

        if path is not None:
            try:
                resolved = (self.workspace / path).resolve()
                if not resolved.is_relative_to(self.workspace):
                    logger.warning("Playbook path outside workspace rejected.")
                    path, source = None, None
                else:
                    exists = resolved.is_file() and bool(
                        resolved.read_text(encoding="utf-8").strip()
                    )
            except (OSError, UnicodeError, RuntimeError):
                exists = False

        return PlaybookResolution(path, source, is_required, exists)

    def _find_test(self, class_name: str) -> str | None:
        root = self.workspace / "tests"
        if not root.is_dir():
            return None
        snake = _snake(class_name)
        candidates = [
            f"test_{snake}.py",
            f"test_{snake.replace('_', '')}.py",
            f"{snake}_test.py",
        ]
        for candidate in candidates:
            matches = sorted(root.rglob(candidate))
            if matches:
                return matches[0].relative_to(self.workspace).as_posix()
        return None

    def _scan_utils(self, errors: list[ScanError]) -> list[UtilContract]:
        root = self.workspace / "src" / "utils"
        if not root.is_dir():
            errors.append(ScanError("src/utils", "DirectoryMissing", "warning"))
            return []

        utils: list[UtilContract] = []
        for path in sorted(root.glob("*.py")):
            if path.name == "__init__.py":
                continue
            relative = path.relative_to(self.workspace).as_posix()
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
            except (SyntaxError, OSError, UnicodeError) as exc:
                errors.append(ScanError(relative, type(exc).__name__, "error"))
                continue

            exports: list[str] = []
            declared = False
            for stmt in tree.body:
                targets: list[ast.expr] = []
                if isinstance(stmt, ast.Assign):
                    targets = list(stmt.targets)
                elif isinstance(stmt, ast.AnnAssign):
                    targets = [stmt.target]
                if any(
                    isinstance(t, ast.Name) and t.id == "__all__" for t in targets
                ):
                    value = getattr(stmt, "value", None)
                    if isinstance(value, (ast.List, ast.Tuple)):
                        exports = [
                            element.value
                            for element in value.elts
                            if isinstance(element, ast.Constant)
                            and isinstance(element.value, str)
                        ]
                    declared = True
                    break

            if not declared:
                exports = sorted(
                    n.name for n in tree.body
                    if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                    and not n.name.startswith("_")
                )

            utils.append(UtilContract(path.stem, relative, exports))
        return utils

    def _scan_pipelines(
        self, agents: dict[str, AgentContract], errors: list[ScanError]
    ) -> tuple[list[PipelineContract], list[str]]:
        config = self.workspace / "config" / "pipelines.json"
        if not config.is_file():
            return [], []

        try:
            data = json.loads(config.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            errors.append(ScanError("config/pipelines.json", type(exc).__name__, "error"))
            return [], []

        definitions = data.get("pipelines", {}) if isinstance(data, dict) else {}
        if not isinstance(definitions, dict):
            return [], []

        pipelines: list[PipelineContract] = []
        missing: set[str] = set()

        for name in sorted(definitions):
            definition = definitions[name]
            if not isinstance(definition, dict):
                continue
            raw_steps = definition.get("steps", [])
            if not isinstance(raw_steps, list):
                continue
            steps = [
                str(step["name"])
                for step in raw_steps
                if isinstance(step, dict) and isinstance(step.get("name"), str)
            ]
            refs = [
                str(step["agent"])
                for step in raw_steps
                if isinstance(step, dict) and isinstance(step.get("agent"), str)
            ]
            missing.update(ref for ref in refs if ref and ref not in agents)
            pipelines.append(PipelineContract(name, steps, refs))

        return pipelines, sorted(missing)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    _engine = DiscoveryEngine(Path(__file__).resolve().parents[2])
    print(_engine.report(_engine.scan()))