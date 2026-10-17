# src/companion/__init__.py
"""Jarvi Companion: the brain that discovers and operates the toolchain."""
from src.companion.discovery import (
    AgentContract,
    DiscoveryEngine,
    DiscoveryError,
    ParamSpec,
    PipelineContract,
    PlaybookResolution,
    ScanError,
    SystemRegistry,
    UtilContract,
)

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