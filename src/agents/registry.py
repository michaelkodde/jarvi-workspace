# src/agents/registry.py
"""
Agent Registry.
Single source of truth for agent classes. 
Add new agents here; main.py will automatically instantiate them.
"""
from typing import Dict, Type

from .analyst import Analyst
from .coder import Coder
from .creator import Creator
from .publisher import Publisher
from .scout import Scout
from .trend_spotter import TrendSpotter

AGENT_REGISTRY: Dict[str, Type] = {
    "scout": Scout,
    "analyst": Analyst,
    "creator": Creator,
    "publisher": Publisher,
    "trend_spotter": TrendSpotter,
    "coder": Coder,
}