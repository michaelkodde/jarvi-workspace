# main.py
"""
Gold Standard Entry Point.
Uses Dependency Injection and Factory patterns to decouple component initialization.
Strictly manages the Daemon lifecycle to prevent double-execution.
"""
import logging
import os
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
@dataclass(frozen=True)
class AppConfig:
    auth_token: str = os.getenv("JARVI_AUTH_TOKEN", "")
    router_model: str = os.getenv("JARVI_ROUTER_MODEL", "qwen2.5:3b")
    agent_model: str = os.getenv("JARVI_AGENT_MODEL", "qwen2.5:7b")
    playbook_path: str = "src/agents/playbooks/router.md"
    pipeline_config: str = "config/pipelines.json"
    memory_scope: str = "business"
    workspace: Path = Path(__file__).parent.resolve()

CONFIG = AppConfig()

# ============================================================================
# 2. DEPENDENCY INJECTION CONTAINER
# ============================================================================
@dataclass
class JarviContainer:
    """Holds all initialized core components."""
    memory_engine: Any
    context_injector: Any
    router: Any
    pipeline_flow: Any
    query_api: Any
    ollama_client: Any
    self_corrector: Any

def _build_agents(config: AppConfig, ollama_client: Any, token_ledger: Any) -> Dict[str, Any]:
    """
    Factory pattern for agents. 
    Uses AgentClientAdapter to bridge the new Gold Standard client with legacy BaseAgent.
    """
    from src.agents.registry import AGENT_REGISTRY
    from src.utils.llm_client import AgentClientAdapter

    playbook_dir = config.workspace / "src" / "agents" / "playbooks"
    agents = {}
    
    for name, agent_class in AGENT_REGISTRY.items():
        # Bind the model name to the client for this specific agent
        bound_client = AgentClientAdapter(ollama_client, config.agent_model)
        
        agents[name] = agent_class(
            llm_client=bound_client,
            token_ledger=token_ledger,
            model_name=config.agent_model,
            playbook_path=str(playbook_dir / f"{name}.md")
        )
    return agents

def initialize_brain(config: AppConfig) -> JarviContainer:
    """Pure function to initialize the core 'Brain' components."""
    from src.companion.context_injector import ContextInjector
    from src.companion.memory import MemoryEngine
    from src.companion.query_api import QueryAPI
    from src.services.pipeline_flow import PipelineFlow
    from src.services.self_corrector import SelfCorrector
    from src.agents.router import Router
    from src.utils.llm_client import get_sync_ollama_client
    from src.utils.token_ledger import TokenLedger

    logger.info("🧠 Initializing Jarvi Brain Components...")

    memory_engine = MemoryEngine(workspace_path=config.workspace, scope=config.memory_scope)
    ollama_client = get_sync_ollama_client()
    token_ledger = TokenLedger()
    
    playbook_path = config.workspace / config.playbook_path
    pipeline_config_path = config.workspace / config.pipeline_config
    
    if not playbook_path.exists() or not pipeline_config_path.exists():
        raise FileNotFoundError("Critical playbook or pipeline config missing.")

    router = Router(
        ollama_model=config.router_model,
        playbook_path=str(playbook_path),
        pipeline_config_path=str(pipeline_config_path),
    )

    agents = _build_agents(config, ollama_client, token_ledger)

    # Initialize Self Corrector (Module 3)
    self_corrector = SelfCorrector(
        llm_client=ollama_client,
        model_name=config.agent_model,
    )

    # GOLD STANDARD: Pass self_corrector directly to PipelineFlow
    # PipelineFlow owns execution and error recovery.
    pipeline_flow = PipelineFlow(
        agents=agents,
        pipeline_definitions_path=str(pipeline_config_path),
        self_corrector=self_corrector,
    )

    query_api = QueryAPI(pipeline_flow=pipeline_flow)
    
    context_injector = ContextInjector(
        memory_engine=memory_engine,
        max_context_memories=3,
        max_context_chars=2000,
    )

    logger.info("✅ Brain initialization complete.")
    return JarviContainer(
        memory_engine=memory_engine,
        context_injector=context_injector,
        router=router,
        pipeline_flow=pipeline_flow,
        query_api=query_api,
        ollama_client=ollama_client,
        self_corrector=self_corrector,
    )

# ============================================================================
# 3. LIFESPAN & APP FACTORY
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages the lifecycle of the background Daemon."""
    logger.info("🔄 Starting Jarvi Background Services...")
    container: JarviContainer = app.state.container
    
    from src.daemon.daemon_loop import DaemonLoop
    
    # DaemonLoop receives context_injector, but NOT self_corrector.
    # Self-correction is handled internally by PipelineFlow during execution.
    daemon = DaemonLoop(
        workspace_path=CONFIG.workspace,
        pipeline_flow=container.pipeline_flow,
        poll_interval_sec=2.0,
        queue_depth_threshold=50,
        enabled=True,
        context_injector=container.context_injector,
    )
    
    daemon_thread = threading.Thread(target=daemon.start, daemon=True, name="Jarvi-Daemon")
    daemon_thread.start()
    app.state.daemon = daemon
    logger.info("🌟 Jarvi is awake and running in the background.")
    
    yield
    
    logger.info("🛑 Shutting down Jarvi...")
    daemon.stop(timeout_sec=5.0)
    daemon_thread.join(timeout=5.0)
    logger.info("✅ Jarvi has gone to sleep.")

def create_app() -> FastAPI:
    """Factory function. Wires the Container into the FastAPI app."""
    from src.companion.chat_server import ChatServer
    
    try:
        container = initialize_brain(CONFIG)
        
        # Initialize the chat server WITHOUT passing lifespan to it.
        # We manage the lifecycle entirely in the main app factory to prevent double-execution.
        chat_server = ChatServer(
            workspace_path=CONFIG.workspace,
            pipeline_flow=container.pipeline_flow,
            query_api=container.query_api,
            router=container.router,
            memory_engine=container.memory_engine,
            auth_token=CONFIG.auth_token,
            lifespan=None, 
        )
        
        # Create our main app with the single source of truth for lifespan
        app = FastAPI(lifespan=lifespan)
        app.state.container = container
        app.state.pipeline_flow = container.pipeline_flow
        app.state.memory_engine = container.memory_engine
        
        # Merge routes from chat_server into our main app
        app.include_router(chat_server.app.router)
        
        return app
    except Exception as e:
        logger.critical(f"Failed to initialize Jarvi: {e}")
        raise

app = create_app()

if __name__ == "__main__":
    import uvicorn
    logger.info("🌐 Starting Uvicorn on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)