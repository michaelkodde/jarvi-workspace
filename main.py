# main.py
"""
Entry point for running the Jarvi Chat Server with real dependencies.
"""
import asyncio
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from src.companion.chat_server import ChatServer

# ============================================================================
# 1. CONFIGURATION & SECURITY
# ============================================================================
AUTH_TOKEN = os.getenv("JARVI_AUTH_TOKEN", None)

CONFIG = {
    "router_model": os.getenv("JARVI_ROUTER_MODEL", "qwen2.5:3b"),
    "agent_model": os.getenv("JARVI_AGENT_MODEL", "qwen2.5:7b"),
    "playbook_path": "src/agents/playbooks/router.md", 
    "pipeline_config": "config/pipelines.json",                  
    "memory_scope": "business",
}

# ============================================================================
# 2. IMPORTS & LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from src.companion.memory import MemoryEngine
from src.companion.query_api import QueryAPI
from src.companion.context_injector import ContextInjector
from src.daemon.daemon_loop import DaemonLoop
from src.services.pipeline_flow import PipelineFlow
from src.agents.router import Router
from src.agents.analyst import Analyst
from src.agents.creator import Creator
from src.agents.publisher import Publisher
from src.agents.scout import Scout
from src.agents.trend_spotter import TrendSpotter
from src.agents.coder import Coder

from src.utils.llm_client import query_ollama, OllamaResponse
from src.utils.token_ledger import TokenLedger

# --- Adapter Class ---
class OllamaClient:
    """Wrapper to bind the model name to the functional query_ollama."""
    def __init__(self, model_name: str):
        self.model_name = model_name
        
    def __call__(self, prompt: str, system_prompt: str = None, **kwargs) -> OllamaResponse:
        kwargs.pop('model', None)
        return query_ollama(model=self.model_name, prompt=prompt, system_prompt=system_prompt, **kwargs)

WORKSPACE = Path(__file__).parent.resolve()

# ============================================================================
# 2.5. BACKGROUND TASKS (LIFESPAN)
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage background tasks for the Jarvi application.
    Wakes up the Daemon Loop (Jarvi's brain) and serves the Chat UI.
    """
    logger.info("🔄 Starting Jarvi (Daemon + Chat Server)...")
    
    # Initialize Context Injector
    context_injector = ContextInjector(
        memory_engine=app.state.memory_engine,
        max_context_memories=3,
        max_context_chars=2000,
    )
    logger.info("✅ ContextInjector initialized.")
    
    # Initialize the Daemon Loop
    daemon = DaemonLoop(
        workspace_path=WORKSPACE,
        pipeline_flow=app.state.pipeline_flow,
        poll_interval_sec=2.0,
        queue_depth_threshold=50,
        enabled=True,
        context_injector=context_injector,
    )
    
    # Run the blocking Daemon Loop in a background thread
    daemon_thread = threading.Thread(target=daemon.start, daemon=True, name="Jarvi-Daemon")
    daemon_thread.start()
    
    # Save reference for graceful shutdown
    app.state.daemon = daemon
    logger.info("🌟 Jarvi is awake and running in the background.")
    
    yield  # <-- FastAPI runs here, serving the chat UI
    
    # Graceful Shutdown
    logger.info("🛑 Shutting down Jarvi...")
    daemon.stop(timeout_sec=5.0)
    daemon_thread.join(timeout=5.0)
    logger.info("✅ Jarvi has gone to sleep.")


# ============================================================================
# 3. APP FACTORY
# ============================================================================
def create_app() -> FastAPI:
    """Factory function to create and configure the ChatServer FastAPI app."""
    logger.info("🚀 Initializing Jarvi Real Components...")

    try:
        # 1. Memory Engine (Persistence & Context)
        memory_engine = MemoryEngine(
            workspace_path=WORKSPACE, 
            scope=CONFIG["memory_scope"]
        )
        logger.info(f"✅ MemoryEngine initialized (scope: {CONFIG['memory_scope']}).")

        # 2. Router (Intent Classification)
        playbook_path = WORKSPACE / CONFIG["playbook_path"]
        pipeline_config_path = WORKSPACE / CONFIG["pipeline_config"]

        if not playbook_path.exists():
            raise FileNotFoundError(f"Router playbook not found at {playbook_path}")
        
        router = Router(
            ollama_model=CONFIG["router_model"],
            playbook_path=str(playbook_path),
            pipeline_config_path=str(pipeline_config_path),
        )
        logger.info(f"✅ Router initialized (Model: {CONFIG['router_model']}).")

        # 3. Pipeline Flow (Agent Execution)
        config_path = WORKSPACE / CONFIG["pipeline_config"]
        if not config_path.exists():
            raise FileNotFoundError(f"Pipeline config not found at {config_path}")
            
        # Initialize shared dependencies
        token_ledger = TokenLedger()
        
        # Create ONE client instance bound to the agent model
        agent_client = OllamaClient(CONFIG["agent_model"])
        
        playbook_dir = WORKSPACE / "src" / "agents" / "playbooks"
        
        # Wire the bound client and ledger into each agent
        agents = {
            "scout": Scout(llm_client=agent_client, token_ledger=token_ledger, model_name=CONFIG["agent_model"], playbook_path=str(playbook_dir / "scout.md")),
            "analyst": Analyst(llm_client=agent_client, token_ledger=token_ledger, model_name=CONFIG["agent_model"], playbook_path=str(playbook_dir / "analyst.md")),
            "creator": Creator(llm_client=agent_client, token_ledger=token_ledger, model_name=CONFIG["agent_model"], playbook_path=str(playbook_dir / "creator.md")),
            "publisher": Publisher(llm_client=agent_client, token_ledger=token_ledger, model_name=CONFIG["agent_model"], playbook_path=str(playbook_dir / "publisher.md")),
            "trend_spotter": TrendSpotter(llm_client=agent_client, token_ledger=token_ledger, model_name=CONFIG["agent_model"], playbook_path=str(playbook_dir / "trend_spotter.md")),
            "coder": Coder(
                llm_client=agent_client,
                token_ledger=token_ledger,
                model_name=CONFIG["agent_model"],
                playbook_path=str(playbook_dir / "coder.md")
            ),
        }
        logger.info(f"✅ Initialized {len(agents)} agents (Model: {CONFIG['agent_model']}).")

        # Wire agents and config into PipelineFlow
        pipeline_flow = PipelineFlow(
            agents=agents,
            pipeline_definitions_path=str(config_path)
        )

        # NOTE: We do NOT call pipeline_flow.start_worker() here.
        # The DaemonLoop is now the sole executor to prevent race conditions.
        logger.info("✅ PipelineFlow initialized (DaemonLoop will drive execution).")

        # 4. Query API (HITL & Status)
        query_api = QueryAPI(pipeline_flow=pipeline_flow)
        logger.info("✅ QueryAPI initialized.")

        # --- Wire into Chat Server ---
        chat_server = ChatServer(
            workspace_path=WORKSPACE,
            pipeline_flow=pipeline_flow,
            query_api=query_api,
            router=router,
            memory_engine=memory_engine,
            auth_token=AUTH_TOKEN,
            lifespan=lifespan,
        )
        chat_server.app.state.pipeline_flow = pipeline_flow
        chat_server.app.state.memory_engine = memory_engine  # Needed for ContextInjector in lifespan

        if AUTH_TOKEN:
            logger.info("🔒 Auth token detected. WebSocket connections will require authentication.")
        else:
            logger.warning("⚠️ No JARVI_AUTH_TOKEN set. Server is open to local network only.")

        logger.info("🎉 Jarvi Chat Server fully wired and ready.")
        return chat_server.app

    except Exception as e:
        logger.critical(f"Failed to initialize Jarvi: {e}")
        raise


# Expose the FastAPI app for uvicorn at the module level
app = create_app()


if __name__ == "__main__":
    import uvicorn
    logger.info("🌐 Starting Uvicorn on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)