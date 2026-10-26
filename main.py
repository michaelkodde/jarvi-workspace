# main.py
"""
Entry point for running the Jarvi Chat Server with real dependencies.
"""
import asyncio
import logging
import os
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
from src.services.pipeline_flow import PipelineFlow
from src.agents.router import Router
from src.agents.analyst import Analyst
from src.agents.creator import Creator
from src.agents.publisher import Publisher
from src.agents.scout import Scout
from src.agents.trend_spotter import TrendSpotter
from src.agents.coder import Coder  # <-- NEW: Import Coder Agent

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
REVIEW_INTERVAL_SECONDS = 86400  

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage background tasks for the Jarvi application.
    Ensures graceful startup and shutdown of asynchronous maintenance loops.
    """
    logger.info("🔄 Starting background maintenance tasks...")
    
    async def periodic_code_review():
        """Background loop that periodically submits code review jobs."""
        logger.info(f"🕒 Background review task STARTED. Interval: {REVIEW_INTERVAL_SECONDS}s")
        try:
            while True:
                logger.info(f"💤 Sleeping for {REVIEW_INTERVAL_SECONDS} seconds...")
                await asyncio.sleep(REVIEW_INTERVAL_SECONDS)
                
                logger.info("⏰ Triggering periodic code review...")
                try:
                    # Access pipeline_flow safely from app state (no globals)
                    pf = getattr(app.state, "pipeline_flow", None)
                    
                    if pf is not None:
                        logger.info("📝 Submitting job to PipelineFlow...")
                        pf.submit_job(
                            job_type="code_review_pipeline",
                            parameters={
                                "target_files": "src/agents/analyst.py,src/agents/base_agent.py,src/companion/chat_server.py",
                                "focus": "Check for exception chaining, token ledger usage, and BaseAgent contract compliance."
                            },
                            priority=1,
                            description="Automated daily codebase health check"
                        )
                        logger.info("✅ Code review job submitted to queue.")
                    else:
                        logger.error("❌ pipeline_flow not found in app.state. Cannot submit job.")
                        
                except Exception as e:
                    # Log full traceback for debugging, but do not crash the background loop
                    logger.error(f"❌ Failed to submit periodic review job: {type(e).__name__}: {str(e)}", exc_info=True)
                    
        except asyncio.CancelledError:
            logger.info("🛑 Background review task gracefully cancelled.")
        except Exception as e:
            # Catch-all for unexpected loop crashes
            logger.error(f"💥 Background review task crashed: {type(e).__name__}: {str(e)}", exc_info=True)

    # Start the background task
    logger.info("🚀 Creating asyncio task for periodic review...")
    review_task = asyncio.create_task(periodic_code_review())
    
    yield  # Application runs here
    
    # Shutdown: Gracefully cancel and await the task
    logger.info("🛑 Cancelling periodic review task...")
    review_task.cancel()
    try:
        await review_task
    except asyncio.CancelledError:
        pass  # Expected behavior during shutdown
    logger.info("🛑 Stopped periodic code review task.")


# ============================================================================
# 3. APP FACTORY
# ============================================================================
def create_app() -> FastAPI:
    """Factory function to create and configure the ChatServer FastAPI app."""
    global _pipeline_flow_ref
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
            # --- NEW: Coder Agent ---
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

        pipeline_flow.start_worker()

        # Assign to global reference so the lifespan background task can access it
        _pipeline_flow_ref = pipeline_flow
        logger.info("✅ PipelineFlow initialized.")

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