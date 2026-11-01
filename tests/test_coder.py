"""
Isolated test script for the Coder Agent.
Uses absolute paths to eliminate any working-directory ambiguity.
"""
import logging
import traceback
from pathlib import Path
from typing import Any

from src.agents.coder import Coder
from src.utils.token_ledger import TokenLedger
from src.utils.llm_client import query_ollama, OllamaResponse

# Configure logging to match Jarvi standards
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Dynamically find the workspace root (the directory containing this script)
WORKSPACE = Path(__file__).parent.resolve()


class TestOllamaClient:
    """Test wrapper to bind the model name to the functional query_ollama."""
    def __init__(self, model_name: str):
        self.model_name = model_name
        
    def __call__(self, prompt: str, system_prompt: str | None = None, **kwargs: Any) -> OllamaResponse:
        kwargs.pop('model', None)
        return query_ollama(
            model=self.model_name, 
            prompt=prompt, 
            system_prompt=system_prompt, 
            **kwargs
        )


def main() -> None:
    logger.info(f"🚀 Starting isolated Coder Agent test from: {WORKSPACE}")
    
    try:
        # 1. Setup dependencies using ABSOLUTE paths
        ledger = TokenLedger()
        client = TestOllamaClient("qwen2.5:7b")
        
        playbook_path = WORKSPACE / "src" / "agents" / "playbooks" / "coder.md"
        logger.info(f"🔍 Looking for playbook at: {playbook_path}")
        
        # Defensive check: Ensure playbook exists before initialization
        if not playbook_path.exists():
            raise FileNotFoundError(f"Playbook not found at {playbook_path}. Please verify the file exists and is named exactly 'coder.md' (lowercase).")
        
        # 2. Initialize Agent
        logger.info("📦 Initializing Coder Agent...")
        coder = Coder(
            llm_client=client,
            token_ledger=ledger,
            model_name="qwen2.5:7b",
            playbook_path=str(playbook_path)
        )
        logger.info("✅ Agent initialized successfully.")
        
        # 3. Execute using absolute paths for target files
        target_1 = str(WORKSPACE / "src" / "agents" / "base_agent.py")
        target_2 = str(WORKSPACE / "src" / "agents" / "analyst.py")
        
        logger.info(f"⚡ Executing agent on: {target_1}, {target_2}")
        result = coder.execute(
            task_prompt="Check for exception chaining and BaseAgent contract compliance.",
            context=f"{target_1},{target_2}"
        )
        
        logger.info("🎉 SUCCESS! Agent returned artifact:")
        logger.info("-" * 80)
        logger.info(result.artifact[:1000])  # Print first 1000 chars
        logger.info("-" * 80)
        
    except Exception as e:
        logger.error("💥 FAILED with traceback:")
        traceback.print_exc()


if __name__ == "__main__":
    main()