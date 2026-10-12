import requests
import logging
import json
from pathlib import Path
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging for this module
logger = logging.getLogger(__name__)

def _get_ollama_url() -> str:
    """Dynamically load Ollama URL from config.json, with a safe fallback."""
    config_path = Path(__file__).parent.parent.parent / "config.json"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                # Fallback to default if key is missing
                return config.get("ollama_url", "http://localhost:11434")
        except Exception as e:
            logger.warning(f"Failed to read config.json, using default Ollama URL: {e}")
    return "http://localhost:11434"

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def query_ollama(
    model: str, 
    prompt: str, 
    images: Optional[list[str]] = None, 
    stream: bool = False
) -> str:
    """
    Gold Standard LLM Client: Queries local Ollama via /api/chat with exponential backoff.
    
    Args:
        model (str): The Ollama model to use (e.g., 'qwen2.5:7b', 'minicpm-v').
        prompt (str): The user prompt.
        images (Optional[list[str]]): List of base64 encoded image strings for vision models.
        stream (bool): Whether to stream the response. Default False.
        
    Returns:
        str: The generated text content.
    """
    base_url = _get_ollama_url()
    url = f"{base_url}/api/chat" # Always use /api/chat for modern compatibility
    
    message_payload = {"role": "user", "content": prompt}
    if images:
        message_payload["images"] = images

    payload = {
        "model": model,
        "messages": [message_payload],
        "stream": stream
    }
    
    logger.info(f"Querying Ollama: {model} | Prompt length: {len(prompt)} chars")
    
    try:
        # 120s timeout is safe for local CPU/GPU inference
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        
        result = response.json().get('message', {}).get('content', '')
        logger.info(f"Ollama query successful. Response length: {len(result)} chars")
        return result
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama request failed for model {model}: {e}")
        raise  # Tenacity will catch this and trigger a retry