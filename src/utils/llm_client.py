import requests
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging for this module
logger = logging.getLogger(__name__)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def query_ollama(model: str, prompt: str, images: list = None, stream: bool = False) -> str:
    """
    Gold Standard LLM Client: Queries local Ollama with exponential backoff retry logic.
    
    Args:
        model (str): The Ollama model to use (e.g., 'qwen2.5:7b', 'minicpm-v').
        prompt (str): The user prompt.
        images (list, optional): List of base64 encoded image strings for vision models.
        stream (bool, optional): Whether to stream the response. Default False.
        
    Returns:
        str: The generated text content.
    """
    url = "http://localhost:11434/api/chat" if images else "http://localhost:11434/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt if not images else None,
        "messages": [{"role": "user", "content": prompt, "images": images}] if images else None,
        "stream": stream
    }
    
    # Clean up None values from payload
    payload = {k: v for k, v in payload.items() if v is not None}
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        
        if images:
            return response.json().get('message', {}).get('content', '')
        else:
            return response.json().get('response', '')
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama request failed for model {model}: {e}")
        raise # Tenacity will catch this and retry