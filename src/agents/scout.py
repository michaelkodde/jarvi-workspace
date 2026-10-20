"""
Scout Agent: Orchestrates dynamic saturation loop for product discovery.
"""
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Callable, List

from src.agents.base_agent import AgentResult, BaseAgent
from src.utils.llm_client import OllamaResponse
from src.utils.product_scraper import ProductData, search_products
from src.utils.scout_checkpoint import ScoutCheckpoint
from src.utils.token_ledger import TokenLedger, TokenUsage

logger = logging.getLogger(__name__)

__all__ = ["Scout"]

# Circuit breaker - NOT a search limit. LLM decides convergence.
MAX_SAFETY_CEILING = 10


class Scout(BaseAgent):
    """
    Concrete specialist agent for product discovery and validation.
    
    Inherits execution and logging logic from BaseAgent, but overrides
    execute() to implement a dynamic saturation loop with SQLite checkpointing.
    """

    def __init__(
        self,
        llm_client: Callable[..., OllamaResponse],
        token_ledger: TokenLedger,
        model_name: str,
        playbook_path: str = "src/agents/playbooks/scout.md",
        banned_brands_path: str = "brain/memories/banned_brands.json",
    ) -> None:
        super().__init__(
            llm_client=llm_client,
            token_ledger=token_ledger,
            playbook_path=playbook_path,
            agent_name="Scout",
            model_name=model_name,
        )
        self.checkpoint = ScoutCheckpoint()
        self.banned_brands_path = Path(banned_brands_path)
        self.banned_brands = self._load_banned_brands()

    def _load_banned_brands(self) -> set:
        """Loads banned brands from the configured path."""
        if not self.banned_brands_path.exists():
            return set()
        
        try:
            with open(self.banned_brands_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                banned = set()
                for entry in data.get("banned", []):
                    banned.add(entry["name"].lower())
                    banned.update(alias.lower() for alias in entry.get("aliases", []))
                return banned
        except Exception as e:
            logger.warning("Failed to load banned brands: %s", e)
            return set()

    def _is_brand_banned(self, brand: str) -> bool:
        """Checks if a brand is in the banned list."""
        return brand.lower() in self.banned_brands

    def execute(self, task_prompt: str, context: str) -> AgentResult:
        """
        Executes the dynamic saturation loop to find verified products.
        
        Args:
            task_prompt: The specific product type to search for.
            context: The Analyst's Ethical Validation and proof criteria ONLY.
            
        Returns:
            AgentResult containing a strict JSON array of verified products.
        """
        run_id = hashlib.md5(f"{task_prompt}_{context}".encode()).hexdigest()
        
        # Load checkpoint if exists
        state = self.checkpoint.load_state(run_id)
        if state:
            iteration, current_query, accumulated_products = state
            logger.info("Resuming Scout saturation loop from iteration %d", iteration)
            all_scraped = [ProductData(**p) for p in accumulated_products]
        else:
            iteration = 0
            current_query = task_prompt
            all_scraped = []

        circuit_breaker_triggered = False

        while iteration < MAX_SAFETY_CEILING:
            products = search_products(current_query)
            
            # Filter out banned brands in Python (defense-in-depth)
            filtered_products = [p for p in products if not self._is_brand_banned(p.brand)]
            
            # Deduplicate and accumulate
            existing_asins = {p.asin for p in all_scraped}
            for p in filtered_products:
                if p.asin not in existing_asins:
                    all_scraped.append(p)
                    existing_asins.add(p.asin)

            # Save checkpoint atomically with full product data
            self.checkpoint.save_state(
                run_id, 
                iteration + 1, 
                current_query, 
                [p.__dict__ for p in all_scraped]
            )

            # Saturation assessment
            scraped_json = json.dumps([p.__dict__ for p in all_scraped])
            assessment_prompt = (
                f"Context:\n{context}\n\n"
                f"Scraped Data:\n{scraped_json}\n\n"
                f"Assess if we have enough verified products (at least 3). "
                f"Output ONLY strict JSON: {{\"saturated\": bool, \"next_query\": str}}"
            )

            response = self.llm_client(
                model=self.model_name,
                prompt=assessment_prompt,
                system_prompt=self.playbook,
                options={"keep_alive": "5m"},
            )
            
            self.token_ledger.log_usage(
                TokenUsage(
                    agent_name="Scout",
                    model_name=self.model_name,
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    total_tokens=response.total_tokens,
                )
            )

            try:
                assessment = json.loads(response.content.strip())
                if assessment.get("saturated"):
                    break
                
                next_query = str(assessment.get("next_query", current_query)).strip()
                if next_query.lower() == current_query.lower():
                    logger.warning("Saturation loop stagnated. Breaking.")
                    break
                    
                current_query = next_query
            except json.JSONDecodeError:
                logger.warning("Failed to parse saturation assessment. Breaking loop.")
                break

            iteration += 1
        else:
            # Circuit breaker triggered
            circuit_breaker_triggered = True
            logger.warning(
                "Circuit breaker triggered after %d iterations. Returning best set found.",
                MAX_SAFETY_CEILING
            )

        # Final formatting and filtering
        scraped_json = json.dumps([p.__dict__ for p in all_scraped])
        final_prompt = (
            f"Context:\n{context}\n\n"
            f"Accumulated Scraped Data:\n{scraped_json}\n\n"
            f"Filter to ONLY include verified brands. Format remaining into strict JSON array. "
            f"NEVER invent/guess. If nothing survives, output: "
            f"[{{\"product_name\": \"N/A\", \"brand\": \"N/A\", \"price\": 0.0, \"currency\": \"USD\", \"url\": \"N/A\", \"asin\": \"N/A\"}}]"
        )

        final_response = self.llm_client(
            model=self.model_name,
            prompt=final_prompt,
            system_prompt=self.playbook,
            options={"keep_alive": "5m"},
        )
        
        self.token_ledger.log_usage(
            TokenUsage(
                agent_name="Scout",
                model_name=self.model_name,
                prompt_tokens=final_response.prompt_tokens,
                completion_tokens=final_response.completion_tokens,
                total_tokens=final_response.total_tokens,
            )
        )

        # Post-validation gate
        try:
            content = final_response.content.strip()
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()
                
            final_products = json.loads(content)
        except json.JSONDecodeError:
            logger.error("Final LLM output is not valid JSON.")
            final_products = [
                {"product_name": "N/A", "brand": "N/A", "price": 0.0, "currency": "USD", "url": "N/A", "asin": "N/A"}
            ]

        scraped_asins = {p.asin for p in all_scraped if p.asin != "N/A"}
        scraped_urls = {p.asin: p.url for p in all_scraped if p.asin != "N/A"}

        validated_products = []
        if isinstance(final_products, list):
            for p in final_products:
                asin = str(p.get("asin", ""))
                brand = str(p.get("brand", ""))
                
                # Double-check banned brands (defense-in-depth)
                if self._is_brand_banned(brand):
                    continue
                
                if asin in scraped_asins and p.get("url") == scraped_urls.get(asin):
                    try:
                        price = float(p.get("price", 0.0))
                        if not math.isfinite(price):
                            price = 0.0
                    except (ValueError, TypeError):
                        price = 0.0

                    validated_products.append(
                        {
                            "product_name": str(p.get("product_name", "N/A")),
                            "brand": brand,
                            "price": price,
                            "currency": str(p.get("currency", "USD")),
                            "url": str(p.get("url", "N/A")),
                            "asin": asin,
                        }
                    )

        if not validated_products:
            validated_products = [
                {"product_name": "N/A", "brand": "N/A", "price": 0.0, "currency": "USD", "url": "N/A", "asin": "N/A"}
            ]

        # Clear checkpoint ONLY after successful validation
        self.checkpoint.clear_state(run_id)

        return AgentResult(
            agent_name="Scout",
            artifact=json.dumps(validated_products),
            token_usage={"total_tokens": 0},  # Logged manually per call
        )

    def execute(self, task_prompt: str, context: str | None = None) -> "AgentResult":
        """
        Executes the agent's primary task using inherited helpers.
        """
        from src.agents.base_agent import AgentResult
        
        # 1. Call the LLM using the inherited helper (playbook is auto-applied as system prompt)
        response = self._call_llm(prompt=task_prompt, context=context)
        
        # 2. Clean the output using the inherited helper
        cleaned_content = self._strip_markdown_fences(response.content)
        
        # 3. Return the standardized result
        return AgentResult(
            artifact=cleaned_content,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
        )