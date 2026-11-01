"""
Creator Agent: TEXT-ONLY content strategist and copywriter.

Generates marketing copy, scripts, listings, and EDLs based on verified
product data and structured context. Never generates actual media files.
Returns content with routing instructions for the orchestrator.
"""
import dataclasses
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from src.agents.base_agent import AgentResult, BaseAgent
from src.utils.llm_client import OllamaResponse
from src.utils.token_ledger import TokenLedger

logger = logging.getLogger(__name__)

__all__ = ["Creator", "ContentRequestError"]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ContentRequestError(Exception):
    """Raised when input validation fails before LLM invocation."""
    pass


# ---------------------------------------------------------------------------
# Dataclasses (output schema)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ImageSuggestion:
    """Suggested image for downstream design/asset pipeline."""
    description: str
    alt_text: str
    aspect_ratio: str = "16:9"


@dataclass(frozen=True)
class DesignMetadata:
    """Layout hints for downstream Design agent."""
    layout: str = ""
    visual_hierarchy: tuple[str, ...] = ()
    suggested_palette: tuple[str, ...] = ()


@dataclass(frozen=True)
class DownstreamContract:
    """Contract for downstream agent consumption."""
    format_for: str
    schema_version: str = "1.0"
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RoutingInstructions:
    """Instructions for orchestrator to deliver output."""
    destination: str = "orchestrator"
    file_path: Optional[str] = None
    next_agent: Optional[str] = None
    format_for: Optional[str] = None
    validate_before_send: bool = True
    fallback_destination: str = "orchestrator"
    validation_passed: bool = True
    downstream_contract: Optional[DownstreamContract] = None


@dataclass(frozen=True)
class OutputMetadata:
    """Metadata about the generated content."""
    content_hash: str
    version: int
    parent_hash: Optional[str]
    timestamp: str
    model_name: str
    token_count: int
    product_index: int
    content_type: str
    platform: Optional[str]
    routing: RoutingInstructions


@dataclass(frozen=True)
class ContentOutput:
    """Single content output. Fields are optional based on content_type."""
    content_type: str
    platform: Optional[str] = None
    title: Optional[str] = None
    meta_description: Optional[str] = None
    content: Optional[str] = None
    product_description: Optional[str] = None
    script: Optional[str] = None
    affiliate_disclosure: Optional[str] = None
    cta: Optional[str] = None
    hashtags: tuple[str, ...] = ()
    tone_notes: Optional[str] = None
    estimated_duration_sec: float = 0.0
    total_duration_sec: float = 0.0
    image_suggestions: tuple[ImageSuggestion, ...] = ()
    design_metadata: Optional[DesignMetadata] = None
    scenes: tuple[dict[str, Any], ...] = ()
    output_metadata: Optional[OutputMetadata] = None


@dataclass(frozen=True)
class ContentEnvelope:
    """Top-level output wrapper. Always contains variants list for uniform shape."""
    request_id: str
    variant_count: int
    variants: tuple[ContentOutput, ...]
    output_metadata: OutputMetadata


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_CONTENT_TYPES = frozenset(
    ["seo_listing", "social_post", "voiceover_script", "video_edl", "blog_post", "email"]
)
ALLOWED_PLATFORMS = frozenset(
    ["twitter", "linkedin", "instagram", "facebook", "blog", "youtube", "email"]
)
ALLOWED_DESTINATIONS = frozenset(
    ["file", "agent", "publisher", "orchestrator"]
)
REQUIRED_PRODUCT_FIELDS = frozenset(
    ["url", "asin", "brand", "price", "currency", "product_name"]
)
COMMERCIAL_CONTENT_TYPES = frozenset(
    ["seo_listing", "social_post", "blog_post", "email"]
)
VISUAL_CONTENT_TYPES = frozenset(
    ["seo_listing", "social_post", "blog_post"]
)
MAX_SIZES = {
    "seo_listing": {"title_chars": 60, "meta_chars": 160, "description_words": 500},
    "social_post": {},
    "voiceover_script": {"script_words": 1000},
    "video_edl": {},
    "blog_post": {"content_words": 2000},
    "email": {"content_words": 500},
}
PLATFORM_MAX_CHARS = {
    "twitter": 280,
    "linkedin": 1300,
    "instagram": 2200,
    "facebook": 63206,
}
WORDS_PER_SECOND = 2.5


_MINIMAL_BRAND_VOICE = {
    "version": "fallback",
    "brand_name": "Jarvi",
    "tones": {
        "professional": {"description": "Formal, authoritative", "forbidden_phrases": [], "preferred_phrases": []},
        "casual": {"description": "Friendly, conversational", "forbidden_phrases": [], "preferred_phrases": []},
        "persuasive": {"description": "Compelling, benefit-focused", "forbidden_phrases": [], "preferred_phrases": []},
        "technical": {"description": "Precise, specification-focused", "forbidden_phrases": [], "preferred_phrases": []},
    },
    "templates": {},
    "conversion_frameworks": {"AIDA": "Attention-Interest-Desire-Action"},
    "platform_guidelines": {},
    "affiliate_disclosure_default": "As an affiliate, we may earn a commission from qualifying purchases. This does not affect the price you pay.",
    "image_suggestion_defaults": {"product_hero": "Professional product photo on clean background"},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_words(text: str) -> int:
    """Robust word count using regex word boundaries."""
    if not text:
        return 0
    return len(re.findall(r"\b\w+\b", text))


def _truncate_at_word_boundary(text: str, max_words: int) -> str:
    """Truncate text at word boundary."""
    words = text.split()
    if len(words) <= max_words:
        return text
    truncated = " ".join(words[:max_words])
    last_period = truncated.rfind(".")
    if last_period > max_words * 0.7:
        truncated = truncated[: last_period + 1]
    return truncated


def _compute_content_hash(content: ContentOutput) -> str:
    """Deterministic hash of canonical content fields only."""
    canonical = {
        "content_type": content.content_type,
        "platform": content.platform,
        "title": content.title,
        "content": content.content,
        "product_description": content.product_description,
        "script": content.script,
        "cta": content.cta,
        "scenes": content.scenes,
    }
    serialized = json.dumps(canonical, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Creator Agent
# ---------------------------------------------------------------------------

class Creator(BaseAgent):
    """
    TEXT-ONLY content strategist and copywriter.

    Generates one piece of content per call targeting one product. Returns
    content wrapped in a ContentEnvelope with routing instructions.

    Args:
        llm_client: Callable returning OllamaResponse.
        token_ledger: TokenLedger for usage tracking.
        model_name: Ollama model identifier.
        playbook_path: Path to creator playbook markdown.
        brand_voice_path: Path to brand voice JSON config.

    Raises:
        FileNotFoundError: If playbook does not exist.
        ContentRequestError: If BaseAgent API is incompatible.
    """

    def __init__(
        self,
        llm_client: Callable[..., OllamaResponse],
        token_ledger: TokenLedger,
        model_name: str,
        playbook_path: str = "src/agents/playbooks/creator.md",
        brand_voice_path: str = "brain/memories/brand_voice.json",
    ) -> None:
        super().__init__(
            llm_client=llm_client,
            token_ledger=token_ledger,
            playbook_path=playbook_path,
            agent_name="Creator",
            model_name=model_name,
        )
        # Store our own copies to avoid depending on BaseAgent's private attributes
        self._creator_model_name = model_name
        self._creator_agent_name = "Creator"
        
        # Pre-flight: verify BaseAgent exposes the methods we depend on
        for required_method in ("_call_llm", "_strip_markdown_fences"):
            if not hasattr(self, required_method):
                raise ContentRequestError(
                    f"BaseAgent is missing required method '{required_method}'. "
                    "The locked BaseAgent API has changed."
                )
        self._brand_voice = self._load_brand_voice(brand_voice_path)

    def _load_brand_voice(self, path: str) -> dict[str, Any]:
        """Load brand voice config; fall back to minimal on failure."""
        brand_path = Path(path)
        if not brand_path.exists():
            logger.warning("Brand voice file not found at %s; using minimal fallback.", path)
            return dict(_MINIMAL_BRAND_VOICE)
        try:
            with open(brand_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "tones" not in data or not isinstance(data["tones"], dict):
                raise ValueError("brand_voice.json missing 'tones' dict")
            return data
        except (OSError, json.JSONDecodeError, ValueError) as e:
            logger.warning("Failed to load brand voice from %s (%s); using minimal fallback.", path, e)
            return dict(_MINIMAL_BRAND_VOICE)

    def _validate_inputs(
        self,
        task_prompt: str,
        context: dict[str, Any] | str,
        revision_feedback: Optional[str],
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        """Validate inputs fail-fast. Raises ContentRequestError on any failure."""
        try:
            products = json.loads(task_prompt)
        except json.JSONDecodeError as e:
            raise ContentRequestError(f"task_prompt is not valid JSON: {e.__class__.__name__}") from e
        if not isinstance(products, list) or len(products) < 1:
            raise ContentRequestError("task_prompt must be a JSON array with at least 1 product.")

        if isinstance(context, str):
            try:
                context_dict = json.loads(context)
            except json.JSONDecodeError as e:
                raise ContentRequestError(f"context is not valid JSON: {e.__class__.__name__}") from e
        elif isinstance(context, dict):
            context_dict = context
        else:
            raise ContentRequestError(f"context must be a JSON string or dict, got {type(context).__name__}")

        content_request = context_dict.get("content_request")
        if not isinstance(content_request, dict):
            raise ContentRequestError("context.content_request must be a dict.")

        product_index = content_request.get("product_index", 0)
        if not isinstance(product_index, int) or product_index < 0:
            raise ContentRequestError(f"product_index must be a non-negative int, got {product_index!r}")
        if product_index >= len(products):
            raise ContentRequestError(f"product_index {product_index} out of bounds for {len(products)} products.")

        target_product = products[product_index]
        if not isinstance(target_product, dict):
            raise ContentRequestError(f"Product at index {product_index} is not a dict.")
        missing = REQUIRED_PRODUCT_FIELDS - set(target_product.keys())
        if missing:
            raise ContentRequestError(f"Product at index {product_index} missing required fields: {sorted(missing)}")

        content_type = content_request.get("content_type")
        if content_type not in ALLOWED_CONTENT_TYPES:
            raise ContentRequestError(f"content_type {content_type!r} not in allowlist: {sorted(ALLOWED_CONTENT_TYPES)}")

        platform = content_request.get("platform")
        if platform is not None and platform not in ALLOWED_PLATFORMS:
            raise ContentRequestError(f"platform {platform!r} not in allowlist: {sorted(ALLOWED_PLATFORMS)}")

        tone = content_request.get("tone", "professional")
        available_tones = set(self._brand_voice.get("tones", {}).keys())
        if tone not in available_tones:
            raise ContentRequestError(f"tone {tone!r} not in brand_voice tones: {sorted(available_tones)}")

        template = content_request.get("template")
        if template is not None:
            available_templates = set(self._brand_voice.get("templates", {}).keys())
            if template not in available_templates:
                raise ContentRequestError(f"template {template!r} not in brand_voice templates: {sorted(available_templates)}")

        variant_count = content_request.get("variant_count", 1)
        if not isinstance(variant_count, int) or not (1 <= variant_count <= 5):
            raise ContentRequestError(f"variant_count must be int 1-5, got {variant_count!r}")

        routing = content_request.get("output_routing", {})
        if not isinstance(routing, dict):
            raise ContentRequestError("output_routing must be a dict.")
        destination = routing.get("destination", "orchestrator")
        if destination not in ALLOWED_DESTINATIONS:
            raise ContentRequestError(f"destination {destination!r} not in allowlist: {sorted(ALLOWED_DESTINATIONS)}")
        if destination == "file":
            file_path = routing.get("file_path")
            if not isinstance(file_path, str) or not file_path:
                raise ContentRequestError("destination 'file' requires non-empty 'file_path'.")
        if destination == "agent":
            next_agent = routing.get("next_agent")
            if not isinstance(next_agent, str) or not next_agent:
                raise ContentRequestError("destination 'agent' requires non-empty 'next_agent'.")

        return products, context_dict, content_request

    def _build_prompt(
        self,
        product: dict[str, Any],
        context_dict: dict[str, Any],
        content_request: dict[str, Any],
        revision_feedback: Optional[str],
        variant_index: int,
    ) -> str:
        """Build the LLM prompt for one variant."""
        tone = content_request.get("tone", "professional")
        tone_rules = self._brand_voice.get("tones", {}).get(tone, {})
        template = content_request.get("template")
        template_rules = self._brand_voice.get("templates", {}).get(template, {}) if template else {}
        framework = content_request.get("conversion_framework")
        framework_rules = self._brand_voice.get("conversion_frameworks", {}).get(framework, "") if framework else ""
        platform_guidelines = (
            self._brand_voice.get("platform_guidelines", {}).get(content_request.get("platform"), {})
            if content_request.get("platform") else {}
        )

        sections = [
            f"# Target Product\n```json\n{json.dumps(product, indent=2)}\n```",
            f"# Context\n```json\n{json.dumps({k: v for k, v in context_dict.items() if k != 'content_request'}, indent=2)}\n```",
            f"# Content Request\n```json\n{json.dumps(content_request, indent=2)}\n```",
            f"# Brand Voice: {tone}\n{json.dumps(tone_rules, indent=2)}",
        ]
        if template_rules:
            sections.append(f"# Template: {template}\n{json.dumps(template_rules, indent=2)}")
        if framework_rules:
            sections.append(f"# Conversion Framework: {framework}\n{framework_rules}")
        if platform_guidelines:
            sections.append(f"# Platform Guidelines\n{json.dumps(platform_guidelines, indent=2)}")
        if revision_feedback:
            sections.append(f"# Revision Feedback (INCORPORATE THIS)\n{revision_feedback}")
        if variant_index > 0:
            sections.append(
                f"# Variant Instruction\nThis is variant {variant_index + 1}. "
                f"Use a DIFFERENT angle, hook, or framing than other variants."
            )

        sections.append(
            "# Output Rules\n"
            "- Output ONLY a single raw JSON object matching the schema for "
            f"content_type '{content_request.get('content_type')}'.\n"
            "- NO markdown fences, NO commentary, NO trailing commas.\n"
            "- Embed the verified product URL in the content.\n"
            "- Include affiliate_disclosure (MANDATORY for commercial content).\n"
            "- Include image_suggestions (MANDATORY for visual content types).\n"
            "- Include design_metadata with layout hints.\n"
            "- Respect all max size limits.\n"
            "- Never invent product features or ethical claims."
        )

        return "\n\n".join(sections)

    def _post_validate(
        self,
        raw_output: dict[str, Any],
        product: dict[str, Any],
        content_request: dict[str, Any],
        brand_voice: dict[str, Any],
        token_count: int,
        variant_index: int,
        parent_hash: Optional[str],
    ) -> ContentOutput:
        """Validate and normalize LLM output; inject defaults for missing mandatory fields."""
        content_type = content_request.get("content_type")
        platform = content_request.get("platform")
        output_kwargs: dict[str, Any] = {"content_type": content_type, "platform": platform}

        title = raw_output.get("title")
        if isinstance(title, str):
            max_title = MAX_SIZES.get(content_type, {}).get("title_chars")
            if max_title and len(title) > max_title:
                title = title[:max_title]
            output_kwargs["title"] = title

        meta = raw_output.get("meta_description")
        if isinstance(meta, str):
            max_meta = MAX_SIZES.get(content_type, {}).get("meta_chars")
            if max_meta and len(meta) > max_meta:
                meta = meta[:max_meta]
            output_kwargs["meta_description"] = meta

        for field_name, limit_key in [
            ("content", "content_words"),
            ("product_description", "description_words"),
            ("script", "script_words"),
        ]:
            value = raw_output.get(field_name)
            if isinstance(value, str):
                max_words = MAX_SIZES.get(content_type, {}).get(limit_key)
                if max_words and _count_words(value) > max_words:
                    logger.warning("Truncating %s from %d to %d words.", field_name, _count_words(value), max_words)
                    value = _truncate_at_word_boundary(value, max_words)
                output_kwargs[field_name] = value

        if content_type == "social_post" and platform in PLATFORM_MAX_CHARS:
            content_val = raw_output.get("content")
            if isinstance(content_val, str):
                max_chars = PLATFORM_MAX_CHARS[platform]
                if len(content_val) > max_chars:
                    logger.warning("Truncating social_post content from %d to %d chars.", len(content_val), max_chars)
                    output_kwargs["content"] = content_val[:max_chars]

        hashtags = raw_output.get("hashtags")
        if isinstance(hashtags, list):
            output_kwargs["hashtags"] = tuple(str(h) for h in hashtags)

        cta = raw_output.get("cta")
        if isinstance(cta, str):
            output_kwargs["cta"] = cta

        tone_notes = raw_output.get("tone_notes")
        if isinstance(tone_notes, str):
            output_kwargs["tone_notes"] = tone_notes

        scenes = raw_output.get("scenes")
        if isinstance(scenes, list):
            validated_scenes = []
            for s in scenes:
                if isinstance(s, dict):
                    scene = dict(s)
                    vo_text = scene.get("voiceover_text", "")
                    if isinstance(vo_text, str) and vo_text.strip():
                        word_count = _count_words(vo_text)
                        scene["duration_sec"] = max(1.0, word_count / WORDS_PER_SECOND)
                    validated_scenes.append(scene)
            output_kwargs["scenes"] = tuple(validated_scenes)
            output_kwargs["total_duration_sec"] = sum(float(s.get("duration_sec", 0.0)) for s in validated_scenes)

        script = output_kwargs.get("script") or raw_output.get("script")
        if isinstance(script, str) and script.strip():
            word_count = _count_words(script)
            output_kwargs["estimated_duration_sec"] = max(1.0, word_count / WORDS_PER_SECOND)

        image_suggestions = raw_output.get("image_suggestions")
        validated_images: list[ImageSuggestion] = []
        if isinstance(image_suggestions, list):
            for img in image_suggestions:
                if isinstance(img, dict) and img.get("description"):
                    validated_images.append(
                        ImageSuggestion(
                            description=str(img.get("description", "")),
                            alt_text=str(img.get("alt_text", img.get("description", ""))),
                            aspect_ratio=str(img.get("aspect_ratio", "16:9")),
                        )
                    )
        if content_type in VISUAL_CONTENT_TYPES and not validated_images:
            logger.warning("LLM omitted image_suggestions for %s; injecting defaults.", content_type)
            defaults = brand_voice.get("image_suggestion_defaults", {})
            for key, desc in list(defaults.items())[:2]:
                validated_images.append(ImageSuggestion(description=str(desc), alt_text=str(desc)))
        output_kwargs["image_suggestions"] = tuple(validated_images)

        design_meta = raw_output.get("design_metadata")
        if isinstance(design_meta, dict):
            output_kwargs["design_metadata"] = DesignMetadata(
                layout=str(design_meta.get("layout", "")),
                visual_hierarchy=tuple(str(x) for x in design_meta.get("visual_hierarchy", [])),
                suggested_palette=tuple(str(x) for x in design_meta.get("suggested_palette", [])),
            )

        disclosure = raw_output.get("affiliate_disclosure")
        if not isinstance(disclosure, str) or not disclosure.strip():
            if content_type in COMMERCIAL_CONTENT_TYPES:
                logger.warning("LLM omitted affiliate_disclosure for %s; injecting default (FTC).", content_type)
                disclosure = brand_voice.get("affiliate_disclosure_default", _MINIMAL_BRAND_VOICE["affiliate_disclosure_default"])
            else:
                disclosure = None
        output_kwargs["affiliate_disclosure"] = disclosure

        preliminary = ContentOutput(**output_kwargs)
        content_hash = _compute_content_hash(preliminary)

        routing_dict = content_request.get("output_routing", {})
        format_for = routing_dict.get("format_for")
        downstream_contract = None
        if format_for:
            contract_params: dict[str, Any] = {}
            if format_for == "tts_agent":
                contract_params = {
                    "script": output_kwargs.get("script") or output_kwargs.get("content") or "",
                    "tone": content_request.get("tone", "professional"),
                    "language": "en-US",
                    "voice_id": "default",
                    "speed": 1.0,
                }
            elif format_for == "video_assembler":
                contract_params = {
                    "scenes": output_kwargs.get("scenes", ()),
                    "output_format": "mp4",
                    "resolution": "1080p",
                    "fps": 30,
                }
            downstream_contract = DownstreamContract(format_for=format_for, parameters=contract_params)

        routing = RoutingInstructions(
            destination=routing_dict.get("destination", "orchestrator"),
            file_path=routing_dict.get("file_path"),
            next_agent=routing_dict.get("next_agent"),
            format_for=format_for,
            validate_before_send=bool(routing_dict.get("validate_before_send", True)),
            fallback_destination=routing_dict.get("fallback_destination", "orchestrator"),
            validation_passed=True,
            downstream_contract=downstream_contract,
        )

        output_metadata = OutputMetadata(
            content_hash=content_hash,
            version=1 if parent_hash is None else 2,
            parent_hash=parent_hash,
            timestamp=datetime.utcnow().isoformat() + "Z",
            model_name=self._creator_model_name,
            token_count=token_count,
            product_index=content_request.get("product_index", 0),
            content_type=content_type,
            platform=platform,
            routing=routing,
        )
        output_kwargs["output_metadata"] = output_metadata

        return ContentOutput(**output_kwargs)

    def _safe_fallback_envelope(
        self,
        content_request: dict[str, Any],
        product: dict[str, Any],
        token_count: int,
        error_reason: str,
    ) -> ContentEnvelope:
        """Return a minimal valid envelope when LLM output is unusable."""
        content_type = content_request.get("content_type", "seo_listing")
        fallback_output = ContentOutput(
            content_type=content_type,
            platform=content_request.get("platform"),
            title="Content Generation Failed",
            content=f"Error: {error_reason}. Please retry.",
            affiliate_disclosure=self._brand_voice.get(
                "affiliate_disclosure_default", _MINIMAL_BRAND_VOICE["affiliate_disclosure_default"]
            ),
            output_metadata=OutputMetadata(
                content_hash="fallback",
                version=1,
                parent_hash=None,
                timestamp=datetime.utcnow().isoformat() + "Z",
                model_name=self._creator_model_name,
                token_count=token_count,
                product_index=content_request.get("product_index", 0),
                content_type=content_type,
                platform=content_request.get("platform"),
                routing=RoutingInstructions(
                    destination=content_request.get("output_routing", {}).get("fallback_destination", "orchestrator"),
                    validation_passed=False,
                ),
            ),
        )
        return ContentEnvelope(
            request_id=content_request.get("request_id", "unknown"),
            variant_count=1,
            variants=(fallback_output,),
            output_metadata=fallback_output.output_metadata,
        )

    def execute(
        self,
        task_prompt: str,
        context: str,
        revision_feedback: Optional[str] = None,
    ) -> AgentResult:
        """
        Execute content generation for one product.

        Args:
            task_prompt: JSON string of Scout's verified product array.
            context: JSON string of structured Analyst context + content_request.
            revision_feedback: Optional feedback for iterative refinement.

        Returns:
            AgentResult with artifact = JSON-serialized ContentEnvelope.

        Raises:
            ContentRequestError: If input validation fails.
        """
        products, context_dict, content_request = self._validate_inputs(task_prompt, context, revision_feedback)

        product_index = content_request.get("product_index", 0)
        target_product = products[product_index]
        variant_count = content_request.get("variant_count", 1)
        request_id = content_request.get("request_id", "unknown")

        variants: list[ContentOutput] = []
        parent_hash: Optional[str] = None
        total_tokens = 0

        for variant_idx in range(variant_count):
            prompt = self._build_prompt(target_product, context_dict, content_request, revision_feedback, variant_idx)
            try:
                response: OllamaResponse = self._call_llm(prompt=prompt, system_prompt=None, context=None)
                total_tokens += response.total_tokens
            except Exception as e:
                logger.error("LLM call failed for variant %d: %s", variant_idx, e)
                envelope = self._safe_fallback_envelope(
                    content_request, target_product, total_tokens, f"LLM call failed: {e.__class__.__name__}"
                )
                return AgentResult(
                    agent_name=self._creator_agent_name,
                    artifact=json.dumps(dataclasses.asdict(envelope)),
                    token_usage={"total_tokens": total_tokens},
                )

            raw_content = self._strip_markdown_fences(response.content)
            try:
                raw_output = json.loads(raw_content)
            except json.JSONDecodeError as e:
                logger.warning("LLM output not valid JSON for variant %d: %s", variant_idx, e)
                envelope = self._safe_fallback_envelope(
                    content_request, target_product, total_tokens, "LLM output was not valid JSON"
                )
                return AgentResult(
                    agent_name=self._creator_agent_name,
                    artifact=json.dumps(dataclasses.asdict(envelope)),
                    token_usage={"total_tokens": total_tokens},
                )

            if not isinstance(raw_output, dict):
                logger.warning("LLM output was not a JSON object for variant %d.", variant_idx)
                envelope = self._safe_fallback_envelope(
                    content_request, target_product, total_tokens, "LLM output was not a JSON object"
                )
                return AgentResult(
                    agent_name=self._creator_agent_name,
                    artifact=json.dumps(dataclasses.asdict(envelope)),
                    token_usage={"total_tokens": total_tokens},
                )

            content_output = self._post_validate(
                raw_output=raw_output,
                product=target_product,
                content_request=content_request,
                brand_voice=self._brand_voice,
                token_count=response.total_tokens,
                variant_index=variant_idx,
                parent_hash=parent_hash,
            )
            variants.append(content_output)
            parent_hash = content_output.output_metadata.content_hash

        envelope_metadata = variants[-1].output_metadata
        envelope = ContentEnvelope(
            request_id=request_id,
            variant_count=variant_count,
            variants=tuple(variants),
            output_metadata=envelope_metadata,
        )

        return AgentResult(
            agent_name=self._creator_agent_name,
            artifact=json.dumps(dataclasses.asdict(envelope)),
            token_usage={"total_tokens": total_tokens},
        )