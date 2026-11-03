"""
AI Learning Roadmap Generator
=============================

A modular, production-ready Streamlit application powered by Google's
`gemini-3.8-flash` model via the official `google-genai` SDK.

Architecture:
- Configuration & Security Layer: Environment variable resolution (`GEMINI_API_KEY`)
  with zero hardcoded secrets or UI exposure.
- Domain Layer: Strongly typed dataclasses and validation for roadmap inputs.
- Prompt Engineering Layer: Deterministic builders for system instructions and
  structured curriculum prompts.
- Service Layer: `RoadmapGeneratorService` wrapping `google-genai` with
  dependency injection, thinking-level configuration, and safe error handling.
- Export Layer: Pure utilities for generating downloadable Markdown and JSON.
- Presentation Layer: Modular Streamlit UI components invoked via `main()`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from dotenv import load_dotenv

try:
    import streamlit as st
except ImportError:  # pragma: no cover - fallback for bare test environments
    st = None  # type: ignore[assignment]

try:
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover - fallback when SDK is mocked in tests
    genai = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# 1. Configuration & Environment Security Layer
# ---------------------------------------------------------------------------

MODEL_ID: str = "gemini-3.8-flash"
ENV_API_KEY_NAMES: Tuple[str, ...] = ("GEMINI_API_KEY", "GOOGLE_API_KEY")

VALID_LEVELS: Tuple[str, ...] = ("Beginner", "Intermediate", "Advanced")
VALID_LEARNING_STYLES: Tuple[str, ...] = (
    "Hands-on Project-Based",
    "Balanced (Theory + Projects)",
    "Theory & Documentation First",
    "Fast-Track Crash Course",
)
# Note: `gemini-3.8-flash` supports low, medium, and high thinking levels;
# setting `minimal` returns an API validation error.
VALID_THINKING_LEVELS: Tuple[str, ...] = ("low", "medium", "high")


class ConfigurationError(Exception):
    """Raised when required environment configuration is missing or invalid."""


class RoadmapGenerationError(Exception):
    """Raised when the AI service fails to generate a valid learning roadmap."""


def load_api_key(env_map: Optional[Dict[str, str]] = None) -> str:
    """Load the Gemini API key securely from environment variables.

    The API key is never hardcoded, logged, or rendered in the UI.

    Args:
        env_map: Optional dictionary overriding `os.environ` for unit testing.

    Returns:
        The resolved non-empty API key string.

    Raises:
        ConfigurationError: If no API key is configured in the environment.
    """
    if env_map is None:
        load_dotenv(override=False)
        source: Any = os.environ
    else:
        source = env_map

    for var_name in ENV_API_KEY_NAMES:
        raw_value = source.get(var_name, "").strip()
        if raw_value:
            return raw_value

    raise ConfigurationError(
        "Missing API key. Set the `GEMINI_API_KEY` environment variable in your "
        "`.env` file or shell environment before generating a roadmap."
    )


def has_configured_api_key(env_map: Optional[Dict[str, str]] = None) -> bool:
    """Return True if a valid API key is present in the environment."""
    try:
        load_api_key(env_map=env_map)
        return True
    except ConfigurationError:
        return False


# ---------------------------------------------------------------------------
# 2. Domain & Data Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoadmapRequest:
    """Validated user parameters for generating a personalized learning roadmap."""

    topic: str
    current_level: str = "Beginner"
    target_goal: str = "Build production-ready skills and a portfolio project"
    duration_weeks: int = 8
    hours_per_week: int = 10
    learning_style: str = "Balanced (Theory + Projects)"
    prior_background: str = ""
    thinking_level: str = "medium"
    include_projects: bool = True
    include_assessments: bool = True

    @property
    def total_study_hours(self) -> int:
        """Calculate the total planned study hours across the full duration."""
        return self.duration_weeks * self.hours_per_week

    def validate(self) -> None:
        """Validate all request constraints and raise ValueError on invalid input."""
        cleaned_topic = self.topic.strip()
        if not cleaned_topic:
            raise ValueError("Learning topic cannot be empty.")
        if len(cleaned_topic) < 2 or len(cleaned_topic) > 200:
            raise ValueError("Learning topic must be between 2 and 200 characters.")

        if self.current_level not in VALID_LEVELS:
            raise ValueError(
                f"Invalid current_level '{self.current_level}'. "
                f"Must be one of: {', '.join(VALID_LEVELS)}."
            )

        if not self.target_goal.strip():
            raise ValueError("Target goal cannot be empty.")

        if not 1 <= int(self.duration_weeks) <= 52:
            raise ValueError("Duration in weeks must be between 1 and 52.")

        if not 1 <= int(self.hours_per_week) <= 60:
            raise ValueError("Hours per week must be between 1 and 60.")

        if self.learning_style not in VALID_LEARNING_STYLES:
            raise ValueError(
                f"Invalid learning_style '{self.learning_style}'. "
                f"Must be one of: {', '.join(VALID_LEARNING_STYLES)}."
            )

        normalized_thinking = self.thinking_level.lower().strip()
        if normalized_thinking not in VALID_THINKING_LEVELS:
            raise ValueError(
                f"Invalid thinking_level '{self.thinking_level}' for {MODEL_ID}. "
                f"Supported values are: {', '.join(VALID_THINKING_LEVELS)}."
            )


@dataclass(frozen=True)
class RoadmapResult:
    """Encapsulates the generated roadmap content and generation metadata."""

    request: RoadmapRequest
    markdown_content: str
    model_id: str = MODEL_ID
    generated_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
    )


# ---------------------------------------------------------------------------
# 3. Prompt Engineering Layer
# ---------------------------------------------------------------------------


class PromptBuilder:
    """Constructs structured system instructions and user prompts for Gemini."""

    SYSTEM_INSTRUCTION: str = (
        "You are a Principal Technical Curriculum Architect and Senior Engineering "
        "Mentor. Your task is to design realistic, deeply practical, week-by-week "
        "learning roadmaps tailored to a learner's exact skill level, weekly time "
        "budget, and career objective.\n\n"
        "Guidelines:\n"
        "1. Respect the learner's time budget strictly—allocate hours across topics "
        "so the sum of weekly hours matches their stated weekly availability.\n"
        "2. Avoid generic filler. Name concrete concepts, industry-standard tools, "
        "architectural patterns, and specific documentation or book topics.\n"
        "3. Structure the output in clean, scannable Markdown using clear headings, "
        "week-by-week tables or checklists, hands-on milestones, and verification "
        "criteria.\n"
        "4. Adapt depth to the learner's current level and prior knowledge so they "
        "do not waste time repeating fundamentals they already know."
    )

    @classmethod
    def build_user_prompt(cls, request: RoadmapRequest) -> str:
        """Build a detailed curriculum generation prompt from a validated request."""
        request.validate()

        prior_bg_text = (
            request.prior_background.strip()
            if request.prior_background.strip()
            else "None specified"
        )
        projects_directive = (
            "- **Hands-On Projects & Capstone**: Include concrete mini-projects for "
            "each phase plus a final portfolio-grade capstone project with architecture "
            "requirements and evaluation rubric."
            if request.include_projects
            else "- **Hands-On Practice**: Focus on concise concept exercises."
        )
        assessments_directive = (
            "- **Checkpoints & Self-Assessments**: Provide concrete review questions "
            "and practical competency checks at the end of each phase."
            if request.include_assessments
            else "- **Checkpoints**: Provide brief milestone completion criteria."
        )

        return (
            f"Design a complete, personalized learning roadmap with the following "
            f"parameters:\n\n"
            f"### Learner Profile & Constraints\n"
            f"- **Target Skill / Subject**: {request.topic.strip()}\n"
            f"- **Current Proficiency**: {request.current_level}\n"
            f"- **Primary Goal / Outcome**: {request.target_goal.strip()}\n"
            f"- **Timeline**: {request.duration_weeks} weeks "
            f"({request.hours_per_week} hours/week, "
            f"{request.total_study_hours} total study hours)\n"
            f"- **Preferred Learning Style**: {request.learning_style}\n"
            f"- **Prior Background / Prerequisites**: {prior_bg_text}\n\n"
            f"### Required Roadmap Sections\n"
            f"1. **Executive Overview & Prerequisites Check**: Summary of the learning "
            f"strategy, time-allocation breakdown (theory vs. practice), and setup "
            f"requirements.\n"
            f"2. **Phased & Week-by-Week Curriculum ({request.duration_weeks} Weeks)**: "
            f"For each week (or logical weekly sprint), list specific core concepts, "
            f"hour allocation totaling {request.hours_per_week} hrs/week, and concrete "
            f"deliverables.\n"
            f"3. {projects_directive}\n"
            f"4. {assessments_directive}\n"
            f"5. **Curated Learning Stack & Best Practices**: Recommended official "
            f"documentation, foundational books/courses, tooling setup, and common "
            f"pitfalls to avoid.\n"
        )


# ---------------------------------------------------------------------------
# 4. AI Service Layer (Gemini 3.8 Flash via `google-genai`)
# ---------------------------------------------------------------------------


class RoadmapGeneratorService:
    """Service wrapper around the Google GenAI SDK (`gemini-3.8-flash`)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
        model_id: str = MODEL_ID,
        temperature: float = 0.4,
    ) -> None:
        """Initialize the service with an injected client or environment API key.

        Args:
            api_key: Optional API key override (defaults to `load_api_key()`).
            client: Optional pre-configured `genai.Client` instance for testing.
            model_id: Gemini model identifier (defaults to `gemini-3.8-flash`).
            temperature: Sampling temperature between 0.0 and 2.0.
        """
        if not 0.0 <= float(temperature) <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0.")

        self.model_id = model_id
        self.temperature = float(temperature)
        self._client = client
        self._api_key = api_key

    def _get_client(self) -> Any:
        """Lazily initialize and return the Google GenAI client."""
        if self._client is not None:
            return self._client

        resolved_key = self._api_key if self._api_key else load_api_key()
        if genai is None:
            raise RoadmapGenerationError(
                "The `google-genai` package is not installed. Run "
                "`pip install -r requirements.txt`."
            )
        self._client = genai.Client(api_key=resolved_key)
        return self._client

    def _build_generation_config(self, thinking_level: str) -> Any:
        """Build `GenerateContentConfig` for `gemini-3.8-flash`."""
        normalized_thinking = thinking_level.lower().strip()
        if normalized_thinking not in VALID_THINKING_LEVELS:
            raise ValueError(
                f"Unsupported thinking_level '{thinking_level}'. "
                f"Must be one of {VALID_THINKING_LEVELS}."
            )

        if types is not None:
            return types.GenerateContentConfig(
                system_instruction=PromptBuilder.SYSTEM_INSTRUCTION,
                temperature=self.temperature,
                thinking_config=types.ThinkingConfig(
                    thinking_level=normalized_thinking
                ),
            )

        # Fallback dictionary config when `google.genai.types` is mocked in tests
        return {
            "system_instruction": PromptBuilder.SYSTEM_INSTRUCTION,
            "temperature": self.temperature,
            "thinking_config": {"thinking_level": normalized_thinking},
        }

    def generate_roadmap(self, request: RoadmapRequest) -> RoadmapResult:
        """Generate a complete learning roadmap synchronously.

        Args:
            request: Validated `RoadmapRequest` domain object.

        Returns:
            `RoadmapResult` containing the generated Markdown roadmap and metadata.

        Raises:
            ValueError: If request parameters fail validation.
            ConfigurationError: If `GEMINI_API_KEY` is missing.
            RoadmapGenerationError: If the API call fails or returns empty text.
        """
        prompt = PromptBuilder.build_user_prompt(request)
        config = self._build_generation_config(request.thinking_level)
        client = self._get_client()

        try:
            response = client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=config,
            )
        except Exception as exc:
            raise RoadmapGenerationError(
                "Failed to generate roadmap from the Gemini API. Verify that your "
                "`GEMINI_API_KEY` environment variable is valid and has quota "
                f"available ({type(exc).__name__})."
            ) from exc

        text = getattr(response, "text", None)
        if not text or not str(text).strip():
            raise RoadmapGenerationError(
                "The Gemini model returned an empty response. Try refining your "
                "topic or learning goal."
            )

        return RoadmapResult(
            request=request,
            markdown_content=str(text).strip(),
            model_id=self.model_id,
        )

    def stream_roadmap(self, request: RoadmapRequest) -> Iterator[str]:
        """Stream roadmap chunks from `gemini-3.8-flash` for responsive UI updates."""
        prompt = PromptBuilder.build_user_prompt(request)
        config = self._build_generation_config(request.thinking_level)
        client = self._get_client()

        try:
            stream = client.models.generate_content_stream(
                model=self.model_id,
                contents=prompt,
                config=config,
            )
            for chunk in stream:
                chunk_text = getattr(chunk, "text", None)
                if chunk_text:
                    yield str(chunk_text)
        except Exception as exc:
            raise RoadmapGenerationError(
                "Interrupted while streaming roadmap from the Gemini API. "
                f"Check your network connection and API quota ({type(exc).__name__})."
            ) from exc


# ---------------------------------------------------------------------------
# 5. Export & Formatting Layer
# ---------------------------------------------------------------------------


class RoadmapExporter:
    """Pure formatting utilities for exporting generated roadmaps."""

    @staticmethod
    def to_markdown_document(result: RoadmapResult) -> str:
        """Format a `RoadmapResult` as a standalone downloadable Markdown file."""
        req = result.request
        header = (
            f"# Learning Roadmap: {req.topic}\n\n"
            f"- **Model**: `{result.model_id}`\n"
            f"- **Generated At**: {result.generated_at_utc}\n"
            f"- **Current Level**: {req.current_level}\n"
            f"- **Target Goal**: {req.target_goal}\n"
            f"- **Timeline**: {req.duration_weeks} weeks "
            f"({req.hours_per_week} hrs/week — {req.total_study_hours} total hours)\n"
            f"- **Learning Style**: {req.learning_style}\n\n"
            f"---\n\n"
        )
        return header + result.markdown_content + "\n"

    @staticmethod
    def to_json_document(result: RoadmapResult) -> str:
        """Serialize a `RoadmapResult` to formatted JSON for archiving or tooling."""
        payload = {
            "model_id": result.model_id,
            "generated_at_utc": result.generated_at_utc,
            "parameters": {
                **asdict(result.request),
                "total_study_hours": result.request.total_study_hours,
            },
            "roadmap_markdown": result.markdown_content,
        }
        return json.dumps(payload, indent=2, ensure_ascii=False)

    @staticmethod
    def build_safe_filename(topic: str, extension: str) -> str:
        """Create a filesystem-safe slug filename for downloads."""
        cleaned = "".join(
            ch.lower() if ch.isalnum() else "_" for ch in topic.strip()
        )
        slug = "_".join(part for part in cleaned.split("_") if part)[:40]
        if not slug:
            slug = "learning_roadmap"
        ext = extension.lstrip(".")
        return f"roadmap_{slug}.{ext}"


# ---------------------------------------------------------------------------
# 6. Streamlit Presentation Layer
# ---------------------------------------------------------------------------


class RoadmapUI:
    """Modular Streamlit UI controller for the AI Learning Roadmap Generator."""

    SESSION_RESULT_KEY: str = "latest_roadmap_result"
    SESSION_HISTORY_KEY: str = "roadmap_history"

    @staticmethod
    def configure_page() -> None:
        """Configure Streamlit page metadata and layout."""
        st.set_page_config(
            page_title="AI Learning Roadmap Generator",
            page_icon="🧭",
            layout="wide",
            initial_sidebar_state="expanded",
        )

    @staticmethod
    def init_session_state() -> None:
        """Initialize Streamlit session state keys if absent."""
        if RoadmapUI.SESSION_RESULT_KEY not in st.session_state:
            st.session_state[RoadmapUI.SESSION_RESULT_KEY] = None
        if RoadmapUI.SESSION_HISTORY_KEY not in st.session_state:
            st.session_state[RoadmapUI.SESSION_HISTORY_KEY] = []

    @staticmethod
    def render_sidebar() -> Tuple[str, float, bool]:
        """Render sidebar settings for model reasoning and environment status.

        Returns:
            A tuple of `(thinking_level, temperature, use_streaming)`.
        """
        with st.sidebar:
            st.header("⚙️ Model & Generation Settings")
            st.caption(f"Powered by **`{MODEL_ID}`**")

            api_ready = has_configured_api_key()
            if api_ready:
                st.success("🔒 API Key loaded from environment (`GEMINI_API_KEY`).")
            else:
                st.error(
                    "⚠️ `GEMINI_API_KEY` not detected in environment. "
                    "Add it to your `.env` file or export it in your terminal."
                )

            thinking_level = st.selectbox(
                "Reasoning Effort (`thinking_level`)",
                options=list(VALID_THINKING_LEVELS),
                index=1,
                help=(
                    "Controls the reasoning depth of `gemini-3.8-flash`. "
                    "Supported values: `low`, `medium` (default), and `high`."
                ),
            )

            temperature = st.slider(
                "Creativity (`temperature`)",
                min_value=0.0,
                max_value=1.0,
                value=0.4,
                step=0.1,
                help="Lower values produce more structured, deterministic curricula.",
            )

            use_streaming = st.toggle(
                "Stream response in real time",
                value=True,
                help="Display the roadmap progressively as `gemini-3.8-flash` generates it.",
            )

            st.divider()
            st.subheader("📚 Session History")
            history: List[RoadmapResult] = st.session_state.get(
                RoadmapUI.SESSION_HISTORY_KEY, []
            )
            if not history:
                st.caption("No roadmaps generated in this session yet.")
            else:
                for idx, item in enumerate(reversed(history)):
                    label = (
                        f"{item.request.topic} ({item.request.current_level}, "
                        f"{item.request.duration_weeks}w)"
                    )
                    if st.button(label, key=f"history_btn_{idx}", use_container_width=True):
                        st.session_state[RoadmapUI.SESSION_RESULT_KEY] = item

        return thinking_level, float(temperature), bool(use_streaming)

    @staticmethod
    def render_input_form(default_thinking_level: str) -> Tuple[bool, Optional[RoadmapRequest]]:
        """Render the roadmap parameter form and return `(submitted, request)`."""
        with st.form("roadmap_parameters_form", clear_on_submit=False):
            st.subheader("🎯 Define Your Learning Goal")

            col_left, col_right = st.columns(2)
            with col_left:
                topic = st.text_input(
                    "Skill, Technology, or Subject *",
                    placeholder="e.g., Agentic AI Engineering with Python, Rust Systems Programming",
                )
                current_level = st.selectbox(
                    "Current Proficiency Level",
                    options=list(VALID_LEVELS),
                    index=0,
                )
                learning_style = st.selectbox(
                    "Preferred Learning Style",
                    options=list(VALID_LEARNING_STYLES),
                    index=0,
                )

            with col_right:
                target_goal = st.text_input(
                    "Target Career or Project Outcome *",
                    value="Build production-ready skills and a portfolio capstone project",
                )
                duration_weeks = st.slider(
                    "Timeline (Weeks)",
                    min_value=1,
                    max_value=52,
                    value=8,
                    step=1,
                )
                hours_per_week = st.slider(
                    "Weekly Study Commitment (Hours / Week)",
                    min_value=1,
                    max_value=40,
                    value=10,
                    step=1,
                )

            prior_background = st.text_area(
                "Prior Knowledge / Tools Already Familiar With (Optional)",
                placeholder=(
                    "e.g., Comfortable with intermediate Python, Git, and REST APIs; "
                    "new to vector databases and orchestration."
                ),
                height=80,
            )

            opt_col1, opt_col2 = st.columns(2)
            with opt_col1:
                include_projects = st.checkbox(
                    "Include hands-on phase projects & capstone rubric",
                    value=True,
                )
            with opt_col2:
                include_assessments = st.checkbox(
                    "Include self-assessment checkpoints per phase",
                    value=True,
                )

            submitted = st.form_submit_button(
                "🚀 Generate Personalized Roadmap",
                type="primary",
                use_container_width=True,
            )

        if not submitted:
            return False, None

        request = RoadmapRequest(
            topic=topic,
            current_level=current_level,
            target_goal=target_goal,
            duration_weeks=int(duration_weeks),
            hours_per_week=int(hours_per_week),
            learning_style=learning_style,
            prior_background=prior_background,
            thinking_level=default_thinking_level,
            include_projects=bool(include_projects),
            include_assessments=bool(include_assessments),
        )
        return True, request

    @staticmethod
    def render_roadmap_result(result: RoadmapResult) -> None:
        """Render the generated roadmap along with summary metrics and export buttons."""
        req = result.request
        st.divider()
        st.subheader(f"🗺️ Personalized Roadmap: {req.topic}")

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Current Level", req.current_level)
        m_col2.metric("Timeline", f"{req.duration_weeks} Weeks")
        m_col3.metric("Weekly Commitment", f"{req.hours_per_week} hrs/wk")
        m_col4.metric("Total Study Budget", f"{req.total_study_hours} Hours")

        dl_col1, dl_col2 = st.columns(2)
        md_export = RoadmapExporter.to_markdown_document(result)
        json_export = RoadmapExporter.to_json_document(result)

        with dl_col1:
            st.download_button(
                label="📥 Download as Markdown (.md)",
                data=md_export,
                file_name=RoadmapExporter.build_safe_filename(req.topic, "md"),
                mime="text/markdown",
                use_container_width=True,
            )
        with dl_col2:
            st.download_button(
                label="📦 Download as JSON (.json)",
                data=json_export,
                file_name=RoadmapExporter.build_safe_filename(req.topic, "json"),
                mime="application/json",
                use_container_width=True,
            )

        with st.container(border=True):
            st.markdown(result.markdown_content)


def main(service_factory: Optional[Callable[[float], RoadmapGeneratorService]] = None) -> None:
    """Application entrypoint for the Streamlit UI.

    Args:
        service_factory: Optional factory taking `temperature` and returning a
            `RoadmapGeneratorService` (used for dependency injection in tests).
    """
    RoadmapUI.configure_page()
    RoadmapUI.init_session_state()

    st.title("🧭 AI Learning Roadmap Generator")
    st.markdown(
        "Design structured, week-by-week technical curricula tailored to your "
        f"schedule and background using **`{MODEL_ID}`**."
    )

    thinking_level, temperature, use_streaming = RoadmapUI.render_sidebar()
    submitted, request = RoadmapUI.render_input_form(
        default_thinking_level=thinking_level
    )

    if submitted and request is not None:
        try:
            request.validate()
            factory = service_factory or (
                lambda temp: RoadmapGeneratorService(temperature=temp)
            )
            service = factory(temperature)

            if use_streaming:
                with st.status(
                    f"Architecting your {request.duration_weeks}-week roadmap with `{MODEL_ID}`...",
                    expanded=True,
                ) as status_box:
                    stream_placeholder = st.empty()
                    accumulated_chunks: List[str] = []
                    for chunk in service.stream_roadmap(request):
                        accumulated_chunks.append(chunk)
                        stream_placeholder.markdown("".join(accumulated_chunks))
                    full_markdown = "".join(accumulated_chunks).strip()
                    if not full_markdown:
                        raise RoadmapGenerationError(
                            "The model stream completed without returning content."
                        )
                    result = RoadmapResult(
                        request=request,
                        markdown_content=full_markdown,
                        model_id=service.model_id,
                    )
                    status_box.update(
                        label="Roadmap generation complete!",
                        state="complete",
                        expanded=False,
                    )
            else:
                with st.spinner(
                    f"Generating {request.duration_weeks}-week roadmap with `{MODEL_ID}`..."
                ):
                    result = service.generate_roadmap(request)

            st.session_state[RoadmapUI.SESSION_RESULT_KEY] = result
            st.session_state[RoadmapUI.SESSION_HISTORY_KEY].append(result)

        except (ValueError, ConfigurationError, RoadmapGenerationError) as err:
            st.error(f"**Could not generate roadmap:** {err}")

    latest_result: Optional[RoadmapResult] = st.session_state.get(
        RoadmapUI.SESSION_RESULT_KEY
    )
    if latest_result is not None:
        RoadmapUI.render_roadmap_result(latest_result)


if __name__ == "__main__":
    main()