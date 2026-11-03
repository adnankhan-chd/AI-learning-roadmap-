# 🧭 AI Learning Roadmap Generator

A modular, production-ready web application built with **Streamlit** and Google's **`gemini-3.8-flash`** model (via the official `google-genai` SDK) that designs personalized, week-by-week technical learning roadmaps tailored to your current skill level, weekly time budget, and career goals.

## ✨ Key Features

- **Powered by `gemini-3.8-flash`**: Uses Google's `gemini-3.8-flash` model with configurable reasoning effort (`thinking_level`: `low`, `medium`, or `high`) and real-time streaming responses.
- **Strict API Key Security**: Reads `GEMINI_API_KEY` exclusively from environment variables or a local `.env` file via `python-dotenv`. The key is never hardcoded, logged, or exposed in the browser UI.
- **Refactored Layered Architecture**:
  - **Configuration & Security Layer**: `load_api_key` and `has_configured_api_key` validate environment secrets without exposing values.
  - **Domain Layer**: Immutable `@dataclass` models (`RoadmapRequest`, `RoadmapResult`) with strict input validation (`validate()`) and study-hour budgeting.
  - **Prompt Engineering Layer**: `PromptBuilder` constructs deterministic system instructions and structured curriculum prompts.
  - **AI Service Layer**: `RoadmapGeneratorService` encapsulates the `google-genai` client with dependency injection for unit testing, streaming support, and sanitized error handling.
  - **Export Layer**: `RoadmapExporter` generates downloadable Markdown (`.md`) and JSON (`.json`) artifacts with metadata headers.
  - **Presentation Layer**: `RoadmapUI` provides a clean Streamlit interface with session history, summary metrics, and one-click exports.

## 📁 Project Structure

| File | Responsibility |
| :--- | :--- |
| `app.py` | Complete refactored application containing configuration, domain models, prompt builder, `gemini-3.8-flash` service client, exporters, and Streamlit UI |
| `requirements.txt` | Pinned Python dependencies (`streamlit`, `google-genai`, `python-dotenv`) |
| `readme.md` | Setup, environment variable configuration, architecture overview, and testing instructions |

## 🚀 Getting Started

### 1. Clone & Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Your Environment Variable (API Key Security)

Never commit your API key to version control. Set `GEMINI_API_KEY` either in your shell environment or in a `.env` file in the project root (and ensure `.env` is listed in your `.gitignore`):

**Option A — Using a `.env` file (recommended for local development):**

Create a file named `.env` in the same directory as `app.py`:

```ini
GEMINI_API_KEY="your_actual_gemini_api_key_here"
```

**Option B — Exporting in your terminal session:**

```bash
export GEMINI_API_KEY="your_actual_gemini_api_key_here"
```

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## 🧠 Model Notes (`gemini-3.8-flash`)

- **Model Identifier**: `gemini-3.8-flash`
- **Thinking Levels**: Supports `low`, `medium` (default), and `high` via `types.ThinkingConfig(thinking_level=...)`. Note that `minimal` is not supported on `gemini-3.8-flash` and is explicitly guarded against by `RoadmapRequest.validate()`.
- **Context Window**: Up to 1,048,576 input tokens and 65,536 output tokens.
