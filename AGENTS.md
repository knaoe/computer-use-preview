# Agents Architecture Overview

This document explains the control flow and major components involved in the browser
automation agent, with an emphasis on the current narration pipeline.

## Core Components

- **`BrowserAgent` (`agent.py`)**
  - Orchestrates the conversation loop with Gemini Computer Use.
  - Maintains conversation history (`self._contents`) and retry logic.
  - Converts model replies into concrete function invocations via `handle_action`.
  - After each tool call returning an `EnvState`, hands the screenshot to the flash-based
    narration flow (`_maybe_request_flash_narration`).

- **`Computer` interface (`computers/computer.py`)**
  - Abstract base class describing browser actions (click, navigate, type, etc.).
  - Concrete implementations (`PlaywrightComputer`, `BrowserbaseComputer`) execute these
    actions using Playwright or Browserbase.
  - Exposes `narrate_text(text, source)` so higher layers can feed narration scripts to
    the presenter without coupling the agent to a single environment.

- **`PlaywrightComputer` (`computers/playwright/playwright.py`)**
  - Manages a local Chromium instance.
  - Wraps Playwright APIs to implement the `Computer` interface.
  - Produces `EnvState` snapshots (PNG + URL) after each action.
  - Forwards narration scripts to the presenter in `narrate_text`.

- **`SlideAudioPresenter` (`presentation/slide_audio.py`)**
  - Owns the speech backend (default: macOS `say`).
  - Receives narration scripts through `ingest_external_text` and handles deduping,
    cooldown, and playback.
  - DOM scrapers/OCR fallbacks have been intentionally retired; the presenter now speaks
    only when given explicit scripts (e.g., from flash narration).

## Narration Pipeline

1. **Action execution**  
   `BrowserAgent` calls `handle_action`, which invokes a Playwright/Browserbase method.
   The environment returns an `EnvState` containing a screenshot and URL.

2. **Flash decision**  
   `_maybe_request_flash_narration` hashes the screenshot to avoid duplicate work, then
   queries a Gemini flash model (default: `gemini-2.5-flash`) with the screenshot and
   high-level task context.

3. **Response parsing**  
   The flash response is expected to be JSON with `should_narrate` and `script`. The
   agent strips code fences when present and parses the JSON payload.

4. **Playback**  
   If `should_narrate` is true and `script` is non-empty, the agent invokes
   `narrate_text` on the active computer. This calls `SlideAudioPresenter.ingest_external_text`,
   which respects cooldown and then plays the script via `say`.

5. **Caching**  
   Screenshot hashes are cached to prevent repeated narration when the model takes
   multiple actions within the same visual state.

## Configuration

- CLI (`main.py`) exposes switches such as:
  - `--slide-audio` to enable narration.
  - `--slide-audio-voice`, `--slide-audio-rate`, `--slide-audio-cooldown`,
    `--slide-audio-debug`, `--slide-audio-warmup`.
- Environment variables:
  - `GEMINI_API_KEY` (required).
  - `FLASH_NARRATION_MODEL` (optional override for the flash model name).

## Known Constraints

- Narration now depends entirely on the flash model generating an affirmative response.
  If the model returns malformed JSON (e.g., fenced blocks), `_parse_flash_response`
  attempts to sanitize and parse the embedded object, but silent failures are possible.
- DOM-based fallbacks were removed; if flash rejects narration for key slides, there is
  no secondary path to force speech.
- Speech backend currently uses macOS `say`. Deployments on non-macOS environments
  require adding alternate synthesizer implementations.

## Hand-off Notes

- Current debugging tip: run with `--slide-audio-debug` to surface the flash raw
  response (`[Narration] flash raw response: ...`) and the `say` invocation logs.
- If narration still fails due to malformed flash responses, consider hardening
  `_parse_flash_response` or establishing a minimal “force speak” command based on
  Computer Use reasoning text.
