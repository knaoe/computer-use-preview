# Agents Architecture Overview

This document explains the control flow and major components involved in the browser
automation agent, with an emphasis on the narration pipeline including the new Gemini Native Audio support.

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
  - Owns the speech backend. Supports two modes:
    - **`say` backend** (default): macOS text-to-speech command
    - **`native-audio` backend**: Gemini Native Audio via Live API
  - Receives narration scripts through `ingest_external_text` and handles deduping,
    cooldown, and playback.
  - DOM scrapers/OCR fallbacks have been intentionally retired; the presenter now speaks
    only when given explicit scripts (e.g., from flash narration) or video frames (native-audio).

- **`NativeAudioPresenter` (`presentation/native_audio_presenter.py`)**
  - Manages Gemini Live API connection for real-time audio generation.
  - Sends video frames (screenshots) continuously to the model.
  - Receives and plays audio streams from the model.
  - Uses `proactivity: {'proactive_audio': True}` to enable autonomous audio generation.
  - Sends periodic silence to keep the Live API session active.

## Narration Pipelines

### Flash Narration (say backend)

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

### Native Audio Pipeline (native-audio backend)

1. **Session initialization**
   On startup, `NativeAudioPresenter` establishes a Live API connection with:
   - `proactivity: {'proactive_audio': True}` for autonomous audio generation
   - Audio-only response modality
   - System instruction for presentation narration

2. **Continuous video streaming**
   After each browser action, `PlaywrightComputer.current_state()` captures a screenshot.
   The screenshot is sent to `NativeAudioPresenter` via `send_video_frame()`, which:
   - Queues the frame for async sending
   - Uses `send_client_content()` with image + text prompt
   - Falls back to `send_realtime_input(media=...)` if needed

3. **Keepalive mechanism**
   A background task sends silent audio every 2 seconds to keep the Live API session active,
   mimicking the continuous audio input in the official examples.

4. **Audio reception and playback**
   The `_receive_audio_loop()` continuously receives responses from the model:
   - Iterates through each turn's responses
   - Extracts audio data chunks
   - Queues audio for playback via PyAudio

5. **Model behavior**
   With `proactive_audio: True`, the model automatically generates narration when it sees
   new slide content. No explicit flash narration is needed - the model directly produces audio.

6. **Flash narration bypass**
   When `native-audio` backend is active, `_maybe_request_flash_narration` is skipped to
   avoid redundant API calls and let the Native Audio model handle narration autonomously.

## Configuration

### CLI Options (`main.py`)

**Common options:**
- `--slide-audio`: Enable narration
- `--slide-audio-backend {say,native-audio}`: Choose backend (default: `say`)
- `--slide-audio-debug`: Enable verbose logging
- `--slide-audio-cooldown`: Minimum seconds between narrations (default: 2.0)
- `--slide-audio-warmup`: Optional phrase to play on initialization

**Say backend options:**
- `--slide-audio-voice`: macOS voice name
- `--slide-audio-rate`: Speech rate

**Native Audio backend options:**
- `--slide-audio-native-model`: Model name (default: `gemini-2.5-flash-native-audio-preview-09-2025`)
- `--slide-audio-native-instruction`: Custom system instruction
- `--slide-audio-frame-rate`: Frames per second to send (default: 1.0)

### Environment Variables

- `GEMINI_API_KEY` (required)
- `FLASH_NARRATION_MODEL` (optional): Override flash model for say backend
- `USE_VERTEXAI`: Set to `1` or `true` to use Vertex AI
- `VERTEXAI_PROJECT`: Vertex AI project ID
- `VERTEXAI_LOCATION`: Vertex AI location

### Example Usage

**Flash narration with say backend:**
```bash
python main.py \
  --query "Present the slides at example.com/presentation" \
  --slide-audio \
  --slide-audio-backend say \
  --slide-audio-voice "Kyoko" \
  --slide-audio-debug
```

**Native Audio backend:**
```bash
python main.py \
  --query "Present the slides at example.com/presentation" \
  --slide-audio \
  --slide-audio-backend native-audio \
  --slide-audio-frame-rate 1.0 \
  --slide-audio-debug
```

**Note:** Native Audio requires `pyaudio`. Install with:
```bash
brew install portaudio  # macOS
pip install pyaudio
```

## Known Constraints

### Say Backend
- Narration depends entirely on the flash model generating an affirmative response.
  If the model returns malformed JSON (e.g., fenced blocks), `_parse_flash_response`
  attempts to sanitize and parse the embedded object, but silent failures are possible.
- DOM-based fallbacks were removed; if flash rejects narration for key slides, there is
  no secondary path to force speech.
- macOS only - requires `say` command. Other platforms need alternate implementations.

### Native Audio Backend
- **No synchronization with Computer Use**: The Computer Use model advances slides independently
  of narration completion. This can cause slides to change mid-narration.
  - **Future enhancement**: Implement Function Calling to coordinate timing between models.
- **Proactive audio required**: Must set `proactivity: {'proactive_audio': True}` or the model
  won't generate audio autonomously.
- **Keepalive mechanism**: Requires periodic silent audio to maintain Live API session activity.
- **Platform requirements**: Requires PyAudio and portaudio library.
- **Playwright only**: Currently only implemented for `PlaywrightComputer`. Browserbase support TBD.

## Token Usage Tracking

The agent tracks token usage separately for each model:
- **Computer Use model**: Main orchestration and browser control
- **Flash Narration model**: Text generation for say backend (skipped in native-audio mode)
- **Native Audio model**: (Future) Live API token tracking

Per-turn and cumulative statistics are displayed in a three-column table:
- This Turn
- Model-specific Total
- Overall Total

Session summary shows per-model breakdown and overall totals.

## Troubleshooting

### Say Backend
- Run with `--slide-audio-debug` to see:
  - Flash raw response: `[Narration] flash raw response: ...`
  - Say invocation logs: `[say] speaking X chars`
- If narration fails, check flash response format

### Native Audio Backend
- Run with `--slide-audio-debug` to see detailed logs:
  - `[Native Audio] Session connected` - Connection established
  - `[Native Audio] Sent video frame #X` - Frame sent successfully
  - `[Native Audio] Sent silence to keep connection alive` - Keepalive working
  - `[Native Audio] Calling session.receive()` - Waiting for response
  - `[Native Audio] ✓ Received audio chunk` - Audio received (should appear!)
  - `[Model Text]: <text>` - Any text responses from model
- If no audio chunks received:
  - Verify `proactivity: {'proactive_audio': True}` is set
  - Check that silence keepalive is sending
  - Ensure PyAudio is properly installed and configured
- Use headphones to prevent echo feedback

## Future Enhancements

### Synchronization Between Models
Implement Function Calling to coordinate Computer Use and Native Audio:

1. **Native Audio Function**: `narration_complete()`
   - Called by Native Audio model when narration finishes
   - Signals that it's safe to advance slides

2. **Computer Use Wait Logic**:
   - After executing slide-related actions, wait for `narration_complete()` signal
   - Use asyncio events or callbacks for synchronization

3. **Prompt Adjustments**:
   - Computer Use: "Wait for narration to complete before advancing"
   - Native Audio: "Call narration_complete() when you finish speaking"

This would ensure proper pacing and prevent slides from changing mid-narration.
