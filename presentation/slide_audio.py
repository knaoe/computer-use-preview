import dataclasses
import hashlib
import os
import shutil
import subprocess
import time
from typing import Any, Callable, Optional, Protocol

import termcolor
from playwright.sync_api import Page


class SlideAudioError(RuntimeError):
    """Raised when slide audio narration cannot be initialized."""


class SpeechSynthesizer(Protocol):
    """Protocol for speech synthesis implementations."""

    def speak(
        self, text: str, *, interrupt: bool = True, wait: bool = False
    ) -> None: ...

    def stop(self) -> None: ...

    def send_video_frame(self, frame_data: bytes, mime_type: str = "image/png") -> None:
        """Optional: Send video frame for native audio models."""
        ...


@dataclasses.dataclass(slots=True)
class SlideAudioConfig:
    """Configuration for slide narration."""

    enabled: bool = False
    backend: str = "say"
    voice: Optional[str] = None
    rate: Optional[int] = None
    min_chars: int = 1
    max_chars: int = 1200
    warmup_phrase: Optional[str] = "それでは、プレゼンテーションを始めます。"
    debug: bool = False
    cooldown_seconds: float = 2.0
    # Native audio specific settings
    native_audio_model: str = "gemini-2.5-flash-native-audio-preview-09-2025"
    native_audio_system_instruction: Optional[str] = None
    native_audio_frame_rate: float = 1.0  # frames per second to send
    native_audio_wait_timeout: float = 30.0  # seconds to wait for narration completion
    native_audio_quiet_window: float = 0.8  # seconds of silence to treat as done
    native_audio_no_response_timeout: float = (
        6.0  # seconds before assuming no narration will start
    )

    def validate(self) -> None:
        if self.min_chars <= 0:
            raise ValueError("min_chars must be positive")
        if self.max_chars <= self.min_chars:
            raise ValueError("max_chars must be greater than min_chars")
        if self.cooldown_seconds < 0:
            raise ValueError("cooldown_seconds cannot be negative")
        if not self.backend:
            raise ValueError("backend must be provided")
        if self.native_audio_frame_rate <= 0:
            raise ValueError("native_audio_frame_rate must be positive")
        if self.native_audio_wait_timeout <= 0:
            raise ValueError("native_audio_wait_timeout must be positive")
        if self.native_audio_quiet_window <= 0:
            raise ValueError("native_audio_quiet_window must be positive")
        if self.native_audio_no_response_timeout <= 0:
            raise ValueError("native_audio_no_response_timeout must be positive")


class _OSSaySpeechSynthesizer:
    """Speech synthesis using macOS `say`."""

    def __init__(self, voice: Optional[str], rate: Optional[int], debug: bool) -> None:
        if shutil.which("say") is None:
            raise SlideAudioError(
                "macOS `say` command not found. Install the Xcode command line tools "
                "or switch to a different speech backend."
            )
        self._voice = voice
        self._rate = rate
        self._debug = debug
        self._current_process: Optional[subprocess.Popen[str]] = None

    def speak(self, text: str, *, interrupt: bool = True, wait: bool = False) -> None:
        if self._debug:
            termcolor.cprint(f"[say] speaking {len(text)} chars", color="cyan")
        if interrupt and self._current_process and self._current_process.poll() is None:
            self._current_process.terminate()
            try:
                self._current_process.wait(timeout=0.25)
            except subprocess.TimeoutExpired:
                self._current_process.kill()
        command = ["say"]
        if self._voice:
            command.extend(["-v", self._voice])
        if self._rate:
            command.extend(["-r", str(self._rate)])
        command.append(text)
        self._current_process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if wait:
            self._current_process.wait()

    def stop(self) -> None:
        if self._current_process and self._current_process.poll() is None:
            self._current_process.terminate()
            try:
                self._current_process.wait(timeout=0.25)
            except subprocess.TimeoutExpired:
                self._current_process.kill()
        self._current_process = None

    def send_video_frame(self, frame_data: bytes, mime_type: str = "image/png") -> None:
        """Not supported for macOS say backend."""
        pass


class _GeminiNativeAudioSynthesizer:
    """Speech synthesis using Gemini Native Audio Live API."""

    def __init__(
        self,
        config: SlideAudioConfig,
        token_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        try:
            from presentation.native_audio_presenter import NativeAudioPresenter
        except ImportError as e:
            raise SlideAudioError(
                "Failed to import native_audio_presenter. "
                "Make sure pyaudio is installed: pip install pyaudio"
            ) from e

        self._config = config
        self._debug = config.debug

        # Get API credentials
        api_key = os.environ.get("GEMINI_API_KEY")
        use_vertexai = os.environ.get("USE_VERTEXAI", "0").lower() in ["true", "1"]
        project = os.environ.get("VERTEXAI_PROJECT")
        location = os.environ.get("VERTEXAI_LOCATION")

        # Initialize presenter
        instruction = (
            config.native_audio_system_instruction or self._get_default_instruction()
        )
        instruction = self._ensure_completion_instruction(instruction)
        self._presenter = NativeAudioPresenter(
            api_key=api_key,
            model=config.native_audio_model,
            system_instruction=instruction,
            debug=config.debug,
            use_vertexai=use_vertexai,
            project=project,
            location=location,
            token_callback=token_callback,
        )
        self._presenter.start()

    def _get_default_instruction(self) -> str:
        return """
あなたはプロフェッショナルなプレゼンターです。
現在、スライドショーのプレゼンテーションを行っています。

スライドが表示されたら、以下のように発表してください：
1. スライドの内容を読み上げるだけは避けよう。内容から考えて共感してもらえるような言葉選びで説明しよう。
2. 自然な日本語で、聴衆に語りかけるように話す
3. 重要なポイントを強調する
4. 各スライドは2-3文程度で短く説明する
5. 最後のスライドになったら挨拶して終わり

重要: 以下の場合は発表しないでください:
- スライド編集画面やサムネイル表示
- ブラウザのナビゲーションページ (Google検索など)
- ローディング画面
- プレゼンテーションモードに入る前のGoogle Slides等のインターフェース

プレゼンテーションモード (フルスクリーンのスライド表示) のときのみ発表してください。
"""

    def speak(self, text: str, *, interrupt: bool = True, wait: bool = False) -> None:
        """Not used for native audio - the model generates speech directly from video."""
        if self._debug:
            termcolor.cprint(
                f"[Native Audio] speak() called but ignored (model generates audio from video)",
                color="yellow",
            )

    def send_video_frame(self, frame_data: bytes, mime_type: str = "image/png") -> None:
        """Send a video frame to the model for processing."""
        if self._debug:
            termcolor.cprint(
                f"[Native Audio] Sending frame ({len(frame_data)} bytes)", color="cyan"
            )
        self._presenter.send_video_frame(frame_data, mime_type)

    def stop(self) -> None:
        """Stop the native audio presenter."""
        self._presenter.stop()

    # Coordination helper for BrowserAgent/Computer
    def wait_for_quiet(
        self,
        *,
        timeout_s: Optional[float] = None,
        quiet_s: Optional[float] = None,
        no_audio_timeout: Optional[float] = None,
    ) -> None:
        if not hasattr(self._presenter, "wait_for_quiet"):
            return

        effective_timeout = (
            timeout_s
            if timeout_s is not None
            else self._config.native_audio_wait_timeout
        )
        effective_quiet = (
            quiet_s if quiet_s is not None else self._config.native_audio_quiet_window
        )
        effective_no_audio = (
            no_audio_timeout
            if no_audio_timeout is not None
            else self._config.native_audio_no_response_timeout
        )

        self._presenter.wait_for_quiet(
            timeout_s=effective_timeout,
            quiet_s=effective_quiet,
            no_audio_timeout=effective_no_audio,
        )

    def _ensure_completion_instruction(self, text: str) -> str:
        reminder = (
            "\n\nCONTROL FLOW (do NOT speak): "
            "After narrating, call narration_complete then advance_slide tools."
        )
        if "narration_complete" in text or "CONTROL FLOW" in text:
            return text
        return text.strip() + reminder


def _create_synthesizer(
    config: SlideAudioConfig,
    token_callback: Optional[Callable[[int, int], None]] = None,
) -> SpeechSynthesizer:
    backend = config.backend.lower()
    if backend == "say":
        return _OSSaySpeechSynthesizer(config.voice, config.rate, config.debug)
    elif backend == "native-audio":
        return _GeminiNativeAudioSynthesizer(config, token_callback)
    raise SlideAudioError(f"Unsupported speech backend: {config.backend}")


class SlideAudioPresenter:
    """Continuously narrates the active slide inside a Playwright page."""

    def __init__(
        self,
        page: Page,
        config: SlideAudioConfig,
        status_callback: Optional[Callable[[str], None]] = None,
        token_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        self._page = page
        self._config = config
        self._config.validate()
        self._status_callback = status_callback or (
            lambda message: termcolor.cprint(message, color="green")
        )
        self._synthesizer = _create_synthesizer(config, token_callback)
        self._active = False
        self._last_hash: Optional[str] = None
        self._last_spoken_at: float = 0.0

    def start(self) -> None:
        if self._active:
            return
        if self._config.warmup_phrase:
            self._synthesizer.speak(self._config.warmup_phrase, interrupt=True)
        self._status_callback("Slide audio presenter activated.")
        self._active = True

    def stop(self) -> None:
        if not self._active:
            return
        self._active = False
        self._synthesizer.stop()

    def process(self, screenshot: Optional[bytes] = None) -> None:
        # DOM-based narration has been disabled intentionally.
        if self._config.debug and screenshot:
            termcolor.cprint(
                "[Narration] process() called but DOM narration is disabled.",
                color="magenta",
            )
        return None

    def ingest_external_text(
        self,
        *,
        text: str,
        source: str = "external",
        url: Optional[str] = None,
    ) -> None:
        if not self._active:
            return
        trimmed = (text or "").strip()
        if not trimmed:
            return
        payload = {
            "text": trimmed,
            "url": url or (self._page.url if not self._page.is_closed() else ""),
            "source": source,
        }
        if self._config.debug:
            excerpt = trimmed if len(trimmed) < 120 else trimmed[:117] + "..."
            termcolor.cprint(
                f"[Narration] ingest_external_text({source}): {excerpt}",
                color="blue",
            )
        self._maybe_speak_payload(payload, source=source)

    def _maybe_speak_payload(
        self, payload: Optional[dict[str, Any]], *, source: str
    ) -> bool:
        if not payload:
            return False
        if payload.get("text") is None:
            return False

        text = str(payload["text"]).strip()
        if not text:
            return False
        if self._config.debug:
            termcolor.cprint(f"[{source}] extracted {len(text)} chars", color="yellow")
        if len(text) < self._config.min_chars and source not in {"flash"}:
            if self._config.debug:
                termcolor.cprint(
                    f"[{source}] skipped: below min_chars ({len(text)} < {self._config.min_chars})",
                    color="magenta",
                )
            return False
        if len(text) > self._config.max_chars:
            text = text[: self._config.max_chars]

        digest_input = f"{payload.get('url', self._page.url)}::{text}"
        digest = hashlib.sha1(digest_input.encode("utf-8")).hexdigest()
        now = time.time()
        if (
            digest == self._last_hash
            and (now - self._last_spoken_at) < self._config.cooldown_seconds
        ):
            if self._config.debug:
                termcolor.cprint(
                    f"[{source}] skipped: within cooldown window.",
                    color="magenta",
                )
            return False

        self._last_hash = digest
        self._last_spoken_at = now
        self._synthesizer.speak(text, interrupt=True, wait=True)
        if self._config.debug:
            termcolor.cprint(f"[{source}] narrated slide", color="green")
        return True
