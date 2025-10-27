import dataclasses
import hashlib
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

    def speak(self, text: str, *, interrupt: bool = True, wait: bool = False) -> None: ...

    def stop(self) -> None: ...


@dataclasses.dataclass(slots=True)
class SlideAudioConfig:
    """Configuration for slide narration."""

    enabled: bool = False
    backend: str = "say"
    voice: Optional[str] = None
    rate: Optional[int] = None
    min_chars: int = 1
    max_chars: int = 1200
    warmup_phrase: Optional[str] = "Slide audio presenter is ready."
    debug: bool = False
    cooldown_seconds: float = 2.0

    def validate(self) -> None:
        if self.min_chars <= 0:
            raise ValueError("min_chars must be positive")
        if self.max_chars <= self.min_chars:
            raise ValueError("max_chars must be greater than min_chars")
        if self.cooldown_seconds < 0:
            raise ValueError("cooldown_seconds cannot be negative")
        if not self.backend:
            raise ValueError("backend must be provided")


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


def _create_synthesizer(config: SlideAudioConfig) -> SpeechSynthesizer:
    backend = config.backend.lower()
    if backend == "say":
        return _OSSaySpeechSynthesizer(config.voice, config.rate, config.debug)
    raise SlideAudioError(f"Unsupported speech backend: {config.backend}")


class SlideAudioPresenter:
    """Continuously narrates the active slide inside a Playwright page."""

    def __init__(
        self,
        page: Page,
        config: SlideAudioConfig,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._page = page
        self._config = config
        self._config.validate()
        self._status_callback = status_callback or (lambda message: termcolor.cprint(message, color="green"))
        self._synthesizer = _create_synthesizer(config)
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

    def _maybe_speak_payload(self, payload: Optional[dict[str, Any]], *, source: str) -> bool:
        if not payload:
            return False
        if payload.get("text") is None:
            return False

        text = str(payload["text"]).strip()
        if not text:
            return False
        if self._config.debug:
            termcolor.cprint(
                f"[{source}] extracted {len(text)} chars", color="yellow"
            )
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
        if digest == self._last_hash and (now - self._last_spoken_at) < self._config.cooldown_seconds:
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
