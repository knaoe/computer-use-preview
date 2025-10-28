# -*- coding: utf-8 -*-
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Native Audio Presenter using Gemini Live API.

This module provides real-time narration for browser presentations
by sending video frames to Gemini Native Audio models and playing
back the audio responses.
"""

import asyncio
import sys
import threading
import time
from collections import deque
from typing import Optional, Callable

import pyaudio
import termcolor
from google import genai
from google.genai import types

if sys.version_info < (3, 11, 0):
    import taskgroup, exceptiongroup
    asyncio.TaskGroup = taskgroup.TaskGroup
    asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup


FORMAT = pyaudio.paInt16
CHANNELS = 1
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

DEFAULT_MODEL = "gemini-2.5-flash-native-audio-preview-09-2025"
DEFAULT_SYSTEM_INSTRUCTION = """
You are a professional presenter giving a live presentation.

When you see a slide in presentation mode (full-screen), immediately speak out loud to narrate it.
Keep your narration VERY CONCISE (1-2 sentences maximum) focusing ONLY on the key message.

IMPORTANT: You MUST speak audio for every slide you see. Do not stay silent.
When you see a slide, always generate audio narration immediately.

Speak in Japanese in a natural, conversational style.

STRUCTURE your narration as:
1. Main point (1-2 sentences) - explain the key takeaway clearly and briefly
2. Transition phrase (brief) - end with a natural connector like "では、次に参りましょう" or "続けてご覧ください"

After you finish narrating (including the transition), you MUST call TWO tools in sequence:
1. First call `narration_complete` - signals that audio generation is complete
2. Then call `advance_slide` - signals that it's time to advance to the next slide

CRITICAL: Always call `advance_slide` after `narration_complete` to allow the presentation to proceed.
Keep it brief and engaging!
"""


class NativeAudioPresenter:
    """Manages Live API connection and audio streaming for presentations."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION,
        status_callback: Optional[Callable[[str], None]] = None,
        debug: bool = False,
        use_vertexai: bool = False,
        project: Optional[str] = None,
        location: Optional[str] = None,
        token_callback: Optional[Callable[[int, int], None]] = None,
    ):
        """Initialize the Native Audio Presenter.

        Args:
            api_key: Gemini API key (if not using Vertex AI)
            model: Model name to use
            system_instruction: System instruction for the model
            status_callback: Callback for status messages
            debug: Enable debug logging
            use_vertexai: Use Vertex AI instead of API key
            project: Vertex AI project
            location: Vertex AI location
            token_callback: Callback for token usage (input_tokens, output_tokens)
        """
        self._model = model
        self._system_instruction = system_instruction
        self._status_callback = status_callback or (
            lambda msg: termcolor.cprint(msg, color="green", attrs=["bold"])
        )
        self._debug = debug
        self._token_callback = token_callback

        # Initialize Gemini client
        self._client = genai.Client(
            api_key=api_key,
            http_options={"api_version": "v1alpha"},
            vertexai=use_vertexai,
            project=project,
            location=location,
        )

        # PyAudio setup
        self._pya = pyaudio.PyAudio()

        # Async components
        self._video_queue: Optional[asyncio.Queue] = None
        self._audio_queue: Optional[asyncio.Queue] = None
        self._session: Optional[genai.types.LiveSession] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._active = False
        self._last_audio_ts = 0.0
        self._pending_frames = 0
        self._pending_lock = threading.Lock()
        self._completion_event = threading.Event()
        self._completion_event.set()
        self._advance_event = threading.Event()
        self._advance_event.set()
        self._pending_frame_times: deque[float] = deque()

    def _debug_print(self, message: str, color: str = "cyan", attrs: Optional[list[str]] = None) -> None:
        if self._debug:
            termcolor.cprint(f"[Native Audio] {message}", color=color, attrs=attrs)

    def start(self) -> None:
        """Start the Live API session in a background thread."""
        if self._active:
            return

        self._active = True
        self._last_audio_ts = 0.0
        with self._pending_lock:
            self._pending_frames = 0
            self._completion_event.set()
            self._advance_event.set()
            self._pending_frame_times.clear()
        self._status_callback("Starting Gemini Native Audio presenter...")
        self._debug_print("Presenter starting; reset state", color="blue")

        # Create a new event loop for the background thread
        self._loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._loop_thread.start()

        # Wait a bit for the session to initialize
        time.sleep(2)
        self._status_callback("Native Audio presenter is ready.")

    def stop(self) -> None:
        """Stop the Live API session."""
        if not self._active:
            return

        self._active = False
        with self._pending_lock:
            self._pending_frames = 0
            self._completion_event.set()
            self._advance_event.set()
            self._pending_frame_times.clear()

        # Cancel all pending tasks gracefully
        if self._loop and self._loop.is_running():
            # Schedule shutdown in the event loop
            asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)

        if self._loop_thread:
            self._loop_thread.join(timeout=5)

        self._status_callback("Native Audio presenter stopped.")
        self._debug_print("Presenter stopped", color="blue")

    async def _shutdown(self) -> None:
        """Gracefully shutdown the async session."""
        # Give tasks a moment to complete
        await asyncio.sleep(0.5)

        # Stop the event loop
        if self._loop:
            self._loop.stop()

    def send_video_frame(self, frame_data: bytes, mime_type: str = "image/png") -> None:
        """Send a video frame to the model.

        Args:
            frame_data: Video frame data (PNG or JPEG)
            mime_type: MIME type of the frame
        """
        if not self._active or not self._loop or not self._video_queue:
            return

        # Track pending narration before enqueueing
        self._mark_frame_pending()
        self._debug_print(
            f"Queueing video frame ({len(frame_data)} bytes)",
            color="cyan"
        )

        # Schedule the coroutine in the background thread's event loop
        future = asyncio.run_coroutine_threadsafe(
            self._video_queue.put({"data": frame_data, "mime_type": mime_type}),
            self._loop
        )

        def _handle_put_result(fut: asyncio.Future) -> None:
            try:
                fut.result()
            except Exception:
                # If enqueue fails, release waiters
                self._mark_narration_complete(source="enqueue-failed")
                self._debug_print("Frame enqueue failed", color="red")

        future.add_done_callback(_handle_put_result)

    def _mark_frame_pending(self) -> None:
        with self._pending_lock:
            self._pending_frames += 1
            self._completion_event.clear()
            self._advance_event.clear()
            self._pending_frame_times.append(time.time())
            self._debug_print(
                f"Frame pending → count={self._pending_frames}",
                color="blue"
            )

    def _mark_narration_complete(self, *, source: str = "tool") -> None:
        with self._pending_lock:
            if self._pending_frames > 0:
                self._pending_frames -= 1
            if self._pending_frame_times:
                self._pending_frame_times.popleft()
            if self._pending_frames == 0:
                self._completion_event.set()
                # Auto-advance when narration completes (hybrid approach)
                self._advance_event.set()
        self._debug_print(
            f"Narration completion via {source}; pending={self._pending_frames}, auto-advancing",
            color="cyan"
        )

    def _run_event_loop(self) -> None:
        """Run the asyncio event loop in a background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._async_session())
        except Exception as e:
            self._debug_print(f"Event loop error: {e}", color="red")
        finally:
            self._loop.close()

    async def _async_session(self) -> None:
        """Main async session handler."""
        config = {
            "system_instruction": self._system_instruction,
            "response_modalities": ["AUDIO"],
            "speech_config": {
                "voice_config": {
                    "prebuilt_voice_config": {
                        "voice_name": "Puck"
                    }
                }
            },
            # Enable proactive audio generation - CRITICAL for audio output!
            "proactivity": {'proactive_audio': True},
            # Enable generation after each input
            "generation_config": {
                "temperature": 1.0,
                "candidate_count": 1,
            },
            "tools": [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name="narration_complete",
                            description="Signal that slide narration for the most recent frame is finished.",
                            parameters=types.Schema(
                                type="OBJECT",
                                properties={},
                                required=[],
                            ),
                        ),
                        types.FunctionDeclaration(
                            name="advance_slide",
                            description="Signal that the presenter is ready to advance to the next slide.",
                            parameters=types.Schema(
                                type="OBJECT",
                                properties={},
                                required=[],
                            ),
                        )
                    ]
                )
            ],
        }

        self._debug_print(
            f"Connecting to model: {self._model}",
            color="cyan"
        )

        try:
            async with (
                self._client.aio.live.connect(model=self._model, config=config) as session,
                asyncio.TaskGroup() as tg,
            ):
                self._session = session
                self._video_queue = asyncio.Queue(maxsize=5)
                self._audio_queue = asyncio.Queue()

                self._debug_print(
                    "Session connected, starting tasks...",
                    color="green"
                )

                tg.create_task(self._send_video_loop())
                tg.create_task(self._send_silence_loop())  # Keep connection alive
                tg.create_task(self._receive_audio_loop())
                tg.create_task(self._play_audio_loop())

        except asyncio.CancelledError:
            pass
        except Exception as e:
            self._debug_print(f"Session error: {e}", color="red")

    async def _send_silence_loop(self) -> None:
        """Send periodic silent audio to keep the connection active."""
        send_rate = 16000  # Audio send sample rate

        while self._active:
            try:
                await asyncio.sleep(2.0)  # Every 2 seconds

                if self._session:
                    # Send 100ms of silence (16000 Hz * 0.1s * 2 bytes)
                    silence = b'\x00' * (send_rate // 10 * 2)

                    await self._session.send_realtime_input(
                        audio={"data": silence, "mime_type": "audio/pcm"}
                    )

                    self._debug_print(
                        "Sent silence to keep connection alive",
                        color="blue"
                    )

            except Exception as e:
                self._debug_print(
                    f"Error sending silence: {e}",
                    color="yellow"
                )
                await asyncio.sleep(1.0)

    async def _send_video_loop(self) -> None:
        """Background task to send video frames to the model."""
        frame_count = 0
        while self._active:
            try:
                # Wait for video frame with timeout
                frame = await asyncio.wait_for(
                    self._video_queue.get(),
                    timeout=1.0
                )

                if self._session:
                    frame_count += 1

                    try:
                        # Try using send_client_content with image as content
                        content = types.Content(
                            role="user",
                            parts=[
                                types.Part.from_bytes(
                                    data=frame["data"],
                                    mime_type=frame["mime_type"]
                                ),
                                types.Part(text="Please narrate this slide in Japanese.")
                            ]
                        )

                        await self._session.send_client_content(
                            turns=content,
                            turn_complete=True
                        )

                        self._debug_print(
                            f"Sent video frame #{frame_count} via send_client_content ({len(frame['data'])} bytes)",
                            color="cyan"
                        )

                    except Exception as e:
                        # Fallback to send_realtime_input
                        self._debug_print(
                            f"send_client_content failed, trying send_realtime_input: {e}",
                            color="yellow"
                        )

                        await self._session.send_realtime_input(
                            media=frame
                        )

                        self._debug_print(
                            f"Sent video frame #{frame_count} via send_realtime_input ({len(frame['data'])} bytes)",
                            color="cyan"
                        )

            except asyncio.TimeoutError:
                # No frame available, continue
                continue
            except Exception as e:
                self._debug_print(
                    f"Error sending video: {e}",
                    color="yellow"
                )
                self._mark_narration_complete(source="send-error")

    async def _receive_audio_loop(self) -> None:
        """Background task to receive audio from the model."""
        turn_count = 0
        last_log_time = time.time()

        while self._active:
            try:
                if not self._session:
                    await asyncio.sleep(0.1)
                    continue

                turn_count += 1

                # Log periodically even if no response
                current_time = time.time()
                if self._debug and (current_time - last_log_time) > 5.0:
                    self._debug_print(
                        f"Still waiting... (turn #{turn_count}, session active: {self._session is not None})",
                        color="yellow"
                    )
                    last_log_time = current_time

                self._debug_print(
                    f"Calling session.receive() for turn #{turn_count}...",
                    color="cyan"
                )

                turn = self._session.receive()

                self._debug_print(
                    "Got turn object, iterating responses...",
                    color="cyan"
                )

                response_count = 0
                chunk_count = 0
                async for response in turn:
                    if not self._active:
                        break

                    response_count += 1
                    if self._debug:
                        # Only print detailed logs for the first few responses and then periodically
                        log_details = response_count <= 3 or response_count % 50 == 0
                        if log_details:
                            self._debug_print(
                                f"Turn #{turn_count}, Response #{response_count}: {type(response).__name__}",
                                color="green"
                            )
                            if hasattr(response, '__dict__'):
                                attrs = {k: type(v).__name__ for k, v in response.__dict__.items()}
                                self._debug_print(
                                    f"Response attributes: {attrs}",
                                    color="cyan"
                                )

                    if data := response.data:
                        chunk_count += 1
                        await self._audio_queue.put(data)

                        if self._debug and (chunk_count == 1 or chunk_count % 20 == 0):
                            self._debug_print(
                                f"Audio chunk #{chunk_count} ({len(data)} bytes)",
                                color="green",
                                attrs=["bold"]
                            )
                        # Record last time we received audio
                        self._last_audio_ts = time.time()
                        # Once audio starts playing, reset completion waiters
                        with self._pending_lock:
                            if self._pending_frames == 0:
                                # Ensure waiting threads do not block needlessly
                                self._completion_event.set()

                    if self._debug and getattr(response, "server_content", None):
                        parts = getattr(response.server_content, "parts", None)
                        if parts:
                            text_parts = [getattr(part, "text", None) for part in parts]
                            text_parts = [t for t in text_parts if t]
                            if text_parts:
                                self._debug_print(
                                    f"Model Text: {' '.join(text_parts)}",
                                    color="magenta",
                                    attrs=["bold"]
                                )

                    if response.tool_call and response.tool_call.function_calls:
                        for function_call in response.tool_call.function_calls:
                            self._debug_print(
                                f"Tool call received: {function_call.name}",
                                color="blue"
                            )
                            if function_call.name == "narration_complete":
                                self._mark_narration_complete(source="tool-call")
                                try:
                                    await self._session.send_tool_response(
                                        function_responses=types.FunctionResponse(
                                            name=function_call.name,
                                            response={"status": "ack"},
                                            id=function_call.id,
                                        )
                                    )
                                except Exception as exc:  # noqa: BLE001
                                    self._debug_print(
                                        f"Failed to send tool response: {exc}",
                                        color="red"
                                    )
                            elif function_call.name == "advance_slide":
                                self._advance_event.set()
                                self._debug_print(
                                    "advance_slide received → signaling ready to advance",
                                    color="green",
                                    attrs=["bold"]
                                )
                                try:
                                    await self._session.send_tool_response(
                                        function_responses=types.FunctionResponse(
                                            name=function_call.name,
                                            response={"status": "ack"},
                                            id=function_call.id,
                                        )
                                    )
                                except Exception as exc:  # noqa: BLE001
                                    self._debug_print(
                                        f"Failed to send tool response: {exc}",
                                        color="red"
                                    )

                self._debug_print(
                    f"Turn #{turn_count} complete ({response_count} responses, {chunk_count} chunks)",
                    color="yellow"
                )

                # Clear audio queue on interruption
                while not self._audio_queue.empty():
                    try:
                        self._audio_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._active:
                    self._debug_print(
                        f"ERROR in receive loop: {e}",
                        color="red",
                        attrs=["bold"]
                    )
                    import traceback
                    traceback.print_exc()
                if self._active:
                    await asyncio.sleep(1.0)
                else:
                    break

    async def _play_audio_loop(self) -> None:
        """Background task to play audio through speakers."""
        stream = None
        try:
            stream = await asyncio.to_thread(
                self._pya.open,
                format=FORMAT,
                channels=CHANNELS,
                rate=RECEIVE_SAMPLE_RATE,
                output=True,
            )

            while self._active:
                try:
                    # Wait for audio with timeout
                    bytestream = await asyncio.wait_for(
                        self._audio_queue.get(),
                        timeout=1.0
                    )
                    await asyncio.to_thread(stream.write, bytestream)

                except asyncio.TimeoutError:
                    # No audio available, continue
                    continue
                except asyncio.CancelledError:
                    break

        finally:
            if stream:
                try:
                    await asyncio.to_thread(stream.close)
                except Exception:
                    pass  # Ignore errors during cleanup

    def wait_for_quiet(
        self,
        *,
        timeout_s: float = 8.0,
        quiet_s: float = 0.6,
        no_audio_timeout: float = 6.0,
    ) -> None:
        """Block until narration completes AND advance_slide is signaled.

        This implements two-stage synchronization:
        1. Wait for narration_complete (audio generation done)
        2. Wait for advance_slide (permission to proceed)

        This is called from a synchronous thread (the Playwright/agent loop).
        """
        start_wait = time.time()
        with self._pending_lock:
            pending_snapshot = self._pending_frames
            first_frame_time = self._pending_frame_times[0] if self._pending_frame_times else None
        self._debug_print(
            f"wait_for_quiet start: pending={pending_snapshot}, last_audio={self._last_audio_ts:.3f}, timeout={timeout_s}, quiet={quiet_s}, no_audio={no_audio_timeout}",
            color="blue"
        )
        deadline = time.time() + timeout_s

        # STAGE 1: Wait for narration_complete
        event_fired = False
        self._debug_print("STAGE 1: Waiting for narration_complete...", color="cyan")

        while time.time() < deadline:
            now = time.time()
            if first_frame_time and no_audio_timeout > 0:
                last_audio_ts = self._last_audio_ts
                if last_audio_ts <= first_frame_time and (now - first_frame_time) >= no_audio_timeout:
                    self._debug_print(
                        f"No audio within {no_audio_timeout:.2f}s of frame; marking completion",
                        color="yellow"
                    )
                    self._mark_narration_complete(source="no-audio")
                    # Also auto-advance if no audio detected
                    self._advance_event.set()
                    if self._completion_event.is_set():
                        event_fired = True
                        break
                    with self._pending_lock:
                        first_frame_time = self._pending_frame_times[0] if self._pending_frame_times else None
                    continue

            remaining = max(0.0, deadline - time.time())
            if self._completion_event.wait(timeout=min(0.1, remaining)):
                event_fired = True
                break

            with self._pending_lock:
                first_frame_time = self._pending_frame_times[0] if self._pending_frame_times else None

        if not event_fired:
            # Timed out waiting entirely; release any pending waiters so the loop can proceed
            self._mark_narration_complete(source="timeout")
            self._advance_event.set()  # Auto-advance on timeout
            self._debug_print(
                f"wait_for_quiet STAGE 1 timeout after {time.time() - start_wait:.2f}s",
                color="red"
            )
            return

        # narration_complete received. Wait briefly for residual audio to clear.
        self._debug_print(
            f"STAGE 1 complete: narration_complete received after {time.time() - start_wait:.2f}s",
            color="green"
        )
        end_deadline = time.time() + quiet_s
        while time.time() < end_deadline:
            last_ts = self._last_audio_ts
            if last_ts > 0 and (time.time() - last_ts) >= quiet_s:
                break
            time.sleep(0.05)

        # STAGE 2: Wait for advance_slide signal
        self._debug_print("STAGE 2: Waiting for advance_slide...", color="cyan")
        stage2_deadline = time.time() + 2.0  # 2 second timeout for stage 2

        if not self._advance_event.wait(timeout=max(0, stage2_deadline - time.time())):
            # Timeout waiting for advance_slide, auto-advance anyway
            self._debug_print(
                f"STAGE 2 timeout waiting for advance_slide after {time.time() - start_wait:.2f}s; auto-advancing",
                color="yellow"
            )
            self._advance_event.set()

        self._debug_print(
            f"wait_for_quiet complete in {time.time() - start_wait:.2f}s; pending={self._pending_frames}",
            color="blue",
            attrs=["bold"]
        )
