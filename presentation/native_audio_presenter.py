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
from typing import Optional, Callable

import pyaudio
import termcolor
from google import genai

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
Keep your narration concise (2-4 sentences) and engaging.

IMPORTANT: You MUST speak audio for every slide you see. Do not stay silent.
When you see a slide, always generate audio narration immediately.

Speak in Japanese in a natural, conversational style.
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

    def start(self) -> None:
        """Start the Live API session in a background thread."""
        if self._active:
            return

        self._active = True
        self._status_callback("Starting Gemini Native Audio presenter...")

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

        # Cancel all pending tasks gracefully
        if self._loop and self._loop.is_running():
            # Schedule shutdown in the event loop
            asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)

        if self._loop_thread:
            self._loop_thread.join(timeout=5)

        self._status_callback("Native Audio presenter stopped.")

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

        # Schedule the coroutine in the background thread's event loop
        asyncio.run_coroutine_threadsafe(
            self._video_queue.put({"data": frame_data, "mime_type": mime_type}),
            self._loop
        )

    def _run_event_loop(self) -> None:
        """Run the asyncio event loop in a background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._async_session())
        except Exception as e:
            if self._debug:
                termcolor.cprint(
                    f"Event loop error: {e}",
                    color="red"
                )
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
            }
        }

        if self._debug:
            termcolor.cprint(
                f"[Native Audio] Connecting to model: {self._model}",
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

                if self._debug:
                    termcolor.cprint(
                        "[Native Audio] Session connected, starting tasks...",
                        color="green"
                    )

                tg.create_task(self._send_video_loop())
                tg.create_task(self._send_silence_loop())  # Keep connection alive
                tg.create_task(self._receive_audio_loop())
                tg.create_task(self._play_audio_loop())

        except asyncio.CancelledError:
            pass
        except Exception as e:
            if self._debug:
                termcolor.cprint(
                    f"Session error: {e}",
                    color="red"
                )

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

                    if self._debug:
                        termcolor.cprint(
                            "[Native Audio] Sent silence to keep connection alive",
                            color="blue"
                        )

            except Exception as e:
                if self._debug:
                    termcolor.cprint(
                        f"[Native Audio] Error sending silence: {e}",
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
                        from google.genai import types

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

                        if self._debug:
                            termcolor.cprint(
                                f"[Native Audio] Sent video frame #{frame_count} via send_client_content ({len(frame['data'])} bytes)",
                                color="cyan"
                            )

                    except Exception as e:
                        # Fallback to send_realtime_input
                        if self._debug:
                            termcolor.cprint(
                                f"[Native Audio] send_client_content failed, trying send_realtime_input: {e}",
                                color="yellow"
                            )

                        await self._session.send_realtime_input(
                            media=frame
                        )

                        if self._debug:
                            termcolor.cprint(
                                f"[Native Audio] Sent video frame #{frame_count} via send_realtime_input ({len(frame['data'])} bytes)",
                                color="cyan"
                            )

            except asyncio.TimeoutError:
                # No frame available, continue
                continue
            except Exception as e:
                if self._debug:
                    termcolor.cprint(
                        f"[Native Audio] Error sending video: {e}",
                        color="yellow"
                    )

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
                    termcolor.cprint(
                        f"[Native Audio] Still waiting... (turn #{turn_count}, session active: {self._session is not None})",
                        color="yellow"
                    )
                    last_log_time = current_time

                if self._debug:
                    termcolor.cprint(
                        f"[Native Audio] Calling session.receive() for turn #{turn_count}...",
                        color="cyan"
                    )

                turn = self._session.receive()

                if self._debug:
                    termcolor.cprint(
                        f"[Native Audio] Got turn object, iterating responses...",
                        color="cyan"
                    )

                response_count = 0
                async for response in turn:
                    if not self._active:
                        break

                    response_count += 1
                    if self._debug:
                        termcolor.cprint(
                            f"[Native Audio] Turn #{turn_count}, Response #{response_count}: {type(response).__name__}",
                            color="green"
                        )
                        # Log all response attributes
                        if hasattr(response, '__dict__'):
                            attrs = {k: type(v).__name__ for k, v in response.__dict__.items()}
                            termcolor.cprint(
                                f"[Native Audio] Response attributes: {attrs}",
                                color="cyan"
                            )

                    if data := response.data:
                        await self._audio_queue.put(data)

                        termcolor.cprint(
                            f"[Native Audio] ✓ Received audio chunk ({len(data)} bytes)",
                            color="green",
                            attrs=["bold"]
                        )

                    if text := response.text:
                        termcolor.cprint(f"[Model Text]: {text}", color="magenta", attrs=["bold"])

                if self._debug:
                    termcolor.cprint(
                        f"[Native Audio] Turn #{turn_count} complete, received {response_count} responses",
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
                    termcolor.cprint(
                        f"[Native Audio] ERROR in receive loop: {e}",
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
