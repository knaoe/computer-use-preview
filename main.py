# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
from agent import BrowserAgent
from computers import BrowserbaseComputer, PlaywrightComputer
from presentation.slide_audio import SlideAudioConfig


PLAYWRIGHT_SCREEN_SIZE = (1440, 900)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the browser agent with a query.")
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="The query for the browser agent to execute.",
    )

    parser.add_argument(
        "--env",
        type=str,
        choices=("playwright", "browserbase"),
        default="playwright",
        help="The computer use environment to use.",
    )
    parser.add_argument(
        "--initial_url",
        type=str,
        default="https://www.google.com",
        help="The inital URL loaded for the computer.",
    )
    parser.add_argument(
        "--highlight_mouse",
        action="store_true",
        default=False,
        help="If possible, highlight the location of the mouse.",
    )
    parser.add_argument(
        "--model",
        default='gemini-2.5-computer-use-preview-10-2025',
        help="Set which main model to use.",
    )
    parser.add_argument(
        "--slide-audio",
        action="store_true",
        default=False,
        help="Enable real-time narration for presentation slides.",
    )
    parser.add_argument(
        "--slide-audio-backend",
        type=str,
        choices=("say",),
        default="say",
        help="Speech backend to use for slide narration.",
    )
    parser.add_argument(
        "--slide-audio-voice",
        type=str,
        default=None,
        help="Optional voice override for slide narration (backend dependent).",
    )
    parser.add_argument(
        "--slide-audio-rate",
        type=int,
        default=None,
        help="Optional speech rate override for slide narration.",
    )
    parser.add_argument(
        "--slide-audio-cooldown",
        type=float,
        default=2.0,
        help="Minimum seconds between narrations of the same slide.",
    )
    parser.add_argument(
        "--slide-audio-warmup",
        type=str,
        default=None,
        help="Optional phrase to play immediately after narration initializes.",
    )
    parser.add_argument(
        "--slide-audio-debug",
        action="store_true",
        default=False,
        help="Enable verbose logging for slide narration.",
    )
    args = parser.parse_args()

    slide_audio_config = None
    if args.slide_audio:
        config_kwargs = dict(
            enabled=True,
            backend=args.slide_audio_backend,
            voice=args.slide_audio_voice,
            rate=args.slide_audio_rate,
            debug=args.slide_audio_debug,
            cooldown_seconds=args.slide_audio_cooldown,
        )
        if args.slide_audio_warmup is not None:
            config_kwargs["warmup_phrase"] = args.slide_audio_warmup
        slide_audio_config = SlideAudioConfig(**config_kwargs)

    if args.env == "playwright":
        env = PlaywrightComputer(
            screen_size=PLAYWRIGHT_SCREEN_SIZE,
            initial_url=args.initial_url,
            highlight_mouse=args.highlight_mouse,
            slide_audio_config=slide_audio_config,
        )
    elif args.env == "browserbase":
        env = BrowserbaseComputer(
            screen_size=PLAYWRIGHT_SCREEN_SIZE,
            initial_url=args.initial_url,
            slide_audio_config=slide_audio_config,
        )
    else:
        raise ValueError("Unknown environment: ", args.env)

    with env as browser_computer:
        agent = BrowserAgent(
            browser_computer=browser_computer,
            query=args.query,
            model_name=args.model,
        )
        agent.agent_loop()
    return 0


if __name__ == "__main__":
    main()
