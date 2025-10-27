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
import hashlib
import json
import os
from typing import Literal, Optional, Union, Any
from google import genai
from google.genai import types
import termcolor
from PIL import Image
import io
from google.genai.types import (
    Part,
    GenerateContentConfig,
    Content,
    Candidate,
    FunctionResponse,
    FinishReason,
)
import time
from rich.console import Console
from rich.table import Table

from computers import EnvState, Computer

MAX_RECENT_TURN_WITH_SCREENSHOTS = 3
PREDEFINED_COMPUTER_USE_FUNCTIONS = [
    "open_web_browser",
    "click_at",
    "hover_at",
    "type_text_at",
    "scroll_document",
    "scroll_at",
    "wait_5_seconds",
    "go_back",
    "go_forward",
    "search",
    "navigate",
    "key_combination",
    "drag_and_drop",
]


console = Console()

# Built-in Computer Use tools will return "EnvState".
# Custom provided functions will return "dict".
FunctionResponseT = Union[EnvState, dict]

# Maximum width for screenshots sent to models (to reduce token cost)
MAX_SCREENSHOT_WIDTH = 800


def resize_screenshot(screenshot_bytes: bytes, max_width: int = MAX_SCREENSHOT_WIDTH) -> bytes:
    """Resize screenshot to reduce token cost while maintaining aspect ratio."""
    img = Image.open(io.BytesIO(screenshot_bytes))
    original_width, original_height = img.size

    # Only resize if image is wider than max_width
    if original_width <= max_width:
        return screenshot_bytes

    # Calculate new height maintaining aspect ratio
    aspect_ratio = original_height / original_width
    new_width = max_width
    new_height = int(new_width * aspect_ratio)

    # Resize using high-quality Lanczos filter
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Convert back to bytes
    output = io.BytesIO()
    img_resized.save(output, format='PNG')
    return output.getvalue()


def multiply_numbers(x: float, y: float) -> dict:
    """Multiplies two numbers."""
    return {"result": x * y}


class BrowserAgent:
    def __init__(
        self,
        browser_computer: Computer,
        query: str,
        model_name: str,
        verbose: bool = True,
    ):
        self._browser_computer = browser_computer
        self._query = query
        self._model_name = model_name
        self._verbose = verbose
        self.final_reasoning = None
        self._client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY"),
            vertexai=os.environ.get("USE_VERTEXAI", "0").lower() in ["true", "1"],
            project=os.environ.get("VERTEXAI_PROJECT"),
            location=os.environ.get("VERTEXAI_LOCATION"),
        )
        self._contents: list[Content] = [
            Content(
                role="user",
                parts=[
                    Part(text=self._query),
                ],
            )
        ]

        # Token usage tracking
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cost = 0.0

        # Pricing for Gemini models (USD per 1M tokens)
        # Adjust these values based on your model pricing
        self._input_token_price = 1.25  # $1.25 per 1M input tokens
        self._output_token_price = 5.0  # $5.00 per 1M output tokens

        # Exclude any predefined functions here.
        excluded_predefined_functions = []

        # Add your own custom functions here.
        custom_functions = [
            # For example:
            types.FunctionDeclaration.from_callable(
                client=self._client, callable=multiply_numbers
            )
        ]

        self._generate_content_config = GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
            tools=[
                types.Tool(
                    computer_use=types.ComputerUse(
                        environment=types.Environment.ENVIRONMENT_BROWSER,
                        excluded_predefined_functions=excluded_predefined_functions,
                    ),
                ),
                types.Tool(function_declarations=custom_functions),
            ],
        )
        self._flash_narration_model = os.environ.get(
            "FLASH_NARRATION_MODEL", "gemini-2.5-flash"
        )
        self._narration_cache: set[str] = set()

    def handle_action(self, action: types.FunctionCall) -> FunctionResponseT:
        """Handles the action and returns the environment state."""
        if action.name == "open_web_browser":
            return self._browser_computer.open_web_browser()
        elif action.name == "click_at":
            x = self.denormalize_x(action.args["x"])
            y = self.denormalize_y(action.args["y"])
            return self._browser_computer.click_at(
                x=x,
                y=y,
            )
        elif action.name == "hover_at":
            x = self.denormalize_x(action.args["x"])
            y = self.denormalize_y(action.args["y"])
            return self._browser_computer.hover_at(
                x=x,
                y=y,
            )
        elif action.name == "type_text_at":
            x = self.denormalize_x(action.args["x"])
            y = self.denormalize_y(action.args["y"])
            press_enter = action.args.get("press_enter", False)
            clear_before_typing = action.args.get("clear_before_typing", True)
            return self._browser_computer.type_text_at(
                x=x,
                y=y,
                text=action.args["text"],
                press_enter=press_enter,
                clear_before_typing=clear_before_typing,
            )
        elif action.name == "scroll_document":
            return self._browser_computer.scroll_document(action.args["direction"])
        elif action.name == "scroll_at":
            x = self.denormalize_x(action.args["x"])
            y = self.denormalize_y(action.args["y"])
            magnitude = action.args.get("magnitude", 800)
            direction = action.args["direction"]

            if direction in ("up", "down"):
                magnitude = self.denormalize_y(magnitude)
            elif direction in ("left", "right"):
                magnitude = self.denormalize_x(magnitude)
            else:
                raise ValueError("Unknown direction: ", direction)
            return self._browser_computer.scroll_at(
                x=x, y=y, direction=direction, magnitude=magnitude
            )
        elif action.name == "wait_5_seconds":
            return self._browser_computer.wait_5_seconds()
        elif action.name == "go_back":
            return self._browser_computer.go_back()
        elif action.name == "go_forward":
            return self._browser_computer.go_forward()
        elif action.name == "search":
            return self._browser_computer.search()
        elif action.name == "navigate":
            return self._browser_computer.navigate(action.args["url"])
        elif action.name == "key_combination":
            return self._browser_computer.key_combination(
                action.args["keys"].split("+")
            )
        elif action.name == "drag_and_drop":
            x = self.denormalize_x(action.args["x"])
            y = self.denormalize_y(action.args["y"])
            destination_x = self.denormalize_x(action.args["destination_x"])
            destination_y = self.denormalize_y(action.args["destination_y"])
            return self._browser_computer.drag_and_drop(
                x=x,
                y=y,
                destination_x=destination_x,
                destination_y=destination_y,
            )
        # Handle the custom function declarations here.
        elif action.name == multiply_numbers.__name__:
            return multiply_numbers(x=action.args["x"], y=action.args["y"])
        else:
            raise ValueError(f"Unsupported function: {action}")

    def _log_token_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Log token usage and cost for this turn and cumulative totals."""
        # Calculate cost for this turn
        turn_cost = (input_tokens * self._input_token_price / 1_000_000) + \
                    (output_tokens * self._output_token_price / 1_000_000)

        # Update cumulative totals
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        self._total_cost += turn_cost

        # Create a table for token usage display
        token_table = Table(show_header=True, header_style="bold yellow", expand=True)
        token_table.add_column("Metric", style="cyan")
        token_table.add_column("This Turn", justify="right", style="green")
        token_table.add_column("Cumulative", justify="right", style="magenta")

        token_table.add_row(
            "Input Tokens",
            f"{input_tokens:,}",
            f"{self._total_input_tokens:,}"
        )
        token_table.add_row(
            "Output Tokens",
            f"{output_tokens:,}",
            f"{self._total_output_tokens:,}"
        )
        token_table.add_row(
            "Total Tokens",
            f"{input_tokens + output_tokens:,}",
            f"{self._total_input_tokens + self._total_output_tokens:,}"
        )
        token_table.add_row(
            "Cost (USD)",
            f"${turn_cost:.6f}",
            f"${self._total_cost:.6f}"
        )

        if self._verbose:
            console.print(token_table)
            print()

    def get_model_response(
        self, max_retries=5, base_delay_s=1
    ) -> types.GenerateContentResponse:
        for attempt in range(max_retries):
            try:
                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=self._contents,
                    config=self._generate_content_config,
                )

                # Log token usage if available
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    usage = response.usage_metadata
                    input_tokens = getattr(usage, 'prompt_token_count', 0)
                    output_tokens = getattr(usage, 'candidates_token_count', 0)
                    self._log_token_usage(input_tokens, output_tokens)

                return response  # Return response on success
            except Exception as e:
                print(e)
                if attempt < max_retries - 1:
                    delay = base_delay_s * (2**attempt)
                    message = (
                        f"Generating content failed on attempt {attempt + 1}. "
                        f"Retrying in {delay} seconds...\n"
                    )
                    termcolor.cprint(
                        message,
                        color="yellow",
                    )
                    time.sleep(delay)
                else:
                    termcolor.cprint(
                        f"Generating content failed after {max_retries} attempts.\n",
                        color="red",
                    )
                    raise

    def get_text(self, candidate: Candidate) -> Optional[str]:
        """Extracts the text from the candidate."""
        if not candidate.content or not candidate.content.parts:
            return None
        text = []
        for part in candidate.content.parts:
            if part.text:
                text.append(part.text)
        return " ".join(text) or None

    def extract_function_calls(self, candidate: Candidate) -> list[types.FunctionCall]:
        """Extracts the function call from the candidate."""
        if not candidate.content or not candidate.content.parts:
            return []
        ret = []
        for part in candidate.content.parts:
            if part.function_call:
                ret.append(part.function_call)
        return ret

    def run_one_iteration(self) -> Literal["COMPLETE", "CONTINUE"]:
        # Generate a response from the model.
        if self._verbose:
            with console.status(
                "Generating response from Gemini Computer Use...", spinner_style=None
            ):
                try:
                    response = self.get_model_response()
                except Exception as e:
                    return "COMPLETE"
        else:
            try:
                response = self.get_model_response()
            except Exception as e:
                return "COMPLETE"

        if not response.candidates:
            print("Response has no candidates!")
            print(response)
            raise ValueError("Empty response")

        # Extract the text and function call from the response.
        candidate = response.candidates[0]
        # Append the model turn to conversation history.
        if candidate.content:
            self._contents.append(candidate.content)

        reasoning = self.get_text(candidate)
        function_calls = self.extract_function_calls(candidate)

        # Retry the request in case of malformed FCs.
        if (
            not function_calls
            and not reasoning
            and candidate.finish_reason == FinishReason.MALFORMED_FUNCTION_CALL
        ):
            return "CONTINUE"

        if not function_calls:
            print(f"Agent Loop Complete: {reasoning}")
            self.final_reasoning = reasoning
            return "COMPLETE"

        function_call_strs = []
        for function_call in function_calls:
            # Print the function call and any reasoning.
            function_call_str = f"Name: {function_call.name}"
            if function_call.args:
                function_call_str += f"\nArgs:"
                for key, value in function_call.args.items():
                    function_call_str += f"\n  {key}: {value}"
            function_call_strs.append(function_call_str)

        table = Table(expand=True)
        table.add_column(
            "Gemini Computer Use Reasoning", header_style="magenta", ratio=1
        )
        table.add_column("Function Call(s)", header_style="cyan", ratio=1)
        table.add_row(reasoning, "\n".join(function_call_strs))
        if self._verbose:
            console.print(table)
            print()

        function_responses = []
        for function_call in function_calls:
            extra_fr_fields = {}
            if function_call.args and (
                safety := function_call.args.get("safety_decision")
            ):
                decision = self._get_safety_confirmation(safety)
                if decision == "TERMINATE":
                    print("Terminating agent loop")
                    return "COMPLETE"
                # Explicitly mark the safety check as acknowledged.
                extra_fr_fields["safety_acknowledgement"] = "true"
            if self._verbose:
                with console.status(
                    "Sending command to Computer...", spinner_style=None
                ):
                    fc_result = self.handle_action(function_call)
            else:
                fc_result = self.handle_action(function_call)
            if isinstance(fc_result, EnvState):
                self._maybe_request_flash_narration(
                    env_state=fc_result,
                    function_call=function_call,
                    reasoning=reasoning,
                )
                # Resize screenshot before sending to Computer Use model
                resized_screenshot = resize_screenshot(fc_result.screenshot)
                function_responses.append(
                    FunctionResponse(
                        name=function_call.name,
                        response={
                            "url": fc_result.url,
                            **extra_fr_fields,
                        },
                        parts=[
                            types.FunctionResponsePart(
                                inline_data=types.FunctionResponseBlob(
                                    mime_type="image/png", data=resized_screenshot
                                )
                            )
                        ],
                    )
                )
            elif isinstance(fc_result, dict):
                function_responses.append(
                    FunctionResponse(name=function_call.name, response=fc_result)
                )

        self._contents.append(
            Content(
                role="user",
                parts=[Part(function_response=fr) for fr in function_responses],
            )
        )

        # only keep screenshots in the few most recent turns, remove the screenshot images from the old turns.
        turn_with_screenshots_found = 0
        for content in reversed(self._contents):
            if content.role == "user" and content.parts:
                # check if content has screenshot of the predefined computer use functions.
                has_screenshot = False
                for part in content.parts:
                    if (
                        part.function_response
                        and part.function_response.parts
                        and part.function_response.name
                        in PREDEFINED_COMPUTER_USE_FUNCTIONS
                    ):
                        has_screenshot = True
                        break

                if has_screenshot:
                    turn_with_screenshots_found += 1
                    # remove the screenshot image if the number of screenshots exceed the limit.
                    if turn_with_screenshots_found > MAX_RECENT_TURN_WITH_SCREENSHOTS:
                        for part in content.parts:
                            if (
                                part.function_response
                                and part.function_response.parts
                                and part.function_response.name
                                in PREDEFINED_COMPUTER_USE_FUNCTIONS
                            ):
                                part.function_response.parts = None

        return "CONTINUE"

    def _maybe_request_flash_narration(
        self,
        env_state: EnvState,
        function_call: types.FunctionCall,
        reasoning: Optional[str],
    ) -> None:
        if (
            not env_state
            or not env_state.screenshot
            or not self._flash_narration_model
        ):
            return

        digest = hashlib.sha1(env_state.screenshot).hexdigest()
        if digest in self._narration_cache:
            if self._verbose:
                termcolor.cprint(
                    f"[Narration] Skipping cached screenshot for {function_call.name}.",
                    color="cyan",
                )
            return

        if self._verbose:
            termcolor.cprint(
                f"[Narration] Evaluating screenshot for {function_call.name}.",
                color="cyan",
            )

        prompt_lines = [
            "You are a professional presenter delivering a live presentation to an engaged audience.",
            "IMPORTANT: Only narrate when viewing an ACTIVE SLIDESHOW in presentation mode (full-screen slides).",
            "DO NOT narrate for:",
            "- Slide editing interfaces or thumbnails",
            "- Browser navigation pages (Google homepage, search results, etc.)",
            "- Loading screens or generic UI",
            "- Google Slides interface BEFORE entering slideshow mode",
            "- Menu bars, toolbars, or any non-presentation content",
            "",
            "When you DO narrate (only in active slideshow mode):",
            "- Open with an engaging hook or transition that connects to the presentation flow",
            "- Explain the key concept in your own words with context and meaning",
            "- Use natural, conversational Japanese as if speaking directly to the audience",
            "- Add brief examples or implications when it helps understanding",
            "- Keep it concise (2-4 sentences) but impactful",
            "",
            'Respond ONLY with JSON: {"should_narrate": true|false, "script": "..."}.',
            "Leave script empty when should_narrate is false.",
        ]

        context_lines: list[str] = []
        if self._query:
            context_lines.append(f"Original task: {self._query}")
        context_lines.append(f"Latest function: {function_call.name}")
        prompt_lines.append("Context:\n" + "\n".join(context_lines))

        # Resize screenshot before sending to flash model
        resized_screenshot = resize_screenshot(env_state.screenshot)

        # Log image resolution for debugging
        if self._verbose:
            try:
                original_img = Image.open(io.BytesIO(env_state.screenshot))
                resized_img = Image.open(io.BytesIO(resized_screenshot))
                termcolor.cprint(
                    f"[Narration] Screenshot resolution: {original_img.size[0]}x{original_img.size[1]} → {resized_img.size[0]}x{resized_img.size[1]} pixels",
                    color="cyan",
                )
            except Exception:
                pass

        parts = [
            Part(text="\n".join(prompt_lines)),
            Part.from_bytes(mime_type="image/png", data=resized_screenshot),
        ]
        try:
            response = self._client.models.generate_content(
                model=self._flash_narration_model,
                contents=[Content(role="user", parts=parts)],
                config=GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=2000,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            if self._verbose:
                termcolor.cprint(
                    f"Flash narration request failed: {exc}",
                    color="yellow",
                )
            return

        text = None
        if response.candidates:
            text = self.get_text(response.candidates[0])
        if self._verbose:
            termcolor.cprint(
                f"[Narration] flash raw response: {text}",
                color="cyan",
            )
        if not text:
            return

        parsed = self._parse_flash_response(text)
        if parsed is None:
            if self._verbose:
                termcolor.cprint(
                    f"Unexpected narration response: {text}",
                    color="yellow",
                )
            return

        should_narrate = bool(parsed.get("should_narrate"))
        script = (parsed.get("script") or "").strip()
        if not should_narrate or not script:
            if self._verbose:
                termcolor.cprint(
                    "[Narration] Flash model chose not to speak.",
                    color="cyan",
                )
            self._narration_cache.add(digest)
            return

        try:
            self._browser_computer.narrate_text(script, source="flash")
            if self._verbose:
                excerpt = script if len(script) < 120 else script[:117] + "..."
                termcolor.cprint(
                    f"[Narration] Speaking flash script: {excerpt}",
                    color="green",
                )
        except Exception as exc:  # noqa: BLE001
            if self._verbose:
                termcolor.cprint(
                    f"Failed to narrate flash script: {exc}",
                    color="yellow",
                )
        finally:
            self._narration_cache.add(digest)

    def _parse_flash_response(self, text: str) -> Optional[dict[str, Any]]:
        cleaned = (text or "").strip()
        if not cleaned:
            return None
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = cleaned[start : end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                return None
        return None

    def _get_safety_confirmation(
        self, safety: dict[str, Any]
    ) -> Literal["CONTINUE", "TERMINATE"]:
        if safety["decision"] != "require_confirmation":
            raise ValueError(f"Unknown safety decision: safety['decision']")
        termcolor.cprint(
            "Safety service requires explicit confirmation!",
            color="yellow",
            attrs=["bold"],
        )
        print(safety["explanation"])
        decision = ""
        while decision.lower() not in ("y", "n", "ye", "yes", "no"):
            decision = input("Do you wish to proceed? [Yes]/[No]\n")
        if decision.lower() in ("n", "no"):
            return "TERMINATE"
        return "CONTINUE"

    def _log_session_summary(self) -> None:
        """Log final session summary with total token usage and cost."""
        if not self._verbose:
            return

        summary_table = Table(
            show_header=True,
            header_style="bold cyan",
            title="[bold yellow]Session Summary[/bold yellow]",
            expand=True
        )
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Total", justify="right", style="bold green")

        summary_table.add_row("Input Tokens", f"{self._total_input_tokens:,}")
        summary_table.add_row("Output Tokens", f"{self._total_output_tokens:,}")
        summary_table.add_row(
            "Total Tokens",
            f"{self._total_input_tokens + self._total_output_tokens:,}"
        )
        summary_table.add_row("Total Cost (USD)", f"${self._total_cost:.6f}")

        console.print()
        console.print(summary_table)
        console.print()

    def agent_loop(self):
        status = "CONTINUE"
        while status == "CONTINUE":
            status = self.run_one_iteration()

        # Log session summary at the end
        self._log_session_summary()

    def denormalize_x(self, x: int) -> int:
        return int(x / 1000 * self._browser_computer.screen_size()[0])

    def denormalize_y(self, y: int) -> int:
        return int(y / 1000 * self._browser_computer.screen_size()[1])
