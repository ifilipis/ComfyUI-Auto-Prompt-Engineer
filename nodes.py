"""ComfyUI nodes for director/actor execution."""

from __future__ import annotations

import base64
import io
import json
import math
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from server import PromptServer

try:
    import google.generativeai as genai
except ImportError as exc:  # pragma: no cover - dependency error surfaced at runtime
    raise ImportError(
        "The 'google-generativeai' package is required. Please install it by running"
        " 'pip install google-generativeai'."
    ) from exc


_EXPAND_SYSTEM_INSTRUCTION = """You are an expert prompt engineer for a state-of-the-art image editing model. Your task is to take a user's simple instruction and a set of input images, and expand the instruction into a detailed, descriptive prompt that will guide the image model to produce a high-quality edited result.

Focus on describing the necessary changes based on the user's instruction while considering the content of the original images. Your prompt must include:
(!) Always describe small, specific, targeted edits that will move you to the desired result.

- Visual Style: Match the existing style (e.g., photorealistic, oil painting).
- Composition & Framing: Describe changes in relation to the existing composition.
- Camera: Describe camera parameters and image style.
- Lighting: Describe how lighting should be altered or added, matching the existing light source.
- Details & Texture: Mention specific details from the original image that should be changed.
- Action: Clearly describe the edit to be performed on the image.
- Resulting image: Describe what the result will look like.

The output must be ONLY the new, detailed prompt. Do not add any conversational text, greetings, or explanations."""

_REVIEW_SYSTEM_INSTRUCTION = """You are a self-critical, multimodal AI assistant. Your primary function is to edit images based on user instructions, but a crucial part of your process is to analyze your own work to ensure quality.

You will be shown a conversation history: the original images, the user's instruction, and your previous attempts. Your final task is to analyze your LATEST generated image.

Compare the latest generated image to the original user instruction, considering the entire conversation history for context.

If you have perfectly followed the instruction, respond with only the word 'SUCCESS'.

If you have failed, you must change your strategy. DO NOT repeat the same prompt. Your output must be an updated prompt that points out the previous failure and suggests a corrective action. Do not add conversational text or formatting—your output must be a PROMPT ONLY."""


_PERSIST_DIR = os.environ.get("DIRECTOR_ACTOR_PERSIST_DIR", "./director_actor")
_HISTORY_FILENAME = "history.json"
_SESSION_FILENAME = "session.json"
_LATEST_IMAGE_NAME = "latest.png"


def _resolve_persist_root() -> Path:
    return Path(os.path.expanduser(_PERSIST_DIR)).resolve()


def _resolve_group_dir(link_id: str) -> Path:
    root = _resolve_persist_root()
    return root / link_id if link_id else root


def _history_path(group_dir: Path) -> Path:
    return group_dir / _HISTORY_FILENAME


def _session_path(group_dir: Path) -> Path:
    return group_dir / _SESSION_FILENAME


def _load_history_entries(group_dir: Path) -> List[Dict[str, Any]]:
    history_file = _history_path(group_dir)
    try:
        with history_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        return []

    if isinstance(data, list):
        return [entry for entry in data if isinstance(entry, dict)]
    return []


def _save_history_entries(group_dir: Path, entries: Iterable[Dict[str, Any]]) -> None:
    group_dir.mkdir(parents=True, exist_ok=True)
    history_file = _history_path(group_dir)
    serializable = [entry for entry in entries if isinstance(entry, dict)]
    with history_file.open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle)


def _load_session_state(group_dir: Path) -> Dict[str, Any]:
    session_file = _session_path(group_dir)
    try:
        with session_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}

    if isinstance(data, dict):
        return data
    return {}


def _save_session_state(group_dir: Path, state: Dict[str, Any]) -> None:
    group_dir.mkdir(parents=True, exist_ok=True)
    session_file = _session_path(group_dir)
    serializable = {k: v for k, v in state.items() if isinstance(k, str)}
    with session_file.open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle)


def _start_new_session(group_dir: Path, instruction: str) -> str:
    session_id = uuid.uuid4().hex
    state = {"session_id": session_id, "instruction": instruction or ""}
    _save_session_state(group_dir, state)
    latest_path = group_dir / _LATEST_IMAGE_NAME
    try:
        latest_path.unlink()
    except FileNotFoundError:
        pass
    return session_id


def _active_session_id(group_dir: Path) -> Optional[str]:
    state = _load_session_state(group_dir)
    session_id = state.get("session_id") if isinstance(state, dict) else None
    return session_id if isinstance(session_id, str) and session_id else None


def _encode_image_file(image_path: Path) -> Optional[str]:
    try:
        with image_path.open("rb") as handle:
            return base64.b64encode(handle.read()).decode("utf-8")
    except FileNotFoundError:
        return None


def _tensor_to_pil_list(tensor: torch.Tensor) -> List[Image.Image]:
    if not isinstance(tensor, torch.Tensor):
        return []

    tensor = torch.clamp(tensor, 0.0, 1.0)
    images: List[Image.Image] = []
    for slice_ in tensor:
        array = (slice_.cpu().numpy() * 255).astype(np.uint8)
        images.append(Image.fromarray(array))
    return images


def _pil_to_inline_image(image: Image.Image) -> Dict[str, Any]:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return {"inline_data": {"mime_type": "image/png", "data": encoded}}


def _debug_log(message: str, **details: Any) -> None:
    prefix = "[DirectorGemini] "
    if details:
        try:
            serialized = json.dumps(details, default=str)
        except Exception:
            serialized = str(details)
        print(f"{prefix}{message}: {serialized}", flush=True)
    else:
        print(f"{prefix}{message}", flush=True)


def _extract_response_text(response: Any) -> str:
    if response is None:
        return ""
    return getattr(response, "text", "")


def _call_gemini(
    api_key: str,
    model_name: str,
    contents: List[Dict[str, Any]],
    system_instruction: str,
) -> str:
    if api_key:
        genai.configure(api_key=api_key)

    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_instruction,
    )

    generation_config = genai.types.GenerationConfig(temperature=0.7)
    response = model.generate_content(contents, generation_config=generation_config)
    return _extract_response_text(response)


class DirectorGemini:
    """Single-output Gemini director node."""

    CATEGORY = "Director/Gemini"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "instruction": ("STRING", {"default": "", "multiline": True}),
                "initial_image": ("IMAGE",),
                "model": ("STRING", {"default": "gemini-1.5-pro"}),
                "api_key": ("STRING", {"default": ""}),
                "link_id": ("STRING", {"default": ""}),
            },
            "optional": {
                "latest_image": ("IMAGE",),
            },
        }

    @staticmethod
    def _build_expand_contents(
        instruction: str,
        initial: Image.Image,
    ) -> Tuple[List[Dict[str, Any]], str]:
        contents = [
            {
                "role": "user",
                "parts": [
                    _pil_to_inline_image(initial),
                    {"text": instruction},
                ],
            }
        ]
        return contents, _EXPAND_SYSTEM_INSTRUCTION

    @staticmethod
    def _add_history_parts(contents: List[Dict[str, Any]], history: Iterable[Dict[str, Any]]) -> None:
        for step in history:
            image_b64 = None
            if isinstance(step, dict):
                image_b64 = step.get("image") or step.get("output") or step.get("initial")
            if isinstance(image_b64, str) and image_b64:
                contents.append(
                    {
                        "role": "model",
                        "parts": [
                            {
                                "inline_data": {
                                    "mime_type": "image/png",
                                    "data": image_b64,
                                }
                            }
                        ],
                    }
                )

            analysis_text = step.get("analysisText") if isinstance(step, dict) else None
            if isinstance(analysis_text, str) and analysis_text.strip():
                contents.append({"role": "model", "parts": [{"text": analysis_text}]})

    @classmethod
    def _build_review_contents(
        cls,
        instruction: str,
        initial: Image.Image,
        latest: Optional[Image.Image],
        history: Iterable[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], str]:
        contents: List[Dict[str, Any]] = [
            {
                "role": "user",
                "parts": [
                    _pil_to_inline_image(initial),
                    {"text": f'My instruction is: "{instruction}"'},
                ],
            }
        ]

        cls._add_history_parts(contents, history)

        if latest is not None:
            contents.append({"role": "model", "parts": [_pil_to_inline_image(latest)]})

        contents.append(
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            "Look at the image and analyze if you failed to follow the instruction. "
                            "If you succeeded, respond with SUCCESS. Otherwise, provide a corrected prompt."
                        )
                    }
                ],
            }
        )

        return contents, _REVIEW_SYSTEM_INSTRUCTION

    @staticmethod
    def _send_event(link_id: str, done: bool, prompt: str) -> None:
        if not link_id:
            return
        _debug_log("Sending director-status", done=done, prompt_preview=prompt[:80])
        try:
            PromptServer.instance.send_sync(
                "director-status",
                {"link_id": link_id, "done": done, "prompt": prompt},
            )
        except Exception:
            # Front-end events should not break execution; swallow errors silently.
            pass

    def _load_history_for_prompt(
        self,
        group_dir: Path,
        session_id: Optional[str],
        include_latest: bool,
    ) -> List[Dict[str, Any]]:
        if not session_id:
            return []

        entries = [
            entry
            for entry in _load_history_entries(group_dir)
            if isinstance(entry, dict)
            and entry.get("session_id") == session_id
        ]

        if not include_latest and entries:
            entries = entries[:-1]

        history: List[Dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            image_key = entry.get("image_path")
            record: Dict[str, Any] = {}
            if isinstance(image_key, str) and image_key:
                encoded = _encode_image_file(group_dir / image_key)
                if encoded:
                    record["image"] = encoded
            analysis_text = entry.get("analysisText")
            if isinstance(analysis_text, str) and analysis_text.strip():
                record["analysisText"] = analysis_text
            if record:
                history.append(record)
        return history

    def _update_latest_history_entry(
        self,
        group_dir: Path,
        session_id: Optional[str],
        response_text: str,
        done: bool,
    ) -> None:
        if not session_id:
            return

        entries = _load_history_entries(group_dir)
        if not entries:
            return

        session_indices = [
            index
            for index, entry in enumerate(entries)
            if isinstance(entry, dict)
            and entry.get("session_id") == session_id
        ]
        if not session_indices:
            return

        latest_index = session_indices[-1]
        latest = entries[latest_index]
        if not isinstance(latest, dict):
            return
        latest["analysisText"] = "SUCCESS" if done else response_text
        entries[latest_index] = latest
        _save_history_entries(group_dir, entries)

    def execute(
        self,
        instruction: str,
        initial_image: torch.Tensor,
        model: str,
        api_key: str,
        link_id: str,
        latest_image: Optional[torch.Tensor] = None,
    ) -> Tuple[str]:
        _debug_log(
            "Execute called",
            link_id=link_id,
            has_latest=latest_image is not None,
            instruction_chars=len(instruction or ""),
        )
        prompt_output = ""
        event_prompt = ""
        done = False

        group_dir = _resolve_group_dir(link_id)
        session_id = _active_session_id(group_dir)
        latest_pil: Optional[Image.Image] = None
        try:
            initial_images = _tensor_to_pil_list(initial_image)
            if not initial_images:
                raise ValueError("Initial image tensor is empty.")
            _debug_log("Initial image processed", count=len(initial_images))

            if latest_image is not None:
                latest_list = _tensor_to_pil_list(latest_image)
                if latest_list:
                    latest_pil = latest_list[0]
            _debug_log("Latest image state", available=latest_pil is not None)

            mode = "expand" if latest_pil is None else "review"
            if mode == "expand":
                session_id = _start_new_session(group_dir, instruction)
            elif not session_id:
                session_id = _start_new_session(group_dir, instruction)

            history = self._load_history_for_prompt(
                group_dir,
                session_id,
                include_latest=latest_pil is None,
            )
            _debug_log("Resolved mode", mode=mode, history_entries=len(history))

            if mode == "expand":
                contents, system_instruction = self._build_expand_contents(
                    instruction, initial_images[0]
                )
            else:
                contents, system_instruction = self._build_review_contents(
                    instruction, initial_images[0], latest_pil, history
                )
            _debug_log("Built contents", parts=len(contents), system_instruction=system_instruction[:40])

            response_text = _call_gemini(api_key, model, contents, system_instruction).strip()
            _debug_log(
                "Gemini response received",
                empty=not bool(response_text),
                preview=response_text[:80],
            )

            if mode == "expand":
                prompt_output = response_text or instruction
                event_prompt = prompt_output
                done = False
            else:
                if response_text.upper() == "SUCCESS":
                    prompt_output = "SUCCESS"
                    event_prompt = ""
                    done = True
                else:
                    prompt_output = response_text or instruction
                    event_prompt = prompt_output
                    done = False

        except Exception as exc:  # pragma: no cover - runtime safety
            prompt_output = f"<director_error:{exc}>"
            event_prompt = prompt_output
            done = True
            _debug_log("Execute failed", error=str(exc))

        self._send_event(link_id, done, event_prompt)
        if latest_pil is not None:
            self._update_latest_history_entry(group_dir, session_id, prompt_output, done)
        _debug_log(
            "Execute completed",
            done=done,
            prompt_preview=prompt_output[:80],
        )
        return (prompt_output,)


class ImageRouterSink:
    """Persist images, update the latest pointer, and notify the front-end."""

    CATEGORY = "Director/IO"
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "link_id": ("STRING", {"default": ""}),
            }
        }

    @staticmethod
    def _notify(link_id: str, iteration_path: Path, latest_path: Path, iteration: int) -> None:
        if not link_id:
            return
        payload = {
            "link_id": link_id,
            "path": str(iteration_path),
            "iter": iteration,
            "latest": str(latest_path),
        }
        try:
            PromptServer.instance.send_sync("img-ready", payload)
        except Exception:
            # Notification failures should not interrupt execution.
            pass

    @staticmethod
    def _atomic_update_latest(image: Image.Image, latest_path: Path) -> None:
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".png", dir=str(latest_path.parent)
            ) as handle:
                tmp_path = Path(handle.name)
                image.save(handle, format="PNG")
            os.replace(tmp_path, latest_path)
        finally:
            if tmp_path and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except FileNotFoundError:
                    pass

    @staticmethod
    def _next_iteration(group_dir: Path, session_id: Optional[str]) -> int:
        if not session_id:
            return 0

        entries = [
            entry
            for entry in _load_history_entries(group_dir)
            if isinstance(entry, dict)
            and entry.get("session_id") == session_id
        ]
        if not entries:
            return 0
        last = entries[-1]
        if isinstance(last, dict) and isinstance(last.get("iteration"), int):
            return last["iteration"] + 1
        return len(entries)

    def execute(
        self,
        image: torch.Tensor,
        link_id: str,
    ) -> Tuple[()]:
        pil_images = _tensor_to_pil_list(image)
        if not pil_images:
            raise ValueError("ImageRouterSink received an empty image tensor.")

        base_path = _resolve_group_dir(link_id)
        base_path.mkdir(parents=True, exist_ok=True)

        session_id = _active_session_id(base_path)
        if not session_id:
            session_id = _start_new_session(base_path, "")

        iteration = self._next_iteration(base_path, session_id)
        filename = f"iter_{int(iteration):04d}.png"
        iteration_path = base_path / filename
        pil_images[0].save(iteration_path, format="PNG")

        latest_path = base_path / _LATEST_IMAGE_NAME
        self._atomic_update_latest(pil_images[0], latest_path)

        entries = _load_history_entries(base_path)
        entries.append(
            {
                "session_id": session_id,
                "iteration": int(iteration),
                "image_path": filename,
            }
        )
        _save_history_entries(base_path, entries)

        self._notify(link_id, iteration_path.resolve(), latest_path.resolve(), int(iteration))
        return tuple()


class LatestImageSource:
    """Load the most recent actor image as a tensor."""

    CATEGORY = "Director/IO"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "link_id": ("STRING", {"default": ""}),
            }
        }

    @staticmethod
    def _resolve_path(link_id: str) -> Path:
        group_dir = _resolve_group_dir(link_id)
        return (group_dir / _LATEST_IMAGE_NAME).resolve()

    @staticmethod
    def _load_tensor_from_image(image_path: Path) -> torch.Tensor:
        with Image.open(image_path) as handle:
            rgb_image = handle.convert("RGB")
            array = np.array(rgb_image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(-1)
        tensor = tensor.unsqueeze(0)
        return tensor

    @classmethod
    def IS_CHANGED(
        cls,
        link_id: str,
    ) -> str:
        resolved = cls._resolve_path(link_id)
        try:
            stat = resolved.stat()
        except FileNotFoundError:
            return "missing"
        return f"{resolved}:{stat.st_mtime_ns}:{stat.st_size}"

    def load(
        self,
        link_id: str,
    ) -> Tuple[torch.Tensor]:
        resolved = self._resolve_path(link_id)
        if not resolved.exists():
            return (torch.zeros((0,), dtype=torch.float32),)
        tensor = self._load_tensor_from_image(resolved)
        return (tensor,)


NODE_CLASS_MAPPINGS = {
    "DirectorGemini": DirectorGemini,
    "ImageRouterSink": ImageRouterSink,
    "LatestImageSource": LatestImageSource,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "DirectorGemini": "Director (Gemini, single prompt)",
    "ImageRouterSink": "Image Router Sink (persist + latest)",
    "LatestImageSource": "Latest Image Source",
}
