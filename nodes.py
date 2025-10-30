import base64
import io
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from server import PromptServer

try:
    import google.generativeai as genai
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "The 'google-generativeai' package is required. Please install it by running 'pip install google-generativeai'."
    ) from exc


def tensor_to_pil(tensor: torch.Tensor) -> List[Image.Image]:
    """Converts a ComfyUI IMAGE tensor (N, H, W, C) in the range [0, 1] to a list of PIL images."""
    if not isinstance(tensor, torch.Tensor):
        return []

    images: List[Image.Image] = []
    for img_tensor in tensor:
        img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
        image_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        images.append(Image.fromarray(image_np))
    return images


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

_FORCE_REVIEW_SYSTEM_INSTRUCTION = """You are a self-critical, multimodal AI assistant. You previously generated an image that the user was unsatisfied with, and you must now re-evaluate your work and find the flaw.

Critically re-examine your last generated image against the original user instruction and the entire conversation history. If the user provided specific feedback, you MUST prioritize addressing it.

You MUST find a flaw. Do not determine that you were successful.

Your output must be a new, updated prompt for the image generation model. This prompt should be written as a humble, self-critical analysis in the first person, explaining what you misinterpreted or could have done better. Start by acknowledging the user's dissatisfaction (e.g., "The image is still not quite right because..."). This analysis itself will be used as the next prompt. Do NOT add any other conversational text or formatting. Your output must be a PROMPT ONLY."""


def _pil_to_inline_image(pil_image: Image.Image) -> Dict[str, Any]:
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    data = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return {"inline_data": {"mime_type": "image/png", "data": data}}


def _safe_json_loads(text: str) -> List[Dict[str, Any]]:
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def _extract_text(response: Any) -> str:
    if not response:
        return ""
    try:
        return response.text or ""
    except AttributeError:
        return ""


def _call_gemini(api_key: str, model_name: str, contents: List[Dict[str, Any]], system_instruction: str) -> str:
    if api_key:
        genai.configure(api_key=api_key)
    else:
        genai.configure()

    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_instruction,
    )

    generation_config = genai.types.GenerationConfig(temperature=0.7)
    response = model.generate_content(contents, generation_config=generation_config)
    return _extract_text(response)


def _collect_history_parts(history: Iterable[Dict[str, Any]]):
    for step in history:
        if not isinstance(step, dict):
            continue
        image_b64 = step.get("image") or step.get("output") or step.get("initial") or step.get("input")
        if isinstance(image_b64, str) and image_b64:
            yield "image", image_b64
        analysis = step.get("analysisText") or step.get("analysis")
        if isinstance(analysis, str) and analysis:
            yield "text", analysis


def _emit_director_status(link_id: str, done: bool, prompt_text: str) -> None:
    if not link_id:
        return
    try:
        PromptServer.instance.send_sync(
            "director-status",
            {"link_id": link_id, "done": done, "prompt": prompt_text},
        )
    except Exception:
        pass


class DirectorGemini:
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
                "model": ("STRING", {"default": "gemini-1.5-pro-latest"}),
                "api_key": ("STRING", {"default": ""}),
                "history_json": ("STRING", {"default": "[]", "multiline": True}),
                "link_id": ("STRING", {"default": ""}),
            },
            "optional": {
                "latest_image": ("IMAGE",),
            },
        }

    @staticmethod
    def _build_expand_contents(instruction: str, initial_image: Image.Image) -> Tuple[List[Dict[str, Any]], str]:
        contents = [
            {
                "role": "user",
                "parts": [
                    _pil_to_inline_image(initial_image),
                    {"text": instruction},
                ],
            }
        ]
        return contents, _EXPAND_SYSTEM_INSTRUCTION

    @staticmethod
    def _build_review_contents(
        instruction: str,
        initial_image: Image.Image,
        latest_image: Optional[Image.Image],
        history: Iterable[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], str]:
        contents: List[Dict[str, Any]] = [
            {
                "role": "user",
                "parts": [
                    _pil_to_inline_image(initial_image),
                    {"text": f'My instruction is: "{instruction}"'},
                ],
            }
        ]

        for kind, value in _collect_history_parts(history):
            if kind == "image":
                contents.append(
                    {
                        "role": "model",
                        "parts": [
                            {"inline_data": {"mime_type": "image/png", "data": value}},
                        ],
                    }
                )
            else:
                contents.append({"role": "model", "parts": [{"text": value}]})

        if latest_image is not None:
            contents.append({"role": "model", "parts": [_pil_to_inline_image(latest_image)]})

        final_text = (
            "Look at the image and analyze if you failed to follow the instruction. "
            "If you succeeded respond with SUCCESS. If not, return the corrected prompt."
        )
        contents.append({"role": "user", "parts": [{"text": final_text}]})
        return contents, _REVIEW_SYSTEM_INSTRUCTION

    def execute(
        self,
        instruction: str,
        initial_image: torch.Tensor,
        model: str,
        api_key: str,
        history_json: str,
        link_id: str,
        latest_image: Optional[torch.Tensor] = None,
    ) -> Tuple[str]:
        prompt_text = ""
        history = _safe_json_loads(history_json)

        print("[DirectorGemini] Starting execution", {
            "instruction": instruction,
            "link_id": link_id,
        })

        try:
            initial_pil_list = tensor_to_pil(initial_image)
            if not initial_pil_list:
                raise ValueError("Initial image tensor is empty.")
            initial_pil = initial_pil_list[0]

            latest_pil: Optional[Image.Image] = None
            if isinstance(latest_image, torch.Tensor):
                latest_list = tensor_to_pil(latest_image)
                if latest_list:
                    latest_pil = latest_list[0]

            is_review = latest_pil is not None
            print("[DirectorGemini] Mode determined", {
                "mode": "review" if is_review else "expand",
                "has_history": bool(history),
            })
            if is_review:
                contents, system_instruction = self._build_review_contents(
                    instruction,
                    initial_pil,
                    latest_pil,
                    history,
                )
            else:
                contents, system_instruction = self._build_expand_contents(instruction, initial_pil)

            response_text = _call_gemini(api_key, model, contents, system_instruction).strip()
            if not response_text:
                response_text = instruction.strip()

            if is_review:
                if response_text.upper() == "SUCCESS":
                    prompt_text = "SUCCESS"
                    _emit_director_status(link_id, True, "")
                    print("[DirectorGemini] Review marked SUCCESS", {"link_id": link_id})
                else:
                    prompt_text = response_text
                    _emit_director_status(link_id, False, prompt_text)
                    print("[DirectorGemini] Review generated prompt", {"prompt": prompt_text})
            else:
                prompt_text = response_text
                _emit_director_status(link_id, False, prompt_text)
                print("[DirectorGemini] Expand generated prompt", {"prompt": prompt_text})
        except Exception as exc:  # pragma: no cover - runtime diagnostics
            prompt_text = f"<director_error:{exc}>"
            _emit_director_status(link_id, True, prompt_text)
            print("[DirectorGemini] Error", {"error": prompt_text})

        return (prompt_text,)


class ImageRouter:
    CATEGORY = "Director/IO"
    RETURN_TYPES: Tuple = ()
    RETURN_NAMES: Tuple = ()
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "link_id": ("STRING", {"default": ""}),
                "persist_dir": ("STRING", {"default": "."}),
                "iter_tag": ("INT", {"default": 0, "min": 0}),
            }
        }

    def execute(self, image: torch.Tensor, link_id: str, persist_dir: str, iter_tag: int) -> Tuple[()]:
        print("[ImageRouter] Received image", {"link_id": link_id, "iter": iter_tag})
        pil_images = tensor_to_pil(image)
        if not pil_images:
            raise ValueError("ImageRouter received no image data.")

        base_dir = persist_dir or "."
        if link_id:
            base_dir = os.path.join(base_dir, link_id)
        os.makedirs(base_dir, exist_ok=True)

        filename = f"iter_{iter_tag:04d}.png"
        path = os.path.abspath(os.path.join(base_dir, filename))
        pil_images[0].save(path, format="PNG")
        print("[ImageRouter] Saved image", {"path": path})

        try:
            PromptServer.instance.send_sync(
                "img-ready",
                {"link_id": link_id, "path": path, "iter": int(iter_tag)},
            )
            print("[ImageRouter] Emitted img-ready", {"link_id": link_id, "iter": int(iter_tag)})
        except Exception:
            pass

        return tuple()


class GeminiDirector:
    CATEGORY = "Director/Gemini"
    RETURN_TYPES = ("STRING", "BOOLEAN")
    RETURN_NAMES = ("prompt", "done")
    FUNCTION = "execute"
    OUTPUT_IS_LIST = (False, False)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["expand", "review", "force_review"],),
                "goal": ("STRING", {"default": "", "multiline": True}),
                "image": ("IMAGE",),
                "model": ("STRING", {"default": "gemini-1.5-pro-latest"}),
                "api_key": ("STRING", {"default": ""}),
                "history_json": ("STRING", {"default": "[]", "multiline": True}),
            },
            "optional": {
                "feedback": ("STRING", {"default": "", "multiline": True}),
            },
        }

    @staticmethod
    def _build_contents(
        mode: str,
        goal: str,
        current_image: Image.Image,
        history: List[Dict[str, Any]],
        feedback: str,
    ) -> Tuple[List[Dict[str, Any]], str]:
        current_image_part = _pil_to_inline_image(current_image)

        if mode == "expand":
            contents = [{"role": "user", "parts": [current_image_part, {"text": goal}]}]
            return contents, _EXPAND_SYSTEM_INSTRUCTION

        initial_image_b64 = ""
        if history:
            for step in history:
                img_b64 = step.get("initial") or step.get("input") or step.get("image")
                if isinstance(img_b64, str) and img_b64:
                    initial_image_b64 = img_b64
                    break

        if not initial_image_b64:
            initial_image_b64 = current_image_part["inline_data"]["data"]

        initial_image_part = {"inline_data": {"mime_type": "image/png", "data": initial_image_b64}}
        contents = [{"role": "user", "parts": [initial_image_part, {"text": f'My instruction is: "{goal}"'}]}]

        for step in history:
            img_b64 = step.get("image")
            if isinstance(img_b64, str):
                image_part = {"inline_data": {"mime_type": "image/png", "data": img_b64}}
                contents.append({"role": "model", "parts": [image_part]})

            analysis = step.get("analysisText")
            if isinstance(analysis, str):
                contents.append({"role": "model", "parts": [{"text": analysis}]})

        contents.append({"role": "model", "parts": [current_image_part]})

        if mode == "review":
            final_text = (
                "Look at this latest image. Analyze if you have perfectly followed the instructions. "
                "If so, say SUCCESS. If not, provide an improved prompt."
            )
            contents.append({"role": "user", "parts": [{"text": final_text}]})
            return contents, _REVIEW_SYSTEM_INSTRUCTION

        final_feedback = feedback.strip() or (
            "The image still failed to meet the instructions. Re-analyze your work, find the flaw, "
            "and provide the detailed, first-person critique as instructed."
        )
        contents.append({"role": "user", "parts": [{"text": final_feedback}]})
        return contents, _FORCE_REVIEW_SYSTEM_INSTRUCTION

    def execute(
        self,
        mode: str,
        goal: str,
        image: torch.Tensor,
        model: str,
        api_key: str,
        history_json: str,
        feedback: str = "",
    ) -> Tuple[str, bool]:
        prompt = ""
        done = True

        print("[GeminiDirector] Starting", {"mode": mode, "goal": goal[:30]})

        try:
            pil_images = tensor_to_pil(image)
            if not pil_images:
                raise ValueError("Input tensor could not be converted to an image.")
            current_pil_image = pil_images[0]

            history = _safe_json_loads(history_json)
            contents, system_instruction = self._build_contents(mode, goal, current_pil_image, history, feedback)
            response_text = _call_gemini(api_key, model, contents, system_instruction).strip()

            if mode == "review" and response_text.upper() == "SUCCESS":
                prompt = ""
                done = True
                print("[GeminiDirector] Review SUCCESS", {})
            else:
                prompt = response_text
                done = not bool(prompt)
                print("[GeminiDirector] Generated prompt", {"prompt": prompt})

        except Exception as exc:
            prompt = f"<error> GeminiDirector failed: {exc}"
            done = True
            print("[GeminiDirector] Error", {"error": prompt})

        return (prompt, done)


class PromptSwitch:
    CATEGORY = "Director/Gemini"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("current_prompt",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "iteration": ("INT", {"default": 0, "min": 0}),
                "first_prompt": ("STRING", {"default": "", "multiline": True}),
                "next_prompt": ("STRING", {"default": "", "multiline": True}),
            }
        }

    def execute(self, iteration: int, first_prompt: str, next_prompt: str) -> Tuple[str]:
        current_prompt = first_prompt if iteration == 0 else next_prompt
        print("[PromptSwitch] Selected prompt", {"iteration": iteration})
        return (current_prompt,)


NODE_CLASS_MAPPINGS = {
    "GeminiDirector": GeminiDirector,
    "PromptSwitch": PromptSwitch,
    "DirectorGemini": DirectorGemini,
    "ImageRouter": ImageRouter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiDirector": "Gemini Director",
    "PromptSwitch": "Prompt Switch",
    "DirectorGemini": "Director (Gemini, single prompt)",
    "ImageRouter": "Image Router (persist + notify)",
}
