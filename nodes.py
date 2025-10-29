import base64
import io
import json
from typing import Any, Dict, List, Tuple

from comfy.utils import tensor_to_pil

try:
    from google import genai  # type: ignore

    _GENAI_MODE = "new"
except ImportError:  # pragma: no cover
    import google.generativeai as genai  # type: ignore

    _GENAI_MODE = "legacy"


_EXPAND_SYSTEM_INSTRUCTION = """You are an expert prompt engineer for a state-of-the-art image editing model. Your task is to take a user's simple instruction and a set of input images, and expand the instruction into a detailed, descriptive prompt that will guide the image model to produce a high-quality edited result.

Focus on describing the necessary changes based on the user's instruction while considering the content of the original images. Your prompt must include:
 (!) **Always** describe small, specific, targeted edits that will move you to the desired result.
- **Visual Style:** Match the existing style (e.g., photorealistic, oil painting).
- **Composition & Framing:** Describe changes in relation to the existing composition.
- **Camera:** Describe camera parameters and image style.
- **Lighting:** Describe how lighting should be altered or added, matching the existing light source.
- **Details & Texture:** Mention specific details from the original image that should be changed.
- **Action:** Clearly describe the edit to be performed on the image.
- **Resulting image:** Describe what the result will look like.

The output must be ONLY the new, detailed prompt. Do not add any conversational text, greetings, or explanations."""

_REVIEW_SYSTEM_INSTRUCTION = """You are a self-critical, multimodal AI assistant. Your primary function is to edit images based on user instructions, but a crucial part of your process is to analyze your own work to ensure quality.

You will be shown a conversation history: the original images, the user's instruction, and your previous attempts and self-critiques. Your final task is to analyze your LATEST generated image.

1.  **Compare** the latest generated image to the original user instruction, considering the entire conversation history for context.
2.  **If you have perfectly followed the instruction**, respond with only the word 'SUCCESS'.
3.  **If you have failed**, it means that the previous prompts did not work, and you must change your strategy. DO NOT repeat the same prompt and reasoning. Do NOT add any other conversational text or formatting.\
At the end of your analysis, you must write an updated prompt, pointing out elements that have failed and corrective action.
Keep analysis in the thinking channel. DO NOT output analysis directly. Your output must be a PROMPT ONLY.

 Focus on describing the necessary changes based on the user's instruction while considering the content of the original image. Your prompt must include:
 (!) **Always** describe small, specific, targeted edits that will move you to the desired result.
- **Visual Style:** Match the existing style (e.g., photorealistic, oil painting).
- **Composition & Framing:** Describe changes in relation to the existing composition.
- **Camera:** Describe camera parameters and image style.
- **Lighting:** Describe how lighting should be altered or added, matching the existing light source.
- **Details & Texture:** Mention specific details from the original image that should be changed.
- **Action:** Clearly describe the edit to be performed on the image.
- **Resulting image:** Describe what the result will look like."""

_FORCE_REVIEW_SYSTEM_INSTRUCTION = """You are a self-critical, multimodal AI assistant. You previously generated an image that you may have thought was correct, but the user was unsatisfied and has forced you to re-evaluate your work. Your task is to find the flaws in your last generated image.

1.  **Critically re-examine** your last generated image against the original user instruction and the entire conversation history. If the user provides specific feedback on the failure, you MUST prioritize addressing it in your analysis.
2.  You **MUST** find a flaw. Do not determine that you were successful.
3.  Write a detailed, humble, and self-critical analysis in the first person explaining what you could have done better or what you misinterpreted. Start by acknowledging the user's dissatisfaction (e.g., "The image is still..."). Your analysis itself will be used as the next prompt. Do NOT add any other conversational text or formatting.\
4.  **You have failed**, it means that the previous prompts did not work, and you must change your strategy. DO NOT repeat the same prompt and reasoning. Do NOT add any other conversational text or formatting.\
At the end of your analysis, you must write an updated prompt, pointing out elements that have failed and corrective action.
Keep analysis in the thinking channel. DO NOT output analysis directly. Your output must be a PROMPT ONLY.

 Focus on describing the necessary changes based on the user's instruction while considering the content of the original image. Your prompt must include:
 (!) **Always** describe small, specific, targeted edits that will move you to the desired result.
- **Visual Style:** Match the existing style (e.g., photorealistic, oil painting).
- **Composition & Framing:** Describe changes in relation to the existing composition.
- **Camera:** Describe camera parameters and image style.
- **Lighting:** Describe how lighting should be altered or added, matching the existing light source.
- **Details & Texture:** Mention specific details from the original image that should be changed.
- **Action:** Clearly describe the edit to be performed on the image.
- **Resulting image:** Describe what the result will look like."""


def _image_tensor_to_base64(image_tensor) -> str:
    pil_images = tensor_to_pil(image_tensor)
    pil_image = pil_images[0] if isinstance(pil_images, list) else pil_images
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _inline_image(b64_data: str) -> Dict[str, Any]:
    return {"inline_data": {"mime_type": "image/png", "data": b64_data}}


def _safe_json_loads(text: str) -> List[Dict[str, Any]]:
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def _extract_text(response: Any) -> str:
    if response is None:
        return ""
    if hasattr(response, "text") and response.text:
        return response.text
    candidates = getattr(response, "candidates", None)
    parts: List[str] = []
    if candidates:
        for cand in candidates:
            content = getattr(cand, "content", None)
            cand_parts = getattr(content, "parts", None)
            if cand_parts:
                for part in cand_parts:
                    text = getattr(part, "text", None)
                    if text:
                        parts.append(text)
            elif hasattr(cand, "text") and cand.text:
                parts.append(cand.text)
    if parts:
        return "".join(parts)
    if isinstance(response, dict):
        return response.get("text") or ""
    return ""


def _call_gemini(api_key: str, model: str, contents: List[Dict[str, Any]], system_instruction: str) -> str:
    temperature = 0.7
    if _GENAI_MODE == "new":
        client = genai.Client(api_key=api_key or None)
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config={"system_instruction": system_instruction, "temperature": temperature},
        )
    else:
        if api_key:
            genai.configure(api_key=api_key)
        else:
            genai.configure()
        model_client = genai.GenerativeModel(model)
        response = model_client.generate_content(
            contents,
            system_instruction=system_instruction,
            temperature=temperature,
        )
    return _extract_text(response)


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
                "mode": ("ENUM", {"options": ["expand", "review", "force_review"], "default": "expand"}),
                "goal": ("STRING", {"default": "", "multiline": True}),
                "image": ("IMAGE",),
                "model": ("STRING", {"default": "gemini-1.5-pro"}),
                "api_key": ("STRING", {"default": ""}),
                "history_json": ("STRING", {"default": "[]", "multiline": True}),
            },
            "optional": {
                "feedback": ("STRING", {"default": "", "multiline": True}),
            },
        }

    @staticmethod
    def _build_contents(mode: str, goal: str, current_b64: str, history: List[Dict[str, Any]], feedback: str) -> Tuple[List[Dict[str, Any]], str]:
        if mode == "expand":
            contents = [
                {
                    "role": "user",
                    "parts": [
                        _inline_image(current_b64),
                        {"text": goal},
                    ],
                }
            ]
            return contents, _EXPAND_SYSTEM_INSTRUCTION

        initial_image_b64 = ""
        for step in history:
            img = step.get("initial") or step.get("input")
            if isinstance(img, str) and img:
                initial_image_b64 = img
                break
        if not initial_image_b64:
            if history and isinstance(history[0].get("image"), str):
                initial_image_b64 = history[0]["image"]
        if not initial_image_b64:
            initial_image_b64 = current_b64

        contents: List[Dict[str, Any]] = [
            {
                "role": "user",
                "parts": [
                    _inline_image(initial_image_b64),
                    {"text": f'My instruction is: "{goal}"'},
                ],
            }
        ]

        for step in history:
            img_b64 = step.get("image")
            if isinstance(img_b64, str) and img_b64:
                contents.append({"role": "model", "parts": [_inline_image(img_b64)]})
            analysis = step.get("analysisText")
            if isinstance(analysis, str) and analysis:
                contents.append({"role": "model", "parts": [{"text": analysis}]})

        contents.append({"role": "model", "parts": [_inline_image(current_b64)]})

        if mode == "review":
            final_text = "Look at the image and analyze if you failed to follow the instructions."
            contents.append({"role": "user", "parts": [{"text": final_text}]})
            return contents, _REVIEW_SYSTEM_INSTRUCTION

        final_feedback = feedback.strip() or "The image still failed to meet the instructions. Re-analyze your work, find the flaw, and provide the detailed, first-person critique as instructed."
        contents.append({"role": "user", "parts": [{"text": final_feedback}]})
        return contents, _FORCE_REVIEW_SYSTEM_INSTRUCTION

    def execute(self, mode: str, goal: str, image, model: str, api_key: str, history_json: str, feedback: str = ""):
        try:
            current_b64 = _image_tensor_to_base64(image)
            history = _safe_json_loads(history_json)
            contents, system_instruction = self._build_contents(mode, goal, current_b64, history, feedback)
            response_text = _call_gemini(api_key, model, contents, system_instruction).strip()

            if mode == "expand":
                prompt = response_text
                done = False
            elif mode == "review":
                if response_text.upper() == "SUCCESS":
                    prompt = ""
                    done = True
                else:
                    prompt = response_text
                    done = False
                    if not prompt:
                        done = True
            else:  # force_review
                prompt = response_text
                done = False

            if mode != "expand" and not response_text and not done:
                done = True

            return prompt, done
        except Exception as error:  # pragma: no cover
            return f"<director_error:{error}>", True


class PromptSwitch:
    CATEGORY = "Director/Gemini"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("current_prompt",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "iter": ("INT", {"default": 0}),
                "first_prompt": ("STRING", {"default": "", "multiline": True}),
                "next_prompt": ("STRING", {"default": "", "multiline": True}),
            }
        }

    def execute(self, iter: int, first_prompt: str, next_prompt: str):
        return (first_prompt if iter == 0 else next_prompt,)


NODE_CLASS_MAPPINGS = {
    "GeminiDirector": GeminiDirector,
    "PromptSwitch": PromptSwitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiDirector": "Gemini Director",
    "PromptSwitch": "Prompt Switch",
}
