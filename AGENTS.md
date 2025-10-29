**Create Minimal ComfyUI Nodes** that use the **Gemini Python SDK in ComfyUI Core** and the **EXACT prompts** provided by the user (NO CLI, NO BLOAT)

You are an expert Python developer. Implement **two** ComfyUI custom nodes that enable an **unsupervised Director→Actor iterative editing loop entirely inside ComfyUI**, using the **Gemini Python SDK already bundled with ComfyUI Core**. Use the **exact system instructions and behaviors** from the user’s TypeScript example below. **Do not** add a CLI, server, or any external orchestrator. **Do not** re-implement KSampler. Keep the code compact and readable.

---

## Files & Location

* Single file: `ComfyUI/custom_nodes/comfy_gemini_director_loop.py`
* No extra dependencies beyond what ComfyUI ships with (PIL, requests, the Gemini Python SDK in core).

---

## Node 1 — `GeminiDirector` (expand / review / force_review)

**Purpose:** Call Gemini to (1) expand the initial instruction into a professional prompt once, (2) review each generated image and either return `SUCCESS` or a **corrective prompt**, and (3) optionally perform a forced self-critique pass.

**Inputs (required):**

* `mode` : CHOICE → `"expand" | "review" | "force_review"`
* `goal` : STRING — the user’s simple instruction (e.g., "Make the cat look like a pirate").
* `image` : IMAGE tensor — PNG-encode and send to Gemini as inline image data.
* `model` : STRING — Gemini reasoning model id (e.g., `gemini-1.5-pro`).
* `api_key` : STRING — may be empty (SDK should also honor env vars in ComfyUI).
* `history_json` : STRING — JSON array of `{ image: base64PNG, analysisText: string }` steps; may be empty `[]`.
* Optional `feedback` : STRING — only used when `mode="force_review"`.

**Outputs:**

* `prompt` : STRING — for `expand` this is the first detailed prompt; for `review`/`force_review` this is either the **corrective prompt** or empty when done.
* `done`   : BOOL — `False` on `expand`; for `review` it is `True` iff Gemini replies exactly `SUCCESS`.

**Gemini SDK usage:** Use the **Python Gemini SDK included in ComfyUI Core**. Prefer the newer `from google import genai` client; fall back to `google.generativeai` if that import fails. Create a client from the API key or environment; call **`generate_content`** with **`system_instruction`** and **multimodal `contents`** that include inline PNG data.

**EXACT system instructions (copy verbatim):**

### EXPAND (`mode=expand`) — use this exact `system_instruction`

```
You are an expert prompt engineer for a state-of-the-art image editing model. Your task is to take a user's simple instruction and a set of input images, and expand the instruction into a detailed, descriptive prompt that will guide the image model to produce a high-quality edited result.

Focus on describing the necessary changes based on the user's instruction while considering the content of the original images. Your prompt must include:
 (!) **Always** describe small, specific, targeted edits that will move you to the desired result.
- **Visual Style:** Match the existing style (e.g., photorealistic, oil painting).
- **Composition & Framing:** Describe changes in relation to the existing composition.
- **Camera:** Describe camera parameters and image style.
- **Lighting:** Describe how lighting should be altered or added, matching the existing light source.
- **Details & Texture:** Mention specific details from the original image that should be changed.
- **Action:** Clearly describe the edit to be performed on the image.
- **Resulting image:** Describe what the result will look like.

The output must be ONLY the new, detailed prompt. Do not add any conversational text, greetings, or explanations.
```

**Contents to send:**

* `role: 'user'`, `parts`: `[ inline PNG of the image, { text: goal } ]`

### REVIEW (`mode=review`) — use this exact `system_instruction`

```
You are a self-critical, multimodal AI assistant. Your primary function is to edit images based on user instructions, but a crucial part of your process is to analyze your own work to ensure quality.

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
- **Resulting image:** Describe what the result will look like.
```

**Contents to send (mimic the TS flow):**

* Start with: `role:'user'` parts: `[ initial image(s) as inline PNG, { text: \"My instruction is: \"<goal>\"\" } ]`
* For each history step: push two `role:'model'` parts: (1) `{ inline image of previous output }`, (2) `{ text: step.analysisText }`
* Add the latest generated image as `role:'model'` with inline PNG.
* Add a final `role:'user'` with text: `Look at the image and analyze if you failed to follow the instructions.`
  **Decision:**
* If the returned text is exactly `SUCCESS`, set `done=True` and output an empty `prompt`.
* Else, `done=False` and output the returned text as the **corrective prompt**.

### FORCE REVIEW (`mode=force_review`) — use this exact `system_instruction`

```
You are a self-critical, multimodal AI assistant. You previously generated an image that you may have thought was correct, but the user was unsatisfied and has forced you to re-evaluate your work. Your task is to find the flaws in your last generated image.

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
- **Resulting image:** Describe what the result will look like.
```

**Contents to send:** Same conversation framing as REVIEW, but the final `role:'user'` text depends on optional `feedback`:

* If `feedback` provided and non-empty: `${feedback}. Re-analyze your work, find the flaw, and provide the detailed, first-person critique as instructed.`
* Else: `The image still failed to meet the instructions. Re-analyze your work, find the flaw, and provide the detailed, first-person critique as instructed.`

**Implementation details:**

* Convert ComfyUI image tensor to PNG base64 via `comfy.utils.tensor_to_pil(...)` then `io.BytesIO()` + `base64.b64encode`.
* For the new SDK (`from google import genai`):

  * `client = genai.Client(api_key=api_key or None)`; rely on env if key empty.
  * `client.models.generate_content(model=model, contents=[...], config={"system_instruction": ..., "temperature": ...})`.
* For legacy (`import google.generativeai as genai`):

  * `genai.configure(api_key=...)` then `model = genai.GenerativeModel(model)` and `model.generate_content([...], system_instruction=...)`.
* Return `(prompt, done)`; when a network/SDK error occurs, return `(f"<director_error:{e}>", True)` to break the loop safely.

---

## Node 2 — `PromptSwitch`

**Purpose:** If `iter==0`, output `first_prompt`; else output `next_prompt`.

**Inputs:**

* `iter` : INT (0-based, provided by the loop framework)
* `first_prompt` : STRING (from `GeminiDirector` with `mode=expand`)
* `next_prompt`  : STRING (from `GeminiDirector` with `mode=review/force_review`)

**Output:**

* `current_prompt` : STRING

**Behavior:** Return `first_prompt` when `iter==0` else `next_prompt` (empty string allowed).

---

## Looping **inside** ComfyUI (no external code)

Implement for **ControlFlowUtils** (preferred). The workflow wiring is:

1. `LoopOpen` → outputs `iter`.
2. **Initial pass** (`iter==0`):

   * `GeminiDirector(mode=expand, goal, image=initial, model, api_key)` → `first_prompt`.
3. **Every pass**:

   * `PromptSwitch(iter, first_prompt, next_prompt)` → `current_prompt`.
   * **img2img chain** with stock nodes: `VAEEncode → KSampler → VAEDecode`.
   * Send the decoded image to `GeminiDirector(mode=review, goal, image=decoded, model, api_key, history_json)` → `{done, next_prompt}`.
4. `LoopClose` with `continue = (!done) && (iter < max_loops)`; feed the decoded image forward as the next pass’s input.

**Notes:**

* The actor’s world is **only** the last image + latest prompt each pass.
* Maintain a `history_json` string externally via a simple Text Accumulator node if available; otherwise pass `[]`.
* If `review` returns empty text with `done=false`, coerce `done=true` in the node to avoid dead loops.

---

## Coding Requirements

* Include both classes, `NODE_CLASS_MAPPINGS`, `NODE_DISPLAY_NAME_MAPPINGS`.
* Categories: `Director/Gemini`.
* Use try/catch around SDK calls; timeouts ~60s are reasonable.
* Keep the file < ~250 lines if possible; no helpers split into multiple files.

---

## Acceptance Criteria

* Two nodes only: `GeminiDirector`, `PromptSwitch`.
* Uses Gemini Python SDK in ComfyUI Core (new import with fallback to legacy).
* Uses the **exact** system instructions from the user’s TypeScript for expand/review/force-review.
* Runs in-graph via ControlFlow loop; no CLI, no external orchestrator.
* Loop terminates on `SUCCESS`, `max_loops`, or error; errors surface as `done=true` with a `<director_error:...>` prompt.
* Works with stock `KSampler` as the actor; no custom sampler code.

---

**End of Codex prompt. Implement exactly this and nothing else.**
