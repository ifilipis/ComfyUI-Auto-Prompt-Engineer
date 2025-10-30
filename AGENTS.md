You are creating a custom ComfyUI node. You must update this document with issues, goals, and tasks

## Issues
- Previous nodes (`GeminiDirector`, `PromptSwitch`) did not match the single-output director + image router workflow.
- Front-end lacked an executor to orchestrate director/actor group iterations.

## Goals
- Provide a Gemini-driven director node that emits status events and a tail image router that persists outputs.
- Mirror LG_GroupExecutor-style front-end control with prompt filtering and queue management.

## Tasks
- Replace legacy nodes with `DirectorGemini` and `ImageRouter` implementations that emit the required events.
- Add front-end helpers (`director_actor_executor.js`, `director_actor_prompt_filter.js`, `director_actor_queue_utils.js`) to run director/actor groups with filtered prompts and cancel support.
