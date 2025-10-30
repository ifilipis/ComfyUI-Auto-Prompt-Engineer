You are creating a custom ComfyUI node. You must update this document with issues, goals, and tasks

## Issues
- Director loop missing JS orchestration for Gemini-driven workflow.
- No event-driven image router to notify the front-end when renders finish.

## Goals
- Provide a single-output Gemini director node that emits status events.
- Add an image routing node that persists results and triggers front-end updates.
- Supply matching JavaScript utilities to drive group execution safely.

## Tasks
- [x] Replace legacy Gemini nodes with `DirectorGemini` and `ImageRouter` implementations.
- [x] Emit `director-status` and `img-ready` events from Python nodes.
- [x] Ship `director_actor_executor.js`, `director_actor_prompt_filter.js`, and `director_actor_queue_utils.js` at the package root.
