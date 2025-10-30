You are creating a custom ComfyUI node. You must update this document with issues, goals, and tasks

## Issues
- Need director/actor orchestration to mirror LG_GroupExecutor behaviour without modifying restricted APIs.
- Require lifecycle visibility to confirm Director node stages during execution.
- Director loops must stop once Gemini emits SUCCESS and preserve conversation history across passes.
- Session boundaries must reset with each executor run so prior prompts do not leak.
- Director executor must wait for explicit Gemini status updates before scheduling actors so SUCCESS halts the loop cleanly.
- Hidden link routing inputs should stay internal to avoid exposing manual wiring in the graph UI.
- Fallback queuing must dispatch each phase only once so actor targets do not double-submit and block director reruns.

## Goals
- Provide a single-output Gemini director node that emits control events.
- Persist actor images via an Image Router node and notify the front-end.
- Drive group execution loops and prompt slicing purely from front-end extensions.
- Auto-manage persistence for history snapshots so review context survives between iterations.

## Tasks
- Implement `DirectorGemini` and `ImageRouter` classes with required payloads and events.
- Add `ImageRouterSink` and `LatestImageSource` nodes to persist iterations and publish atomic latest pointers.
- Retire the legacy `ImageRouter` node mappings once the sink/source pair fully replace it.
- Add queue slicing utilities plus `/prompt` interception to scope execution.
- Create an executor node that manages run/cancel loops, queue polling, and debug logging.
- Instrument DirectorGemini and front-end interceptors with concise debug output for every phase.
- Internalize storage parameters, append iteration history automatically, and surface loop count controls on the executor UI.
- Reset executor link IDs on Run to start fresh sessions and mirror the new debug outputs.
- Clamp executor loop counts to integers and log the wait for director status before each actor pass.
- Ensure manual queue fallbacks coalesce their targets per phase before dispatching to avoid duplicate runs.
- Add optional system instruction overrides to the DirectorGemini node.
- Add a force analyze control that reuses the active session, clears SUCCESS history entries, and reruns the review loop.
