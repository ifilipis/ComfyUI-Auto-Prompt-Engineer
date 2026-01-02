You are creating a custom ComfyUI node. You must update this document with issues, goals, and tasks

## Issues
- Need director/actor orchestration to mirror LG_GroupExecutor behaviour without modifying restricted APIs.
- Require lifecycle visibility to confirm Director node stages during execution.
- Director loops must stop once Gemini emits SUCCESS and preserve conversation history across passes.
- Session boundaries must reset with each executor run so prior prompts do not leak.
- Director executor must wait for explicit Gemini status updates before scheduling actors so SUCCESS halts the loop cleanly.
- Hidden link routing inputs should stay internal to avoid exposing manual wiring in the graph UI.
- Fallback queuing must dispatch each phase only once so actor targets do not double-submit and block director reruns.
- Qwen3-VL support is missing, so there are no nodes for loading and running Qwen3-VL checkpoints.
- Model configuration data for Qwen3-VL checkpoints is absent, preventing selection of Qwen3-VL variants.
- The package does not declare transformer-based dependencies required for Qwen3-VL execution.
- Nodes currently register under multiple categories, instead of sharing a single ComfyUI category (AutoPromptEngineer).
- Director-style orchestration is missing for Qwen3-VL, so Qwen nodes cannot drive the same loop behavior as DirectorGemini.

## Goals
- Provide a single-output Gemini director node that emits control events.
- Persist actor images via an Image Router node and notify the front-end.
- Drive group execution loops and prompt slicing purely from front-end extensions.
- Auto-manage persistence for history snapshots so review context survives between iterations.
- Add Qwen3-VL nodes that can run multimodal prompts against Qwen3-VL backends.
- Provide a Qwen3-VL model configuration list so users can select supported checkpoints.
- Ensure required dependencies for Qwen3-VL nodes are documented in the package requirements.
- Align all node categories so every node appears under the same ComfyUI category (AutoPromptEngineer).
- Provide a Qwen3-VL director node that mirrors DirectorGemini behavior.

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
- Expose a force analyze feedback input on the DirectorActorExecutor UI so user critiques persist across reruns.
- Provide a dedicated force analyze system instruction override on DirectorGemini for tailored review prompts.
- Create Qwen3-VL node implementations (basic + advanced) that load checkpoints and run multimodal prompts.
- Add a Qwen3-VL model config file listing the supported checkpoints and defaults.
- Update requirements to include transformer stack dependencies needed for Qwen3-VL execution.
- Register all nodes under the shared ComfyUI category (AutoPromptEngineer).
- Add a DirectorQwen3VL node that mirrors DirectorGemini history handling and director-status events.
