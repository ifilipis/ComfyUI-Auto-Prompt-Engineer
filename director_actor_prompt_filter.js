import { api } from "../scripts/api.js";
import { buildFilteredPrompt } from "./director_actor_queue_utils.js";

if (!window.__DAE_state) {
  window.__DAE_state = { active: false, targetIds: [] };
}

const originalFetchApi = api.fetchApi.bind(api);

api.fetchApi = async function (path, options = {}) {
  if (
    path === "/prompt" &&
    window.__DAE_state.active &&
    options?.body &&
    typeof options.body === "string"
  ) {
    try {
      const payload = JSON.parse(options.body);
      const extra = payload.extra_data || {};
      if (!extra.isDirectorActorRequest) {
        const filtered = buildFilteredPrompt(payload.prompt, window.__DAE_state.targetIds);
        payload.prompt = filtered;
        payload.extra_data = { ...extra, isDirectorActorRequest: true };
        options = { ...options, body: JSON.stringify(payload) };
      }
    } catch (error) {
      console.warn("[DirectorActor] Failed to filter prompt:", error);
    }
  }
  return originalFetchApi(path, options);
};
