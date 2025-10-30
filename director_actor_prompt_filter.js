import { api } from "/scripts/api.js";
import { buildFilteredPrompt } from "./director_actor_queue_utils.js";

const daeState = window.__DAE_state || (window.__DAE_state = {});

daeState.targetIds = daeState.targetIds || new Set();
daeState.activeCount = daeState.activeCount || 0;
daeState.active = daeState.active || false;

if (!daeState.__promptFilterPatched) {
  const originalFetchApi = api.fetchApi.bind(api);

  api.fetchApi = async function patchedFetchApi(path, options = {}) {
    let requestOptions = options;

    if (
      path === "/prompt" &&
      requestOptions?.method?.toUpperCase() === "POST" &&
      daeState.active &&
      daeState.targetIds &&
      daeState.targetIds.size > 0
    ) {
      try {
        const body = requestOptions.body;
        if (body) {
          const parsed = typeof body === "string" ? JSON.parse(body) : body;
          const extra = parsed.extra_data || {};

          if (!extra.isDirectorActorRequest) {
            const filtered = buildFilteredPrompt(parsed, daeState.targetIds);
            filtered.extra_data = { ...extra, isDirectorActorRequest: true };
            requestOptions = {
              ...requestOptions,
              body: JSON.stringify(filtered),
              headers: {
                "Content-Type": "application/json",
                ...requestOptions.headers,
              },
            };
          }
        }
      } catch (error) {
        console.warn("[DirectorActor] Failed to filter prompt payload", error);
      }
    }

    return originalFetchApi(path, requestOptions);
  };

  daeState.__promptFilterPatched = true;
}
