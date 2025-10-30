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
      !daeState.stopRequested &&
      daeState.targetIds &&
      daeState.targetIds.size > 0
    ) {
      try {
        const body = requestOptions.body;
        if (body) {
          const parsed = typeof body === "string" ? JSON.parse(body) : body;
          const extra = parsed.extra_data || {};

          if (!extra.isDirectorActorRequest) {
            const allowedTargets = daeState.allowedTargetIds || new Set();
            const originalTargets = Array.from(daeState.targetIds || []);
            const sanitizedTargets = [];
            for (const id of originalTargets) {
              if (allowedTargets.size === 0 || allowedTargets.has(id)) {
                sanitizedTargets.push(id);
              } else {
                console.error("[DirectorActor] Dropping target outside active group", {
                  id,
                  phase: daeState.phase,
                  group: daeState.phaseGroupName,
                });
              }
            }

            if (daeState.phase === "director" && sanitizedTargets.length > 1) {
              console.error("[DirectorActor] Director phase received multiple targets", {
                phase: daeState.phase,
                group: daeState.phaseGroupName,
                sanitizedTargets,
              });
              sanitizedTargets.splice(1);
            }

            if (sanitizedTargets.length === 0) {
              console.error("[DirectorActor] No valid targets after sanitization; skipping filter", {
                phase: daeState.phase,
                group: daeState.phaseGroupName,
                originalTargets,
              });
              return originalFetchApi(path, requestOptions);
            }

            daeState.targetIds.clear();
            sanitizedTargets.forEach((id) => daeState.targetIds.add(id));

            console.debug("[DirectorActor] Filtering prompt request", {
              path,
              phase: daeState.phase,
              group: daeState.phaseGroupName,
              targetCount: daeState.targetIds.size,
              targetIds: sanitizedTargets,
            });

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
