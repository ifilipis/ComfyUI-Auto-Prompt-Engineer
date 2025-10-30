import { api } from "/scripts/api.js";
import { buildFilteredPrompt } from "./director_actor_queue_utils.js";

const daeState = window.__DAE_state || (window.__DAE_state = {});

daeState.targetIds = daeState.targetIds || new Set();
daeState.activeCount = daeState.activeCount || 0;
daeState.active = daeState.active || false;
daeState.phaseTargetIds = daeState.phaseTargetIds || new Set();
daeState.phase = daeState.phase || null;
daeState.phaseGroupName = daeState.phaseGroupName || null;
daeState.directorOutputId = daeState.directorOutputId || null;
daeState.stopOnSuccess = daeState.stopOnSuccess || false;

if (!daeState.__promptFilterPatched) {
  const originalFetchApi = api.fetchApi.bind(api);

  api.fetchApi = async function patchedFetchApi(path, options = {}) {
    let requestOptions = options;

    if (
      path === "/prompt" &&
      requestOptions?.method?.toUpperCase() === "POST" &&
      daeState.active &&
      !daeState.stopOnSuccess &&
      daeState.targetIds &&
      daeState.targetIds.size > 0
    ) {
      try {
        const body = requestOptions.body;
        if (body) {
          const parsed = typeof body === "string" ? JSON.parse(body) : body;
          const extra = parsed.extra_data || {};

          if (!extra.isDirectorActorRequest) {
            const rawTargets = Array.from(daeState.targetIds || []);
            const normalizedTargets = rawTargets.map((id) => String(id));
            const phaseTargets = new Set();
            const expected = daeState.phaseTargetIds;
            if (expected && expected.size > 0) {
              normalizedTargets.forEach((id) => {
                if (expected.has(id)) {
                  phaseTargets.add(id);
                }
              });
            } else {
              normalizedTargets.forEach((id) => phaseTargets.add(id));
            }

            if (daeState.phase === "director") {
              if (phaseTargets.size === 0 && daeState.directorOutputId) {
                phaseTargets.add(daeState.directorOutputId);
              }
              if (phaseTargets.size !== 1) {
                console.error("[DirectorActor] Invalid director target set", {
                  phaseTargets: Array.from(phaseTargets),
                  normalizedTargets,
                  directorOutputId: daeState.directorOutputId,
                });
                if (daeState.directorOutputId) {
                  phaseTargets.clear();
                  phaseTargets.add(daeState.directorOutputId);
                }
              }
            }

            const targetsForFilter = phaseTargets.size > 0 ? phaseTargets : new Set(normalizedTargets);

            console.debug("[DirectorActor] Filtering prompt request", {
              path,
              targetCount: targetsForFilter.size,
              phase: daeState.phase,
              group: daeState.phaseGroupName,
              targets: Array.from(targetsForFilter),
            });
            const filtered = buildFilteredPrompt(parsed, targetsForFilter);
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
