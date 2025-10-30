const daeState = window.__DAE_state || (window.__DAE_state = {});

daeState.targetIds = daeState.targetIds || new Set();

function normalizeNodeId(nodeId) {
  return nodeId != null ? String(nodeId) : "";
}

function shouldSkipNode(node, id) {
  if (!node) {
    console.warn(`[DirectorActor] Missing node ${id} while slicing prompt`);
    return true;
  }
  if (node.disabled === true) {
    console.warn(`[DirectorActor] Skipping disabled node ${id}`);
    return true;
  }
  if (typeof node.mode === "number" && node.mode === 4) {
    console.warn(`[DirectorActor] Skipping node ${id} with mode=4`);
    return true;
  }
  return false;
}

export function collectRelatedNodes(promptGraph, nodeId, collected = new Set()) {
  const id = normalizeNodeId(nodeId);
  if (!id || collected.has(id)) {
    return collected;
  }

  const node = promptGraph?.[id];
  if (shouldSkipNode(node, id)) {
    return collected;
  }

  collected.add(id);

  const inputs = node.inputs || {};
  for (const value of Object.values(inputs)) {
    if (Array.isArray(value) && value.length > 0) {
      const first = value[0];
      if (Array.isArray(first)) {
        for (const connection of value) {
          if (Array.isArray(connection) && connection.length > 0) {
            collectRelatedNodes(promptGraph, connection[0], collected);
          }
        }
      } else {
        collectRelatedNodes(promptGraph, first, collected);
      }
    }
  }

  return collected;
}

export function buildFilteredPrompt(originalBody, targetIdsIterable) {
  const body = originalBody ? { ...originalBody } : {};
  const promptGraph = body.prompt ? { ...body.prompt } : {};

  const collected = new Set();
  for (const rawId of targetIdsIterable || []) {
    collectRelatedNodes(promptGraph, rawId, collected);
  }

  const filtered = {};
  collected.forEach((id) => {
    if (promptGraph[id] != null) {
      filtered[id] = promptGraph[id];
    }
  });

  body.prompt = filtered;
  return body;
}
