const daeState = window.__DAE_state || (window.__DAE_state = {});

daeState.targetIds = daeState.targetIds || new Set();

function normalizeNodeId(nodeId) {
  return nodeId != null ? String(nodeId) : "";
}

export function collectRelatedNodes(promptGraph, nodeId, collected = new Set()) {
  const id = normalizeNodeId(nodeId);
  if (!id || collected.has(id)) {
    return collected;
  }

  const node = promptGraph?.[id];
  if (!node) {
    return collected;
  }

  collected.add(id);

  const inputs = node.inputs || {};
  for (const value of Object.values(inputs)) {
    if (Array.isArray(value) && value.length > 0) {
      if (Array.isArray(value[0])) {
        for (const connection of value) {
          if (Array.isArray(connection) && connection.length > 0) {
            collectRelatedNodes(promptGraph, connection[0], collected);
          }
        }
      } else {
        collectRelatedNodes(promptGraph, value[0], collected);
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
