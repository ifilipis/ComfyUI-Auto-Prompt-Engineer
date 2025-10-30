import { app } from "/scripts/app.js";

const daeState = window.__DAE_state || (window.__DAE_state = {});

daeState.targetIds = daeState.targetIds || new Set();
daeState.currentLinkId = daeState.currentLinkId || null;

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

  const groupMembership = {};
  const phase = daeState.phase;
  const phaseGroup = daeState.phaseGroupName;
  if (app?.graph) {
    const groups = app.graph._groups || [];
    const getRect = (entity) => {
      if (!entity) {
        return [0, 0, 0, 0];
      }
      if (typeof entity.getBounding === "function") {
        return entity.getBounding();
      }
      const pos = entity.pos || [0, 0];
      const size = entity.size || [0, 0];
      return [pos[0], pos[1], size[0], size[1]];
    };
    const rectContains = (container, inner) => {
      const [cx, cy, cw, ch] = container;
      const [ix, iy, iw, ih] = inner;
      if (cw <= 0 || ch <= 0) {
        return false;
      }
      const insideX = ix >= cx && ix + iw <= cx + cw;
      const insideY = iy >= cy && iy + ih <= cy + ch;
      return insideX && insideY;
    };

    collected.forEach((id) => {
      const numericId = Number(id);
      const node =
        app.graph.getNodeById?.(numericId) ||
        (app.graph._nodes || []).find((candidate) => candidate?.id === numericId);
      if (!node) {
        return;
      }
      const nodeRect = getRect(node);
      for (const group of groups) {
        if (rectContains(getRect(group), nodeRect)) {
          groupMembership[id] = group?.title || null;
          return;
        }
      }
      groupMembership[id] = null;
    });
  }

  console.debug("[DirectorActor] Collected nodes for slicing", {
    phase,
    group: phaseGroup,
    targets: Array.from(targetIdsIterable || []),
    collected: Array.from(collected),
    groups: groupMembership,
  });

  const filtered = {};
  collected.forEach((id) => {
    if (promptGraph[id] != null) {
      filtered[id] = promptGraph[id];
    }
  });

  const linkId = daeState.currentLinkId;
  if (linkId) {
    Object.values(filtered).forEach((nodeData) => {
      if (!nodeData || typeof nodeData !== "object") {
        return;
      }
      if (!nodeData.inputs || typeof nodeData.inputs !== "object") {
        nodeData.inputs = nodeData.inputs ? { ...nodeData.inputs } : {};
      }
      if (
        nodeData.class_type === "DirectorGemini" ||
        nodeData.class_type === "ImageRouterSink" ||
        nodeData.class_type === "LatestImageSource"
      ) {
        nodeData.inputs.link_id = linkId;
      }
    });
  }

  body.prompt = filtered;
  return body;
}
