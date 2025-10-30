import { app } from "../scripts/app.js";

function normalizeIds(nodeIds) {
  return Array.from(new Set((nodeIds || []).map((id) => String(id))));
}

export function collectRelatedNodes(output, nodeId, collected) {
  const id = String(nodeId);
  if (!output || collected.has(id)) {
    return;
  }
  const node = output[id];
  if (!node) {
    return;
  }
  collected.add(id);
  const inputs = node.inputs || {};
  for (const value of Object.values(inputs)) {
    if (Array.isArray(value) && value.length > 0) {
      collectRelatedNodes(output, value[0], collected);
    }
  }
}

export function buildFilteredPrompt(promptPayload, targetNodeIds) {
  if (!promptPayload || !promptPayload.output) {
    return promptPayload;
  }
  const keep = new Set();
  const normalized = normalizeIds(targetNodeIds);
  for (const id of normalized) {
    collectRelatedNodes(promptPayload.output, id, keep);
  }
  if (!keep.size) {
    return promptPayload;
  }
  const nextOutput = {};
  for (const id of keep) {
    if (promptPayload.output[id]) {
      nextOutput[id] = promptPayload.output[id];
    }
  }
  return { ...promptPayload, output: nextOutput };
}

export function findGroupByName(groupName) {
  if (!groupName) {
    return null;
  }
  return app.graph?._groups?.find((group) => group?.title === groupName) || null;
}

export function nodesInsideGroup(group) {
  if (!group || !app.graph?._nodes) {
    return [];
  }
  const bounds = group._bounding || group._rect || group.rect;
  if (!bounds) {
    return [];
  }
  const [gx, gy, gw, gh] = bounds;
  return app.graph._nodes.filter((node) => {
    if (!node?.pos || !node?.size) {
      return false;
    }
    const [nx, ny] = node.pos;
    const [nw, nh] = node.size;
    return nx + nw > gx && ny + nh > gy && nx < gx + gw && ny < gy + gh;
  });
}

export function getGroupOutputNodeIds(groupName) {
  const group = findGroupByName(groupName);
  if (!group) {
    return [];
  }
  const nodes = nodesInsideGroup(group);
  const outputs = nodes.filter((node) => {
    if (node?.constructor?.nodeData?.output_node) {
      return true;
    }
    if (!node?.outputs?.length) {
      return false;
    }
    return node.outputs.every((slot) => !slot?.links || slot.links.length === 0);
  });
  return outputs.map((node) => node.id);
}
