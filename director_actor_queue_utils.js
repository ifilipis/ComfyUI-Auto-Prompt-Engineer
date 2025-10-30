import { app } from "../../scripts/app.js";

const DEFAULT_STATE = {
    active: false,
    targetIds: new Set(),
    owner: null,
};

export function getDirectorActorState() {
    if (!window.__DAE_STATE__) {
        window.__DAE_STATE__ = {
            ...DEFAULT_STATE,
            targetIds: new Set(),
        };
    } else {
        window.__DAE_STATE__.targetIds = window.__DAE_STATE__.targetIds || new Set();
        window.__DAE_STATE__.active = window.__DAE_STATE__.active || false;
    }
    return window.__DAE_STATE__;
}

export function collectRelatedNodes(prompt, nodeId, acc) {
    if (!prompt || !prompt[nodeId] || acc.has(nodeId)) {
        return;
    }
    acc.add(nodeId);
    console.debug("[DirectorActor][QueueUtils] Visiting node", nodeId);
    const node = prompt[nodeId];
    if (!node.inputs) {
        return;
    }
    for (const input of Object.values(node.inputs)) {
        if (Array.isArray(input) && input.length > 0) {
            const upstreamId = input[0];
            if (upstreamId !== undefined && upstreamId !== null) {
                collectRelatedNodes(prompt, upstreamId, acc);
            }
        }
    }
}

export function buildFilteredPrompt(prompt, targetIds) {
    if (!prompt || !targetIds?.size) {
        return prompt;
    }
    console.debug("[DirectorActor][QueueUtils] Building filtered prompt", Array.from(targetIds));
    const relevantNodes = new Set();
    for (const id of targetIds) {
        collectRelatedNodes(prompt, id, relevantNodes);
    }
    const filtered = {};
    for (const id of relevantNodes) {
        if (prompt[id]) {
            filtered[id] = prompt[id];
        }
    }
    return filtered;
}

export function getNodesInsideGroup(groupName) {
    const group = app.graph?._groups?.find((g) => g?.title === groupName);
    if (!group) {
        console.warn("[DirectorActor][QueueUtils] Group not found", groupName);
        return [];
    }
    const bounding = group._bounding || group.bounding || group.rect || [group.pos?.[0] ?? 0, group.pos?.[1] ?? 0, group.size?.[0] ?? 0, group.size?.[1] ?? 0];
    const nodes = app.graph._nodes.filter((node) => {
        if (!node?.pos || !node?.size) {
            return false;
        }
        const nodeBounding = node.getBounding?.() || [node.pos[0], node.pos[1], node.size[0], node.size[1]];
        return LiteGraph.overlapBounding?.(bounding, nodeBounding) ?? overlapBoundingFallback(bounding, nodeBounding);
    });
    console.debug("[DirectorActor][QueueUtils] Nodes inside group", groupName, nodes.map((n) => n.id));
    return nodes;
}

function overlapBoundingFallback(a, b) {
    if (!Array.isArray(a) || !Array.isArray(b)) {
        return false;
    }
    const [ax, ay, aw = 0, ah = 0] = a;
    const [bx, by, bw = 0, bh = 0] = b;
    return ax < bx + bw && ax + aw > bx && ay < by + bh && ay + ah > by;
}

export function guessOutputNodes(nodes) {
    if (!Array.isArray(nodes)) {
        return [];
    }
    const outputs = nodes.filter((node) => {
        if (!node?.outputs?.length) {
            return false;
        }
        const isDeclaredOutput = node.constructor?.nodeData?.output_node === true;
        if (isDeclaredOutput) {
            return true;
        }
        const hasLinks = node.outputs.some((output) => Array.isArray(output?.links) && output.links.length > 0);
        return !hasLinks || node.type === "ImageRouter" || node.type === "SaveImage";
    });
    console.debug("[DirectorActor][QueueUtils] Output nodes guessed", outputs.map((n) => n.id));
    return outputs;
}

export function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}
