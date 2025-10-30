import { app } from "/scripts/app.js";

export const directorActorState = (function () {
    if (!window.__DAE_state) {
        window.__DAE_state = {
            active: false,
            targetNodeIds: new Set(),
            linkId: null,
        };
    }
    return window.__DAE_state;
})();

export function collectRelatedNodes(prompt, nodeId, accumulator) {
    if (!prompt || !prompt[nodeId] || accumulator.has(nodeId)) {
        return;
    }

    accumulator.add(nodeId);
    const node = prompt[nodeId];
    const inputs = node.inputs || {};

    Object.values(inputs).forEach((inputValue) => {
        if (Array.isArray(inputValue) && inputValue.length > 0) {
            const upstream = inputValue[0];
            if (upstream !== undefined && upstream !== null) {
                collectRelatedNodes(prompt, upstream, accumulator);
            }
        }
    });
}

export function buildFilteredPrompt(originalPrompt, targetNodeIds) {
    if (!originalPrompt || !targetNodeIds || targetNodeIds.length === 0) {
        return originalPrompt;
    }

    const relevant = new Set();
    for (const id of targetNodeIds) {
        collectRelatedNodes(originalPrompt, String(id), relevant);
    }

    const filtered = {};
    relevant.forEach((id) => {
        if (originalPrompt[id] != null) {
            filtered[id] = originalPrompt[id];
        }
    });

    return filtered;
}

export function getGroupByName(groupName) {
    if (!groupName) {
        return null;
    }
    return app.graph?._groups?.find((group) => group?.title === groupName) ?? null;
}

export function getNodesInsideGroup(group) {
    if (!group) {
        return [];
    }

    const nodes = [];
    const bounding = group._bounding || group._bounds || group.bounding || group.rect;

    for (const node of app.graph._nodes) {
        if (!node?.getBounding) {
            continue;
        }
        const nodeBounding = node.getBounding();
        if (LiteGraph.overlapBounding(bounding, nodeBounding)) {
            nodes.push(node);
        }
    }
    return nodes;
}

export function resolveOutputNodes(nodes) {
    if (!Array.isArray(nodes)) {
        return [];
    }

    return nodes.filter((node) => {
        if (!node || node.mode === LiteGraph.NEVER) {
            return false;
        }
        if (node.constructor?.nodeData?.output_node) {
            return true;
        }
        if (!Array.isArray(node.outputs) || node.outputs.length === 0) {
            return true;
        }
        return node.outputs.every((output) => !output?.links || output.links.length === 0);
    });
}
