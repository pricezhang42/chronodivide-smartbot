import type {
    AdvisorQueueName,
    LivePolicyDebugInfo,
    LivePolicyDebugScore,
    LivePolicyFamily,
    LivePolicyInput,
    LivePolicyOutput,
    LivePolicyTile,
} from "./livePolicyTypes.js";
import { ADVISOR_QUEUE_NAMES } from "./livePolicyTypes.js";

const VALID_FAMILIES = new Set<LivePolicyFamily>([
    "Order",
    "Queue",
    "PlaceBuilding",
    "ActivateSuperWeapon",
    "SellObject",
    "ToggleRepair",
    "ResignGame",
    "Noop",
]);
const VALID_QUEUE_UPDATES = new Set(["Add", "Cancel", "AddNext"]);

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === "object" && value !== null && !Array.isArray(value);
}

function sanitizeNumber(value: unknown, fallback = 0): number {
    return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function sanitizeNullableInteger(value: unknown): number | null {
    return Number.isInteger(value) ? Number(value) : null;
}

function sanitizeTile(value: unknown): LivePolicyTile | null {
    if (!isRecord(value)) {
        return null;
    }
    const rx = sanitizeNullableInteger(value.rx);
    const ry = sanitizeNullableInteger(value.ry);
    if (rx === null || ry === null) {
        return null;
    }
    return { rx, ry };
}

function sanitizeDebugScores(value: unknown): LivePolicyDebugScore[] {
    if (!Array.isArray(value)) {
        return [];
    }
    return value
        .filter((entry): entry is Record<string, unknown> => isRecord(entry))
        .map((entry) => ({
            name: typeof entry.name === "string" ? entry.name : "<unknown>",
            score: sanitizeNumber(entry.score, 0),
        }))
        .slice(0, 5);
}

function sanitizeDebugInfo(value: unknown): LivePolicyDebugInfo {
    if (!isRecord(value)) {
        return {
            familyScore: 0,
            topFamilies: [],
            topSubtypes: [],
        };
    }
    return {
        familyScore: sanitizeNumber(value.familyScore, 0),
        topFamilies: sanitizeDebugScores(value.topFamilies),
        topSubtypes: sanitizeDebugScores(value.topSubtypes),
        notes: Array.isArray(value.notes) ? value.notes.filter((item): item is string => typeof item === "string") : undefined,
    };
}

function sanitizeQueueName(value: unknown): AdvisorQueueName | null {
    if (typeof value !== "string") {
        return null;
    }
    return ADVISOR_QUEUE_NAMES.includes(value as AdvisorQueueName) ? (value as AdvisorQueueName) : null;
}

function sanitizeFamily(value: unknown): LivePolicyFamily {
    return typeof value === "string" && VALID_FAMILIES.has(value as LivePolicyFamily)
        ? (value as LivePolicyFamily)
        : "Noop";
}

export function validateLivePolicyOutput(_input: LivePolicyInput, raw: unknown): LivePolicyOutput {
    if (!isRecord(raw)) {
        return {
            family: "Noop",
            score: 0,
            debug: { familyScore: 0, topFamilies: [], topSubtypes: [], notes: ["non_object_output"] },
            order: null,
            queue: null,
            placeBuilding: null,
            superWeapon: null,
            targetEntityId: null,
        };
    }

    const family = sanitizeFamily(raw.family);
    const score = sanitizeNumber(raw.score, 0);
    const debug = sanitizeDebugInfo(raw.debug);
    const targetEntityId = sanitizeNullableInteger(raw.targetEntityId);

    const order = isRecord(raw.order)
        ? {
              orderType: typeof raw.order.orderType === "string" ? raw.order.orderType : null,
              targetMode: typeof raw.order.targetMode === "string" ? raw.order.targetMode : null,
              queueFlag: raw.order.queueFlag === true,
              unitIds: Array.isArray(raw.order.unitIds)
                  ? raw.order.unitIds
                        .filter((value): value is number => Number.isInteger(value))
                        .map((value) => Number(value))
                        .slice(0, 64)
                  : [],
              targetEntityId: sanitizeNullableInteger(raw.order.targetEntityId),
              targetLocation: sanitizeTile(raw.order.targetLocation),
              targetLocation2: sanitizeTile(raw.order.targetLocation2),
          }
        : null;

    const queue = isRecord(raw.queue)
        ? {
              queue: sanitizeQueueName(raw.queue.queue),
              updateType:
                  typeof raw.queue.updateType === "string" && VALID_QUEUE_UPDATES.has(raw.queue.updateType)
                      ? (raw.queue.updateType as "Add" | "Cancel" | "AddNext")
                      : null,
              objectName: typeof raw.queue.objectName === "string" ? raw.queue.objectName : null,
              quantity: (() => {
                  const numeric = sanitizeNullableInteger(raw.queue.quantity);
                  return numeric === null ? null : Math.max(1, numeric);
              })(),
          }
        : null;

    const placeBuilding = isRecord(raw.placeBuilding)
        ? {
              buildingName: typeof raw.placeBuilding.buildingName === "string" ? raw.placeBuilding.buildingName : null,
              targetLocation: sanitizeTile(raw.placeBuilding.targetLocation),
          }
        : null;

    const superWeapon = isRecord(raw.superWeapon)
        ? {
              typeName: typeof raw.superWeapon.typeName === "string" ? raw.superWeapon.typeName : null,
              targetLocation: sanitizeTile(raw.superWeapon.targetLocation),
              targetLocation2: sanitizeTile(raw.superWeapon.targetLocation2),
          }
        : null;

    return {
        family,
        score,
        debug,
        order,
        queue,
        placeBuilding,
        superWeapon,
        targetEntityId,
    };
}
