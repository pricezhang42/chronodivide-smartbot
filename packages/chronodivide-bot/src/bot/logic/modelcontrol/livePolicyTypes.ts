import { QueueStatus, QueueType } from "@chronodivide/game-api";

export const ADVISOR_QUEUE_NAMES = [
    "Structures",
    "Armory",
    "Infantry",
    "Vehicles",
    "Aircrafts",
    "Ships",
] as const;

export type AdvisorQueueName = typeof ADVISOR_QUEUE_NAMES[number];

export const QUEUE_TYPE_TO_ADVISOR_QUEUE: Record<QueueType, AdvisorQueueName> = {
    [QueueType.Structures]: "Structures",
    [QueueType.Armory]: "Armory",
    [QueueType.Infantry]: "Infantry",
    [QueueType.Vehicles]: "Vehicles",
    [QueueType.Aircrafts]: "Aircrafts",
    [QueueType.Ships]: "Ships",
};

export type LivePolicyAvailableOption = {
    name: string;
    cost: number;
};

export type LivePolicyQueueSnapshot = {
    queue: AdvisorQueueName;
    status: QueueStatus;
    activeItems: string[];
    availableOptions: LivePolicyAvailableOption[];
};

export type LivePolicyThreatSummary = {
    certainty: number;
    enemyLandThreat: number;
    enemyAirThreat: number;
    enemyAntiAirThreat: number;
    enemyDefensiveThreat: number;
    ownDefensivePower: number;
    ownAntiGroundPower: number;
    ownAntiAirPower: number;
    ownAirPower: number;
} | null;

export type LivePolicyMapSummary = {
    width: number;
    height: number;
    tileArea: number;
    startingLocationCount: number;
    nearestStartingLocationDistance: number | null;
    nearestEnemyStartDistance: number | null;
    sizeClass: string;
    rushDistanceClass: string;
    rushRisk: string;
    summary: string;
};

export type LivePolicyRuntimeState = {
    lastActionTick: number | null;
    lastActionTypeNameV1: string | null;
    lastQueueValue: number;
    buildOrderActionTypeNamesV1: string[];
    currentSelectedObjectIds?: number[];
    recentActionFamilyNamesV2?: string[];
    recentOrderTypeNamesV2?: string[];
    pendingBuildingQueueByQueueName?: Record<string, string>;
    lastFailedActionReason?: string | null;
};

export type LivePolicyReplayPlayer = {
    name: string;
    countryName: string | null;
    sideId: number;
};

export type LivePolicyFeaturePayload = {
    playerName: string;
    tick: number;
    featureSchemaObservation: {
        scalarFeatureNames: string[];
        entityFeatureNames: string[];
        spatialChannelNames: string[];
        minimapChannelNames: string[];
        maxEntities: number;
        spatialSize: number;
        minimapSize: number;
    };
    featureTensors: {
        scalar: number[];
        lastActionContext: number[];
        currentSelectionCount: number[];
        currentSelectionResolvedCount: number[];
        currentSelectionOverflowCount: number[];
        currentSelectionIndices: number[];
        currentSelectionMask: number[];
        currentSelectionResolvedMask: number[];
        entityNameTokens: number[];
        entityMask: number[];
        entityFeatures: number[][];
        spatial: number[][][];
        minimap: number[][][];
        mapStatic: number[][][];
    };
    entityObjectIds: number[];
    sharedNameVocabulary: {
        idToName: string[];
        nameToId: Record<string, number>;
    };
    playerProduction: Record<string, unknown> | null;
    playerSuperWeapons: Array<Record<string, unknown>>;
    replayPlayers: LivePolicyReplayPlayer[];
    superWeaponRechargeSecondsByType: Record<string, number | null>;
    runtimeState: LivePolicyRuntimeState;
};

export type LivePolicyInput = {
    featurePayload: LivePolicyFeaturePayload;
};

export type LivePolicyFamily =
    | "Order"
    | "Queue"
    | "PlaceBuilding"
    | "ActivateSuperWeapon"
    | "SellObject"
    | "ToggleRepair"
    | "ResignGame"
    | "Noop";

export type LivePolicyTile = {
    rx: number;
    ry: number;
};

export type LivePolicyDebugScore = {
    name: string;
    score: number;
};

export type LivePolicyDebugInfo = {
    familyScore: number;
    topFamilies: LivePolicyDebugScore[];
    topSubtypes: LivePolicyDebugScore[];
    notes?: string[];
};

export type LivePolicyOrderPayload = {
    orderType: string | null;
    targetMode: string | null;
    queueFlag: boolean;
    unitIds: number[];
    targetEntityId: number | null;
    targetLocation: LivePolicyTile | null;
    targetLocation2: LivePolicyTile | null;
};

export type LivePolicyQueuePayload = {
    queue: AdvisorQueueName | null;
    updateType: "Add" | "Cancel" | "AddNext" | null;
    objectName: string | null;
    quantity: number | null;
};

export type LivePolicyPlaceBuildingPayload = {
    buildingName: string | null;
    targetLocation: LivePolicyTile | null;
};

export type LivePolicySuperWeaponPayload = {
    typeName: string | null;
    targetLocation: LivePolicyTile | null;
    targetLocation2: LivePolicyTile | null;
};

export type LivePolicyOutput = {
    family: LivePolicyFamily;
    score: number;
    debug: LivePolicyDebugInfo;
    order: LivePolicyOrderPayload | null;
    queue: LivePolicyQueuePayload | null;
    placeBuilding: LivePolicyPlaceBuildingPayload | null;
    superWeapon: LivePolicySuperWeaponPayload | null;
    targetEntityId: number | null;
};

export function createInitialLivePolicyRuntimeState(): LivePolicyRuntimeState {
    return {
        lastActionTick: null,
        lastActionTypeNameV1: null,
        lastQueueValue: -1,
        buildOrderActionTypeNamesV1: [],
        currentSelectedObjectIds: [],
        recentActionFamilyNamesV2: [],
        recentOrderTypeNamesV2: [],
        pendingBuildingQueueByQueueName: {},
        lastFailedActionReason: null,
    };
}
