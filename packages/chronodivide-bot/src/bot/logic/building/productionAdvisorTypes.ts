export {
    ADVISOR_QUEUE_NAMES,
    QUEUE_TYPE_TO_ADVISOR_QUEUE,
} from "../modelcontrol/livePolicyTypes.js";
import type {
    AdvisorQueueName,
    LivePolicyAvailableOption,
    LivePolicyFeaturePayload,
    LivePolicyMapSummary,
    LivePolicyQueueSnapshot,
    LivePolicyReplayPlayer,
    LivePolicyRuntimeState,
    LivePolicyThreatSummary,
} from "../modelcontrol/livePolicyTypes.js";

export type { AdvisorQueueName } from "../modelcontrol/livePolicyTypes.js";
export type ProductionAdvisorAvailableOption = LivePolicyAvailableOption;
export type ProductionAdvisorQueueSnapshot = LivePolicyQueueSnapshot;
export type ProductionAdvisorThreatSummary = LivePolicyThreatSummary;
export type ProductionAdvisorMapSummary = LivePolicyMapSummary;
export type CheckpointAdvisorRuntimeState = LivePolicyRuntimeState;
export type CheckpointAdvisorReplayPlayer = LivePolicyReplayPlayer;
export type CheckpointAdvisorFeaturePayload = LivePolicyFeaturePayload;

export type ProductionAdvisorInput = {
    tick: number;
    attackMode: boolean;
    credits: number;
    harvesters: number;
    ownCounts: Record<string, number>;
    enemyCounts: Record<string, number>;
    requestedUnitTypes: Record<string, number>;
    queues: ProductionAdvisorQueueSnapshot[];
    threat: ProductionAdvisorThreatSummary;
    map: ProductionAdvisorMapSummary;
    checkpointFeatures?: CheckpointAdvisorFeaturePayload;
};

export type ProductionAdvisorOutput = Partial<Record<AdvisorQueueName, Record<string, number>>> | null;
