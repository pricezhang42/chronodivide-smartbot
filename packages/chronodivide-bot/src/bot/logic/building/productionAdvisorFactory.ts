import { CheckpointProductionAdvisor } from "./checkpointProductionAdvisor.js";
import type { ProductionAdvisor } from "./productionAdvisor.js";
import { MockProductionAdvisor } from "./mockProductionAdvisor.js";
import { NullProductionAdvisor } from "./nullProductionAdvisor.js";

const isEnabled = () => process.env.LLM_PRODUCTION_ENABLED === "1";

export const createProductionAdvisor = (): ProductionAdvisor => {
    const checkpointPath = process.env.SL_CHECKPOINT_PATH ?? "";
    if (checkpointPath) {
        return new CheckpointProductionAdvisor(checkpointPath);
    }

    if (!isEnabled()) {
        return new NullProductionAdvisor();
    }

    switch ((process.env.LLM_PRODUCTION_PROVIDER ?? "mock").toLowerCase()) {
        case "mock":
            return new MockProductionAdvisor();
        default:
            return new NullProductionAdvisor();
    }
};
