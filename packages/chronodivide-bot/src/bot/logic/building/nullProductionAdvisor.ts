import type { ProductionAdvisor } from "./productionAdvisor.js";
import type { ProductionAdvisorInput, ProductionAdvisorOutput } from "./productionAdvisorTypes.js";

export class NullProductionAdvisor implements ProductionAdvisor {
    public readonly enabled = false;
    public readonly requiresLiveFeaturePayload = false;

    async getRecommendations(_input: ProductionAdvisorInput): Promise<ProductionAdvisorOutput> {
        return null;
    }
}
