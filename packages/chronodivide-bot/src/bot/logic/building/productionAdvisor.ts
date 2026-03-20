import type { ProductionAdvisorInput, ProductionAdvisorOutput } from "./productionAdvisorTypes.js";

export interface ProductionAdvisor {
    readonly enabled: boolean;
    readonly requiresLiveFeaturePayload: boolean;
    getRecommendations(input: ProductionAdvisorInput): Promise<ProductionAdvisorOutput>;
    dispose?(): Promise<void>;
}
