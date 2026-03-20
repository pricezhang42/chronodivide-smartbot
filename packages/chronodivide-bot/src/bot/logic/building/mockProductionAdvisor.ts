import type { ProductionAdvisor } from "./productionAdvisor.js";
import type { ProductionAdvisorInput, ProductionAdvisorOutput } from "./productionAdvisorTypes.js";
import { validateProductionAdvisorOutput } from "./productionAdvisorValidation.js";

const addIfAvailable = (
    recommendations: Record<string, number>,
    availableOptions: string[],
    name: string,
    score: number,
) => {
    if (availableOptions.includes(name) && score !== 0) {
        recommendations[name] = score;
    }
};

export class MockProductionAdvisor implements ProductionAdvisor {
    public readonly enabled = true;
    public readonly requiresLiveFeaturePayload = false;

    async getRecommendations(input: ProductionAdvisorInput): Promise<ProductionAdvisorOutput> {
        const output: Record<string, Record<string, number>> = {};
        input.queues.forEach((queue) => {
            const recommendations: Record<string, number> = {};
            const availableOptions = queue.availableOptions.map((option) => option.name);

            Object.entries(input.requestedUnitTypes).forEach(([unitName, priority]) => {
                if (availableOptions.includes(unitName)) {
                    recommendations[unitName] = Math.round(priority * 0.2);
                }
            });

            if (queue.queue === "Structures") {
                addIfAvailable(recommendations, availableOptions, "GAPOWR", input.credits < 400 ? 8 : 0);
                addIfAvailable(recommendations, availableOptions, "NAPOWR", input.credits < 400 ? 8 : 0);
                addIfAvailable(recommendations, availableOptions, "GAREFN", input.harvesters < 2 ? 6 : 0);
                addIfAvailable(recommendations, availableOptions, "NAREFN", input.harvesters < 2 ? 6 : 0);
            }

            if (queue.queue === "Vehicles") {
                addIfAvailable(recommendations, availableOptions, "MTNK", input.attackMode ? 5 : 2);
                addIfAvailable(recommendations, availableOptions, "HTNK", input.attackMode ? 5 : 2);
                addIfAvailable(
                    recommendations,
                    availableOptions,
                    "FV",
                    input.threat?.enemyAirThreat && input.threat.enemyAirThreat > 0 ? 3 : 1,
                );
                addIfAvailable(
                    recommendations,
                    availableOptions,
                    "HTK",
                    input.threat?.enemyAirThreat && input.threat.enemyAirThreat > 0 ? 3 : 1,
                );
            }

            if (queue.queue === "Infantry") {
                addIfAvailable(recommendations, availableOptions, "E1", input.attackMode ? 2 : 1);
                addIfAvailable(recommendations, availableOptions, "E2", input.attackMode ? 2 : 1);
            }

            if (Object.keys(recommendations).length > 0) {
                output[queue.queue] = recommendations;
            }
        });

        return validateProductionAdvisorOutput(input, output);
    }
}
