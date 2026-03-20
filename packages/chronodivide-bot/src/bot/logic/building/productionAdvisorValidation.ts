import type { AdvisorQueueName, ProductionAdvisorInput, ProductionAdvisorOutput } from "./productionAdvisorTypes.js";
import { ADVISOR_QUEUE_NAMES } from "./productionAdvisorTypes.js";

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function validateProductionAdvisorOutput(
    input: ProductionAdvisorInput,
    raw: unknown,
): ProductionAdvisorOutput {
    if (!isRecord(raw)) {
        return null;
    }

    const availableByQueue = Object.fromEntries(
        input.queues.map((queue) => [
            queue.queue,
            new Set(queue.availableOptions.map((option) => option.name)),
        ]),
    ) as Record<AdvisorQueueName, Set<string>>;

    const output: Partial<Record<AdvisorQueueName, Record<string, number>>> = {};
    ADVISOR_QUEUE_NAMES.forEach((queueName) => {
        const queueValue = raw[queueName];
        if (!isRecord(queueValue)) {
            return;
        }

        const sanitizedEntries = Object.entries(queueValue)
            .filter(([name, score]) => availableByQueue[queueName].has(name) && typeof score === "number" && Number.isFinite(score))
            .map(([name, score]) => {
                const numericScore = score as number;
                return [name, Math.round(Math.max(-10, Math.min(10, numericScore)))];
            });

        if (sanitizedEntries.length > 0) {
            output[queueName] = Object.fromEntries(sanitizedEntries);
        }
    });

    return Object.keys(output).length > 0 ? output : null;
}
