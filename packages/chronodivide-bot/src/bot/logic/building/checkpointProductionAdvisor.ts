import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import path from "node:path";
import * as readline from "node:readline";
import { fileURLToPath } from "node:url";

import type { ProductionAdvisor } from "./productionAdvisor.js";
import type { ProductionAdvisorInput, ProductionAdvisorOutput } from "./productionAdvisorTypes.js";
import { validateProductionAdvisorOutput } from "./productionAdvisorValidation.js";

type PendingRequest = {
    resolve: (value: ProductionAdvisorOutput) => void;
    reject: (error: Error) => void;
};

type ServiceResponse = {
    id?: number;
    type?: string;
    output?: unknown;
    error?: string;
};

const DEFAULT_PYTHON_EXECUTABLE = "python";

export class CheckpointProductionAdvisor implements ProductionAdvisor {
    public readonly enabled = true;
    public readonly requiresLiveFeaturePayload = true;

    private serviceProcess: ChildProcessWithoutNullStreams | null = null;
    private serviceReadline: readline.Interface | null = null;
    private nextRequestId = 1;
    private pendingRequests = new Map<number, PendingRequest>();

    constructor(
        private checkpointPath: string = process.env.SL_CHECKPOINT_PATH ?? "",
        private pythonExecutable: string = process.env.SL_PYTHON_EXECUTABLE ?? DEFAULT_PYTHON_EXECUTABLE,
    ) {
        if (!this.checkpointPath) {
            throw new Error("CheckpointProductionAdvisor requires SL_CHECKPOINT_PATH or an explicit checkpointPath.");
        }
    }

    private getServiceScriptPath(): string {
        const currentDir = path.dirname(fileURLToPath(import.meta.url));
        return path.resolve(currentDir, "../../../../../chronodivide-bot-sl/live_production_advisor_service.py");
    }

    private handleServiceExit = (code: number | null, signal: NodeJS.Signals | null) => {
        const message = `checkpoint production advisor service exited code=${code ?? "null"} signal=${signal ?? "null"}`;
        const error = new Error(message);
        this.pendingRequests.forEach((pending) => pending.reject(error));
        this.pendingRequests.clear();
        this.serviceReadline?.close();
        this.serviceReadline = null;
        this.serviceProcess = null;
        console.log(`[production-advisor] ${message}`);
    };

    private handleServiceLine = (line: string) => {
        if (!line.trim()) {
            return;
        }

        let response: ServiceResponse;
        try {
            response = JSON.parse(line) as ServiceResponse;
        } catch (error) {
            console.log(`[production-advisor] failed to parse service line: ${line}`);
            return;
        }

        if (response.type === "ready") {
            console.log(`[production-advisor] checkpoint service ready checkpoint=${this.checkpointPath}`);
            return;
        }

        if (!Number.isInteger(response.id)) {
            console.log(`[production-advisor] ignoring malformed service response: ${line}`);
            return;
        }

        const pending = this.pendingRequests.get(response.id as number);
        if (!pending) {
            return;
        }
        this.pendingRequests.delete(response.id as number);

        if (response.error) {
            pending.reject(new Error(response.error));
            return;
        }

        pending.resolve((response.output ?? null) as ProductionAdvisorOutput);
    };

    private ensureService(): ChildProcessWithoutNullStreams {
        if (this.serviceProcess !== null) {
            return this.serviceProcess;
        }

        const serviceScriptPath = this.getServiceScriptPath();
        const child = spawn(
            this.pythonExecutable,
            [serviceScriptPath, "--checkpoint-path", this.checkpointPath],
            {
                env: {
                    ...process.env,
                    PYTHONUNBUFFERED: "1",
                },
                stdio: ["pipe", "pipe", "pipe"],
            },
        );

        this.serviceProcess = child;
        this.serviceReadline = readline.createInterface({ input: child.stdout });
        this.serviceReadline.on("line", this.handleServiceLine);
        child.stderr.on("data", (chunk: Buffer | string) => {
            const text = typeof chunk === "string" ? chunk : chunk.toString("utf8");
            const trimmed = text.trim();
            if (trimmed) {
                console.log(`[production-advisor] service stderr: ${trimmed}`);
            }
        });
        child.on("exit", this.handleServiceExit);
        return child;
    }

    async getRecommendations(input: ProductionAdvisorInput): Promise<ProductionAdvisorOutput> {
        const child = this.ensureService();
        const requestId = this.nextRequestId;
        this.nextRequestId += 1;

        return new Promise<ProductionAdvisorOutput>((resolve, reject) => {
            this.pendingRequests.set(requestId, { resolve, reject });
            child.stdin.write(`${JSON.stringify({ id: requestId, input })}\n`, (error) => {
                if (!error) {
                    return;
                }
                this.pendingRequests.delete(requestId);
                reject(error instanceof Error ? error : new Error(String(error)));
            });
        }).then((output) => validateProductionAdvisorOutput(input, output));
    }

    async dispose(): Promise<void> {
        if (this.serviceProcess === null) {
            return;
        }
        const child = this.serviceProcess;
        this.serviceProcess = null;
        this.serviceReadline?.close();
        this.serviceReadline = null;
        child.removeAllListeners("exit");
        child.kill();
    }
}
