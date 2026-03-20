import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import path from "node:path";
import * as readline from "node:readline";
import { fileURLToPath } from "node:url";

import type { LivePolicyInput, LivePolicyOutput } from "./livePolicyTypes.js";
import { validateLivePolicyOutput } from "./livePolicyValidation.js";

type PendingRequest = {
    resolve: (value: LivePolicyOutput) => void;
    reject: (error: Error) => void;
};

type ServiceResponse = {
    id?: number;
    type?: string;
    output?: unknown;
    error?: string;
};

const DEFAULT_PYTHON_EXECUTABLE = "python";

export class CheckpointPolicyClient {
    private serviceProcess: ChildProcessWithoutNullStreams | null = null;
    private serviceReadline: readline.Interface | null = null;
    private nextRequestId = 1;
    private pendingRequests = new Map<number, PendingRequest>();

    constructor(
        private checkpointPath: string = process.env.SL_CHECKPOINT_PATH ?? "",
        private pythonExecutable: string = process.env.SL_PYTHON_EXECUTABLE ?? DEFAULT_PYTHON_EXECUTABLE,
    ) {
        if (!this.checkpointPath) {
            throw new Error("CheckpointPolicyClient requires SL_CHECKPOINT_PATH or an explicit checkpointPath.");
        }
    }

    private getServiceScriptPath(): string {
        const currentDir = path.dirname(fileURLToPath(import.meta.url));
        return path.resolve(currentDir, "../../../../../chronodivide-bot-sl/live_policy_service.py");
    }

    private handleServiceExit = (code: number | null, signal: NodeJS.Signals | null) => {
        const message = `checkpoint policy service exited code=${code ?? "null"} signal=${signal ?? "null"}`;
        const error = new Error(message);
        this.pendingRequests.forEach((pending) => pending.reject(error));
        this.pendingRequests.clear();
        this.serviceReadline?.close();
        this.serviceReadline = null;
        this.serviceProcess = null;
        console.log(`[checkpoint-policy] ${message}`);
    };

    private handleServiceLine = (line: string) => {
        if (!line.trim()) {
            return;
        }

        let response: ServiceResponse;
        try {
            response = JSON.parse(line) as ServiceResponse;
        } catch {
            console.log(`[checkpoint-policy] failed to parse service line: ${line}`);
            return;
        }

        if (response.type === "ready") {
            console.log(`[checkpoint-policy] service ready checkpoint=${this.checkpointPath}`);
            return;
        }

        if (!Number.isInteger(response.id)) {
            console.log(`[checkpoint-policy] ignoring malformed service response: ${line}`);
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

        pending.resolve((response.output ?? null) as LivePolicyOutput);
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
                console.log(`[checkpoint-policy] service stderr: ${trimmed}`);
            }
        });
        child.on("exit", this.handleServiceExit);
        return child;
    }

    async getAction(input: LivePolicyInput): Promise<LivePolicyOutput> {
        const child = this.ensureService();
        const requestId = this.nextRequestId;
        this.nextRequestId += 1;

        return new Promise<LivePolicyOutput>((resolve, reject) => {
            this.pendingRequests.set(requestId, { resolve, reject });
            child.stdin.write(`${JSON.stringify({ id: requestId, input })}\n`, (error) => {
                if (!error) {
                    return;
                }
                this.pendingRequests.delete(requestId);
                reject(error instanceof Error ? error : new Error(String(error)));
            });
        }).then((output) => validateLivePolicyOutput(input, output));
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
