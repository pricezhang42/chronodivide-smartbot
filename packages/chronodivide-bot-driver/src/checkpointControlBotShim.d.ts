declare module "@supalosa/chronodivide-bot/dist/bot/checkpointControlBot.js" {
    import { ApiEvent, Bot, GameApi } from "@chronodivide/game-api";

    export class CheckpointControlBot extends Bot {
        constructor(
            name: string,
            country: string,
            tryAllyWith?: string[],
            enableLogging?: boolean,
            checkpointPath?: string,
            pythonExecutable?: string,
        );
        onGameStart(game: GameApi): void;
        onGameTick(game: GameApi): void;
        onGameEvent(ev: ApiEvent): void;
        hasPendingPolicyRequest(): boolean;
        hasPendingProductionAdvisorRequest(): boolean;
        waitForPendingPolicyRequest(): Promise<void>;
        waitForPendingProductionAdvisorRequest(): Promise<void>;
        dispose(): Promise<void>;
    }
}

declare module "../../chronodivide-bot/dist/bot/checkpointControlBot.js" {
    export { CheckpointControlBot } from "@supalosa/chronodivide-bot/dist/bot/checkpointControlBot.js";
}
