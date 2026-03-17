import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";

const require = createRequire(import.meta.url);

const INTERNAL_EXPORTS = [
  "Replay",
  "Parser",
  "Serializer",
  "Engine",
  "GameFactory",
  "ActionFactory",
  "ActionFactoryReg",
  "GameApi",
  "MapFile",
  "TileSets",
  "BoxedVar",
  "AppLogger",
  "ReplayRecorder",
  "ReplayEventFactory",
  "TurnActionsReplayEvent",
  "MixinRules",
];

const PUBLIC_EXPORT_PATTERN = /export\{[^}]*\bcdapi\b\};/;
const BRIDGE_FILENAME = "index.replay-bridge.mjs";

function getGameApiSourcePath() {
  return require.resolve("@chronodivide/game-api");
}

function buildBridgeSource(sourceText) {
  const exportLine = `export { ${INTERNAL_EXPORTS.join(", ")} };`;
  if (sourceText.includes(exportLine)) {
    return sourceText;
  }

  const match = sourceText.match(PUBLIC_EXPORT_PATTERN);
  if (!match) {
    throw new Error("Could not find the public export block in @chronodivide/game-api/dist/index.js.");
  }

  return sourceText.replace(match[0], `${match[0]}\n${exportLine}`);
}

export function ensureGameApiBridge() {
  const sourcePath = getGameApiSourcePath();
  const bridgePath = path.join(path.dirname(sourcePath), BRIDGE_FILENAME);
  const sourceText = fs.readFileSync(sourcePath, "utf8");
  const bridgeText = buildBridgeSource(sourceText);

  if (!fs.existsSync(bridgePath) || fs.readFileSync(bridgePath, "utf8") !== bridgeText) {
    fs.writeFileSync(bridgePath, bridgeText, "utf8");
  }

  return bridgePath;
}

export async function loadGameApiBridge({ cacheBust = true } = {}) {
  const bridgePath = ensureGameApiBridge();
  const bridgeUrl = pathToFileURL(bridgePath).href;
  const href = cacheBust ? `${bridgeUrl}?v=${Date.now()}` : bridgeUrl;
  return import(href);
}
