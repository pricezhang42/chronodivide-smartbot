import fs from "node:fs";
import { once } from "node:events";
import path from "node:path";

import { parseArgs, parseIntegerArg, requireArg } from "./cli_args.mjs";
import { extractReplaySupervisedDataset } from "./sl_dataset.mjs";

function usage() {
  return [
    "Usage:",
    "  node extract_sl_tensors.mjs --replay <path> --data-dir <path>",
    "                              [--player <name|all>]",
    "                              [--include-no-action true|false]",
    "                              [--include-ui-actions true|false]",
    "                              [--max-actions <n>]",
    "                              [--max-tick <n>]",
    "                              [--max-entities <n>]",
    "                              [--max-selected-units <n>]",
    "                              [--spatial-size <n>]",
    "                              [--minimap-size <n>]",
    "                              [--include-flat true|false]",
    "                              [--include-debug true|false]",
    "                              [--output-format json|binary]",
    "                              [--output <path>]",
    "",
    "Extracts an action-aligned supervised-learning tensor dataset from a replay.",
    "The output can be streamed JSON or a binary-backed directory with section-wise tensor dumps.",
  ].join("\n");
}

async function writeChunk(stream, chunk) {
  if (!stream.write(chunk)) {
    await once(stream, "drain");
  }
}

async function endStream(stream) {
  stream.end();
  await once(stream, "finish");
}

async function writeDatasetJson(outputPath, result) {
  const stream = fs.createWriteStream(path.resolve(outputPath), { encoding: "utf-8" });
  stream.on("error", (error) => {
    throw error;
  });

  try {
    await writeChunk(stream, "{");
    const topLevelEntries = [
      ["replay", result.replay],
      ["sampledPlayers", result.sampledPlayers],
      ["options", result.options],
      ["schema", result.schema],
      ["superWeaponSchema", result.superWeaponSchema],
      ["staticMapSchema", result.staticMapSchema],
      ["staticMapByPlayer", result.staticMapByPlayer],
      ["counts", result.counts],
    ];

    for (let index = 0; index < topLevelEntries.length; index += 1) {
      const [key, value] = topLevelEntries[index];
      if (index > 0) {
        await writeChunk(stream, ",");
      }
      await writeChunk(stream, JSON.stringify(key));
      await writeChunk(stream, ":");
      await writeChunk(stream, JSON.stringify(value));
    }

    await writeChunk(stream, ',\"samples\":[');
    for (let index = 0; index < result.samples.length; index += 1) {
      if (index > 0) {
        await writeChunk(stream, ",");
      }
      await writeChunk(stream, JSON.stringify(result.samples[index]));
    }
    await writeChunk(stream, "]}");
    await endStream(stream);
  } catch (error) {
    stream.destroy();
    throw error;
  }
}

function ensureCleanDirectory(outputDir) {
  const resolvedOutputDir = path.resolve(outputDir);
  if (fs.existsSync(resolvedOutputDir)) {
    const stat = fs.statSync(resolvedOutputDir);
    if (!stat.isDirectory()) {
      throw new Error(`Binary output path must be a directory: ${resolvedOutputDir}`);
    }
    fs.rmSync(resolvedOutputDir, { recursive: true, force: true });
  }
  fs.mkdirSync(resolvedOutputDir, { recursive: true });
  return resolvedOutputDir;
}

function getTypedArrayConstructor(dtype) {
  switch (dtype) {
    case "float32":
      return Float32Array;
    case "int32":
      return Int32Array;
    default:
      throw new Error(`Unsupported binary section dtype: ${dtype}`);
  }
}

function getSectionElementCount(shape) {
  return shape.reduce((product, dimension) => product * Number(dimension), 1);
}

function writeTypedArrayToFile(filePath, typedArray) {
  const buffer = Buffer.from(typedArray.buffer, typedArray.byteOffset, typedArray.byteLength);
  fs.writeFileSync(filePath, buffer);
}

function flattenSectionValue(value) {
  if (!Array.isArray(value)) {
    return [Number.isFinite(value) ? Number(value) : 0];
  }

  const flattened = [];
  const stack = [value];
  while (stack.length) {
    const current = stack.pop();
    if (Array.isArray(current)) {
      for (let index = current.length - 1; index >= 0; index -= 1) {
        stack.push(current[index]);
      }
      continue;
    }
    flattened.push(Number.isFinite(current) ? Number(current) : 0);
  }
  return flattened.reverse();
}

function writeTensorSectionFiles(outputDir, schemaSections, samples, tensorKey, filePrefix) {
  const sectionFiles = {};
  const sampleCount = samples.length;

  for (const section of schemaSections) {
    const TypedArrayCtor = getTypedArrayConstructor(section.dtype);
    const elementCount = getSectionElementCount(section.shape);
    const typedArray = new TypedArrayCtor(sampleCount * elementCount);

    for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex += 1) {
      const flattened = flattenSectionValue(samples[sampleIndex][tensorKey][section.name]);
      if (flattened.length !== elementCount) {
        throw new Error(
          `Section ${tensorKey}.${section.name} expected ${elementCount} values, got ${flattened.length}.`,
        );
      }

      const baseOffset = sampleIndex * elementCount;
      for (let valueIndex = 0; valueIndex < elementCount; valueIndex += 1) {
        typedArray[baseOffset + valueIndex] = flattened[valueIndex];
      }
    }

    const fileName = `${filePrefix}__${section.name}.bin`;
    writeTypedArrayToFile(path.join(outputDir, fileName), typedArray);
    sectionFiles[section.name] = fileName;
  }

  return sectionFiles;
}

async function writeDatasetBinary(outputPath, result) {
  const outputDir = ensureCleanDirectory(outputPath);
  const supportSamples = result.samples.map((sample) => {
    const support = {
      tick: sample.tick,
      playerId: sample.playerId,
      playerName: sample.playerName,
      playerProduction: sample.playerProduction ?? null,
      playerSuperWeapons: sample.playerSuperWeapons ?? [],
    };

    if ("debug" in sample) {
      support.debug = sample.debug;
    }
    if ("flatFeatureTensor" in sample) {
      support.flatFeatureTensor = sample.flatFeatureTensor;
    }
    if ("flatLabelTensor" in sample) {
      support.flatLabelTensor = sample.flatLabelTensor;
    }
    return support;
  });

  const featureSectionFiles = writeTensorSectionFiles(
    outputDir,
    result.schema.featureSections,
    result.samples,
    "featureTensors",
    "feature",
  );
  const labelSectionFiles = writeTensorSectionFiles(
    outputDir,
    result.schema.labelSections,
    result.samples,
    "labelTensors",
    "label",
  );

  const supportFile = "support.json";
  fs.writeFileSync(
    path.join(outputDir, supportFile),
    JSON.stringify(
      {
        samples: supportSamples,
      },
      null,
      2,
    ),
    "utf-8",
  );

  const manifest = {
    format: "sl_dataset_binary_v1",
    replay: result.replay,
    sampledPlayers: result.sampledPlayers,
    options: result.options,
    schema: result.schema,
    superWeaponSchema: result.superWeaponSchema,
    staticMapSchema: result.staticMapSchema,
    staticMapByPlayer: result.staticMapByPlayer,
    counts: result.counts,
    sampleCount: result.samples.length,
    supportFile,
    featureSectionFiles,
    labelSectionFiles,
  };

  fs.writeFileSync(path.join(outputDir, "manifest.json"), JSON.stringify(manifest, null, 2), "utf-8");
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    console.log(usage());
    return;
  }

  const replayPath = requireArg(args, "replay");
  const dataDir = requireArg(args, "data-dir");
  const player = args.player ? String(args.player) : null;
  const includeNoAction = Boolean(args["include-no-action"] ?? false);
  const includeUiActions = Boolean(args["include-ui-actions"] ?? false);
  const maxActions = args["max-actions"] === undefined ? null : parseIntegerArg(args["max-actions"], 0, "max-actions");
  const maxTick = args["max-tick"] === undefined ? null : parseIntegerArg(args["max-tick"], 0, "max-tick");
  const maxEntities = parseIntegerArg(args["max-entities"], 128, "max-entities");
  const maxSelectedUnits = parseIntegerArg(args["max-selected-units"], 64, "max-selected-units");
  const spatialSize = parseIntegerArg(args["spatial-size"], 32, "spatial-size");
  const minimapSize = parseIntegerArg(args["minimap-size"], 64, "minimap-size");
  const includeFlat = Boolean(args["include-flat"] ?? false);
  const includeDebug = Boolean(args["include-debug"] ?? false);
  const outputFormat = args["output-format"] ? String(args["output-format"]).toLowerCase() : "json";
  const outputPath = args.output ? String(args.output) : null;

  if (!["json", "binary"].includes(outputFormat)) {
    throw new Error(`Unsupported output format: ${outputFormat}`);
  }

  const result = await extractReplaySupervisedDataset({
    dataDir,
    replayPath,
    player,
    includeNoAction,
    includeUiActions,
    maxActions,
    maxTick,
    maxEntities,
    maxSelectedUnits,
    spatialSize,
    minimapSize,
    includeFlat,
    includeDebug,
  });

  if (outputPath) {
    if (outputFormat === "binary") {
      await writeDatasetBinary(outputPath, result);
    } else {
      await writeDatasetJson(outputPath, result);
    }
  } else {
    if (outputFormat !== "json") {
      throw new Error("Binary output requires --output so the extractor can write a directory.");
    }
    console.log(JSON.stringify(result));
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : error);
  process.exit(1);
});
