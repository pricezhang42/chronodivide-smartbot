# RA2 SL Feature Schema

This note describes the first supervised-learning feature extractor for Chronodivide replay re-simulation.

## Goal

Produce feature records that are close in spirit to mini-AlphaStar's supervised-learning inputs:

- scalar features
- entity features
- spatial planes

while staying strictly on the player-correct observation side.

## Safe Observation Rule

The extractor only uses observation-safe APIs:

- `getVisibleUnits(playerName, ...)`
- `map.isVisibleTile(tile, playerName)`
- `getPlayerData(playerName)` for the acting player
- `getUnitData(id)` / `getGameObjectData(id)` only for IDs returned by `getVisibleUnits(...)`

It intentionally avoids:

- `getAllUnits()`
- `getPlayerData(enemyName)`
- any global unit scan used as a source of training features

That rule is enforced by the implementation in [features.mjs](D:/workspace/supalosa-chronodivide-bot/packages/py-chronodivide/features.mjs).

## Output Structure

The extractor CLI is [extract_features.mjs](D:/workspace/supalosa-chronodivide-bot/packages/py-chronodivide/extract_features.mjs).

Each sampled timestep produces:

- `scalarFeatures`
- `entityFeatures`
- `entityMask`
- `entityNameTokens`
- `spatial`
- `countsByName`

## Scalar Features

Current scalar features include:

- time and map shape
- player start position
- credits
- power totals / drain / margin / low-power flag
- radar disabled flag
- visible tile count and fraction
- self visible composition counts
- allied visible count
- visible enemy composition counts
- visible neutral / hostile-other count
- self and visible-enemy HP / purchase-value totals

These are exposed as numeric arrays with feature names attached in the schema.

## Entity Features

Entities are built from visible units only and padded to a fixed `maxEntities` length.

Each entity vector currently contains:

- relation one-hot: self / allied / enemy / neutral-or-other-hostile
- coarse object-type one-hot: aircraft / building / infantry / vehicle / other
- tile position and normalized tile position
- HP, max HP, HP ratio
- sight
- veteran level
- purchase value
- facing and turret facing as sin/cos
- velocity
- foundation size
- idle / movement / guard / bridge flags
- build-status one-hot
- attack-state one-hot
- power / repair / warp / TNT flags
- garrison / passenger fill ratios
- harvester ore / gems
- ammo

The specific object name is emitted separately as `entityNameTokens`, with a vocabulary built from the extracted samples.

## Spatial Planes

The extractor also builds coarse spatial planes at a configurable square resolution, default `32 x 32`.

Current channels:

- visible tiles
- visible ore
- visible gems
- visible ore spawners
- self presence
- allied presence
- enemy presence
- neutral / other hostile presence
- self HP
- allied HP
- enemy HP
- neutral / other hostile HP
- self mobile presence
- self building presence
- enemy mobile presence
- enemy building presence

These are map-state planes, not UI minimap images.

## Caveats

- This is the feature side only. Action labels are still a separate phase.
- Object-name tokens are built per extraction run right now. A later dataset-wide pass should build a stable vocabulary shared across all training shards.
- Neutral map objects visible to the player are currently grouped into `neutralOrHostileOther`. If you want neutrals separated from hostile-but-not-enemy units, that split should be added before large-scale dataset generation.
