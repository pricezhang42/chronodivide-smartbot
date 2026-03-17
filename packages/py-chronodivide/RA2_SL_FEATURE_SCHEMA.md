# RA2 SL Feature Schema

This note describes the first supervised-learning feature extractor for Chronodivide replay re-simulation.

## Goal

Produce feature records that are close in spirit to mini-AlphaStar's supervised-learning inputs:

- scalar features
- entity features
- spatial planes
- minimap planes

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
- `minimap`
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
- visible neutral count
- visible other-hostile count
- self and visible-enemy HP / purchase-value totals

These are exposed as numeric arrays with feature names attached in the schema.

## Entity Features

Entities are built from visible units only and padded to a fixed `maxEntities` length.

Each entity vector currently contains:

- relation one-hot: self / allied / enemy / neutral / other-hostile
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
- neutral presence
- other hostile presence
- self HP
- allied HP
- enemy HP
- neutral HP
- other hostile HP
- self mobile presence
- self building presence
- enemy mobile presence
- enemy building presence

These are map-state planes, not UI minimap images.

## Minimap Planes

The extractor now also emits a separate minimap-style tensor at a configurable square resolution, default `64 x 64`.

This is still observation-safe and still derived from world state, not from rendered UI pixels. The goal is to provide a compact top-down summary that is easier to feed into an SL model as a dedicated minimap branch.

Current minimap channels:

- visible tiles
- hidden tiles
- visible ore
- visible gems
- visible ore spawners
- self presence
- allied presence
- enemy presence
- neutral presence
- other hostile presence
- self building presence
- self mobile presence
- enemy building presence
- enemy mobile presence
- self HP
- allied HP
- enemy HP
- self start location

Compared to `spatial`, the minimap block is more explicitly "map summary" oriented and includes a hidden-tile mask and self start-location marker.

## Caveats

- This is the feature side only. Action labels are still a separate phase.
- Object-name tokens are built per extraction run right now. A later dataset-wide pass should build a stable vocabulary shared across all training shards.
- Visible hostile-but-not-enemy objects are now split by visible owner into `neutral` and `otherHostile`.
- In common 1v1 ladder replays, `otherHostile` may stay empty for long stretches or for the whole game.
- The minimap is a semantic tensor, not a screenshot of the in-game minimap UI.
