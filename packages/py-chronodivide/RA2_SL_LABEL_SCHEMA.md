# RA2 SL Label Schema

This note describes the first supervised-learning label extractor for Chronodivide replay re-simulation.

## Goal

Turn replay actions into structured labels that can be paired with replay-time player observations.

The current implementation lives in:

- [labels.mjs](D:/workspace/supalosa-chronodivide-bot/packages/py-chronodivide/labels.mjs)
- [extract_labels.mjs](D:/workspace/supalosa-chronodivide-bot/packages/py-chronodivide/extract_labels.mjs)

## Dataset-Level Action Audit

The replay folder `packages/chronodivide-bot-sl/ladder_replays_top50` currently contains `2064` `.rpl` files.

Across those replays, the observed raw action IDs were:

- `0` `NoAction`: `1,464,360`
- `1` `DropPlayerAction`: `175`
- `3` `ResignGameAction`: `1,835`
- `5` `PlaceBuildingAction`: `58,724`
- `6` `SellObjectAction`: `7,538`
- `7` `ToggleRepairAction`: `3,537`
- `8` `SelectUnitsAction`: `872,050`
- `9` `OrderUnitsAction`: `877,493`
- `10` `UpdateQueueAction`: `276,600`
- `12` `ActivateSuperWeaponAction`: `3,338`
- `13` `PingLocationAction`: `13`

Important takeaway:

- the real gameplay surface is dominated by `select`, `order`, and `queue` actions
- `DropPlayerAction` and `PingLocationAction` exist, but are rare and not useful for SL imitation targets

## Default Filtering

By default, the label extractor excludes:

- `NoAction`
- `DropPlayerAction`
- `PingLocationAction`

Those can still be included with CLI flags if needed for auditing.

## Structured Label Fields

Each extracted sample currently contains:

- replay tick
- player id / player name
- raw action id and class name
- dense `actionFamily` / `actionFamilyId`
- `delayFromPreviousAction`
- `delayToNextAction`
- `selectionBeforeActionIds`
- `selectionAfterActionIds`
- `actionSelectedUnitIds`

Action-specific heads are then filled depending on the action family.

## Action Families

Current dense label families are:

- `no_action`
- `select_units`
- `order_units`
- `update_queue`
- `place_building`
- `sell_object`
- `toggle_repair`
- `activate_super_weapon`
- `resign_game`
- `drop_player`
- `ping_location`
- `unknown`

## Family-Specific Fields

### `select_units`

- `actionSelectedUnitIds`

This is the explicit unit-id list present in `SelectUnitsAction`.

### `order_units`

- `actionSelectedUnitIds`
- `queue`
- `orderTypeId`
- `orderTypeName`
- `targetMode`
- `targetTile`
- `targetObjectId`
- `targetObjectName`
- `targetObjectOwner`
- `targetObjectType`
- `targetIsOre`

Selection for orders is inferred from the latest prior `SelectUnitsAction` for that player.

### `update_queue`

- `queueTypeId`
- `queueTypeName`
- `queueUpdateTypeId`
- `queueUpdateTypeName`
- `quantity`
- `itemName`
- `itemType`
- `itemTypeName`
- `itemCost`

Observed queue update types so far:

- `0 = Add`
- `1 = Cancel`
- `2 = Hold`

### `place_building`

- `buildingName`
- `buildingType`
- `buildingTypeName`
- `buildingCost`
- `buildingTile`

### `sell_object`

- `objectId`
- `objectName`
- `objectOwner`
- `objectType`
- `objectTile`

### `toggle_repair`

- `objectId`
- `objectName`
- `objectOwner`
- `objectType`
- `objectTile`

### `activate_super_weapon`

- `superWeaponTypeId`
- `superWeaponTypeName`
- `superWeaponTile`
- `superWeaponTile2`

Observed super-weapon types in the sample replay included `ParaDrop`, `AmerParaDrop`, `LightningStorm`, and `ChronoSphere`.

## Important Design Choice

The label extractor walks the replay tick-by-tick and decodes actions against the replay-time game state.

That matters because:

- object-target actions need the current game state to resolve object references
- sell / repair labels need the current object id to still exist in the game state
- unit selection for orders has to be inferred from the replay action stream

## Caveats

- This is still JSON-level structured output, not final tensor serialization.
- `actionSelectedUnitIds` is a variable-length list and will need padding / truncation policy during tensorization.
- Large selections do occur. In the sample replay, `SelectUnitsAction` sizes ranged up to `60`.
- `PingLocationAction` and `DropPlayerAction` are intentionally treated as UI-only and excluded by default.
- The final training pipeline still needs a combined feature+label serializer so these labels line up directly with the safe observation features.
