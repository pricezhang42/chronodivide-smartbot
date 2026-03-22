# Feature Improvement Plan: mini-AlphaStar vs RA2 Bot

## Feature Comparison

### Scalar Features

| Feature | mini-AlphaStar | Ours | Gap? |
|---|---|---|---|
| Game time | Binary + positional encoding (64 dims) | Raw tick + game_time (2 dims) | **Yes — time encoding is much richer in mAS** |
| Resources | Per-entity mineral/vespene counts | Credits (1 scalar) | Comparable (RA2 has simpler economy) |
| Race/faction | Home + away race one-hot | Self/enemy side + country one-hot | OK |
| Upgrades | 320-dim binary vector | 26-dim tech state flags | **Yes — we only track building presence, not individual upgrades** |
| Enemy upgrades | 320-dim binary vector | 25-dim enemy tech flags | Same gap |
| Available actions | 591-dim binary mask | ~100-dim action mask | OK (different action spaces) |
| Unit counts BoW | 285-dim with sqrt transform | ownedCompositionBow (vocab-sized) | OK |
| MMR/skill rating | 7-dim bucketed one-hot | Not present | Minor (bot vs bot doesn't need this) |
| Build order | 20 units x 285 one-hot -> **Transformer** (3 layers) -> 320 dims | 20 action IDs (raw integers) | **Yes — mAS uses a learned transformer over the build order sequence; we just use raw IDs** |
| Last delay | 128-dim one-hot | 1 raw integer | **Yes — one-hot bucketing gives much richer signal** |
| Last action type | 591-dim one-hot | 1 raw integer | **Yes — same issue** |
| Last repeat/queued | 2-dim one-hot | 1 raw queue flag | Minor |
| Agent statistics | 10 features (units lost, resources spent) | Not present | **Yes — cumulative game statistics missing** |
| Effects/buffs | 276-dim binary vector | Not present | **Moderate — RA2 has fewer effects but some matter (Iron Curtain, Chronoshift)** |

### Entity Features (per unit)

| Feature | mini-AlphaStar (3,585 dims) | Ours (74 dims) | Gap? |
|---|---|---|---|
| Unit type | 231-dim one-hot | Name token (vocabulary index) | **Yes — mAS uses full one-hot, we use a token that needs embedding lookup** |
| Position | 8-bit binary x + 8-bit binary y | tile_x/y raw + normalized | **Yes — binary encoding provides multi-resolution signal** |
| Health | 39-dim one-hot (sqrt bucketed) | 3 raw values (hp, max_hp, ratio) | **Yes — bucketed one-hot is much richer than raw** |
| Shield/Energy | 32+15 dim one-hot | Not applicable (RA2 has no shields) | N/A |
| Weapon cooldown | 32-dim one-hot | Raw ticks (2 values) | **Yes — bucketed is better** |
| Order queue length | 9-dim one-hot | Not present | **Yes — knowing how many queued orders a unit has** |
| Current orders | 2 x 591-dim one-hot (order ID 0 & 1) | attack_state (6-dim one-hot) | **Yes — we only encode attack state, not the full current order** |
| Order progress | Continuous + bucketed one-hot | Not present | **Yes — how far along is a unit's current action** |
| Buffs/debuffs | 276-dim one-hot | Not present | **Moderate** |
| Was selected | 2-dim one-hot | Via selection mask | OK |
| Was targeted | 2-dim one-hot | Not present | **Yes — useful for attention** |
| Cargo space | 9+9 dim one-hot | garrison/passenger fill ratio | OK |
| Alliance | 5-dim one-hot | 5-dim one-hot | OK |
| Attributes | 13 binary flags | 5-dim object type | OK (different games) |
| Upgrades per unit | 3 x 4-dim one-hot (weapon/armor/shield) | veteran_level (1 value) | **Minor** |

### Spatial Features

| Feature | mini-AlphaStar (24 planes, 64x64) | Ours (18+18+5 planes, 32x32 / 64x64) | Gap? |
|---|---|---|---|
| Height map | Yes | terrain_height_norm | OK |
| Visibility | 4-state one-hot | visible_tiles + hidden_tiles | OK |
| Pathability | Foot passable | foot + track passable | OK (we have more) |
| Buildable | Yes | buildable_reference | OK |
| Entity scatter | **Entity embeddings scattered onto map** | Presence density maps | **Yes — mAS puts learned entity representations directly on the spatial grid** |
| Player relative | 5-plane ownership | Per-relation presence maps | OK |
| Resources | Not directly on map | ore/gems/spawners on map | OK (we have more) |
| Camera/screen | 2 planes | Not present | N/A (no camera in bot) |

## Top Improvement Opportunities (ranked by expected impact)

### 1. Richer Time Encoding (Easy, High Impact)

mAS uses 64-dim binary + positional encoding for game time. We use raw tick values. The model can't easily learn "early game vs mid game vs late game" from a raw integer.

**Fix**: Add sinusoidal positional encoding of game time (like transformer position encoding). E.g., 32 dims of `sin(tick / 10000^(2i/d))` and `cos(tick / 10000^(2i/d))`.

### 2. Bucketed One-Hot for Continuous Values (Easy, High Impact)

mAS one-hot encodes nearly everything — health, cooldowns, delays, positions. Raw continuous values force the model to learn nonlinear boundaries itself. One-hot bucketing gives the model a "free" nonlinearity.

**Key targets**: `hit_points`, `weapon_cooldown`, `last_delay`, `last_action_type`, `purchase_value`, `veteran_level`, `tick`.

### 3. Entity Current Order Encoding (Medium, High Impact)

mAS encodes each unit's current order (move, attack, patrol, etc.) as a 591-dim one-hot — **twice** (primary and secondary order). We only have a 6-class `attack_state`. This means our model can't distinguish "this tank is moving" from "this tank is attacking" from "this tank is idle" at the entity level.

**Fix**: Add `current_order_type` (one-hot or token) per entity from the game engine. RA2 units have order types visible in their state.

### 4. Entity Scatter on Spatial Map (Medium, High Impact)

mAS scatters learned entity embeddings onto the spatial grid, so the spatial encoder sees rich per-unit information at each location — not just "there's a unit here." Our spatial maps are just density counts.

**Fix**: After entity encoding, scatter entity embeddings onto the spatial grid (index by tile position, sum/max where multiple units overlap). Feed both the original spatial planes and the scattered embeddings into the spatial encoder.

### 5. Cumulative Game Statistics (Easy, Medium Impact)

mAS tracks 10 agent statistics: units killed, units lost, structures destroyed, resources collected/spent, etc. These help the model understand "am I winning or losing?"

**Fix**: Add running totals: `units_killed`, `units_lost`, `buildings_destroyed`, `buildings_lost`, `total_credits_spent`, `total_credits_earned`. Most of these can be derived from game state diffs.

### 6. Build Order Transformer (Hard, Medium Impact)

mAS processes the 20-step build order through a 3-layer transformer, producing a 320-dim contextual encoding. We pass raw action IDs. The transformer lets the model learn build order patterns (e.g., "Barracks -> Refinery -> War Factory" is a tank rush opener).

**Fix**: Add a small transformer (2-3 layers, 16-dim) over the build order token sequence inside the model's scalar encoder. This is a model architecture change, not a feature extraction change.

### 7. "Was Targeted" Flag per Entity (Easy, Low-Medium Impact)

mAS marks entities that were targeted by the last action. Helps the model track action-object relationships.

**Fix**: Set a binary flag on the entity that was the target of the previous action.

## Recommended Priority

### Immediate (before next training run)

1. **Bucketed one-hot encoding** for last_delay, last_action_type, health, cooldowns — biggest bang for buck
2. **Sinusoidal time encoding** — trivial to add
3. **Cumulative game statistics** — easy if the game engine exposes them

### Next iteration

4. Entity current order encoding
5. Entity scatter on spatial
6. Build order transformer
