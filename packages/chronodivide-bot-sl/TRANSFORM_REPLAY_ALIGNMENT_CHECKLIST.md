# Transform Replay Alignment Checklist

This note compares the current Chronodivide Python SL transformer with mini-AlphaStar's replay transformer:

- `D:\workspace\mini-AlphaStar\alphastarmini\core\sl\transform_replay_data.py`
- `D:\workspace\supalosa-chronodivide-bot\packages\chronodivide-bot-sl\transform_replay_data.py`

The goal is to check whether the RA2 transformer is aligned to the same pipeline steps as mini-AlphaStar, and to call out where it still differs.

## Current Conclusion

The RA2 transformer is aligned with the same major pipeline stages:

- iterate replay files
- choose a player perspective
- step through replay actions
- pair the current observation with the current kept action
- encode previous-action context
- build feature and label tensors
- save replay-level shards

However, it is not yet aligned with all of mini-AlphaStar's policy choices. The main differences are:

- player-selection policy
- action-filtering and downsampling policy
- build-order style context
- transport boundary between Python and the reusable replay extractor

## Status Legend

- `[x]` aligned
- `[~]` partially aligned
- `[ ]` not aligned yet

## Checklist

- `[x]` Replay-file iteration and bounded replay selection are aligned in spirit.
  mini-AlphaStar slices replay files with `DATA_FROM/DATA_NUM`.
  The RA2 transformer does replay glob/start/max selection.

- `[x]` Action-centric observation/label pairing is aligned.
  mini-AlphaStar pairs the current observation with the current kept action.
  The RA2 transformer does the same through the action-aligned tensor extractor.

- `[x]` Previous-action context is aligned conceptually.
  mini-AlphaStar builds `last_list = [last_delay, last_action_type, last_repeat_queued]`.
  The RA2 pipeline carries `lastActionContext` with previous delay/action/queue-style context.

- `[x]` Delay supervision is aligned conceptually.
  mini-AlphaStar uses next-action delay in the label-building path.
  The RA2 pipeline includes `delayToNextAction` and previous-action delay context.

- `[x]` Flat replay-player `(features, labels)` shard saving is aligned.
  mini-AlphaStar saves replay-level tensor shards as `(features, labels)`.
  The RA2 Python transformer also saves flat `(features, labels)` shards.

- `[x]` Structured feature and label section preservation is now present on the RA2 side.
  This goes beyond the current mini-AlphaStar save format by also storing schema-preserving `.sections.pt` sidecars.

- `[x]` Schema-level validation is present on the RA2 side.
  The RA2 transformer validates section presence, section shape, and flat/structured consistency before saving.

- `[~]` Player-selection policy is not aligned.
  mini-AlphaStar explicitly chooses the winning Protoss player.
  The RA2 transformer currently supports `first`, `all`, or an explicit player name.

- `[x]` Action-filtering and downsampling are now aligned in spirit.
  mini-AlphaStar probabilistically downsamples `no_op`, camera, `Smart_*`, gather, and attack-point actions.
  The RA2 transformer now applies a mini-AlphaStar-style post-extraction filter profile that downsamples common RA2 action patterns such as `select_units`, move-style orders, gather orders, and attack-style orders, while rewriting temporal fields on the final kept stream.

- `[~]` Stage ordering is aligned, but the execution boundary is different.
  mini-AlphaStar does replay stepping and tensor building inside one Python process.
  The current RA2 transformer shells out from Python to `py-chronodivide`, which does the replay extraction.

- `[ ]` A true build-order context equivalent is not implemented yet on the RA2 side.
  mini-AlphaStar explicitly computes and feeds build-order state before tensorization.
  The RA2 pipeline does not yet expose an equivalent build-order feature section.

- `[ ]` A winner-only replay policy is not implemented yet on the RA2 side.

- `[~]` Exact mini-AlphaStar threshold parity is not implemented.
  The RA2 side now has the same style of frequency-based downsampling, but the mapped RA2 action categories and default keep probabilities are Chronodivide-specific equivalents rather than one-to-one StarCraft II action IDs.

## Practical Read

If the question is:

`"Did we align to the same steps as mini-AlphaStar's transform_replay_data.py?"`

then the honest answer is:

- `Yes` for the major replay-to-sample pipeline stages.
- `Not yet` for some of the replay-selection and filtering policies that shape the final dataset.

## Highest-Value Next Parity Items

1. Add a winner-player selection policy for RA2 replay extraction.
2. Add a build-order or production-history feature section if we want closer parity in temporal context.
3. Replace the JSON bridge with a more direct binary transport once feature/label parity is stable.
