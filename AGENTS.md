# AGENTS.md

## Scope

This repository is a Rust simulator and search tool for Minecraft item/waterway behavior.

There are two distinct codepaths:

- `search`: heuristic search for candidate item waterways.
- `verify`: schematic-backed motion simulator that is being pushed toward vanilla-aligned behavior.

Do not treat the repository as "just a lane searcher" anymore. The motion simulator is first-class work.

## Files That Matter

- `src/lib.rs`: CLI parsing, shared constants, search entrypoints, verify command wiring.
- `src/verify.rs`: entity motion simulation, dynamic block/fluid ticking, vanilla-alignment tests.
- `src/litematic.rs`: schematic loading, collision shapes, fluid cell decoding, support-shape helpers.
- `tools/prepare-vanilla-probe-root.sh`: clone a clean vanilla probe root.
- `tools/run-vanilla-probe.sh`: run command-scripted vanilla server probes.
- `tools/probes/*.txt`: reusable vanilla probe scripts.

## Ground Rules

### 1. Do not claim 1:1 without evidence

If behavior is described as vanilla-aligned, prove it with at least one of:

- direct guardian source inspection,
- a new regression test,
- a vanilla probe log produced by the scripts in `tools/`.

If evidence is weak or indirect, say so and keep working.

### 2. Motion fidelity beats feature count

The user only cares about entity motion and velocity-vector correctness.

Prioritize:

- `x/y/z`
- `vx/vy/vz`
- `onGround`
- collision outcomes
- fluid interaction timing
- load/reload semantics when they affect the first ticks of motion

Do not spend time expanding unrelated gameplay logic unless it changes movement.

### 3. Java float semantics matter

Cross-language drift is a real bug here.

When porting vanilla logic, be explicit about whether a value is effectively:

- Java `float`
- Java `double`
- integer/clamped intermediate state

Watch especially for:

- block offset math,
- damping/friction/gravity constants,
- fluid current accumulation,
- threshold comparisons,
- repeated per-tick multiplication.

Do not silently replace float-like behavior with pure `f64` math if vanilla uses `float` truncation in the hot path.

### 4. Static schematic load is not the same as live `setblock`

A repeated source of bugs in this repo is assuming that an impossible block state placed into a schematic behaves like a live world edit.

When dealing with unsupported or neighbor-sensitive blocks, distinguish:

- live placement/update behavior,
- saved-and-reloaded world behavior,
- first active tick after load.

If a block disappears on reload in vanilla, the simulator must not keep it around for the first motion tick.

## Required Workflow For Motion Changes

When changing motion-relevant behavior:

1. Inspect the relevant guardian source files first.
2. Add or update a focused test in `src/verify.rs`.
3. If the behavior is tricky, add or update a probe in `tools/probes/`.
4. Run:
   - `cargo fmt`
   - `cargo test --quiet`
5. If the claim is "matches vanilla", keep the probe log path in your notes or final summary.

## Probe Workflow

Use these scripts instead of ad hoc server handling:

```bash
tools/prepare-vanilla-probe-root.sh <base-root> <dest-root> <world-name> <port>
tools/run-vanilla-probe.sh --server-root <root> --jar <server-jar> --commands <probe.txt> --java-home <java-home>
```

Probe scripts should be checked into `tools/probes/` if they are reusable or back a regression.

## Search Discipline

Do not grep the entire `guardian` tree or the entire filesystem unless there is no narrower route.
Target the specific vanilla block/entity classes you need.

Good examples:

- `.../BambooStalkBlock.java`
- `.../CactusBlock.java`
- `.../ItemEntity.java`
- `.../LivingEntity.java`

Bad approach:

- broad global grep over the whole source tree for every turn

## Testing Expectations

At minimum, before closing a non-trivial simulator change, run:

```bash
cargo fmt
cargo test --quiet
```

For changes that alter vanilla-alignment claims, also run the relevant probe script and compare the resulting log.

## Coding Conventions

- Keep code, comments, test names, and commit messages in English.
- Comments should explain why the vanilla behavior is modeled that way, not restate the code.
- Prefer small, targeted helpers over giant branches.
- Do not add new approximation paths if the exact vanilla behavior is practical to model.

## Current Known Risk Areas

These areas deserve extra skepticism:

- snapshot reload normalization for impossible states,
- neighbor-sensitive blocks with scheduled ticks,
- bubble-column and fluid/block startup interaction,
- multi-tick drift from float/double mismatches,
- any scenario still reported through `approximate_collision_blocks`.

If you touch one of those, assume you need new evidence.
