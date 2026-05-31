# item-waterway-solver

`item-waterway-solver` is a Rust search tool for Minecraft 1.17.1 item waterways launched by a slime-piston collision.

It searches transition prefixes and repeating lane backbones, then evaluates whether a launched item can enter a stable two-game-tick cadence and maintain long-run transport close to `0.5 blocks/tick`.

## What It Does

- Models a slime-piston-launched item with configurable starting state.
- Searches compact lane prefixes in front of reusable backbone cycles.
- Scores early cadence quality, long-window speed stability, and full-run cadence consistency.
- Writes CSV, Markdown, and JSON reports for the best candidates.

## Project Status

This tool is a high-throughput search model, not a full Minecraft engine reimplementation.

The current Rust model matches the important item-water interaction order more closely than the earlier JavaScript prototype:

- Item underwater movement uses the same `> 0.1` height threshold as Minecraft item entities.
- Fluid current is applied once after movement, following the `ItemEntity.tick()` and `Entity.updateFluidInteraction()` order.
- The item interaction box uses the same `0.001` deflate margin and `0.25 x 0.25` item dimensions.

Remaining caveats:

- The solver is still a 1D abstraction.
- It does not fully reproduce the complete `FlowingFluid.getFlow()` 3D behavior.
- Final top candidates should still be checked in a real probe world.

## Build

```bash
cargo build --release
```

The release binary is written to:

```text
target/release/item-waterway-solver
```

## Usage

Run the Rust solver:

```bash
cargo run --release -- --early-only --ticks 25 --max-prefix 4 --cadence-pairs 6 --start-samples 9 --top 20
```

Show help:

```bash
cargo run --release -- --help
```

## Important Arguments

- `--mode early|full`: choose early filtering only or full verification.
- `--ticks <n>`: total simulated ticks.
- `--max-prefix <n>`: maximum prefix cell count.
- `--cadence-pairs <n>`: consecutive 2gt cadence checks.
- `--cadence-tolerance <f64>`: allowed per-pair 2gt distance error.
- `--full-cadence-pairs <n>`: long-run cadence verification window.
- `--start-samples <n>`: number of sampled start offsets.
- `--fixed-start-offsets a,b,c`: explicit start offsets.
- `--top <n>`: number of top results to export.

## Output Files

By default the tool writes to:

```text
artifacts/item-waterway-launch-search/
```

Generated files:

- `launch-search-results.csv`
- `launch-search-summary.md`
- `launch-top-candidates.json`

## Release Asset

The release package contains:

- the `item-waterway-solver` release binary
- this `README.md`
- `RELEASE_NOTES.md`

## Development Notes

Run tests with:

```bash
cargo test
```

Format the code with:

```bash
cargo fmt
```
