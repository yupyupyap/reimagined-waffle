# Local OpenBench Template

This directory contains the LocalChessEngine files that should be copied into a local
OpenBench checkout. It does not vendor OpenBench into this repository.

## Files

- `Makefile` is the engine build wrapper. OpenBench can run it with `make CXX=... EXE=...`.
- `Config/config.json` is a minimal local instance config.
- `Engines/LocalChessEngine.json` defines STC/LTC, regression and fixed-game presets.
- `Books/UHO_4060_v2.epd.json` points OpenBench at the upstream UHO book zip and checksum.

## Local Wrapper Check

From the repository root:

```sh
make -C tools/openbench EXE=LocalChessEngine-test CXX=clang++
./tools/openbench/LocalChessEngine-test bench
```

The bench must finish with a line shaped like:

```text
18417 nodes <nps> nps
```

`nps` is hardware and load dependent. The node count is the deterministic compatibility value for
the current `TASK-024`/`TASK-025` baseline.

## OpenBench Instance Setup

1. Fork or clone `AndyGrant/OpenBench` outside this repository.
2. Copy `Config/config.json`, `Engines/LocalChessEngine.json`, and
   `Books/UHO_4060_v2.epd.json` into the matching directories of that checkout.
3. Replace `source` in `Engines/LocalChessEngine.json` with the actual GitHub repository URL that
   OpenBench workers can clone.
4. Keep `build.path` as `tools/openbench` so workers build through this wrapper.
5. Start the OpenBench server and clients using the upstream OpenBench runbook.

The repository currently has no Git remote configured, so this task validates the wrapper and local
Cute Chess fallback rather than launching a complete OpenBench server.
