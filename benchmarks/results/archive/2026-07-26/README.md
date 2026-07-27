# Raw TPU experiment archive — 2026-07-26

This is a byte-for-byte snapshot of the local experiment artifacts collected
during TPU PD bring-up, selective-pull optimization, and protocol hardening.
The raw payload contains 95 files and is approximately 2.6 MiB.

The archive is intentionally lossless:

- unsuccessful and superseded experiments are retained;
- producer and consumer logs are retained alongside client results;
- per-request records are retained alongside summaries;
- environment and package snapshots needed to interpret early measurements
  are retained;
- no measurements were recalculated or rewritten during archival.

## Directory guide

- `phase0/`: initial TPU smoke and HBM bandwidth characterization.
- `phase1/`: original five-run load-generator baseline.
- `phase1-network/`: separately retrieved Phase 1 results.
- `v6e/`: v6e-labeled Phase 1 results.
- `correctness/`: direct reference output records.
- `selective-pull/`: primary control/optimized response files and worker logs.
- `selective-stress/`: concurrency, race, expiration, recovery, and final
  production-mode tests.

`MANIFEST.sha256` covers the 95 raw payload files, excluding this README and
the manifest itself. From this directory, verify it with:

```bash
shasum -a 256 -c MANIFEST.sha256
```

## Provenance and caveats

The compact, publication-oriented comparison is maintained separately at
`benchmarks/results/selective-pull/`. This archive is the supporting evidence
for that summary and for the implementation-hardening claims.

Some files contain expected errors. In particular, names containing
`delay`, `expiration`, `recompute`, `case6`, or `retry` may represent an
injected failure or an intermediate implementation. Consult the JSON status
fields and paired worker logs before drawing conclusions.

Before archival, the payload was scanned for common API-key, access-token,
private-key, and bearer-token patterns; no matches were found. Synthetic token
IDs, request UUIDs, private/localhost addresses, process IDs, and remote
experiment paths were preserved because they are useful for trace correlation.

