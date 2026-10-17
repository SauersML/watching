# Native reader 055r: infrastructure-only retry

The scientific code and all numerical gates remain exactly those of commit
`514c41d44cd9a4951327f4ca88a84d190420abb2` and the [055 protocol](native-read-055.md).
Run 055 never loaded the model or started the scientific module: every allocation
returned a community host on which CUDA failed. Its five allocations were deleted,
with rounded charges totaling $0.0123. This retry does not respond to a scientific
failure and does not relax any equality, identity, or compilation gate.

## Infrastructure change

Exact run name: `watching-native-read-055r`. Request **SECURE L40S only**, maximum
$1.20/hour, at most 15 minutes from the runner's initial allocation, and at most
$0.30 for this run. Retain 80 GB disk, `RP_PACK=0`, `RP_IDLE_PCT=0`, the runner's
independent local/remote wall-time watchdogs, price ceiling, account checks, exact
approval, and confirmed deletion procedure. No automatic follow-up run.

The installed runner has no secure-only environment option. Use a local copy with
[one additional offer filter](native-read-055r-runner.patch) in `cap_offers`, retaining
only `SECURE` offer kinds. No shared runner file is modified. Original runner SHA256:
`1993dc2857a750d26033fa72c534a3bca8ec692ae3dad039c41031b94f9ed36a`.
Scoped copy SHA256:
`b10dc69df38f76c6361fc0877ca45508416b5d59152f2df611591f4d692dbaa1`.
The source bundle and bootstrap remain byte-identical to the previously committed
scientific snapshot. The runner must pass its CUDA health check on the new host
before launching the module; record host-change verification without publishing
host addresses. A new infrastructure failure is reported, not treated as a result.

## Aggregate budget prerequisite

Do not launch until run 053 has finished and every allocation has confirmed deletion.
Record its actual rounded charges and require:

`053 actual + 054 series reserved $0.60 + 055 actual $0.0123 + 055r reserved $0.30 <= $1.40`.

The reservation for 054 includes its earlier failed allocations, not just its active
retry. Unused 053 budget funds this infrastructure retry. Keep the exact 055r approval
only for the duration of the run. Publish the actual aggregate calculation before
launch and the final charges/deletion afterward.

## Evidence and limits

Report each of the four locked prompts as soon as its on/off bytewise checks finish.
The target remains the original base model and one fixed compiled program; this is
not a trained-detector evaluation, fresh-compilation invariance claim, or universal
non-observability guarantee. Preserve failed or partial results without changing the
predeclared gate. Existing 050r passive-capture results remain separate evidence.
