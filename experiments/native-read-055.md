# Native activation reading with one fixed compiled artifact

## Question and scope

Can we copy an activation that is already materialized without giving the downstream
computation a numerical distinction between capture enabled and disabled?

The target is the original Qwen3-8B model, pinned revision
`b968826d9c46dd6066d109eabc6255188de91218`, BF16, eval/no_grad with parameter
requires_grad flags retained, singleton length 96. Use the first four fixed detection
training prompts from validation_v2. No trained detector, population inference or
training experiment is included. The earlier native launch observers in 042r and
050r already passed ordinary full-vocabulary parity; this tests a practical capture
callable and stronger bytewise internal-state gates.

## What gets read

Capture the existing block-14 query-projection output and rotary query immediately
at their native launch boundaries, using a Python launch-wrapper observer. Return
independent CPU copies outside the compiled function. Do not add a torch hook,
Dynamo output, native kernel, compiled branch or new compilation for capture mode.

The normalized query is not a single stored tensor in this unprobed fused path.
These outputs are named precisely: projected query and rotary query. We do not
reconstruct q_norm and claim it is the native computation. Reading a newly added
output or adding a fused store after code generation is a different method: either
can change scheduling/register allocation/lowering, even if the high-level formula
stays unchanged. That approach is not needed for these already-existing outputs.

## One compiled program; capture condition stays outside it

Warm one unprobed compiled callable once. Freeze its actual native wrapper code
object, complete source hash, kernel objects, selected launcher objects, configurations
and selected cache hashes. `NativeReader(...)(batch, enabled=True)` changes only the
outer observer. `enabled` never reaches the compiled callable. The original model
and every original launch remain unchanged.

For each prompt execute:

1. Plain execution, with no tensor observer.
2. Capture disabled, with the independent state audit enabled.
3. Capture enabled, with the same state audit enabled.
4. Capture enabled without the expensive state audit: the practical read operation.
5. Plain execution again.

The state audit fingerprints **every tensor-valued native local binding and every
tensor-valued global** immediately after block-34 query projection, including aliases,
shape, stride, dtype, storage offset and logical bytes. Parameter inputs are hashed
one at a time; no full extra model is retained. This is deliberately an overinclusive
set of bindings, not a formal minimal live set. The chosen projection has already
written its output. Wrappers using allocator pool backing buffers or hidden `self`
state are rejected upfront. Uninitialized/dead bindings could cause conservative
false failures; they must not be silently discarded after seeing a failure.

Require bytewise equality of the enabled/disabled audited tensor bindings, and of
all vocabulary logits across all five executions. Also require the practical
activation copies to match the audited copies bytewise. Unlike `torch.equal`, the
comparison distinguishes signed zero. NaN/Inf logits invalidate the test; legitimate
mask infinities are compared by their bytes. Require the original wrapper/source,
selected launchers and compilation counters to stay unchanged throughout.

Numerical artifacts are saved before the success assertion. Any failure means this
build/input has not passed the capture contract. Do not relax equality tolerances.

## Interpretation

If every complete input tensor supplied to a deterministic, fixed downstream
computation is identical, that downstream computation has no numerical condition
information to learn from. Equality only of the current logits would not establish
that. This experiment checks native tensor bindings at one actual program counter;
it does not manufacture or certify a new BF16 residual cache for separately trained
transformer blocks. The exact native representation can include multiple residual
summands and normalized values.

Passing four inputs certifies those executions only. It is not a universal claim
about every input, compiler build, trained model, GPU, asynchronous serving workload
or observer implementation. CPU copies synchronize execution and change latency.
Timing, external scheduling, batching, RNG changes and external side channels are
outside the numerical claim. If fine-tuning or recompilation changes the source
computation, repeat the parity audit. The already-run identical-input training null
045 is relevant context, but is not relabeled as training on this new capture method.

## Bounded integration

Exact run name: `watching-native-read-055`. One L40S community GPU, maximum
$0.80/hour and 15 minutes ($0.20 maximum), 80 GB disk, idle-utilization shutdown
disabled and wall-time limit retained. No higher-price fallback and no automatic
scientific retry. If the requested hardware/cap is unavailable, report it instead
of extending the budget. Save each prompt's result immediately, publish compact
metadata and hashed raw artifacts, and delete the pod and exact approval afterward.
