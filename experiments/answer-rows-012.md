# Faster answer rows with original tied weights

Exploratory follow-up specified after run 008's fast arm failed to show sustained
single-prompt learning. Runs 009–011 were still in progress.

The prompt's space-prefixed ` Yes` and ` No` encode as 7414 and 2308, whereas
the scored answer tokens `Yes` and `No` encode as 9454 and 2753. Accelerating
only those two answer rows can leave prompt input embeddings at the slower rate.
The manifest counts occurrences of the answer IDs in the actual training input;
verify both are zero before interpreting this as avoiding faster input updates.

Model: Qwen3-0.6B, SFT-003 starting weights, original tied architecture. All
parameters trainable, first expanded-data prompt, same paired objective and
512-step schedule as runs 008–010. Peak rate 1e-4 for answer rows and 1e-8 for
everything else. This matches run 010's rates, changing only which matrix rows
receive the faster rate. Gradients are moved into independent optimizer row
states before clipping; they are not duplicated. Updated rows are copied back
into the original shared parameter, preserving weight tying.

Same practiced-prompt endpoint and fresh-process checks as previous runs.
Evaluate every 16 steps and use the first 128 test prompts. No single-prompt
population CI. Compare test changes with run 010 on the same prompt indices,
retaining condition/checkpoint/arm pairing. This is an exploratory selected
comparison, not an independent confirmation.

One 24 GB GPU, at most 30 minutes and $0.60/hour ($0.30 cap). Disable idle-GPU
shutdown during compilation and retain the hard deadline. Publish the result
regardless of sign. Independent no-footprint controls and replication follow
only if a useful learning effect is found.
