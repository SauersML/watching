# Data coverage audit

This read-only audit uses the actual saved 034 input data, before final scores.
The counts and file hashes are in [data_audit.json](data_audit.json). No training
selection was changed, and the proposed change below is not implemented in 035.

- **Training:** 256 distinct contexts, all 12 training question families
  (21–22 examples each), all 48 question phrasings, and all 20 content families.
  Content-family counts range from 5 (`short_code`) to 21 (`creative_fragments`).
- **Development:** 64 contexts, all 48 training phrasings, but only the eight
  original content families (eight contexts each).
- **Comparison:** 64 contexts, four question families absent from training
  (XML requests, Markdown tables, footnotes, multilingual requests), all their
  16 phrasings, and 16 content families. This is an already inspected project
  development pool, not an untouched confirmatory test.

There are no normalized exact context duplicates within a split or across the
three selected splits. A conservative lexical near-duplicate screen finds one
training pair: “The sum of two odd integers is always even.” and “The sum of two
odd integers is even.” The heuristic and both records are retained in the JSON;
it does not establish semantic independence for all other pairs.

The 256 examples are **not 256 independent question patterns**: they reuse 48
phrasings organized into 12 named structures. Contexts and question templates
are explicitly AI-authored, not independently sampled human requests. Both
numerical conditions receive identical text for each prompt.

## One proposed improvement

Keep the 256-prompt budget, but select jointly by question and content family.
The full bank contains all 240 combinations of 12 question families and 20
content families. The current question-only stratification selects only 163 of
those cells. Select one distinct context from each cell, followed by 16 balanced
extras, using metadata alone. This would cover the available combinations more
evenly without more templates or more optimizer visits.

This is an untested coverage improvement, not evidence that missing cells are
the cause of poor generalization. It must be evaluated separately rather than
silently altering the current matched experiments.
