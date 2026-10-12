# Implementation plan

## Implementation status

The CLI and offline regressions now cover bounded pinned source acquisition,
local source screening/deduplication/splitting, token-length preflight, SQLite
checkpoints and reservations, strict OpenRouter responses, deterministic review
samples, staged approvals, reconciliation, model-neutral id/source/summary
export, and optional token-checked prompt/completion adapters.
See `README.md` for executable commands and limitations.

No 10k paid job or source download is authorized by implementing this pipeline.
The production teacher/provider, real source license/revision, spending cap,
and human pilot review still need explicit decisions. A local training tokenizer
is required only for tokenizer-specific runs or optional adapter exports.
A same-model second-pass assessor now provides automated support/coverage and
disposition screening, sharing the paid-request ledger without changing human
approvals. Independent model verification, an empirically validated source-quality
protocol, and the actual Alloy loader/template adapter remain follow-up work.

An explicit automatic-collection policy can extend candidate pools, targets and
the shared total budget while retaining previous split assignments and charges.
Operational stage checks and quality holds govern unreviewed collection; they
do not grant human approval or authorize public release.

An explicit transient-retry policy can continue collection while uncertain charges
remain conservatively reserved. Per-source retry caps and a shared durable
60-second retry interval prevent retry storms; failed samples are retained and
skipped after exhaustion. Billing confirmation and representative human review
remain necessary before final release.

## Decisions before paid execution

Approve the teacher model, allowed providers, total budget, and pilot size.
Check provider synthetic-training-data and retention terms. Obtain the key
through the local environment, not chat. No paid job is authorized by a configuration alone.

## Milestone 1: sampling

Pin a FineWeb-Edu revision and use bounded streaming or selected shards, not a
full-corpus download. Select diverse topics, domains, and lengths with a fixed
seed. Screen boilerplate, incomplete pages, sensitive data, exact and near
duplicates. Split documents before generation; retain stable IDs and hashes.
Check source plus instruction plus summary against Alloy's 2,048-token context
using its exact tokenizer. Do not silently truncate complete documents.

## Milestone 2: generation

Implement an OpenRouter client with redacted errors, timeouts, bounded retries,
explicit provider routing, supported structured output, and local validation.
Enforce accepted/summary/rejection_reason consistency beyond the JSON schema.
Track actual usage and use a dedicated API-key credit limit as a billing backstop.
Persist each result durably. Do not retry ambiguous paid requests indefinitely.
Run a 100-document teacher comparison before choosing the production teacher.

## Milestone 3: quality gates

Assess source-supported claims, key-information coverage, entities/numbers,
attribution, uncertainty, repetition, and concision. Use an independent verifier
to flag questionable pairs, not as unquestioned ground truth. Track rejections
and review a stratified random sample including accepted and rejected examples.
Expand to 1,000 only after review; then target 10,000 accepted training pairs
plus separate held-out documents. Teacher-generated references alone do not
establish evaluation validity; human review is required.

## Milestone 4: export and training handoff

Export only summary text as the SFT target, not JSON or reviewer metadata.
Keep source and generation provenance in a companion manifest. Inspect Alloy's
actual loader and reproduce its schema and tokenizer checks before claiming
compatibility. Keep news evaluation as a separate domain check. Start training
only under a separate explicit request.

## Initial downstream SFT evaluation

Use the 10,000 training pairs as the first supervised-fine-tuning baseline for an
already-pretrained roughly 300M-parameter model, not as evidence that this size
guarantees adequate quality. Measure actual input/target token counts and context
fit with the consumer's tokenizer and full training template.

Compare the base model and a one-epoch SFT run on validation data; try additional
epochs only if quality improves, and keep the final test set for the frozen
protocol. Assess source support, coverage, entities/numbers, attribution,
uncertainty, and repetition, including representative human checks. Larger,
more diverse datasets are a follow-up hypothesis to test against this baseline,
not an automatic collection target or spending authorization.

This plan does not add a training implementation to the dataset builder or modify
Alloy. See the dataset-size guidance in `README.md` for the handoff boundaries.
