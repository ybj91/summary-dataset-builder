# Implementation plan

## Decisions before paid execution

Approve the teacher model, allowed providers, total budget, and pilot size.
Check provider synthetic-training-data and retention terms. Obtain the key
through the local environment, not chat. No paid job is authorized by this scaffold.

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
