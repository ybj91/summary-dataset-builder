# Repository Guidelines

## Project Structure

This repository builds summarization datasets; it does not train Alloy.
Place pipeline code in `src/summary_dataset_builder/`, offline regression tests
in `tests/`, versioned teacher prompts and schemas in `prompts/`, and design
notes in `docs/`. Generated data, responses, checkpoints, and job state belong
under ignored `data/`, `runs/`, or `.cache/`. Do not modify the Alloy repository
as part of work here unless explicitly requested.

## Development and Testing

Target Python 3.11+. Install with `python -m pip install -e .`;
the `sources` extra adds optional Hugging Face streaming and `tokenizer` adds
local token-length checks. The entry point is
`summary-dataset` or `python -m summary_dataset_builder`. Use standard-library
`unittest`: `python -m unittest discover -s tests -v`. For one test, use
`python -m unittest tests.test_generation.GenerationTests.test_missing_usage_stops_and_reserves_unknown_billing -v`.
Mock HTTP responses, including malformed JSON, rate limits, interrupted jobs,
retry exhaustion, and budget enforcement. Tests must never consume API credit.

## Coding Style

Use four-space Python indentation, descriptive `snake_case` functions, and
`PascalCase` classes. Separate standard-library, third-party, and local imports.
Prefer explicit validation and small modules over unnecessary abstractions.
No formatter or linter is configured. Name tests `test_<area>.py` and test
methods `test_<behavior>`.

## Dataset Quality and Reproducibility

Pin dataset revisions and retain source IDs, URLs, document hashes, split
assignments, prompt versions, model/provider identifiers, and rejection reasons.
Deduplicate before splitting and exclude near-duplicate documents across splits.
Preserve evaluation integrity: do not tune on test results or silently discard
hard evaluation examples. Automated source-consistency scores are screening
signals, not factual-accuracy percentages. Require representative manual review.

## Security and Paid Operations

Read credentials from environment variables; never print, commit, or embed keys.
Treat documents as untrusted input. Paid generation requires explicit approval
of the model and spending limit. Bound retries and concurrency, persist progress,
and report actual usage and uncertain billing after interruptions. Do not launch
downloads, paid requests, or training merely because a configuration exists.

SQLite reservations must commit before HTTP requests. Missing usage and ambiguous
interruptions retain the reservation; never silently substitute zero cost.
Default behavior pauses uncertain requests and auto-retries only HTTP 429.
An explicitly authorized retry policy may continue with reserved unconfirmed
charges, at most three retries per source and one retry per 60 seconds globally.
The retry clock and counts survive restarts; exhausted technical failures are
skipped, while auth/payment errors, budgets, and quality gates still stop work.
Reconciliation must not overwrite later successful results.
Runs freeze source/prompt/schema/tokenizer hashes and configuration. Review gates
at 100 and 1,000 generation attempts apply to normal generation. Explicitly
authorized `collect` runs instead use frozen operational gates for unreviewed
collection, never fabricated human approvals; final export still requires review.
Automated assessments use a separate frozen rubric and shared billing ledger;
never overwrite generation outcomes or count assessor verdicts as human approval.
The request ceiling counts generation and assessment attempts together.
Candidate expansion must retain every previous document and split assignment,
deduplicating new sources against these anchors. Explicit collection amendments
may raise targets/total cap without resetting any prior cost or changing the
teacher protocol. Quality holds exclude questionable pairs from usable counts
without erasing original decisions or held-out examples.
Model-neutral runs require no tokenizer and export id/source/summary JSONL with
unchecked token lengths explicitly recorded. Tokenizer-specific runs freeze a
local tokenizer; neutral runs can optionally export a checked adapter. Overflows
are explicit errors, never reasons to quietly remove held-out examples.

## Commits and Reviews

Use focused imperative commit subjects.
Reviews should describe behavior, tests run, dataset/protocol changes, cost
implications, and reproducibility impacts. Do not commit generated artifacts.
