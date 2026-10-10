# Repository Guidelines

## Project Structure

This repository builds summarization datasets; it does not train Alloy.
Place pipeline code in `src/summary_dataset_builder/`, offline regression tests
in `tests/`, versioned teacher prompts and schemas in `prompts/`, and design
notes in `docs/`. Generated data, responses, checkpoints, and job state belong
under ignored `data/`, `runs/`, or `.cache/`. Do not modify the Alloy repository
as part of work here unless explicitly requested.

## Development and Testing

Target Python 3.11+. The scaffold has no implemented CLI or selected dependency
stack yet; document setup and commands when those are added. Use standard-library
`unittest` for small offline tests: `python -m unittest discover -s tests -v`.
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

## Commits and Reviews

There is no established Git history yet. Use focused imperative commit subjects.
Reviews should describe behavior, tests run, dataset/protocol changes, cost
implications, and reproducibility impacts. Do not commit generated artifacts.
