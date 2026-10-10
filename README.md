# Summary Dataset Builder

A standalone project for building source-grounded summarization SFT datasets.
The initial target is 10,000 accepted FineWeb-Edu training pairs, generated
through OpenRouter and exported for Alloy. This repository currently contains
the project scaffold and a draft prompt, not a working generation pipeline.

## Planned workflow

1. Pin the source revision and sample diverse, complete educational documents.
2. Deduplicate and split training, validation, and test documents before generation.
3. Compare teachers on 100 documents; review quality before scaling to 1,000 and 10,000.
4. Generate summaries with a versioned prompt and validated JSON responses.
5. Verify source support and key-information coverage; manually audit a representative sample.
6. Export accepted pairs with reproducible provenance and Alloy-tokenizer length checks.

FineWeb-Edu is educational web content, not a guarantee of factual correctness
or summarization suitability. Preserve source URLs, dataset IDs, hashes,
revision, and licensing information. Check provider terms before synthetic-data
generation or redistribution. Generated summaries are not independently verified truth.

## Setup and secrets

Use Python 3.11 or newer. No runtime dependencies are selected yet.
Store the OpenRouter key in `OPENROUTER_API_KEY`; never commit or log it.
An environment file template is provided for reference; no automatic `.env`
loader is implemented. No API calls or downloads run automatically.

## Structure

- `prompts/`: versioned teacher prompts and response schemas.
- `docs/`: implementation plan and quality requirements.
- `src/summary_dataset_builder/`: future pipeline package.
- `tests/`: offline regression tests as implementation progresses.
- `data/`, `runs/`, `.cache/`: ignored datasets, job state, and caches.

## Safety and reproducibility

Require an explicit model/provider selection and spending cap before paid jobs.
Persist each completed sample, bound retries and concurrency, and record billed
usage where available. Retrying interrupted requests may incur duplicate charges.
Keep rejection reasons, prompt hashes, model/provider identifiers, and request IDs.
Do not send private data. Treat source text as untrusted input, not instructions.

The Alloy training repository remains independent and consumes exported data only.
