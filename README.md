# Summary Dataset Builder

A local-first pipeline for building **10,000 accepted summarization training
pairs**, plus separate validation/test pairs, using OpenRouter. It includes
pinned source acquisition, deduplication before splitting, durable generation,
budget accounting, review gates, and reusable exports. No command downloads
data or generates paid completions merely because a configuration exists.

**Model-neutral datasets** contain `id`, `source`, and `summary` strings in JSONL.
They do not require a particular tokenizer or training framework. Optional
token-checked exports produce `prompt`/`completion` JSONL for a local tokenizer.
No dataset fits every model's context window automatically. The Alloy loader
adapter is not verified; consumers must check their training format, template,
special tokens, and source redistribution rights. This repository does not start
training or publish datasets.

## Dataset size and 300M-model SFT

The initial 10,000-training-pair target is a practical **first SFT baseline for
an already-pretrained model of roughly 300 million parameters**, focused on
short, source-grounded summaries. It is not a pretraining corpus or a guarantee
of production quality. Base-model understanding, source difficulty, summary
quality, and context length matter more than parameter count alone.

Measure sequence lengths with the downstream model's exact tokenizer and
training template. For illustration, 10,000 pairs averaging 1,000 combined
source/instruction/summary tokens represent about 10 million sequence tokens
per epoch. The number of supervised target tokens can be much smaller when
source/prompt tokens are excluded from the loss.

For the downstream training experiment:

1. Compare the pretrained model against a one-epoch SFT baseline on validation
   data. Consider further epochs, initially up to three, only while validation
   quality improves; training loss alone is insufficient.
2. Check source support, key-information coverage, names/numbers, attribution,
   uncertainty, and repetition. Automated acceptance and same-model assessment
   are screening signals, not verified factual-accuracy scores.
3. Keep validation/test sources disjoint from training, preserve difficult
   examples in the evaluation protocol, and leave the final test set untouched
   until training choices are fixed.
4. Expand to a more diverse 30k-50k training set only if validation comparisons
   show a benefit. More similar examples will not necessarily overcome a weak
   base model or a context-window limitation.

These are experiment guidelines, not tuned training settings. The dataset stays
model-neutral; training and any Alloy-specific integration remain separate work
requiring an explicit request.

## Setup and commands

Use Python 3.11+. From this repository's root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
summary-dataset --help
python -m summary_dataset_builder --help
```

Core orchestration and model-neutral exports use the standard library. The optional `tokenizer` extra loads a
**local** Hugging Face `tokenizer.json`; `sources` adds optional Hugging Face
streaming. There is no automatic tokenizer or model download.

```powershell
python -m unittest discover -s tests -v
python -m unittest tests.test_generation -v
python -m unittest tests.test_generation.GenerationTests.test_missing_usage_stops_and_reserves_unknown_billing -v
python -m pip wheel --no-deps --wheel-dir .cache\wheels .
```

Tests mock network requests and never consume API credit. Tests requiring the
optional tokenizer skip if it is not installed. No linter is configured.
The longer durability/scale regression is opt-in:

```powershell
$env:RUN_SCALE_TESTS = "1"
python -m unittest tests.test_scale -v
Remove-Item Env:\RUN_SCALE_TESTS
```

The client reads `OPENROUTER_API_KEY` from the process environment, never from
source files. If you already have the ignored local `ApiKey.txt`, load it without
printing it:

```powershell
$env:OPENROUTER_API_KEY = (Get-Content -LiteralPath .\ApiKey.txt -Raw).Trim()
```

There is no automatic `.env` loader. Never commit keys or place them in prompts,
source JSONL, command-line arguments, or job configurations.

## 1. Acquire and prepare source documents

For an existing local JSONL file, each line needs `id`, `url`, and `text` strings.
Use HTTP(S) source URLs and retain source licensing information. For a local
snapshot, `--revision` can be `sha256:<64-hex-file-hash>`; for Hugging Face it must
be an immutable 40-character commit SHA, not `main`.

Optional, explicitly authorized FineWeb-Edu acquisition:

```powershell
python -m pip install -e ".[sources]"
$revision = "<40-character dataset commit SHA>"
summary-dataset download --output data\source.jsonl --revision $revision --max-records 100000 --allow-download
```

The default subset is `sample-10BT`. The record limit bounds iteration, not the
network bytes needed to read remote shards. Downloads are not automatically
resumed. Invalid source URLs are rejected with safe IDs/ordinals in a checksummed
audit sidecar; the record limit counts these rejected rows too. Metadata reports
inspected and retained counts separately. Check dataset, source-site, provider retention, synthetic-training-data,
and redistribution terms before acquisition or generation.

Prepare extra candidates to allow for teacher rejections and quality flags:

```powershell
$licenseName = "<verified source license/terms>"
summary-dataset prepare --input data\source.jsonl --output data\prepared --dataset HuggingFaceFW/fineweb-edu --revision $revision --license $licenseName --train-count 15000 --validation-count 400 --test-count 400 --max-records 100000 --max-words 700 --seed 42
```

Preparation screens sources, removes exact/near duplicates globally **before**
assigning splits, and bounds domain representation. It writes `documents.jsonl`,
`rejections.jsonl`, and a manifest with parameters, counts, and hashes.
Inspect shortfalls before proceeding. `--max-words` is a source filter, not a
token estimate. Supplying `--tokenizer` excludes sources that cannot fit the
training prompt plus summary reserve **before** splitting, recording their
rejection reasons. Documents are never truncated. Simple privacy/boilerplate filters
do not replace representative manual source review.

## 2. Initialize and inspect an immutable run

Select a provider explicitly; provider slugs and support for strict JSON output
must be checked in OpenRouter. `openai/gpt-oss-120b` is the initial smoke-tested
candidate, not a validated production teacher.

```powershell
$provider = "<approved OpenRouter provider slug>"
$budgetUsd = "<approved total USD for this NEW run>"
summary-dataset init --prepared data\prepared --run runs\gpt-oss --model openai/gpt-oss-120b --provider $provider --budget-usd $budgetUsd --input-price-per-million 1 --output-price-per-million 2 --model-neutral
summary-dataset plan --run runs\gpt-oss
summary-dataset status --run runs\gpt-oss
```

The prices above are **illustrative routing ceilings in USD per million tokens**,
not advertised model prices. Use reviewed ceilings for the selected provider.
An endpoint without the required parameters/price/privacy constraints will fail
rather than silently route elsewhere. Repeat `--provider` only for other approved
providers.

Defaults target 10,000 train + 200 validation + 200 test accepted pairs.
Initialization freezes sources, dataset mode, teacher prompt/schema hashes,
targets, provider selection, and budget. `--model-neutral` does not tokenize
sources or promise compatibility with a downstream context limit.

For a tokenizer-specific run, install `python -m pip install -e ".[tokenizer]"`
and use `--tokenizer <local-tokenizer.json>` instead of `--model-neutral`.
Supply the same tokenizer during preparation to exclude overlong sources before
splitting. Initialization copies the tokenizer and checks every training prompt,
reserving 256 summary tokens within a 2,048-token context by default. A source
that does not fit aborts initialization. Context/reserve overrides must match
preparation. Changed run settings require a new run; none are silently updated.

`plan` is offline and reports conservative reservation bounds, not a price
forecast. Set a **dedicated OpenRouter key credit limit** as a billing backstop.
Budget accounting is per run; other runs and earlier ad-hoc smoke-test charges
are not included. Previous smoke-test approval does not authorize this run.

## 3. Generate, review, and scale

Only after model/provider, total spend, and terms have been explicitly approved:

```powershell
summary-dataset generate --run runs\gpt-oss --max-requests 100 --approve-paid --terms-reviewed
summary-dataset review-sample --run runs\gpt-oss --output runs\gpt-oss\review-100.json
```

To simply look through source/summary pairs, without verdicts, notes, or review tasks:

```powershell
summary-dataset sample-report --run runs\gpt-oss --output runs\gpt-oss\samples.html
Start-Process .\runs\gpt-oss\samples.html
```

This read-only viewer needs no saved review sample. It starts with accepted
summaries, offers search and navigation, and can show other generated outcomes.
Browsing does not submit approvals or modify data.

For an optional interactive review form:

```powershell
summary-dataset review-report --run runs\gpt-oss --review runs\gpt-oss\review-100.json --output runs\gpt-oss\review-100.html
Start-Process .\runs\gpt-oss\review-100.html
```

The report shows all generated results, with editable verdicts/notes only for the
saved review sample. Filter by disposition, split, review state, or full-text
search; compare each original source with its summary and raw completion.
Everything is embedded locally, with no server or external assets. Source text
is displayed as text, never executed as HTML. Browser drafts are local to that
browser; **Download review JSON** saves a portable copy containing every sample
item, including those hidden by filters. Import that file to resume elsewhere.
Use the downloaded `review-100.completed.json` with the existing `approve`
command after reviewing it. The report never approves, exports training data, or
sends paid requests. It is a snapshot; regenerate it from a fresh sample if the
run advances. Do not share the HTML without reviewing source redistribution rights.

### Automated assessment before scaling

A second-pass model assessment can check source support, coverage, suitability,
and whether the original disposition looks appropriate. It assesses a frozen
snapshot of completed generation outcomes without overwriting their statuses or
pretending a human reviewed them:

```powershell
summary-dataset assess --run runs\gpt-oss --max-requests 100 --max-total-requests 1000 --budget-usd $budgetUsd --approve-paid --terms-reviewed
summary-dataset assessment-status --run runs\gpt-oss
```

These flags are **new paid authorization**, not merely an offline check.
`--max-total-requests` caps generation plus assessment attempts together;
`--budget-usd` must equal the frozen total run cap, including earlier charges.
Both operations share one durable SQLite ledger. Assessment can run while the
generation quality gate is paused, but does not approve or bypass that gate.
The assessor uses the same explicitly approved model/provider with a separate
versioned rubric; its verdict is **not independent factual verification** or a
factual-accuracy percentage. Unknown billing pauses both operations and uses
the same `recover`/`reconcile` commands. Prompts and the assessment population
are frozen on the first invocation; reruns resume only unfinished assessments.
Automatic HTTP 429 retries count against the approved total request ceiling.

Completed assessments appear as optional automated notes in newly generated
sample viewers. Original generation responses, review decisions, and split
assignments remain unchanged. Use the aggregate results and representative
sample inspection to decide whether to scale; do not tune on held-out tests.

**A human must read the review sample.** It is stratified by split, disposition,
and source-length band, including rejected/flagged/error cases when present.
Assess source support, key-information coverage, quantities, attribution,
uncertainty, suitability, and privacy. Edit only each item's `verdict` (`pass` or
`fail`) and substantive `notes`. `pass` means the stored disposition is correct;
a correctly rejected document can pass review. Do not delete difficult cases or
change the reference source. Any failed disposition blocks approval until the
protocol is investigated; use a new run for a revised prompt.

Compare teachers on the same prepared sources in separate, separately budgeted
pilot runs before selecting the production teacher. Never tune on test results.

```powershell
summary-dataset approve --run runs\gpt-oss --review runs\gpt-oss\review-100.json --stage scale --acknowledge-review
summary-dataset generate --run runs\gpt-oss --max-requests 900 --approve-paid --terms-reviewed
summary-dataset review-sample --run runs\gpt-oss --output runs\gpt-oss\review-1000.json
```

Review that sample before approving the next stage:

```powershell
summary-dataset approve --run runs\gpt-oss --review runs\gpt-oss\review-1000.json --stage full --acknowledge-review
summary-dataset generate --run runs\gpt-oss --max-requests 30000 --approve-paid --terms-reviewed
summary-dataset status --run runs\gpt-oss
```

Generation stage gates count attempted generation requests, including rate-limit attempts, rather than
assuming each call produces a usable summary. Generation stops at exact accepted
split targets, the invocation limit, review gates, budget exhaustion, candidate
exhaustion, or a blocking error. Reinvoke the same command to resume; completed
documents are not regenerated. One request at a time is intentional: a 10k job
can take hours. No independent model-based verifier is implemented; automated
flags are screening signals, not factual-accuracy measurements.

Each generation or assessment attempt is durably reserved in SQLite **before** sending. Known charges,
response text, IDs, provider/model, and dispositions are persisted immediately.
By default, only explicit HTTP 429 rate limits auto-retry, with bounded backoff
and at most three attempts per document. Malformed/truncated output is retained
as an error, not exported or silently retried. The opt-in transient retry policy
below changes this behavior explicitly. The 2,048-token generation budget
includes reasoning as well as output; the smoke tests exhausted smaller limits.

## Optional: explicitly authorized automatic collection

The normal `generate` workflow retains manual stage approvals. A separately
authorized `collect` workflow can advance **unreviewed collection** automatically
using operational thresholds. It never creates human approvals, publishes a
dataset, or removes the final human-review requirement for release.

To expand an existing pilot, prepare a larger pool while preserving every
existing candidate and its split. Use the same pinned source revision, license,
screening parameters, and tokenizer settings:

```powershell
summary-dataset prepare --input data\expanded-source.jsonl --output data\expanded-prepared --preserve data\prepared\documents.jsonl --dataset HuggingFaceFW/fineweb-edu --revision $revision --license $licenseName --train-count 15000 --validation-count 400 --test-count 400 --max-records 100000 --max-words 700 --seed 42
summary-dataset extend-candidates --run runs\gpt-oss --prepared data\expanded-prepared --acknowledge-extension
```

Existing candidates act as deduplication anchors before selecting new documents.
Their content, provenance, and train/validation/test assignment cannot change.
The manifest records the preserved snapshot hash, and importing the expansion
does not reset completed requests, charges, or outcomes.

Only with explicit approval of the larger total cap, targets, and conditions:

```powershell
summary-dataset authorize-collection --run runs\gpt-oss --budget-usd 5 --target-train 10000 --target-validation 200 --target-test 200 --structural-threshold 0.98 --usable-threshold 0.75 --max-total-requests 30000 --approve-auto-collection
summary-dataset collect --run runs\gpt-oss --approve-paid --terms-reviewed
```

The cap above includes all earlier generation and assessment charges; it is not
an additional $5. Authorization is saved separately from the original run
configuration. Only targets and total budget are extended; model, provider,
prompts, source provenance, and tokenizer remain unchanged. The automatic policy is frozen; the collector never weakens it automatically
after a failed gate. If the user explicitly approves a lower operational
acceptance threshold, stop the worker and record an audited amendment:

```powershell
summary-dataset amend-collection-policy --run runs\gpt-oss --usable-threshold 0.70 --reason "Explicit user approval to lower usable yield to 70%" --approve-policy-change
summary-dataset collect --run runs\gpt-oss --approve-paid --terms-reviewed
```

This command changes only the usable-pair threshold. Previous policies and failed
checkpoint results remain in the SQLite amendment history and
`collection-policy-amendment-<id>.json`. Affected checkpoints are re-evaluated
using the same recorded attempts, not relabeled as passing. The structural gate,
budget, targets, retry limits, source records, and final release safeguards do
not change. This is a disclosed operating-policy decision, not evidence of
improved model quality.

Checkpoints occur after 100 generation attempts, then 1,000 and every subsequent
1,000, plus a final check. Each stage needs at least 98% structurally valid
model responses and 75% usable pairs among all its attempts, under the example
policy. HTTP 429 and other failures that return no model completion are transport
failures, not malformed model JSON. They remain in retry, spending, and overall
usable-yield counts. An incomplete or malformed model completion still fails
structural validation. A stage with no completions cannot pass.
Retries count as attempts. A usable pair passes the existing automated filters
and has no quality hold. Known assessor concerns and assessment errors are
quarantined separately, without overwriting original teacher dispositions.
These percentages measure operational screening, **not factual accuracy**.

For older runs that incorrectly counted transport failures in the structural
denominator, the paused checkpoint can be corrected with
`correct-gate-metrics --run <run> --approve-correction`. The correction archives
the previous failed calculation and re-evaluates the same attempts without
changing thresholds, accepted samples, spending, or review approvals.

Collection stops on a failed gate, exhausted candidate pool, budget/request cap,
blocking billing uncertainty, reservation overrun, interruption, or exact usable split
targets. No training data is released automatically. Resume with the same
`collect` command after resolving an operational interruption; do not erase
failed gate records or reset the ledger.

The command prints periodic progress and persists `collection-progress.json`,
`collection-events.jsonl`, and checkpoint summaries in the run directory.
Track live state from another terminal:

For an automatically refreshing browser dashboard:

```powershell
python -m summary_dataset_builder dashboard --run runs\gpt-oss --port 8765
# Open http://127.0.0.1:8765/ in your browser.
```

The dashboard refreshes every five seconds, showing usable split counts, budget
reservations, retries, and checkpoints. Remaining time and approximate finish are
estimated from up to 15 minutes of recent usable-pair throughput for unfinished
splits, starting no earlier than the current collector invocation. Estimates
require at least one minute and 10 usable pairs, and are hidden when paused or
without recent progress. They include observed retry delays but cannot predict
future quality gates, yield changes, or budget stops. It binds only to localhost and reads the
existing SQLite ledger in read-only mode; it cannot start/stop generation, approve
reviews, or serve arbitrary repository files. Closing the dashboard does not stop
the collector. Use `--port 0` to select a free port; the chosen URL is printed.
`--ready-file <path.json>` can save that URL for a launcher. Counts are live, while
the worker's reported state/time comes from its latest batch snapshot.

For a terminal-only display:

```powershell
python -m summary_dataset_builder collection-status --run runs\gpt-oss
# Optional refreshing display; Ctrl+C stops only this display:
while ($true) {
    Clear-Host
    python -m summary_dataset_builder collection-status --run runs\gpt-oss
    Start-Sleep -Seconds 5
}
```

`usable_counts` is the count toward the final targets; original accepted counts
can be larger when some pairs are quarantined. Final export includes quality-hold
reasons and operational stage history alongside the original outcomes.

## 4. Recover interrupted requests

To explicitly allow bounded retries and continuation after transient failures:

```powershell
summary-dataset configure-retries --run runs\gpt-oss --max-retries-per-sample 3 --min-retry-interval-seconds 60 --approve-retry-policy
summary-dataset collect --run runs\gpt-oss --approve-paid --terms-reviewed
```

This permits **at most three retries after the initial request** (up to four
attempts for a generation operation). The three-retry allowance is shared per
source across generation and assessment. Retry starts are persisted in SQLite
and spaced at least 60 seconds apart across the whole run, including after a
restart. First attempts for other sources are not subject to this retry throttle.
Exhausted samples are recorded as errors and skipped; teacher rejections and
quality flags are not repeatedly generated to get a different answer.

The policy applies to network failures, transient provider errors/rate limits,
and unusable structured responses. Authentication, payment/configuration errors,
budget exhaustion, reservation overruns, and failed quality gates still stop the
worker. Stage thresholds are unchanged and include failed/retried attempts.

Unknown charges remain `billing_unconfirmed` with their **entire reservation**
counting against the shared cap. They are never recorded as zero. A usable
response with missing usage is retained without duplicate generation. Already
stopped uncertain requests are requeued when this policy is explicitly applied,
unless a usable response was saved or retry capacity is exhausted. The original
attempts, costs/reservations, and retry history are preserved.

`status` distinguishes blocking uncertainty from `unconfirmed_billing_attempts`.
Final review/export still requires billing reconciliation. Verifying an earlier
unconfirmed charge does not overwrite a newer successful response.

Without that opt-in policy, uncertain requests retain the strict pause behavior:

```powershell
summary-dataset status --run runs\gpt-oss
# Only after confirming no worker is still using this run:
summary-dataset recover --run runs\gpt-oss --attempt 17 --worker-stopped
# Look up and verify the actual charge in the provider dashboard first:
summary-dataset reconcile --run runs\gpt-oss --attempt 17 --cost-usd 0.0012 --billing-verified
```

Timeouts, ambiguous server failures, and missing usage retain the full reservation
and pause generation by default. `recover` does not send anything. Reconciliation restores a
saved valid response, or records an error if no usable response exists. Add
`--retry` only when consciously authorizing another attempt after verifying the
old charge; it can be billed again. Never enter zero merely because cost is
unknown. Do not recover an active worker. Provider charges exceeding a reservation
block further generation in that run for investigation.

## 5. Review and export

After the exact accepted counts are reached, create and complete a fresh final
review. Older samples become stale when more requests are made.

```powershell
summary-dataset review-sample --run runs\gpt-oss --output runs\gpt-oss\review-final.json
# Human review and editing happens here.
summary-dataset approve --run runs\gpt-oss --review runs\gpt-oss\review-final.json --stage export --acknowledge-review
summary-dataset export --run runs\gpt-oss --output data\export
```

Model-neutral exports contain train/validation/test JSONL:

```json
{"id":"<document SHA-256>","source":"The original document text.","summary":"Its plain-text summary."}
```

Each export includes row-aligned provenance manifests, all candidate outcomes (including
held-out rejections), an attempt ledger, file hashes, the reviewed protocol and
review snapshot, and accounting. Model-neutral manifests explicitly record that
token lengths are unchecked; `source` is not wrapped in a training template.
For a model-specific adapter from the same reviewed neutral run:

```powershell
python -m pip install -e ".[tokenizer]"
summary-dataset export --run runs\gpt-oss --output data\export-model --tokenizer "<local-tokenizer.json>" --context-limit 4096
```

This writes `prompt`/`completion` pairs without altering the canonical run or its
splits. It checks `tokenizer.encode(prompt + completion, add_special_tokens=True)`
without padding/truncation; any overflow aborts the entire adapter export rather
than silently discarding examples. Tokenizer-specific runs use their frozen
tokenizer/context by default and do not accept overrides at export. These checks
do not cover a different loader's chat template or extra EOS tokens.

Reusable formatting does **not** mean unrestricted licensing. FineWeb-Edu carries
ODC-By attribution requirements and Common Crawl terms; original page rights
still apply. Preserve provenance and review redistribution rights before making
any dataset public.

FineWeb-Edu suitability and teacher summaries are not guarantees of truth.
Keep representative human review and separate-domain evaluation. Generated
sources, responses, job state, and exports belong under ignored `data/`, `runs/`,
or `.cache/`, never in Git.
