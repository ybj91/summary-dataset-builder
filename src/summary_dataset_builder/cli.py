import argparse
import json
from decimal import Decimal
from pathlib import Path
import sqlite3
import sys

from .acquire import download_fineweb
from .export import export_dataset
from .generation import generate
from .openrouter import OpenRouterError, estimate_input_tokens
from .prepare import prepare_jsonl
from .review import approve_review, review_sample
from .review_html import create_html_report
from .store import RunStore, initialize_run, money


def _path(value: str) -> Path:
    return Path(value).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build resumable, source-grounded summarization datasets")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Explicitly stream a bounded pinned FineWeb-Edu sample")
    download.add_argument("--output", type=_path, required=True)
    download.add_argument("--revision", required=True)
    download.add_argument("--max-records", type=int, required=True)
    download.add_argument("--subset", default="sample-10BT")
    download.add_argument("--split", default="train")
    download.add_argument("--allow-download", action="store_true")

    prepare = commands.add_parser("prepare", help="Screen, deduplicate, sample and split local source JSONL")
    prepare.add_argument("--input", type=_path, required=True)
    prepare.add_argument("--output", type=_path, required=True)
    prepare.add_argument("--dataset", required=True)
    prepare.add_argument("--revision", required=True)
    prepare.add_argument("--license", dest="license_name", required=True)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--train-count", type=int, default=15000)
    prepare.add_argument("--validation-count", type=int, default=400)
    prepare.add_argument("--test-count", type=int, default=400)
    prepare.add_argument("--max-records", type=int, default=100000)
    prepare.add_argument("--min-words", type=int, default=120)
    prepare.add_argument("--max-words", type=int, default=700)
    prepare.add_argument("--near-duplicate-threshold", type=float, default=0.8)
    prepare.add_argument("--max-per-domain", type=int, default=200)
    prepare.add_argument("--tokenizer", type=_path)
    prepare.add_argument("--context-limit", type=int, default=2048)
    prepare.add_argument("--summary-token-reserve", type=int, default=256)
    prepare.add_argument("--preserve", type=_path, help="Existing prepared documents whose splits must be preserved")

    init = commands.add_parser("init", help="Validate sources and freeze a new run; no API calls")
    init.add_argument("--prepared", type=_path, required=True)
    init.add_argument("--run", type=_path, required=True)
    init.add_argument("--model", required=True)
    init.add_argument("--provider", action="append", required=True, dest="providers")
    init.add_argument("--budget-usd", required=True)
    init.add_argument("--input-price-per-million", required=True)
    init.add_argument("--output-price-per-million", required=True)
    mode = init.add_mutually_exclusive_group(required=True)
    mode.add_argument("--tokenizer", type=_path, help="Local tokenizer for a model-specific run")
    mode.add_argument("--model-neutral", action="store_true", help="Reusable source/summary pairs without tokenization")
    init.add_argument("--prompt", type=_path, default=_path(str(Path("prompts") / "teacher-v2.txt")))
    init.add_argument("--schema", type=_path, default=_path(str(Path("prompts") / "teacher-v2.schema.json")))
    init.add_argument("--target-train", type=int, default=10000)
    init.add_argument("--target-validation", type=int, default=200)
    init.add_argument("--target-test", type=int, default=200)
    init.add_argument("--max-tokens", type=int, default=2048)
    init.add_argument("--max-attempts", type=int, default=3)
    init.add_argument("--context-limit", type=int, default=2048)
    init.add_argument("--summary-token-reserve", type=int, default=256)

    for name, help_text in (
        ("status", "Show progress, cost, reservations and unresolved attempts"),
        ("plan", "Estimate conservative reservations without a key or network"),
    ):
        subparser = commands.add_parser(name, help=help_text)
        subparser.add_argument("--run", type=_path, required=True)

    generation = commands.add_parser("generate", help="Run or resume explicitly approved paid requests")
    generation.add_argument("--run", type=_path, required=True)
    generation.add_argument("--max-requests", type=int, default=100)
    generation.add_argument("--approve-paid", action="store_true")
    generation.add_argument("--terms-reviewed", action="store_true")

    sample = commands.add_parser("review-sample", help="Write an editable stratified human-review sample")
    sample.add_argument("--run", type=_path, required=True)
    sample.add_argument("--output", type=_path, required=True)
    sample.add_argument("--per-stratum", type=int, default=5)
    sample.add_argument("--seed", type=int, default=42)

    report = commands.add_parser("review-report", help="Create a self-contained HTML review workspace")
    report.add_argument("--run", type=_path, required=True)
    report.add_argument("--review", type=_path, required=True)
    report.add_argument("--output", type=_path, required=True)

    browser = commands.add_parser("sample-report", help="Create a read-only source/summary sample viewer")
    browser.add_argument("--run", type=_path, required=True)
    browser.add_argument("--output", type=_path, required=True)

    assessment = commands.add_parser("assess", help="Run automated source-support checks without granting human approval")
    assessment.add_argument("--run", type=_path, required=True)
    assessment.add_argument("--max-requests", type=int, default=100)
    assessment.add_argument("--max-total-requests", type=int, required=True)
    assessment.add_argument("--budget-usd", required=True)
    assessment.add_argument("--approve-paid", action="store_true")
    assessment.add_argument("--terms-reviewed", action="store_true")
    assessment.add_argument("--prompt", type=_path, default=_path(str(Path("prompts") / "assessor-v1.txt")))
    assessment.add_argument("--schema", type=_path, default=_path(str(Path("prompts") / "assessor-v1.schema.json")))
    assessment_status = commands.add_parser("assessment-status", help="Show automated assessment outcomes and shared costs")
    assessment_status.add_argument("--run", type=_path, required=True)

    extend = commands.add_parser("extend-candidates", help="Import a globally deduplicated expansion preserving existing splits")
    extend.add_argument("--run", type=_path, required=True)
    extend.add_argument("--prepared", type=_path, required=True)
    extend.add_argument("--acknowledge-extension", action="store_true")

    authorize = commands.add_parser("authorize-collection", help="Explicitly authorize automatic operational gates, not release")
    authorize.add_argument("--run", type=_path, required=True)
    authorize.add_argument("--budget-usd", required=True)
    authorize.add_argument("--target-train", type=int, default=10000)
    authorize.add_argument("--target-validation", type=int, default=200)
    authorize.add_argument("--target-test", type=int, default=200)
    authorize.add_argument("--structural-threshold", default="0.98")
    authorize.add_argument("--usable-threshold", default="0.75")
    authorize.add_argument("--max-total-requests", type=int, default=30000)
    authorize.add_argument("--approve-auto-collection", action="store_true")

    amend = commands.add_parser("amend-collection-policy", help="Audit an explicitly approved reduction of the usable threshold")
    amend.add_argument("--run", type=_path, required=True)
    amend.add_argument("--usable-threshold", required=True)
    amend.add_argument("--reason", required=True)
    amend.add_argument("--approve-policy-change", action="store_true")

    correct = commands.add_parser("correct-gate-metrics", help="Audit correction of transport failures miscounted as invalid model JSON")
    correct.add_argument("--run", type=_path, required=True)
    correct.add_argument("--approve-correction", action="store_true")

    collection = commands.add_parser("collect", help="Resume authorized collection until targets or a safety stop")
    collection.add_argument("--run", type=_path, required=True)
    collection.add_argument("--max-requests", type=int, default=30000)
    collection.add_argument("--batch-size", type=int, default=25)
    collection.add_argument("--approve-paid", action="store_true")
    collection.add_argument("--terms-reviewed", action="store_true")
    collection_status = commands.add_parser("collection-status", help="Show usable counts, operational gates and spending")
    collection_status.add_argument("--run", type=_path, required=True)

    dashboard = commands.add_parser("dashboard", help="Serve a live read-only dashboard on localhost")
    dashboard.add_argument("--run", type=_path, required=True)
    dashboard.add_argument("--port", type=int, default=0, help="Local port; 0 selects a free port")
    dashboard.add_argument("--ready-file", type=_path, help="Write the local dashboard URL to this JSON file")

    retry_policy = commands.add_parser("configure-retries", help="Authorize bounded transient retries with reserved uncertain costs")
    retry_policy.add_argument("--run", type=_path, required=True)
    retry_policy.add_argument("--max-retries-per-sample", type=int, default=3)
    retry_policy.add_argument("--min-retry-interval-seconds", type=int, default=60)
    retry_policy.add_argument("--approve-retry-policy", action="store_true")

    approve = commands.add_parser("approve", help="Record a completed human review for scaling or export")
    approve.add_argument("--run", type=_path, required=True)
    approve.add_argument("--review", type=_path, required=True)
    approve.add_argument("--stage", choices=("scale", "full", "export"), required=True)
    approve.add_argument("--acknowledge-review", action="store_true")

    recover = commands.add_parser("recover", help="Mark a stopped worker's in-flight request as billing-uncertain")
    recover.add_argument("--run", type=_path, required=True)
    recover.add_argument("--attempt", type=int, required=True)
    recover.add_argument("--worker-stopped", action="store_true")

    reconcile = commands.add_parser("reconcile", help="Record a verified provider charge for an uncertain request")
    reconcile.add_argument("--run", type=_path, required=True)
    reconcile.add_argument("--attempt", type=int, required=True)
    reconcile.add_argument("--cost-usd", required=True)
    reconcile.add_argument("--billing-verified", action="store_true")
    reconcile.add_argument("--retry", action="store_true")

    export = commands.add_parser("export", help="Export exact accepted targets after review and token checks")
    export.add_argument("--run", type=_path, required=True)
    export.add_argument("--output", type=_path, required=True)
    export.add_argument("--tokenizer", type=_path, help="Optional token-checked adapter for a model-neutral run")
    export.add_argument("--context-limit", type=int, help="Adapter context limit (default: 2048)")
    return parser


def _plan(store: RunStore) -> dict:
    config = store.config
    reserves = []
    for row in store.records(status="pending"):
        messages = [
            {"role": "system", "content": store.prompt},
            {"role": "user", "content": json.dumps({"document": row["record"]["text"]}, ensure_ascii=False)},
        ]
        reserves.append((
            Decimal(estimate_input_tokens(messages, store.schema)) * money(config["input_price_per_million"])
            + Decimal(config["max_tokens"]) * money(config["output_price_per_million"])
        ) / Decimal(1000000))
    return {
        **store.status(), "pending_requests": len(reserves),
        "largest_request_reservation_usd": str(max(reserves, default=Decimal(0))),
        "all_pending_one_attempt_reservation_usd": str(sum(reserves, Decimal(0))),
        "all_pending_max_attempts_reservation_usd": str(sum(reserves, Decimal(0)) * config["max_attempts"]),
        "note": "Conservative reservation bounds at approved provider price caps, not an expected invoice.",
    }


def dispatch(args: argparse.Namespace) -> dict:
    command = args.command
    if command == "dashboard":
        from .dashboard import serve_dashboard

        return serve_dashboard(args.run, port=args.port, ready_file=args.ready_file)
    if command == "download":
        if not args.allow_download:
            raise ValueError("Downloading requires --allow-download; no data was requested")
        return download_fineweb(
            args.output, revision=args.revision, max_records=args.max_records,
            subset=args.subset, split=args.split,
        )
    if command == "prepare":
        return prepare_jsonl(
            args.input, args.output, dataset=args.dataset, revision=args.revision,
            license_name=args.license_name, seed=args.seed, train_count=args.train_count,
            validation_count=args.validation_count, test_count=args.test_count,
            max_records=args.max_records, min_words=args.min_words, max_words=args.max_words,
            near_duplicate_threshold=args.near_duplicate_threshold, max_per_domain=args.max_per_domain,
            tokenizer_path=args.tokenizer, context_limit=args.context_limit,
            summary_token_reserve=args.summary_token_reserve,
            preserve_path=args.preserve,
        )
    if command == "init":
        return initialize_run(
            args.prepared, args.run, model=args.model, providers=args.providers,
            budget_usd=args.budget_usd, input_price_per_million=args.input_price_per_million,
            output_price_per_million=args.output_price_per_million, prompt_path=args.prompt,
            schema_path=args.schema, tokenizer_path=args.tokenizer,
            model_neutral=args.model_neutral,
            target_train=args.target_train, target_validation=args.target_validation, target_test=args.target_test,
            max_tokens=args.max_tokens, max_attempts=args.max_attempts,
            context_limit=args.context_limit, summary_token_reserve=args.summary_token_reserve,
        )
    with RunStore(args.run) as store:
        if command == "status":
            return store.status()
        if command == "plan":
            return _plan(store)
        if command == "generate":
            if not args.terms_reviewed:
                raise ValueError("Confirm provider synthetic-data and retention terms with --terms-reviewed")
            return generate(store, approve_paid=args.approve_paid, max_requests=args.max_requests)
        if command == "review-sample":
            return review_sample(store, args.output, per_stratum=args.per_stratum, seed=args.seed)
        if command == "review-report":
            return create_html_report(store, args.review, args.output)
        if command == "sample-report":
            return create_html_report(store, None, args.output, browse_only=True)
        if command == "assessment-status":
            return store.assessment_status()
        if command == "extend-candidates":
            return store.extend_candidates(args.prepared, acknowledge=args.acknowledge_extension)
        if command == "authorize-collection":
            return store.authorize_collection(
                budget_usd=money(args.budget_usd, positive=True), target_train=args.target_train,
                target_validation=args.target_validation, target_test=args.target_test,
                structural_threshold=money(args.structural_threshold, positive=True),
                usable_threshold=money(args.usable_threshold, positive=True),
                max_total_requests=args.max_total_requests, acknowledge=args.approve_auto_collection,
            )
        if command == "collection-status":
            return store.collection_status()
        if command == "amend-collection-policy":
            return store.amend_usable_threshold(
                money(args.usable_threshold, positive=True),
                reason=args.reason, acknowledge=args.approve_policy_change,
            )
        if command == "correct-gate-metrics":
            return store.correct_structural_metric(acknowledge=args.approve_correction)
        if command == "configure-retries":
            return store.authorize_retry_policy(
                max_retries_per_sample=args.max_retries_per_sample,
                min_retry_interval_seconds=args.min_retry_interval_seconds,
                acknowledge=args.approve_retry_policy,
            )
        if command == "collect":
            if not args.approve_paid or not args.terms_reviewed:
                raise ValueError("Collection requires --approve-paid and --terms-reviewed")
            from .collection import collect

            return collect(
                store, approve_paid=args.approve_paid, max_requests=args.max_requests,
                batch_size=args.batch_size,
                progress=lambda event: print(json.dumps(event, ensure_ascii=False), flush=True),
            )
        if command == "assess":
            if not args.terms_reviewed or not args.approve_paid:
                raise ValueError("Assessment requests require --approve-paid and --terms-reviewed")
            from .assessment import assess

            store.authorize_request_limit(
                args.max_total_requests, budget_usd=money(args.budget_usd, positive=True),
                acknowledge=args.approve_paid,
            )
            return assess(
                store, approve_paid=args.approve_paid, max_requests=args.max_requests,
                prompt_path=args.prompt, schema_path=args.schema,
            )
        if command == "approve":
            return approve_review(store, args.review, stage=args.stage, acknowledge=args.acknowledge_review)
        if command == "recover":
            if not args.worker_stopped:
                raise ValueError("Stop the worker first, then confirm --worker-stopped")
            store.recover(args.attempt)
            return store.status()
        if command == "reconcile":
            if not args.billing_verified:
                raise ValueError("Verify the charge in the provider dashboard, then confirm --billing-verified")
            store.reconcile(args.attempt, money(args.cost_usd), retry=args.retry)
            return store.status()
        if command == "export":
            return export_dataset(
                store, args.output, tokenizer_path=args.tokenizer, context_limit=args.context_limit
            )
    raise ValueError("Unknown command")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = dispatch(args)
    except (ValueError, OSError, RuntimeError, sqlite3.Error, OpenRouterError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted. Check status; unresolved requests remain reserved, never auto-retried.", file=sys.stderr)
        return 130
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.command == "generate" and result["stop_reason"] not in ("request_limit", "targets_met"):
        return 2
    if args.command == "assess" and result["stop_reason"] not in ("request_limit", "assessment_complete"):
        return 2
    if args.command == "collect" and result["stop_reason"] not in ("request_limit", "targets_met"):
        return 2
    if args.command == "prepare" and result["shortfall"] > 0:
        return 2
    return 0
