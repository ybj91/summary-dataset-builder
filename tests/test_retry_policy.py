from decimal import Decimal
import json
from unittest.mock import patch

from summary_dataset_builder.assessment import assess
from summary_dataset_builder.collection import collect
from summary_dataset_builder.cli import main
from summary_dataset_builder.generation import generate
from summary_dataset_builder.openrouter import OpenRouterError
from summary_dataset_builder.retries import RetryRateLimited
from summary_dataset_builder.store import RunStore
from tests.helpers import completion, FakeClient, ROOT, RunCase


class Clock:
    def __init__(self):
        self.now = 1000.0
        self.waits = []

    def time(self):
        return self.now

    def sleep(self, duration):
        self.waits.append(duration)
        self.now += duration


class RetryPolicyTests(RunCase):
    def enable(self, store, **kwargs):
        return store.authorize_retry_policy(acknowledge=True, **kwargs)

    def test_three_retries_then_next_sample_with_one_global_retry_per_minute(self):
        self.initialize()
        store = self.store()
        self.enable(store)
        error = OpenRouterError(status_code=429, retryable=True)
        client = FakeClient([error, error, error, error, error, completion()])
        clock = Clock()
        with patch("summary_dataset_builder.store.time.time", clock.time):
            result = generate(store, approve_paid=True, max_requests=6, client=client, sleep=clock.sleep)
        self.assertEqual(len(client.calls), 6)
        failed = list(store.records(status="error"))
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["attempts"], 4)
        self.assertEqual(failed[0]["reason"], "retry_exhausted:http_429")
        self.assertEqual(len(list(store.records(status="accepted"))), 1)
        retries = [
            row[0] for row in store.db.execute(
                "SELECT retry_started_at FROM attempts WHERE retry_started_at IS NOT NULL ORDER BY id"
            )
        ]
        self.assertEqual(retries, [1000.0, 1060.0, 1120.0, 1180.0])
        self.assertEqual(clock.waits, [60.0, 60.0, 60.0])
        self.assertEqual(result["stop_reason"], "request_limit")

    def test_rate_limit_survives_reopening_and_reapplying_policy(self):
        self.initialize()
        store = self.store()
        self.enable(store)
        error = OpenRouterError(status_code=429, retryable=True)
        clock = Clock()
        with patch("summary_dataset_builder.store.time.time", clock.time):
            generate(store, approve_paid=True, max_requests=2, client=FakeClient([error, error]), sleep=clock.sleep)
            self.enable(store)
            with RunStore(self.run) as reopened:
                document = reopened.next_document()
                with self.assertRaises(RetryRateLimited) as rate:
                    reopened.reserve(document["document_sha256"], Decimal(".01"))
                self.assertEqual(rate.exception.retry_after, 60.0)
                clock.now += 59.9
                with self.assertRaises(RetryRateLimited):
                    reopened.reserve(document["document_sha256"], Decimal(".01"))
                clock.now = 1060.0
                reopened.reserve(document["document_sha256"], Decimal(".01"))
        self.assertEqual(store.accounting()["attempts"], 3)

    def test_uncertain_network_failures_keep_reservations_and_do_not_halt_collection(self):
        self.initialize(model_neutral=True, tokenizer_path=None)
        store = self.store()
        self.enable(store)
        store.authorize_collection(
            budget_usd=Decimal("5"), target_train=2, target_validation=1, target_test=1, acknowledge=True,
        )
        error = OpenRouterError(uncertain=True)
        clock = Clock()
        with patch("summary_dataset_builder.store.time.time", clock.time):
            result = collect(
                store, approve_paid=True, max_requests=5, batch_size=5,
                client=FakeClient([error, error, error, error, completion()]), sleep=clock.sleep,
            )
        self.assertEqual(result["stop_reason"], "request_limit")
        self.assertEqual(store.accounting()["unconfirmed_billing_attempts"], 4)
        self.assertEqual(store.accounting()["blocking_uncertain_attempts"], 0)
        self.assertEqual(store.accounting()["known_cost_usd"], "0.0001")
        reserved = sum(
            (Decimal(row[0]) for row in store.db.execute("SELECT reserved_usd FROM attempts WHERE cost_usd IS NULL")),
            Decimal(0),
        )
        self.assertEqual(Decimal(store.accounting()["uncertain_reserved_usd"]), reserved)
        self.assertEqual(len(list(store.records(status="accepted"))), 1)
        self.assertEqual(len(list(store.records(status="error"))), 1)

    def test_retry_policy_recovers_stopped_request_without_inventing_its_cost(self):
        self.initialize()
        store = self.store()
        error = OpenRouterError(status_code=200, uncertain=True)
        original = generate(store, approve_paid=True, client=FakeClient([error]))
        self.assertEqual(original["stop_reason"], "uncertain_billing")
        reserved = store.accounting()["uncertain_reserved_usd"]
        report = self.enable(store)
        self.assertEqual(report["requeued_samples"], 1)
        self.assertEqual(report["charges_retained_as_unconfirmed"], 1)
        generate(store, approve_paid=True, max_requests=1, client=FakeClient())
        saved = list(store.records(status="accepted"))
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["attempts"], 2)
        self.assertEqual(store.accounting()["uncertain_reserved_usd"], reserved)
        self.assertEqual(store.accounting()["known_cost_usd"], "0.0001")
        self.assertIsNone(store.db.execute("SELECT cost_usd FROM attempts WHERE id=1").fetchone()[0])
        store.reconcile(1, Decimal(".002"))
        self.assertEqual(list(store.records(status="accepted")), saved)
        self.assertEqual(store.accounting()["known_cost_usd"], "0.0021")

    def test_unconfirmed_reservations_can_exhaust_budget_before_retry(self):
        self.initialize(budget_usd=".02")
        store = self.store()
        attempt = store.reserve(store.next_document()["document_sha256"], Decimal(".015"))
        store.finish(attempt, status="error", cost=None, reason="transport_error")
        self.enable(store)
        with self.assertRaisesRegex(ValueError, "Budget"):
            store.reserve(store.next_document()["document_sha256"], Decimal(".015"))
        self.assertEqual(store.accounting()["attempts"], 1)
        self.assertEqual(store.accounting()["committed_usd"], "0.015")

    def test_malformed_output_retries_but_quality_flags_and_rejections_do_not(self):
        self.initialize()
        store = self.store()
        self.enable(store)
        rejected = json.dumps({"accepted": False, "summary": "", "rejection_reason": "Incomplete"})
        flagged = json.dumps({"accepted": True, "summary": "It grew 999999 plants.", "rejection_reason": ""})
        clock = Clock()
        with patch("summary_dataset_builder.store.time.time", clock.time):
            generate(
                store, approve_paid=True, max_requests=4,
                client=FakeClient([completion("bad JSON"), completion(), completion(rejected), completion(flagged)]),
                sleep=clock.sleep,
            )
        self.assertEqual(next(store.records(status="accepted"))["attempts"], 2)
        self.assertEqual(next(store.records(status="rejected"))["attempts"], 1)
        self.assertEqual(next(store.records(status="flagged"))["attempts"], 1)

    def test_valid_response_without_usage_is_saved_without_generating_a_duplicate(self):
        self.initialize()
        store = self.store()
        self.enable(store)
        result = generate(store, approve_paid=True, max_requests=1, client=FakeClient([completion(cost=None)]))
        self.assertEqual(result["stop_reason"], "request_limit")
        self.assertEqual(len(list(store.records(status="accepted"))), 1)
        self.assertEqual(store.accounting()["unconfirmed_billing_attempts"], 1)
        self.assertEqual(store.accounting()["retry_attempts"], 0)

    def test_authentication_failure_still_stops_instead_of_hammering_other_samples(self):
        self.initialize()
        store = self.store()
        self.enable(store)
        client = FakeClient([OpenRouterError(status_code=401)])
        result = generate(store, approve_paid=True, max_requests=20, client=client)
        self.assertEqual(result["stop_reason"], "http_401")
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(len(list(store.records(status="error"))), 1)

    def test_retry_allowance_and_throttle_are_shared_across_generation_and_assessment(self):
        self.initialize()
        store = self.store()
        self.enable(store)
        error = OpenRouterError(status_code=429, retryable=True)
        clock = Clock()
        with patch("summary_dataset_builder.store.time.time", clock.time):
            generate(store, approve_paid=True, max_requests=2, client=FakeClient([error, completion()]), sleep=clock.sleep)
            result = assess(
                store, approve_paid=True, max_requests=10,
                prompt_path=ROOT / "prompts" / "assessor-v1.txt",
                schema_path=ROOT / "prompts" / "assessor-v1.schema.json",
                client=FakeClient([error, error, error]), sleep=clock.sleep,
            )
        self.assertEqual(result["stop_reason"], "assessment_complete")
        self.assertEqual(store.assessment_status()["counts"], {"error": 1})
        self.assertEqual(store.accounting()["generation_attempts"], 2)
        self.assertEqual(store.accounting()["assessment_attempts"], 3)
        self.assertEqual(store.accounting()["retry_attempts"], 3)
        self.assertEqual(clock.waits, [60.0, 60.0])

    def test_retry_policy_does_not_weaken_operational_gates(self):
        self.initialize()
        store = self.store()
        self.enable(store)
        store.authorize_collection(
            budget_usd=Decimal("5"), target_train=2, target_validation=1, target_test=1, acknowledge=True,
        )
        generate(store, approve_paid=True, max_requests=1, client=FakeClient([completion("malformed JSON")]))
        generate(store, approve_paid=True, max_requests=4, client=FakeClient())
        self.assertTrue(store.status()["generation_targets_met"])
        gate = store.evaluate_collection_gate(final=True)
        self.assertEqual(gate["structural_fraction"], "0.8")
        self.assertFalse(gate["passed"])
        self.assertEqual(store.status()["approvals"], [])

    def test_policy_requires_approval_and_cannot_reset_retry_history(self):
        self.initialize()
        store = self.store()
        with self.assertRaisesRegex(ValueError, "explicit approval"):
            store.authorize_retry_policy(acknowledge=False)
        for options in ({"max_retries_per_sample": 4}, {"min_retry_interval_seconds": 59}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.enable(store, **options)
        self.enable(store)
        with self.assertRaisesRegex(ValueError, "already frozen"):
            self.enable(store, min_retry_interval_seconds=120)

    def test_cli_policy_is_explicit(self):
        self.initialize()
        with patch("builtins.print"):
            self.assertEqual(main(["configure-retries", "--run", str(self.run)]), 2)
            self.assertEqual(main([
                "configure-retries", "--run", str(self.run), "--max-retries-per-sample", "3",
                "--min-retry-interval-seconds", "60", "--approve-retry-policy",
            ]), 0)
        self.assertEqual(self.store().retry_policy["max_retries_per_sample"], 3)
