import json
from unittest.mock import Mock

from summary_dataset_builder.generation import generate
from summary_dataset_builder.openrouter import OpenRouterError
from tests.helpers import completion, FakeClient, RunCase


class GenerationTests(RunCase):
    def test_requires_explicit_paid_approval(self):
        self.initialize()
        client = FakeClient()
        with self.assertRaisesRegex(ValueError, "approve-paid"):
            generate(self.store(), approve_paid=False, client=client)
        self.assertEqual(client.calls, [])

    def test_generation_resumes_and_stops_at_exact_split_targets(self):
        self.initialize()
        store = self.store()
        first_client = FakeClient()
        first = generate(store, approve_paid=True, max_requests=2, client=first_client)
        self.assertFalse(first["generation_targets_met"])
        self.assertEqual(first["requests_this_invocation"], 2)
        second_client = FakeClient()
        second = generate(store, approve_paid=True, max_requests=20, client=second_client)
        self.assertEqual(second["stop_reason"], "targets_met")
        self.assertEqual(len(second_client.calls), 2)
        self.assertEqual(second["accounting"]["attempts"], 4)
        self.assertEqual(second["counts"]["train"]["accepted"], 2)
        self.assertEqual(first_client.calls[0]["messages"][0]["role"], "system")
        self.assertIn("document", json.loads(first_client.calls[0]["messages"][1]["content"]))

    def test_truncation_malformed_json_rejection_and_quality_flags_are_retained(self):
        self.initialize()
        responses = [
            completion("", finish_reason="length"), completion("not JSON"),
            completion(json.dumps({"accepted": False, "summary": "", "rejection_reason": "Incomplete source"})),
            completion(json.dumps({"accepted": True, "summary": "It produced 999999 tons.", "rejection_reason": ""})),
        ]
        store = self.store()
        generate(store, approve_paid=True, max_requests=4, client=FakeClient(responses))
        self.assertEqual(len(list(store.records(status="error"))), 2)
        self.assertEqual(len(list(store.records(status="rejected"))), 1)
        self.assertEqual(len(list(store.records(status="flagged"))), 1)
        self.assertEqual(store.accounting()["known_cost_usd"], "0.0004")

    def test_missing_usage_stops_and_reserves_unknown_billing(self):
        self.initialize()
        client = FakeClient([completion(cost=None)])
        store = self.store()
        result = generate(store, approve_paid=True, max_requests=10, client=client)
        self.assertEqual(result["stop_reason"], "uncertain_billing")
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(store.accounting()["uncertain_attempts"], 1)
        with self.assertRaisesRegex(ValueError, "uncertain"):
            generate(store, approve_paid=True, client=FakeClient())

    def test_budget_too_small_sends_no_request(self):
        self.initialize(budget_usd="0.000001")
        client = FakeClient()
        with self.assertRaisesRegex(ValueError, "Budget"):
            generate(self.store(), approve_paid=True, client=client)
        self.assertEqual(client.calls, [])

    def test_interrupt_leaves_uncertain_billing(self):
        self.initialize()
        store = self.store()
        with self.assertRaises(KeyboardInterrupt):
            generate(store, approve_paid=True, client=FakeClient([KeyboardInterrupt()]))
        self.assertEqual(store.accounting()["uncertain_attempts"], 1)
        self.assertEqual(store.status()["inflight_attempts"], [])

    def test_rate_limits_are_bounded_and_charged_separately(self):
        self.initialize()
        error = OpenRouterError(status_code=429, retryable=True, uncertain=False)
        client = FakeClient([error, error, error])
        sleep = Mock()
        store = self.store()
        result = generate(store, approve_paid=True, max_requests=10, client=client, sleep=sleep)
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(result["stop_reason"], "rate_limit_retry_exhausted")
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(store.accounting()["known_cost_usd"], "0")

    def test_timeout_does_not_retry(self):
        self.initialize()
        client = FakeClient([OpenRouterError(uncertain=True, retryable=False)])
        result = generate(self.store(), approve_paid=True, max_requests=10, client=client)
        self.assertEqual(result["stop_reason"], "uncertain_billing")
        self.assertEqual(len(client.calls), 1)

    def test_cost_overrun_is_not_hidden_by_reaching_targets(self):
        self.initialize(target_train=1)
        store = self.store()
        result = generate(
            store, approve_paid=True, max_requests=3,
            client=FakeClient([completion(), completion(), completion(cost="0.1")]),
        )
        self.assertTrue(result["generation_targets_met"])
        self.assertEqual(result["stop_reason"], "provider_cost_exceeded_reservation")
        self.assertEqual(result["reservation_overruns"], [3])
