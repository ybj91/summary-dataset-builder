from copy import deepcopy
from decimal import Decimal
import json
import os
import shutil
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

from summary_dataset_builder.collection import collect
from summary_dataset_builder.files import load_json
from summary_dataset_builder.generation import generate
from tests.helpers import completion, FakeClient, ROOT, RunCase, TOKENIZER


class FakeStore:
    def __init__(self, run_dir, *, count=0, target=10400, passed_checkpoints=()):
        self.run_dir = run_dir
        self.run_dir.mkdir()
        self.count = count
        self.target = target
        self.checkpoints = set(passed_checkpoints)
        self.gates = []
        self.fail_at = None
        self.state = {
            "model": "openai/gpt-oss-120b", "providers": ["deepinfra/bf16"],
            "collection_policy": {
                "budget_usd": "5", "targets": {"train": 10000, "validation": 200, "test": 200},
                "structural_threshold": "0.98", "usable_threshold": "0.75",
                "max_total_requests": 30000,
            },
            "accounting": {
                "known_cost_usd": "0.015827429", "budget_usd": "5",
                "remaining_usd": "4.984172571", "uncertain_attempts": 0,
                "budget_exceeded": False,
            },
            "inflight_attempts": [], "uncertain_attempt_ids": [], "reservation_overruns": [],
            "approvals": [], "failed_gate": False, "quarantined_counts": {},
        }

    def collection_status(self):
        status = deepcopy(self.state)
        checkpoint = 100
        while checkpoint in self.checkpoints:
            checkpoint = 1000 if checkpoint == 100 else checkpoint + 1000
        status.update({
            "generation_targets_met": self.count >= self.target,
            "usable_counts": {"train": min(self.count, self.target), "validation": 0, "test": 0},
            "required_checkpoint": checkpoint if self.count >= checkpoint else None,
            "next_checkpoint": checkpoint, "stages": deepcopy(self.gates),
        })
        status["accounting"].update({"attempts": self.count, "generation_attempts": self.count})
        return status

    def evaluate_collection_gate(self, *, final=False):
        status = self.collection_status()
        through = self.count if final else status["required_checkpoint"]
        gate = {
            "passed": through != self.fail_at, "generation_attempts": through,
            "window_start": max(self.checkpoints, default=0),
            "attempted": through - max(self.checkpoints, default=0),
            "structurally_valid": through - max(self.checkpoints, default=0),
            "usable": through - max(self.checkpoints, default=0), "final": final,
        }
        self.gates.append(gate)
        if gate["passed"] and not final:
            self.checkpoints.add(through)
        if not gate["passed"]:
            self.state["failed_gate"] = True
        return gate


def fake_generate(store, *, max_requests, **kwargs):
    amount = min(max_requests, store.target - store.count)
    store.count += amount
    return {
        "requests_this_invocation": amount,
        "stop_reason": "targets_met" if store.count >= store.target else "request_limit",
    }


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / ".cache" / "collection-tests" / uuid4().hex
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)
        self.store = FakeStore(self.root / "run")

    def read_progress(self):
        return load_json(self.store.run_dir / "collection-progress.json")

    def read_events(self):
        return [
            json.loads(line)
            for line in (self.store.run_dir / "collection-events.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        ]

    def run_collection(self, **kwargs):
        with patch("summary_dataset_builder.collection.generate", side_effect=fake_generate) as call:
            output = collect(self.store, approve_paid=True, **kwargs)
        return output, call

    def test_parameters_require_explicit_paid_approval_and_positive_integers(self):
        for options in (
            {"approve_paid": False}, {"approve_paid": 1},
            {"approve_paid": True, "max_requests": 0},
            {"approve_paid": True, "max_requests": True},
            {"approve_paid": True, "batch_size": -1},
            {"approve_paid": True, "batch_size": 1.5},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                collect(self.store, **options)
        self.assertEqual(list(self.store.run_dir.iterdir()), [])

    def test_authorization_must_already_exist(self):
        self.store.state["collection_policy"] = None
        output, call = self.run_collection()
        self.assertEqual(output["stop_reason"], "collection_not_authorized")
        self.assertEqual(output["state"], "stopped")
        call.assert_not_called()

    def test_resuming_at_pilot_runs_gate_before_any_request_and_reaches_exact_goals(self):
        self.store.count = 100
        events = []

        def assert_gate_then_generate(store, **kwargs):
            self.assertIn(100, store.checkpoints)
            return fake_generate(store, **kwargs)

        with patch(
            "summary_dataset_builder.collection.generate", side_effect=assert_gate_then_generate
        ) as call:
            result = collect(self.store, approve_paid=True, batch_size=450, progress=events.append)
        self.assertEqual(result["stop_reason"], "targets_met")
        self.assertEqual(result["requests_this_invocation"], 10300)
        self.assertEqual(self.store.count, 10400)
        self.assertEqual([event["event"] for event in events[:2]], ["start", "gate"])
        self.assertEqual(
            [gate["generation_attempts"] for gate in self.store.gates],
            [100, *range(1000, 10001, 1000), 10400],
        )
        self.assertTrue(self.store.gates[-1]["final"])
        self.assertEqual(self.store.state["approvals"], [])
        sizes = [item.kwargs["max_requests"] for item in call.call_args_list]
        self.assertEqual(sizes[:5], [450, 450, 450, 450, 100])
        self.assertEqual(result, self.read_progress())
        self.assertFalse((self.store.run_dir / ".collection.lock").exists())

    def test_partial_batches_count_actual_requests_and_resume_without_reset(self):
        self.store.count = 95
        first, call = self.run_collection(max_requests=12, batch_size=10)
        self.assertEqual([item.kwargs["max_requests"] for item in call.call_args_list], [5, 7])
        self.assertEqual(first["requests_this_invocation"], 12)
        self.assertEqual(first["generation_through"], 107)
        second, _ = self.run_collection(max_requests=3)
        self.assertEqual(second["requests_this_invocation"], 3)
        self.assertEqual(second["generation_through"], 110)
        self.assertEqual(len(self.store.gates), 1)
        self.assertEqual([row["event"] for row in self.read_events()].count("start"), 2)

    def test_checkpoint_is_evaluated_even_at_invocation_request_limit(self):
        result, _ = self.run_collection(max_requests=100)
        self.assertEqual(result["stop_reason"], "request_limit")
        self.assertEqual(self.store.gates[0]["generation_attempts"], 100)
        self.assertEqual([row["event"] for row in self.read_events()][-2:], ["gate", "stop"])

    def test_failed_gate_sends_no_additional_requests_and_remains_latched(self):
        self.store.count = 100
        self.store.fail_at = 100
        result, call = self.run_collection()
        self.assertEqual(result["stop_reason"], "stage_failed")
        call.assert_not_called()
        result, call = self.run_collection()
        self.assertEqual(result["stop_reason"], "stage_failed")
        self.assertEqual(len(self.store.gates), 1)
        call.assert_not_called()

    def test_failed_final_gate_is_not_reported_as_targets_met(self):
        self.store.target = 4
        self.store.fail_at = 4
        result, call = self.run_collection()
        self.assertEqual(result["stop_reason"], "stage_failed")
        self.assertEqual(result["state"], "stopped")
        self.assertEqual(call.call_count, 1)
        self.assertTrue(self.store.gates[-1]["final"])

    def test_final_gate_that_quarantines_a_record_requires_replacement(self):
        self.store.target = 4
        evaluate = self.store.evaluate_collection_gate

        def quarantine(*, final=False):
            gate = evaluate(final=final)
            if len(self.store.gates) == 1:
                self.store.target = 5
            return gate

        self.store.evaluate_collection_gate = quarantine
        result, call = self.run_collection()
        self.assertEqual(result["stop_reason"], "targets_met")
        self.assertEqual(result["requests_this_invocation"], 5)
        self.assertEqual(call.call_count, 2)

    def test_startup_safety_blocks_before_generation(self):
        cases = (
            ("inflight_attempts", [1], "inflight_requests"),
            ("uncertain_attempt_ids", [1], "uncertain_billing"),
            ("reservation_overruns", [1], "provider_cost_exceeded_reservation"),
            ("failed_gate", True, "stage_failed"),
        )
        for field, value, reason in cases:
            with self.subTest(field=field):
                previous = self.store.state[field]
                self.store.state[field] = value
                result, call = self.run_collection()
                self.assertEqual(result["stop_reason"], reason)
                call.assert_not_called()
                self.store.state[field] = previous
        for field, value, reason in (
            ("uncertain_attempts", 1, "uncertain_billing"),
            ("budget_exceeded", True, "budget_exceeded"),
        ):
            with self.subTest(field=field):
                previous = self.store.state["accounting"][field]
                self.store.state["accounting"][field] = value
                result, call = self.run_collection()
                self.assertEqual(result["stop_reason"], reason)
                call.assert_not_called()
                self.store.state["accounting"][field] = previous

    def test_generation_fatal_stop_is_not_hidden_by_targets_or_retried(self):
        for reason in (
            "uncertain_billing", "provider_cost_exceeded_reservation",
            "rate_limit_retry_exhausted", "candidates_exhausted",
        ):
            with self.subTest(reason=reason):
                self.store.count = 0
                self.store.target = 1

                def fail(store, **kwargs):
                    store.count += 1
                    return {"requests_this_invocation": 1, "stop_reason": reason}

                with patch("summary_dataset_builder.collection.generate", side_effect=fail) as call:
                    output = collect(self.store, approve_paid=True)
                self.assertEqual(output["stop_reason"], reason)
                self.assertEqual(output["requests_this_invocation"], 1)
                self.assertEqual(call.call_count, 1)
                self.assertEqual(self.store.gates, [])

    def test_lifetime_request_limit_is_shared_with_prior_attempts(self):
        self.store.count = 98
        self.store.state["collection_policy"]["max_total_requests"] = 102
        result, call = self.run_collection(batch_size=25)
        self.assertEqual(result["stop_reason"], "total_request_limit")
        self.assertEqual(result["requests_this_invocation"], 4)
        self.assertEqual([item.kwargs["max_requests"] for item in call.call_args_list], [2, 2])
        self.assertEqual(result["accounting"]["known_cost_usd"], "0.015827429")

    def test_exhausted_budget_does_not_invoke_generation(self):
        self.store.state["accounting"]["remaining_usd"] = "0"
        result, call = self.run_collection()
        self.assertEqual(result["stop_reason"], "budget_exhausted")
        call.assert_not_called()

    def test_gate_validation_failure_is_persisted_without_another_request(self):
        self.store.count = 100
        self.store.evaluate_collection_gate = Mock(side_effect=ValueError("Collection gate failed"))
        result, call = self.run_collection()
        self.assertEqual(result["stop_reason"], "stage_failed")
        self.assertEqual(self.read_progress()["stop_reason"], "stage_failed")
        call.assert_not_called()

    def test_budget_exception_preserves_partial_batch_count_without_raw_error(self):
        def fail(store, **kwargs):
            store.count += 2
            raise ValueError("Budget cannot cover reservation: SENSITIVE_HTTP_BODY")

        with patch("summary_dataset_builder.collection.generate", side_effect=fail) as call:
            result = collect(self.store, approve_paid=True)
        self.assertEqual(result["stop_reason"], "budget_exhausted")
        self.assertEqual(result["requests_this_invocation"], 2)
        self.assertEqual(call.call_count, 1)
        self.assertNotIn("SENSITIVE", json.dumps(self.read_progress()))
        self.assertNotIn("SENSITIVE", json.dumps(self.read_events()))
        self.assertFalse((self.store.run_dir / ".collection.lock").exists())

    def test_unexpected_failure_and_interrupt_persist_failed_state_and_bubble(self):
        for error, reason in (
            (RuntimeError("SENSITIVE_HTTP_BODY"), "unexpected_error"),
            (KeyboardInterrupt(), "interrupted"),
        ):
            with self.subTest(error=type(error).__name__):
                def fail(store, **kwargs):
                    store.count += 1
                    raise error

                with patch("summary_dataset_builder.collection.generate", side_effect=fail):
                    with self.assertRaises(type(error)):
                        collect(self.store, approve_paid=True)
                result = self.read_progress()
                self.assertEqual(result["state"], "failed")
                self.assertEqual(result["stop_reason"], reason)
                self.assertEqual(result["requests_this_invocation"], 1)
                self.assertNotIn("SENSITIVE", json.dumps(result))
                self.assertFalse((self.store.run_dir / ".collection.lock").exists())

    def test_unexpected_failures_notify_progress_and_keep_last_status_if_refresh_fails(self):
        events = []
        status_reader = self.store.collection_status

        def fail(store, **kwargs):
            store.collection_status = Mock(side_effect=RuntimeError("status unavailable"))
            raise RuntimeError("SENSITIVE_HTTP_BODY")

        with patch("summary_dataset_builder.collection.generate", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "SENSITIVE_HTTP_BODY"):
                collect(self.store, approve_paid=True, progress=events.append)
        self.store.collection_status = status_reader
        self.assertEqual(events[-1]["event"], "stop")
        self.assertEqual(events[-1]["state"], "failed")
        self.assertEqual(self.read_progress()["accounting"]["known_cost_usd"], "0.015827429")

    def test_callback_value_error_is_unexpected_and_not_a_success_shaped_stop(self):
        def fail(event):
            raise ValueError("callback bug")

        with self.assertRaisesRegex(ValueError, "callback bug"):
            self.run_collection(progress=fail)
        self.assertEqual(self.read_progress()["state"], "failed")
        self.assertEqual(self.read_progress()["stop_reason"], "unexpected_error")
        self.assertFalse((self.store.run_dir / ".collection.lock").exists())

    def test_callback_cannot_turn_failed_final_gate_into_success(self):
        self.store.target = 1
        self.store.fail_at = 1

        def mutate(event):
            if event["stage"]:
                event["stage"]["passed"] = True

        result, _ = self.run_collection(progress=mutate)
        self.assertEqual(result["stop_reason"], "stage_failed")
        self.assertFalse(self.store.gates[-1]["passed"])

    def test_existing_lock_never_overwrites_active_progress_or_removes_lock(self):
        lock = self.store.run_dir / ".collection.lock"
        lock.write_text(json.dumps({"pid": os.getpid(), "token": "other"}), encoding="utf-8")
        path = self.store.run_dir / "collection-progress.json"
        path.write_text('{"active": true}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Collection lock exists"):
            collect(self.store, approve_paid=True)
        self.assertEqual(path.read_text(encoding="utf-8"), '{"active": true}')
        self.assertTrue(lock.exists())
        self.assertFalse((self.store.run_dir / "collection-events.jsonl").exists())

    def test_replaced_lock_is_not_removed_by_previous_owner(self):
        def replace_owner(store, **kwargs):
            (store.run_dir / ".collection.lock").write_text(
                json.dumps({"pid": os.getpid(), "token": "new-owner"}), encoding="utf-8"
            )
            return {"requests_this_invocation": 0, "stop_reason": "candidates_exhausted"}

        with patch("summary_dataset_builder.collection.generate", side_effect=replace_owner):
            collect(self.store, approve_paid=True)
        self.assertEqual(load_json(self.store.run_dir / ".collection.lock")["token"], "new-owner")

    def test_callback_observes_durable_progress_and_client_is_only_passed_when_provided(self):
        events = []

        def progress(value):
            persisted = self.read_progress()
            for key in ("event", "state", "requests_this_invocation", "generation_through", "accounting"):
                self.assertEqual(value[key], persisted[key])
            self.assertEqual(value, self.read_events()[-1])
            self.assertNotIn("stages", value)
            self.assertNotIn("collection_policy", value)
            events.append(value)

        output, call = self.run_collection(max_requests=1, progress=progress)
        self.assertNotIn("client", call.call_args.kwargs)
        self.assertEqual([row["event"] for row in events], ["start", "batch", "stop"])
        self.assertTrue(output["updated_at"].endswith("+00:00"))
        client = FakeClient()
        sleep = Mock()
        _, call = self.run_collection(max_requests=1, client=client, sleep=sleep)
        self.assertIs(call.call_args.kwargs["client"], client)
        self.assertIs(call.call_args.kwargs["sleep"], sleep)

    def test_progress_is_compact_even_after_many_completed_stages(self):
        self.store.count = 10000
        self.store.checkpoints = {100, *range(1000, 10001, 1000)}
        self.store.gates = [
            {"passed": True, "generation_attempts": count, "final": False}
            for count in sorted(self.store.checkpoints)
        ]
        events = []
        result, _ = self.run_collection(max_requests=25, progress=events.append)
        self.assertEqual(result["stage_count"], 11)
        self.assertEqual(result["stage"]["generation_attempts"], 10000)
        self.assertNotIn("stages", result)
        for event in events:
            self.assertNotIn("stages", event)
            self.assertNotIn("collection_policy", event)
            self.assertEqual(event["stage_count"], 11)
            self.assertLess(len(json.dumps(event)), 1500)

    def test_no_progress_is_not_an_infinite_loop(self):
        with patch("summary_dataset_builder.collection.generate", return_value={
            "requests_this_invocation": 0, "stop_reason": "request_limit",
        }) as call:
            output = collect(self.store, approve_paid=True)
        self.assertEqual(output["stop_reason"], "no_progress")
        self.assertEqual(call.call_count, 1)


class CollectionIntegrationTests(RunCase):
    def setUp(self):
        self.root = ROOT / ".cache" / "collection-tests" / uuid4().hex
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)
        self.prepared = self.root / "prepared"
        self.prepared.mkdir()
        self.run = self.root / "run"
        self.tokenizer = self.root / "tokenizer.json"
        self.tokenizer.write_text(json.dumps(TOKENIZER), encoding="utf-8")
        self.write_sources()

    def authorize(self, store, **overrides):
        options = {
            "budget_usd": Decimal("5"), "target_train": 2, "target_validation": 1,
            "target_test": 1, "max_total_requests": 30000, "acknowledge": True,
        }
        options.update(overrides)
        return store.authorize_collection(**options)

    def test_real_store_completes_exact_targets_and_preserves_original_configuration(self):
        self.initialize()
        store = self.store()
        original = (self.run / "config.json").read_bytes()
        self.authorize(store)
        result = collect(store, approve_paid=True, client=FakeClient(), batch_size=3)
        self.assertEqual(result["stop_reason"], "targets_met")
        self.assertEqual(result["requests_this_invocation"], 4)
        self.assertEqual(result["usable_counts"], {"train": 2, "validation": 1, "test": 1})
        self.assertEqual(store.status()["approvals"], [])
        self.assertEqual((self.run / "config.json").read_bytes(), original)
        self.assertEqual(load_json(self.run / "collection-progress.json"), result)
        self.assertFalse((self.run / ".collection.lock").exists())

    def test_real_store_resume_counts_prior_cost_and_only_new_requests(self):
        self.initialize()
        store = self.store()
        generate(store, approve_paid=True, max_requests=1, client=FakeClient())
        self.authorize(store)
        first = collect(store, approve_paid=True, max_requests=1, client=FakeClient())
        self.assertEqual(first["stop_reason"], "request_limit")
        self.assertEqual(first["requests_this_invocation"], 1)
        store.db.close()
        store = self.store()
        second = collect(store, approve_paid=True, client=FakeClient())
        self.assertEqual(second["stop_reason"], "targets_met")
        self.assertEqual(second["requests_this_invocation"], 2)
        self.assertEqual(Decimal(second["accounting"]["known_cost_usd"]), Decimal("0.0004"))
        self.assertEqual(second["accounting"]["generation_attempts"], 4)

    def test_real_uncertain_completion_stops_and_does_not_retry_on_resume(self):
        self.initialize()
        store = self.store()
        self.authorize(store)
        client = FakeClient([completion(cost=None)])
        result = collect(store, approve_paid=True, client=client)
        self.assertEqual(result["stop_reason"], "uncertain_billing")
        self.assertEqual(len(client.calls), 1)
        resumed_client = FakeClient()
        resumed = collect(store, approve_paid=True, client=resumed_client)
        self.assertEqual(resumed["stop_reason"], "uncertain_billing")
        self.assertEqual(resumed["requests_this_invocation"], 0)
        self.assertEqual(resumed_client.calls, [])

    def test_real_budget_reservation_failure_stops_without_sending_a_request(self):
        self.initialize()
        store = self.store()
        self.authorize(store, budget_usd=Decimal("0.000001"))
        client = FakeClient()
        result = collect(store, approve_paid=True, client=client)
        self.assertEqual(result["stop_reason"], "budget_exhausted")
        self.assertEqual(result["state"], "stopped")
        self.assertEqual(result["requests_this_invocation"], 0)
        self.assertEqual(client.calls, [])
        self.assertEqual(load_json(self.run / "collection-progress.json"), result)

    def test_real_final_gate_validates_all_responses_before_reporting_success(self):
        self.initialize()
        store = self.store()
        self.authorize(store)
        client = FakeClient([completion("not JSON"), *[completion() for _ in range(4)]])
        result = collect(store, approve_paid=True, client=client, batch_size=2)
        self.assertEqual(result["stop_reason"], "stage_failed")
        self.assertEqual(result["requests_this_invocation"], 5)
        self.assertEqual(result["usable_counts"], {"train": 2, "validation": 1, "test": 1})
        self.assertEqual(result["stage"]["attempted"], 5)
        self.assertEqual(result["stage"]["structurally_valid"], 4)
        self.assertTrue(result["stage"]["final"])
        self.assertTrue(result["failed_gate"])
        self.assertTrue(result["human_review_required_before_release"])
        self.assertEqual(result["approvals"], [])
        resumed = FakeClient()
        self.assertEqual(
            collect(store, approve_paid=True, client=resumed)["stop_reason"], "stage_failed"
        )
        self.assertEqual(resumed.calls, [])
