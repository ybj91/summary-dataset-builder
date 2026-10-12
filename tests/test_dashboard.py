import base64
from datetime import datetime, timedelta, timezone
import hashlib
from http.client import HTTPConnection
from importlib.resources import files
import json
import sqlite3
import threading
from unittest.mock import patch

from summary_dataset_builder.dashboard import create_dashboard_server, dashboard_status, estimate_remaining_time
from summary_dataset_builder.generation import generate
from summary_dataset_builder.store import RunStore
from tests.helpers import FakeClient, RunCase


class DashboardTests(RunCase):
    def setUp(self):
        super().setUp()
        self.initialize(model_neutral=True, tokenizer_path=None)

    def start_server(self):
        server = create_dashboard_server(self.run, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def cleanup():
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        self.addCleanup(cleanup)
        return server

    def request(self, server, path, *, method="GET", headers=None):
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        try:
            connection.request(method, path, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def test_read_only_store_cannot_write_or_initialize_tables(self):
        with RunStore(self.run, read_only=True) as store:
            self.assertEqual(store.accounting()["attempts"], 0)
            with self.assertRaises(sqlite3.OperationalError):
                store.db.execute("UPDATE documents SET status='error'")
            with self.assertRaises(sqlite3.OperationalError):
                store.db.execute("CREATE TABLE unexpected (id INTEGER)")
        self.assertEqual(len(list(self.store().records(status="pending"))), 12)

    def test_status_reads_fresh_progress_without_changing_the_run(self):
        before = dashboard_status(self.run)
        self.assertIsNone(before["worker"])
        self.assertEqual(before["status"]["usable_counts"]["train"], 0)
        generate(self.store(), approve_paid=True, max_requests=1, client=FakeClient())
        after = dashboard_status(self.run)
        self.assertEqual(after["status"]["accounting"]["generation_attempts"], 1)
        self.assertTrue(after["read_only"])
        self.assertEqual(sum(after["status"]["usable_counts"].values()), 1)
        serialized = json.dumps(after)
        self.assertNotIn("The town greenhouse", serialized)
        self.assertNotIn("OPENROUTER_API_KEY", serialized)
        self.assertFalse((self.run / "collection-progress.json").exists())

    def test_server_is_local_read_only_and_serves_current_status(self):
        server = self.start_server()
        self.assertEqual(server.server_address[0], "127.0.0.1")
        code, headers, body = self.request(server, "/api/status")
        self.assertEqual(code, 200)
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(json.loads(body)["status"]["accounting"]["attempts"], 0)
        generate(self.store(), approve_paid=True, max_requests=1, client=FakeClient())
        code, _, body = self.request(server, "/api/status")
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(body)["status"]["accounting"]["attempts"], 1)
        code, _, body = self.request(server, "/api/start", method="POST")
        self.assertEqual(code, 405)
        self.assertIn("read-only", json.loads(body)["error"])

    def test_page_has_matching_script_hash_and_no_external_assets(self):
        server = self.start_server()
        code, headers, body = self.request(server, "/")
        self.assertEqual(code, 200)
        page = body.decode("utf-8")
        script = page.split('<script id="dashboard-code">')[1].split("</script>")[0]
        digest = base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
        self.assertIn(f"script-src 'sha256-{digest}'", headers["Content-Security-Policy"])
        self.assertIn("connect-src 'self'", headers["Content-Security-Policy"])
        self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
        self.assertIn('fetch("/api/status"', script)
        self.assertNotIn("innerHTML", script)
        self.assertNotIn('<script src=', page)
        self.assertNotIn('<link ', page)

    def test_other_origins_and_arbitrary_files_are_not_served(self):
        server = self.start_server()
        for request_path in ("/ApiKey.txt", "/../config.json", "/data/source.jsonl"):
            code, _, _ = self.request(server, request_path)
            self.assertEqual(code, 404)
        for headers in ({"Host": "attacker.example"}, {"Origin": "https://attacker.example"}):
            code, response_headers, _ = self.request(server, "/api/status", headers=headers)
            self.assertEqual(code, 403)
            self.assertNotIn("Access-Control-Allow-Origin", response_headers)

    def test_read_errors_are_explicit_without_leaking_details_or_false_counters(self):
        server = self.start_server()
        with patch("summary_dataset_builder.dashboard.dashboard_status", side_effect=ValueError("PRIVATE")):
            with self.assertLogs("summary_dataset_builder.dashboard", level="ERROR"):
                code, _, body = self.request(server, "/api/status")
        self.assertEqual(code, 503)
        self.assertNotIn("PRIVATE", body.decode())
        self.assertNotIn("status", json.loads(body))

    def test_template_is_packaged_and_port_validation_is_explicit(self):
        self.assertIn("dashboard-code", files("summary_dataset_builder").joinpath("progress_dashboard.html").read_text())
        for port in (-1, 65536, True):
            with self.subTest(port=port), self.assertRaises(ValueError):
                create_dashboard_server(self.run, port=port)

    def eta_fixture(self):
        store = self.store()
        store.config["targets"] = {"train": 100, "validation": 100, "test": 100}
        generate(store, approve_paid=True, max_requests=12, client=FakeClient())
        now = datetime(2026, 10, 11, 12, 15, tzinfo=timezone.utc)
        with store.db:
            store.db.execute("UPDATE attempts SET started_at='2026-10-11 12:13:00',finished_at='2026-10-11 12:14:00'")
        worker = {"state": "running", "started_at": (now - timedelta(hours=2)).isoformat()}
        return store, now, worker

    def test_eta_uses_recent_usable_throughput_and_all_unfinished_split_targets(self):
        store, now, worker = self.eta_fixture()
        before = store.accounting()
        estimate = estimate_remaining_time(store, store.collection_status(), worker, now=now)
        self.assertEqual(estimate["state"], "estimating")
        self.assertEqual(estimate["observed_usable_pairs"], 12)
        self.assertEqual(estimate["window_seconds"], 900)
        self.assertEqual(estimate["usable_pairs_per_minute"], 0.8)
        self.assertEqual(estimate["remaining_usable_pairs"], 288)
        self.assertEqual(estimate["remaining_seconds"], 21600)
        self.assertEqual(estimate["estimated_finish_at"], "2026-10-11T18:15:00+00:00")
        self.assertEqual(store.accounting(), before)

    def test_eta_excludes_paused_time_before_resume_and_older_completions(self):
        store, now, worker = self.eta_fixture()
        worker["started_at"] = "2026-10-11T20:10:00+08:00"
        with store.db:
            store.db.execute("UPDATE attempts SET finished_at='2026-10-11 12:09:00' WHERE id IN (1,2)")
        estimate = estimate_remaining_time(store, store.collection_status(), worker, now=now)
        self.assertEqual(estimate["window_seconds"], 300)
        self.assertEqual(estimate["observed_usable_pairs"], 10)
        self.assertEqual(estimate["usable_pairs_per_minute"], 2)
        self.assertEqual(estimate["remaining_seconds"], 8640)

    def test_eta_excludes_assessment_results_and_quarantined_pairs(self):
        store, now, worker = self.eta_fixture()
        with store.db:
            store.db.execute("UPDATE attempts SET kind='assessment' WHERE id=1")
            row = store.db.execute("SELECT document_id FROM attempts WHERE id=2").fetchone()
            store.db.execute(
                "INSERT INTO quality_holds(document_id,reason,details) VALUES (?,'concern','{}')",
                (row["document_id"],),
            )
        estimate = estimate_remaining_time(store, store.collection_status(), worker, now=now)
        self.assertEqual(estimate["observed_usable_pairs"], 10)
        self.assertEqual(estimate["remaining_usable_pairs"], 289)

    def test_eta_requires_minimum_duration_and_sample_size(self):
        store, now, worker = self.eta_fixture()
        worker["started_at"] = (now - timedelta(seconds=30)).isoformat()
        with store.db:
            store.db.execute("UPDATE attempts SET finished_at='2026-10-11 12:14:50'")
        estimate = estimate_remaining_time(store, store.collection_status(), worker, now=now)
        self.assertEqual(estimate["state"], "warming_up")
        self.assertIsNone(estimate["remaining_seconds"])
        worker["started_at"] = (now - timedelta(minutes=15)).isoformat()
        with store.db:
            store.db.execute("UPDATE attempts SET intended_status='error' WHERE id>9")
        estimate = estimate_remaining_time(store, store.collection_status(), worker, now=now)
        self.assertEqual(estimate["state"], "warming_up")
        self.assertEqual(estimate["observed_usable_pairs"], 9)

    def test_eta_hidden_when_paused_failed_or_stalled(self):
        store, now, worker = self.eta_fixture()
        status = store.collection_status()
        for state in ("stopped", "failed"):
            with self.subTest(state=state):
                estimate = estimate_remaining_time(store, status, {**worker, "state": state}, now=now)
                self.assertEqual(estimate["state"], "paused")
                self.assertIsNone(estimate["estimated_finish_at"])
        estimate = estimate_remaining_time(store, {**status, "failed_gate": True}, worker, now=now)
        self.assertEqual(estimate["state"], "paused")
        with store.db:
            store.db.execute("UPDATE attempts SET finished_at='2026-10-11 12:09:00'")
        estimate = estimate_remaining_time(store, status, worker, now=now)
        self.assertEqual(estimate["state"], "stalled")
        self.assertIsNone(estimate["remaining_seconds"])

    def test_eta_does_not_count_progress_toward_already_finished_splits(self):
        store, now, worker = self.eta_fixture()
        status = store.collection_status()
        status["targets"]["train"] = status["usable_counts"]["train"]
        estimate = estimate_remaining_time(store, status, worker, now=now)
        self.assertEqual(estimate["remaining_usable_pairs"], 192)
        self.assertEqual(estimate["observed_usable_pairs"], 8)
        self.assertEqual(estimate["state"], "warming_up")

    def test_eta_completion_no_worker_and_invalid_or_future_start(self):
        store, now, worker = self.eta_fixture()
        status = store.collection_status()
        self.assertEqual(estimate_remaining_time(store, status, None, now=now)["state"], "unavailable")
        with self.assertRaisesRegex(ValueError, "start time is invalid"):
            estimate_remaining_time(store, status, {**worker, "started_at": "bad-date"}, now=now)
        future = {**worker, "started_at": (now + timedelta(minutes=1)).isoformat()}
        self.assertEqual(estimate_remaining_time(store, status, future, now=now)["state"], "warming_up")
        status["targets"] = dict(status["usable_counts"])
        estimate = estimate_remaining_time(store, status, worker, now=now)
        self.assertEqual(estimate["state"], "complete")
        self.assertEqual(estimate["remaining_seconds"], 0)
        self.assertIsNone(estimate["estimated_finish_at"])
