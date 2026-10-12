import base64
import hashlib
from html.parser import HTMLParser
from importlib.resources import files
import json
import shutil
import subprocess
import unittest
from unittest.mock import patch

from summary_dataset_builder.cli import main
from summary_dataset_builder.files import load_json
from summary_dataset_builder.generation import generate
from summary_dataset_builder.review import approve_review, review_sample
from summary_dataset_builder.review_html import create_html_report
from tests.helpers import completion, FakeClient, RunCase


class ReportParser(HTMLParser):
    def __init__(self, content):
        super().__init__()
        self.scripts = {}
        self.current_script = None
        self.csp = ""
        self.external_assets = []
        self.body_class = ""
        self.element_ids = []
        self.review_only_ids = set()
        self.ancestors = []
        self.feed(content)

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        review_only = "review-only" in attributes.get("class", "").split()
        if tag == "body":
            self.body_class = attributes.get("class", "")
        if "id" in attributes:
            self.element_ids.append(attributes["id"])
            if review_only or any(parent[1] for parent in self.ancestors):
                self.review_only_ids.add(attributes["id"])
        if tag not in {"meta", "input", "link", "img", "br", "hr"}:
            self.ancestors.append((tag, review_only))
        if tag == "script":
            self.current_script = attributes.get("id", "")
            self.scripts[self.current_script] = ""
        if tag == "meta" and attributes.get("http-equiv") == "Content-Security-Policy":
            self.csp = attributes["content"]
        if "src" in attributes or tag == "link":
            self.external_assets.append(attributes)

    def handle_endtag(self, tag):
        if tag == "script":
            self.current_script = None
        for index in range(len(self.ancestors) - 1, -1, -1):
            if self.ancestors[index][0] == tag:
                del self.ancestors[index:]
                break

    def handle_data(self, data):
        if self.current_script is not None:
            self.scripts[self.current_script] += data


class HtmlReviewTests(RunCase):
    def prepare_report(self):
        self.initialize(tokenizer_path=None, model_neutral=True)
        store = self.store()
        generate(store, approve_paid=True, max_requests=20, client=FakeClient())
        review = self.root / "review.json"
        review_sample(store, review, per_stratum=1)
        return store, review

    def parse_report(self, output):
        return ReportParser(output.read_text(encoding="utf-8"))

    def test_contains_all_results_and_the_exact_unmodified_sample(self):
        store, review = self.prepare_report()
        before = store.status()
        output = self.root / "review.html"
        result = create_html_report(store, review, output)
        parsed = self.parse_report(output)
        payload = json.loads(parsed.scripts["report-data"])
        self.assertEqual(payload["review"], load_json(review))
        self.assertEqual(len(payload["records"]), 4)
        self.assertEqual(len(payload["review"]["items"]), 3)
        self.assertEqual(result["generated_results"], 4)
        self.assertFalse(result["approval_changed"])
        self.assertEqual(store.status(), before)
        self.assertEqual(payload["status"]["accounting"]["known_cost_usd"], "0.0004")

    def test_source_and_raw_response_cannot_break_out_of_json_script(self):
        store, _ = self.prepare_report()
        attack = '</ScRiPt><script>window.compromised=true</script><img src="https://invalid.example/tracker">'
        row = next(store.records(status="accepted"))
        record = row["record"]
        record["text"] = attack + "\nCharacters: \u2028 \u2029 & < >"
        record["source_url"] = "javascript:alert(1)"
        with store.db:
            store.db.execute("UPDATE documents SET record=? WHERE id=?", (json.dumps(record), row["id"]))
            store.db.execute(
                "UPDATE attempts SET response=? WHERE document_id=?",
                (json.dumps({"content": attack}), row["id"]),
            )
        review = self.root / "hostile-review.json"
        review_sample(store, review, per_stratum=5)
        for browse_only in (False, True):
            with self.subTest(browse_only=browse_only):
                output = self.root / f"hostile-{browse_only}.html"
                create_html_report(store, review, output, browse_only=browse_only)
                parsed = self.parse_report(output)
                self.assertEqual(set(parsed.scripts), {"report-data", "report-code"})
                self.assertEqual(parsed.external_assets, [])
                self.assertNotIn("<", parsed.scripts["report-data"])
                payload = json.loads(parsed.scripts["report-data"])
                displayed = next(item for item in payload["records"] if item["document_id"] == row["id"])
                self.assertEqual(displayed["source"]["text"], record["text"])
                self.assertEqual(displayed["response"]["content"], attack)

    def test_csp_hash_matches_script_and_no_network_assets_are_needed(self):
        store, review = self.prepare_report()
        output = self.root / "review.html"
        create_html_report(store, review, output)
        parsed = self.parse_report(output)
        digest = base64.b64encode(hashlib.sha256(parsed.scripts["report-code"].encode()).digest()).decode()
        self.assertIn(f"script-src 'sha256-{digest}'", parsed.csp)
        self.assertIn("connect-src 'none'", parsed.csp)
        self.assertIn("default-src 'none'", parsed.csp)
        self.assertEqual(parsed.external_assets, [])
        self.assertNotIn("innerHTML", parsed.scripts["report-code"])
        self.assertNotIn("fetch(", parsed.scripts["report-code"])

    def test_stale_tampered_and_uncertain_reports_are_rejected(self):
        self.initialize(tokenizer_path=None, model_neutral=True)
        store = self.store()
        generate(store, approve_paid=True, max_requests=1, client=FakeClient())
        review = self.root / "review.json"
        review_sample(store, review)
        original = load_json(review)
        changed = load_json(review)
        changed["items"][0]["source"]["text"] = "Changed source"
        review.write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Only verdict and notes"):
            create_html_report(store, review, self.root / "tampered.html")
        review.write_text(json.dumps(original), encoding="utf-8")
        generate(store, approve_paid=True, max_requests=1, client=FakeClient([completion(cost=None)]))
        with self.assertRaisesRegex(ValueError, "stale"):
            create_html_report(store, review, self.root / "stale.html")
        self.assertFalse((self.root / "stale.html").exists())

    def test_download_contract_preserves_fields_and_is_accepted_by_approval(self):
        store, review = self.prepare_report()
        output = self.root / "review.html"
        create_html_report(store, review, output)
        draft = json.loads(self.parse_report(output).scripts["report-data"])["review"]
        for item in draft["items"]:
            item["verdict"] = "pass"
            item["notes"] = "Offline fixture: checked source and disposition."
        completed = self.root / "completed.json"
        completed.write_text(json.dumps(draft), encoding="utf-8")
        result = approve_review(store, completed, stage="export", acknowledge=True)
        self.assertEqual(result["reviewed_documents"], 3)

    def test_partially_completed_review_preserves_edits_without_autoapproval(self):
        store, review = self.prepare_report()
        data = load_json(review)
        data["items"][0]["verdict"] = "fail"
        data["items"][0]["notes"] = "Investigate missing attribution."
        review.write_text(json.dumps(data), encoding="utf-8")
        output = self.root / "review.html"
        create_html_report(store, review, output)
        payload = json.loads(self.parse_report(output).scripts["report-data"])
        self.assertEqual(payload["review"], data)
        self.assertEqual(store.status()["approvals"], [])

    def test_cli_and_output_safeguards(self):
        store, review = self.prepare_report()
        output = self.root / "review.html"
        with patch("builtins.print"):
            code = main(["review-report", "--run", str(self.run), "--review", str(review), "--output", str(output)])
        self.assertEqual(code, 0)
        with self.assertRaisesRegex(ValueError, "already exists"):
            create_html_report(store, review, output)
        with self.assertRaisesRegex(ValueError, "extension"):
            create_html_report(store, review, self.root / "new.json")

    def test_template_is_available_as_a_package_resource(self):
        template = files("summary_dataset_builder").joinpath("review_report.html").read_text(encoding="utf-8")
        self.assertIn("__REPORT_DATA__", template)
        self.assertIn("__SCRIPT_HASH__", template)

    def prepare_browse_run(self):
        self.initialize(tokenizer_path=None, model_neutral=True)
        store = self.store()
        generate(store, approve_paid=True, max_requests=20, client=FakeClient())
        records = list(store.records(status="accepted"))
        with store.db:
            for row, status in zip(records, ("flagged", "rejected", "error")):
                result = row["result"] if status == "flagged" else None
                if result:
                    result["flags"] = ["unsupported_number"]
                    result["summary"] = "A flagged greenhouse summary."
                store.db.execute(
                    "UPDATE documents SET status=?, result=?, reason=? WHERE id=?",
                    (status, json.dumps(result) if result else None, f"Fixture {status}", row["id"]),
                )
        return store

    def test_browse_requires_no_saved_sample_and_keeps_every_completed_disposition(self):
        store = self.prepare_browse_run()
        before = tuple(store.db.iterdump())
        status = store.status()
        output = self.root / "samples.html"
        with patch("summary_dataset_builder.review_html.load_review") as load_review:
            result = create_html_report(store, None, output, browse_only=True)
        load_review.assert_not_called()
        parsed = self.parse_report(output)
        payload = json.loads(parsed.scripts["report-data"])
        self.assertIsNone(payload["review"])
        self.assertIsNone(payload["download_name"])
        self.assertNotIn("approvals", payload["status"])
        self.assertEqual(payload["status"]["accounting"], status["accounting"])
        self.assertEqual({row["status"] for row in payload["records"]},
                         {"accepted", "flagged", "rejected", "error"})
        self.assertEqual({row["document_id"] for row in payload["records"]},
                         {row["id"] for row in store.records() if row["status"] != "pending"})
        self.assertEqual(payload["flags"], {"unsupported_number": 1})
        self.assertEqual(result["mode"], "browse")
        self.assertTrue(result["browse_only"])
        self.assertEqual(result["generated_results"], 4)
        self.assertEqual(result["sample_count"], 4)
        self.assertEqual(result["review_documents"], 0)
        self.assertIsNone(result["review_id"])
        self.assertFalse(result["approval_changed"])
        self.assertFalse(result["review_sample_created"])
        self.assertEqual(store.status(), status)
        self.assertEqual(tuple(store.db.iterdump()), before)
        self.assertEqual(store.db.execute("SELECT COUNT(*) FROM reviews").fetchone()[0], 0)
        self.assertFalse((self.root / "review.json").exists())

    def test_browse_never_loads_or_embeds_existing_decisions(self):
        store, review = self.prepare_report()
        decision = load_json(review)
        decision["items"][0]["verdict"] = "fail"
        decision["items"][0]["notes"] = "PRIVATE_REVIEW_NOTE_DO_NOT_EMBED"
        review.write_text(json.dumps(decision), encoding="utf-8")
        before_file = review.read_bytes()
        before_db = tuple(store.db.iterdump())
        for path in (review, self.root / "absent-review.json"):
            with self.subTest(path=path):
                output = self.root / f"{path.stem}.html"
                with patch("summary_dataset_builder.review_html.load_review") as load_review:
                    create_html_report(store, path, output, browse_only=True)
                load_review.assert_not_called()
                html = output.read_text(encoding="utf-8")
                payload = json.loads(self.parse_report(output).scripts["report-data"])
                self.assertIsNone(payload["review"])
                self.assertNotIn(decision["items"][0]["notes"], html)
                self.assertNotIn(decision["review_id"], html)
        self.assertEqual(review.read_bytes(), before_file)
        self.assertEqual(tuple(store.db.iterdump()), before_db)

    def test_browse_snapshots_uncertain_billing_without_reconciling_it(self):
        self.initialize(tokenizer_path=None, model_neutral=True)
        store = self.store()
        generate(
            store, approve_paid=True, max_requests=2,
            client=FakeClient([completion(), completion(cost=None)]),
        )
        status = store.status()
        before = tuple(store.db.iterdump())
        self.assertEqual(status["accounting"]["uncertain_attempts"], 1)
        output = self.root / "uncertain.html"
        create_html_report(store, None, output, browse_only=True)
        payload = json.loads(self.parse_report(output).scripts["report-data"])
        self.assertEqual(payload["status"]["accounting"], status["accounting"])
        self.assertEqual(payload["status"]["accounting"]["known_cost_usd"], "0.0001")
        self.assertEqual(tuple(store.db.iterdump()), before)

    def test_assessment_attempts_do_not_replace_generation_completion_or_provenance(self):
        store = self.prepare_browse_run()
        record = next(store.records(status="accepted"))
        generation_response = json.loads(store.db.execute(
            "SELECT response FROM attempts WHERE document_id=? AND kind='generation' "
            "ORDER BY id DESC LIMIT 1", (record["id"],),
        ).fetchone()["response"])
        assessment_response = {
            "content": "ASSESSMENT_RESPONSE_NOT_A_TEACHER_SUMMARY",
            "model": "assessment-model", "provider": "assessment-provider",
            "request_id": "assessment-request", "cost_usd": "0.0002",
        }
        with store.db:
            store.db.execute(
                "INSERT INTO attempts (document_id,kind,status,reserved_usd,cost_usd,response) "
                "VALUES (?,'assessment','completed','0.0002','0.0002',?)",
                (record["id"], json.dumps(assessment_response)),
            )
        review = self.root / "post-assessment-review.json"
        review_sample(store, review, per_stratum=5)
        before = tuple(store.db.iterdump())
        for browse_only in (True, False):
            with self.subTest(browse_only=browse_only):
                output = self.root / f"assessed-{browse_only}.html"
                create_html_report(store, review, output, browse_only=browse_only)
                parsed = self.parse_report(output)
                payload = json.loads(parsed.scripts["report-data"])
                displayed = next(row for row in payload["records"] if row["document_id"] == record["id"])
                self.assertEqual(displayed["response"], generation_response)
                self.assertEqual(displayed["result"], record["result"])
                self.assertEqual(displayed["status"], "accepted")
                self.assertEqual(payload["status"]["accounting"]["attempts"], 5)
                self.assertEqual(payload["status"]["accounting"]["known_cost_usd"], "0.0006")
                self.assertNotIn(assessment_response["content"], parsed.scripts["report-data"])
        self.assertEqual(tuple(store.db.iterdump()), before)

    def test_review_mode_still_requires_a_review_path(self):
        self.initialize(tokenizer_path=None, model_neutral=True)
        output = self.root / "review.html"
        with self.assertRaisesRegex(ValueError, "review path"):
            create_html_report(self.store(), None, output)
        self.assertFalse(output.exists())

    def test_quality_hold_is_visible_without_changing_original_disposition(self):
        store = self.prepare_browse_run()
        record = next(store.records(status="accepted"))
        hold = {
            "reason": 'Assessment concern </script><img src="https://invalid.example/tracker">',
            "details": {"rationale": "<script>QUALITY_HOLD_INJECTION</script>",
                        "independent_verification": False},
        }
        with store.db:
            store.db.execute(
                "INSERT INTO quality_holds (document_id,reason,details) VALUES (?,?,?)",
                (record["id"], hold["reason"], json.dumps(hold["details"])),
            )
        review = self.root / "held-review.json"
        review_sample(store, review, per_stratum=5)
        before = tuple(store.db.iterdump())
        status = store.status()
        self.assertEqual(sum(status["usable_counts"].values()), 0)
        self.assertEqual(sum(status["quarantined_counts"].values()), 1)
        for browse_only in (True, False):
            with self.subTest(browse_only=browse_only):
                output = self.root / f"held-{browse_only}.html"
                create_html_report(store, review, output, browse_only=browse_only)
                parsed = self.parse_report(output)
                self.assertEqual(set(parsed.scripts), {"report-data", "report-code"})
                self.assertEqual(parsed.external_assets, [])
                self.assertNotIn("<", parsed.scripts["report-data"])
                self.assertNotIn("innerHTML", parsed.scripts["report-code"])
                self.assertIn("Quarantined — excluded from training", output.read_text(encoding="utf-8"))
                payload = json.loads(parsed.scripts["report-data"])
                displayed = next(row for row in payload["records"] if row["document_id"] == record["id"])
                self.assertEqual(displayed["quality_hold"], hold)
                self.assertEqual(displayed["status"], "accepted")
                self.assertEqual(displayed["source"], record["record"])
                self.assertEqual(displayed["result"], record["result"])
                self.assertEqual(payload["status"]["usable_counts"], status["usable_counts"])
                self.assertEqual(payload["status"]["quarantined_counts"], status["quarantined_counts"])
                self.assertTrue(all(row["quality_hold"] is None for row in payload["records"]
                                    if row["document_id"] != record["id"]))
                if not browse_only:
                    self.assertEqual(payload["review"], load_json(review))
        self.assertEqual(tuple(store.db.iterdump()), before)

    def test_optional_assessment_notes_are_browse_only_and_escaped_as_plain_text(self):
        store = self.prepare_browse_run()
        record = next(store.records(status="accepted"))
        assessment = {
            "verdict": "needs_attention",
            "rationale": '</script><script>ASSESSMENT_INJECTION</script><img src="https://invalid.example">',
        }
        source_attempt = store.db.execute(
            "SELECT id FROM attempts WHERE document_id=? AND kind='generation' ORDER BY id DESC LIMIT 1",
            (record["id"],),
        ).fetchone()["id"]
        with store.db:
            store.db.execute(
                "INSERT INTO assessments (document_id,source_attempt_id,status,result) "
                "VALUES (?,?,'completed',?)",
                (record["id"], source_attempt, json.dumps(assessment)),
            )
        review = self.root / "assessment-review.json"
        review_sample(store, review, per_stratum=5)
        before = tuple(store.db.iterdump())
        for browse_only in (True, False):
            with self.subTest(browse_only=browse_only):
                output = self.root / f"assessment-notes-{browse_only}.html"
                create_html_report(store, review, output, browse_only=browse_only)
                parsed = self.parse_report(output)
                self.assertEqual(set(parsed.scripts), {"report-data", "report-code"})
                self.assertEqual(parsed.external_assets, [])
                self.assertNotIn("<", parsed.scripts["report-data"])
                payload = json.loads(parsed.scripts["report-data"])
                displayed = next(row for row in payload["records"] if row["document_id"] == record["id"])
                if browse_only:
                    self.assertEqual(displayed["assessment"], {"status": "completed", "result": assessment})
                else:
                    self.assertNotIn("assessment", displayed)
                    self.assertNotIn("ASSESSMENT_INJECTION", parsed.scripts["report-data"])
                self.assertEqual(displayed["result"], record["result"])
        self.assertEqual(tuple(store.db.iterdump()), before)

    def test_browse_hides_review_controls_and_retains_safe_csp(self):
        store = self.prepare_browse_run()
        output = self.root / "browse.html"
        create_html_report(store, None, output, browse_only=True)
        parsed = self.parse_report(output)
        self.assertEqual(parsed.body_class, "browse-only")
        self.assertTrue({
            "scope-filter", "decision-filter", "verdict", "notes", "review-readonly",
            "download-review", "import-review", "progress", "save-state", "snapshot-info",
        }.issubset(parsed.review_only_ids))
        self.assertFalse({"source-text", "summary-text", "search", "previous", "next",
                          "status-filter", "split-filter"} & parsed.review_only_ids)
        html = output.read_text(encoding="utf-8")
        self.assertIn(".browse-only .review-only{display:none!important}", html)
        self.assertNotIn("__BODY_CLASS__", html)
        script = parsed.scripts["report-code"]
        digest = base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
        self.assertIn(f"script-src 'sha256-{digest}'", parsed.csp)
        self.assertIn("connect-src 'none'", parsed.csp)
        self.assertEqual(parsed.external_assets, [])
        self.assertNotIn("innerHTML", script)
        self.assertNotIn("fetch(", script)

    @unittest.skipUnless(shutil.which("node"), "Node.js is needed for the offline script smoke test")
    def test_browser_navigation_search_and_read_only_storage_contract(self):
        store = self.prepare_browse_run()
        record = next(store.records(status="accepted"))
        with store.db:
            store.db.execute(
                "INSERT INTO quality_holds (document_id,reason,details) VALUES (?,?,?)",
                (record["id"], "Assessment concern (fixture)", json.dumps({"fixture": "held"})),
            )
            store.db.execute(
                "INSERT INTO assessments (document_id,source_attempt_id,status,result) "
                "SELECT ?,id,'completed',? FROM attempts WHERE document_id=? AND kind='generation' "
                "ORDER BY id DESC LIMIT 1",
                (record["id"], json.dumps({"verdict": "needs_attention", "rationale": "Fixture notes"}),
                 record["id"]),
            )
        review = self.root / "mixed-review.json"
        review_sample(store, review, per_stratum=5)
        for browse_only in (True, False):
            with self.subTest(browse_only=browse_only):
                output = self.root / f"browser-{browse_only}.html"
                create_html_report(store, review, output, browse_only=browse_only)
                parsed = self.parse_report(output)
                process = subprocess.run(
                    ["node", "-e", BROWSER_SMOKE_TEST],
                    input=json.dumps({
                        "script": parsed.scripts["report-code"],
                        "payload": json.loads(parsed.scripts["report-data"]),
                        "element_ids": parsed.element_ids,
                    }),
                    capture_output=True, text=True, encoding="utf-8", timeout=20,
                )
                self.assertEqual(process.returncode, 0, process.stdout + process.stderr)


BROWSER_SMOKE_TEST = r"""
const assert = require("node:assert/strict");
const vm = require("node:vm");
let input = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", chunk => { input += chunk; });
process.stdin.on("end", () => {
  const {script, payload, element_ids} = JSON.parse(input);
  class Element {
    constructor() {
      this.value = "";
      this.textContent = "";
      this.listeners = {};
      this.children = [];
      this.options = [{}, {}];
      this.classList = {toggle() {}};
    }
    append(...children) { this.children.push(...children); }
    replaceChildren() { this.children = []; }
    setAttribute(name, value) { this[name] = value; }
    removeAttribute(name) { delete this[name]; }
    addEventListener(type, callback) { this.listeners[type] = callback; }
    dispatch(type) { this.listeners[type]?.({target: this}); }
    querySelector() { return {scrollIntoView() {}}; }
  }
  const elements = Object.fromEntries(element_ids.map(id => [id, new Element()]));
  elements["report-data"].textContent = JSON.stringify(payload);
  elements["scope-filter"].value = "sample";
  const window = new Element();
  const saved = new Map();
  let storageAccesses = 0;
  const context = {
    document: {getElementById: id => elements[id], createElement: () => new Element()},
    window, URL, structuredClone,
  };
  Object.defineProperty(context, "localStorage", {get() {
    storageAccesses++;
    if (payload.browse_only) throw new Error("Read-only browsing must not access storage");
    return {getItem: key => saved.get(key), setItem: (key, value) => saved.set(key, value)};
  }});
  vm.runInNewContext(script, context);
  const e = id => elements[id];
  const filter = (id, value) => { e(id).value = value; e(id).dispatch(id === "search" ? "input" : "change"); };
  if (payload.browse_only) {
    assert.equal(storageAccesses, 0);
    assert.equal(e("status-filter").value, "accepted");
    assert.equal(e("record-list").children.length, payload.records.filter(r => r.status === "accepted").length);
    assert.equal(e("subtitle").textContent, "Generated samples; browsing only");
    for (const id of ["verdict", "notes", "download-review", "import-review"]) {
      assert.deepEqual(Object.keys(e(id).listeners), []);
    }
    assert.deepEqual(Object.keys(window.listeners), []);
    assert.equal(e("reason").textContent, "");
    assert.equal(e("review-readonly").hidden, true);
    const assessed = payload.records.find(r => r.assessment);
    assert.equal(e("assessment-panel").hidden, false);
    assert.equal(e("assessment-panel").open, false);
    assert.deepEqual(JSON.parse(e("assessment-notes").textContent), assessed.assessment);
    assert.equal(e("raw-response").textContent, assessed.response.content);
    assert.equal(e("record-badge").textContent, "accepted · quarantined");
    assert.equal(e("record-badge").className, "badge flagged");
    assert.equal(e("quality-hold").hidden, false);
    assert.equal(e("quality-hold-reason").textContent, assessed.quality_hold.reason);
    assert.deepEqual(JSON.parse(e("provenance").textContent).quality_hold, assessed.quality_hold);
    assert.equal(e("usable-caption").hidden, false);
    assert.match(e("usable-caption").textContent, /0 usable \| 1 quarantined.*excluded from training/);
    assert.doesNotMatch(e("visible-count").textContent, /saved sample|review/i);
    for (const button of e("record-list").children) {
      assert.doesNotMatch(button.children[1].textContent, /not sampled|review|notes/i);
    }
    filter("status-filter", "");
    assert.equal(e("record-list").children.length, payload.records.length);
    e("record-list").children[0].dispatch("click");
    assert.equal(e("source-text").textContent, payload.records[0].source.text);
    assert.equal(e("previous").disabled, true);
    e("next").dispatch("click");
    assert.equal(e("source-text").textContent, payload.records[1].source.text);
    e("previous").dispatch("click");
    assert.equal(e("source-text").textContent, payload.records[0].source.text);
    const flagged = payload.records.find(r => r.status === "flagged");
    filter("search", "flagged greenhouse");
    assert.equal(e("record-list").children.length, 1);
    assert.equal(e("summary-text").textContent, flagged.result.summary);
    assert.equal(e("assessment-panel").hidden, true);
    assert.equal(e("assessment-notes").textContent, "");
    assert.equal(e("quality-hold").hidden, true);
    assert.equal(e("quality-hold-reason").textContent, "");
    filter("search", "unsupported_number");
    assert.equal(e("record-list").children.length, 1);
    filter("search", flagged.source.text);
    assert.equal(e("record-list").children.length, 1);
    filter("search", "");
    filter("split-filter", "train");
    assert.equal(e("record-list").children.length, payload.records.filter(r => r.source.split === "train").length);
    filter("search", "no matching source or summary");
    assert.equal(e("empty").hidden, false);
    assert.equal(e("empty").textContent, "No examples match these filters.");
    assert.equal(storageAccesses, 0);
  } else {
    assert.equal(storageAccesses, 1);
    assert.equal(e("scope-filter").value, "sample");
    assert.equal(e("status-filter").value, "");
    assert.equal(e("record-list").children.length, payload.review.items.length);
    assert.equal(e("review-editor").hidden, false);
    assert.equal(e("assessment-panel").hidden, true);
    e("verdict").value = "fail";
    e("notes").value = "Offline fixture: investigate source grounding.";
    e("verdict").dispatch("change");
    assert.equal(storageAccesses, 2);
    assert.equal(saved.size, 1);
    const decision = JSON.parse([...saved.values()][0]);
    assert.equal(decision.decisions.filter(item => item.verdict === "fail").length, 1);
    assert.equal(typeof e("download-review").listeners.click, "function");
    assert.equal(typeof e("import-review").listeners.change, "function");
  }
});
"""
