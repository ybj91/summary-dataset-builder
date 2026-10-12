import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.resources import files
import json
from pathlib import Path

from .files import atomic_text
from .review import load_review
from .store import RunStore


def create_html_report(
    store: RunStore, review_path: Path | None, output_path: Path, *, browse_only: bool = False,
) -> dict:
    """Write an offline snapshot, optionally without loading or collecting reviews."""
    if output_path.exists():
        raise ValueError("HTML report already exists; choose a new output path")
    if output_path.suffix.lower() not in (".html", ".htm"):
        raise ValueError("Report output must have an .html or .htm extension")
    if not browse_only and review_path is None:
        raise ValueError("A review path is required unless browse_only=True")
    with store.db:
        store.db.execute("BEGIN")
        review = None if browse_only else load_review(store, review_path)
        status = store.status()
        if not browse_only and status["accounting"]["uncertain_attempts"]:
            raise ValueError("Reconcile in-flight or uncertain billing before creating a review report")
        if browse_only:
            status.pop("approvals", None)
        assessments = {}
        if browse_only:
            assessments = {
                row["document_id"]: {"status": row["status"], "result": json.loads(row["result"])}
                for row in store.db.execute(
                    "SELECT document_id,status,result FROM assessments WHERE result IS NOT NULL"
                )
            }
        records = []
        flags = Counter()
        for row in store.records():
            if row["status"] == "pending":
                continue
            attempt = store.db.execute(
                "SELECT response FROM attempts WHERE document_id=? AND kind='generation' "
                "ORDER BY id DESC LIMIT 1",
                (row["id"],),
            ).fetchone()
            response = json.loads(attempt["response"]) if attempt and attempt["response"] else None
            records.append({
                "document_id": row["id"], "source": row["record"], "status": row["status"],
                "result": row["result"], "reason": row["reason"], "response": response,
                "quality_hold": store.quality_hold(row["id"]),
            })
            if row["id"] in assessments:
                records[-1]["assessment"] = assessments[row["id"]]
            if row["result"]:
                flags.update(row["result"].get("flags", []))
    payload = {
        "review": review, "records": records, "status": status, "flags": dict(flags),
        "browse_only": browse_only, "mode": "browse" if browse_only else "review",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_name": store.run_dir.name, "run_path": str(store.run_dir),
        "download_name": None if browse_only else review_path.stem + ".completed.json",
        "prompt_version": store.config["prompt_version"],
    }
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    # The payload is inert JSON, but HTML parsers still recognize closing script tags.
    for character, escape in (
        ("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e"),
        ("\u2028", "\\u2028"), ("\u2029", "\\u2029"),
    ):
        data = data.replace(character, escape)
    template = files("summary_dataset_builder").joinpath("review_report.html").read_text(encoding="utf-8")
    script = template.split('<script id="report-code">', 1)[1].split("</script>", 1)[0]
    digest = base64.b64encode(hashlib.sha256(script.encode("utf-8")).digest()).decode("ascii")
    template = template.replace("__SCRIPT_HASH__", digest)
    template = template.replace("__BODY_CLASS__", "browse-only" if browse_only else "")
    template = template.replace("__WORKSPACE_LABEL__", "Sample browser" if browse_only else "Review workspace")
    template = template.replace("__REPORT_TITLE__", "Sample browser" if browse_only else "Human review")
    template = template.replace(
        "__REPORT_HEADING__", "Generated summaries" if browse_only else "Source-grounded summarization pilot"
    )
    document = template.replace("__REPORT_DATA__", data)
    atomic_text(output_path, document)
    return {
        "path": str(output_path), "generated_results": len(records),
        "review_documents": len(review["items"]) if review is not None else 0,
        "review_id": review["review_id"] if review is not None else None,
        "mode": "browse" if browse_only else "review", "browse_only": browse_only,
        "sample_count": len(records), "review_sample_created": False, "approval_changed": False,
    }
