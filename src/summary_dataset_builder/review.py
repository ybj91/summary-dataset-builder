import hashlib
import json
from pathlib import Path

from .files import atomic_json, load_json, sha256_file
from .store import RunStore


def review_sample(store: RunStore, output_path: Path, *, per_stratum: int = 5, seed: int = 42) -> dict:
    if type(per_stratum) is not int or not 1 <= per_stratum <= 100:
        raise ValueError("per_stratum must be 1..100")
    if output_path.exists():
        raise ValueError("Review output already exists")
    if store.accounting()["uncertain_attempts"]:
        raise ValueError("Reconcile uncertain billing before sampling a review")
    selected = {}
    for row in store.records():
        if row["status"] in ("pending", "inflight", "uncertain"):
            continue
        source = row["record"]
        words = len(source["text"].split())
        length_band = "short" if words < 300 else "medium" if words < 600 else "long"
        stratum = f"{row['split']}:{row['status']}:{length_band}"
        rank = hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest()
        bucket = selected.setdefault(stratum, [])
        bucket.append((rank, row))
        bucket.sort(key=lambda item: item[0])
        del bucket[per_stratum:]
    if not selected:
        raise ValueError("No completed documents to review")
    items = []
    for stratum, bucket in sorted(selected.items()):
        for _, row in bucket:
            item = {
                "document_id": row["id"], "stratum": stratum, "source": row["record"],
                "status": row["status"], "result": row["result"], "reason": row["reason"],
                "verdict": None, "notes": "",
            }
            hold = store.quality_hold(row["id"])
            if hold is not None:
                item["quality_hold"] = hold
            items.append(item)
    attempt_count = store.accounting()["attempts"]
    identity = hashlib.sha256(json.dumps(
        {"attempts": attempt_count, "items": items}, sort_keys=True
    ).encode()).hexdigest()
    sample = {
        "review_id": identity, "attempt_count": attempt_count, "seed": seed,
        "per_stratum": per_stratum,
        "instructions": (
            "Review source support, central information, numbers, attribution, uncertainty, "
            "privacy, and suitability. Set verdict to pass only if the stored disposition is "
            "correct (including correct rejection/flagging); otherwise fail. Add substantive notes. "
            "A quality_hold excludes a pair from training even if its original status was accepted; "
            "review that quarantine action as well. "
            "Do not change source, results, IDs, or omit hard examples."
        ),
        "items": items,
    }
    with store.db:
        store.db.execute(
            "INSERT OR REPLACE INTO reviews (id,attempt_count,sample) VALUES (?,?,?)",
            (identity, attempt_count, json.dumps(sample)),
        )
    atomic_json(output_path, sample)
    return {"review_id": identity, "documents": len(items), "strata": len(selected), "path": str(output_path)}


def load_review(store: RunStore, review_path: Path) -> dict:
    review = load_json(review_path)
    if not isinstance(review.get("review_id"), str):
        raise ValueError("Review ID must identify a saved sample")
    row = store.db.execute(
        "SELECT sample, attempt_count FROM reviews WHERE id=?", (review.get("review_id"),)
    ).fetchone()
    if row is None:
        raise ValueError("Review sample does not belong to this run")
    original = json.loads(row["sample"])
    if row["attempt_count"] != store.accounting()["attempts"]:
        raise ValueError("Review sample is stale; sample again after the latest generation")
    if set(review) != set(original) or any(
        review[key] != original[key] for key in original if key != "items"
    ):
        raise ValueError("Review metadata was changed")
    if not isinstance(review.get("items"), list) or len(review["items"]) != len(original["items"]):
        raise ValueError("Review items were added or removed")
    for edited, expected in zip(review["items"], original["items"]):
        if not isinstance(edited, dict) or set(edited) != set(expected) or any(
            edited[key] != expected[key] for key in expected if key not in ("verdict", "notes")
        ):
            raise ValueError("Only verdict and notes may be edited in a review")
        if edited["verdict"] not in (None, "pass", "fail") or not isinstance(edited["notes"], str):
            raise ValueError("Review verdict must be null/pass/fail and notes must be text")
    return review


def approve_review(store: RunStore, review_path: Path, *, stage: str, acknowledge: bool) -> dict:
    if not acknowledge:
        raise ValueError("Approval requires --acknowledge-review after a human has reviewed the sample")
    if stage not in ("scale", "full", "export"):
        raise ValueError("Unknown review stage")
    accounting = store.accounting()
    minimum = {"scale": 100, "full": 1000, "export": 1}[stage]
    if accounting["generation_attempts"] < minimum:
        raise ValueError(f"Stage {stage} requires at least {minimum} generation attempts")
    if accounting["uncertain_attempts"]:
        raise ValueError("Reconcile all uncertain billing before approval")
    if accounting["budget_exceeded"]:
        raise ValueError("Budget overrun must be investigated; this run cannot be approved")
    review = load_review(store, review_path)
    for edited in review["items"]:
        if edited["verdict"] != "pass" or not edited["notes"].strip():
            raise ValueError("All reviewed dispositions must pass with notes; investigate failures before scaling/export")
    if not any(item["status"] == "accepted" for item in review["items"]):
        raise ValueError("Approval needs at least one accepted example")
    snapshot = store.run_dir / f"approved-{stage}.json"
    atomic_json(snapshot, review)
    with store.db:
        store.db.execute(
            "INSERT OR REPLACE INTO approvals (stage,attempt_count,review_sha256) VALUES (?,?,?)",
            (stage, accounting["attempts"], sha256_file(snapshot)),
        )
    return {"approved_stage": stage, "reviewed_documents": len(review["items"])}
