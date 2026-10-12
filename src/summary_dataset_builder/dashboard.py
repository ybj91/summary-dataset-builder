import base64
from datetime import datetime, timedelta, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import json
import logging
import math
import os
from pathlib import Path
import sqlite3
from urllib.parse import urlsplit

from .files import atomic_json, load_json
from .store import RunStore


LOGGER = logging.getLogger(__name__)


def estimate_remaining_time(
    store: RunStore, status: dict, worker: dict | None, *, now: datetime,
) -> dict:
    remaining_by_split = {
        split: max(0, target - status["usable_counts"][split])
        for split, target in status["targets"].items()
    }
    remaining = sum(remaining_by_split.values())
    result = {
        "state": "unavailable", "remaining_seconds": None, "estimated_finish_at": None,
        "usable_pairs_per_minute": None, "observed_usable_pairs": 0,
        "window_seconds": None, "remaining_usable_pairs": remaining,
        "note": "Waiting for an active collector and enough recent usable pairs.",
    }
    if status["failed_gate"] or (worker and worker["state"] in ("stopped", "failed")):
        result.update(state="paused", note="Collection is paused; no finish time is projected.")
        return result
    if remaining == 0:
        result.update(
            state="complete", remaining_seconds=0,
            note="All usable split targets reached; final checks and release are separate.",
        )
        return result
    if not worker or worker["state"] != "running" or not worker.get("started_at"):
        return result
    try:
        resumed = datetime.fromisoformat(worker["started_at"])
        if resumed.tzinfo is None:
            raise ValueError
        resumed = resumed.astimezone(timezone.utc)
    except (ValueError, TypeError):
        raise ValueError("Collector start time is invalid; cannot estimate remaining time") from None
    start = max(resumed, now - timedelta(minutes=15))
    seconds = (now - start).total_seconds()
    if seconds <= 0:
        result.update(state="warming_up", note="Waiting for observations after the collector resumed.")
        return result
    timestamp = lambda value: value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    needed = [split for split, count in remaining_by_split.items() if count]
    placeholders = ",".join("?" for _ in needed)
    usable = store.db.execute(
        "SELECT COUNT(DISTINCT a.document_id) FROM attempts a JOIN documents d ON d.id=a.document_id "
        "WHERE a.kind='generation' AND a.intended_status='accepted' "
        "AND a.status IN ('completed','reconciled','billing_unconfirmed') "
        "AND d.status='accepted' AND a.finished_at>=? AND a.finished_at<=? "
        f"AND d.split IN ({placeholders}) "
        "AND NOT EXISTS (SELECT 1 FROM quality_holds h WHERE h.document_id=d.id)",
        (timestamp(start), timestamp(now), *needed),
    ).fetchone()[0]
    latest = store.db.execute(
        "SELECT MAX(finished_at) FROM attempts WHERE kind='generation' AND finished_at>=? AND finished_at<=?",
        (timestamp(resumed), timestamp(now)),
    ).fetchone()[0]
    result.update(observed_usable_pairs=usable, window_seconds=round(seconds, 1))
    if latest is None or (
        now - datetime.fromisoformat(latest).replace(tzinfo=timezone.utc)
    ).total_seconds() > 300:
        result.update(
            state="stalled" if seconds >= 300 else "warming_up",
            note="No recent completed generation request; waiting for fresh progress.",
        )
        return result
    if seconds < 60 or usable < 10:
        result.update(state="warming_up", note="Estimating after at least one minute and 10 recent usable pairs.")
        return result
    rate = usable / seconds
    duration = math.ceil(remaining / rate)
    result.update(
        state="estimating", remaining_seconds=duration,
        estimated_finish_at=(now + timedelta(seconds=duration)).isoformat(),
        usable_pairs_per_minute=round(rate * 60, 2),
        note="Approximate, based on recent usable pairs for unfinished splits. Retries, yield changes, gates or budget stops may delay completion.",
    )
    return result


def dashboard_status(run_dir: Path) -> dict:
    progress_path = run_dir / "collection-progress.json"
    progress = load_json(progress_path) if progress_path.is_file() else None
    worker = {
        key: progress.get(key)
        for key in ("state", "stop_reason", "updated_at", "started_at", "pid")
    } if progress is not None else None
    now = datetime.now(timezone.utc)
    with RunStore(run_dir, read_only=True) as store, store.db:
        store.db.execute("BEGIN")
        status = store.collection_status()
        estimate = estimate_remaining_time(store, status, worker, now=now)
    return {
        "run_name": run_dir.name, "status": status, "worker": worker, "estimate": estimate,
        "read_only": True, "refreshed_at": now.isoformat(),
        "refresh_interval_seconds": 5,
    }


def create_dashboard_server(run_dir: Path, *, port: int = 0) -> ThreadingHTTPServer:
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("Dashboard port must be between 0 and 65535")
    run_dir = run_dir.resolve()
    dashboard_status(run_dir)
    page = files("summary_dataset_builder").joinpath("progress_dashboard.html").read_text(encoding="utf-8")
    script = page.split('<script id="dashboard-code">', 1)[1].split("</script>", 1)[0]
    digest = base64.b64encode(hashlib.sha256(script.encode("utf-8")).digest()).decode("ascii")
    csp = (
        f"default-src 'none'; script-src 'sha256-{digest}'; style-src 'unsafe-inline'; "
        "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    )
    html = page.encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status_code: int, body: bytes, content_type: str) -> None:
            self.send_response(status_code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", csp)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, status_code: int, value: dict) -> None:
            self._reply(
                status_code, json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"),
                "application/json; charset=utf-8",
            )

        def do_GET(self):
            hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            if self.headers.get("Host") not in hosts or (
                self.headers.get("Origin") is not None
                and self.headers["Origin"] not in {f"http://{host}" for host in hosts}
            ):
                self._json(403, {"error": "Only same-origin localhost access is allowed."})
                return
            route = urlsplit(self.path).path
            if route == "/":
                self._reply(200, html, "text/html; charset=utf-8")
            elif route == "/api/status":
                try:
                    result = dashboard_status(run_dir)
                except (OSError, ValueError, sqlite3.Error) as error:
                    LOGGER.error("Dashboard cannot read run status (%s)", type(error).__name__)
                    self._json(503, {"error": "Could not read run status. Check the run files and retry."})
                    return
                self._json(200, result)
            elif route == "/favicon.ico":
                self._reply(204, b"", "image/x-icon")
            else:
                self._json(404, {"error": "Not found."})

        def do_HEAD(self):
            self.do_GET()

        def do_POST(self):
            self._json(405, {"error": "This dashboard is read-only."})

        do_PUT = do_POST
        do_PATCH = do_POST
        do_DELETE = do_POST

        def log_message(self, format, *args):
            # Polling has no access log; errors are reported without request contents.
            if args and str(args[1] if len(args) > 1 else "").startswith("5"):
                LOGGER.warning("Dashboard returned a server error")

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve_dashboard(run_dir: Path, *, port: int = 0, ready_file: Path | None = None) -> dict:
    with create_dashboard_server(run_dir, port=port) as server:
        connection = {
            "url": f"http://127.0.0.1:{server.server_port}/", "pid": os.getpid(),
            "run_name": run_dir.name, "read_only": True,
        }
        if ready_file is not None:
            atomic_json(ready_file, connection)
        print(json.dumps(connection), flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            return {"dashboard_stopped": True, "collection_stopped": False}
