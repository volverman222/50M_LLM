from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Mapping

from .contract import ContractError, parse_external_analysis
from .security import verify_bearer_token, verify_wandb_signature
from .store import AnalysisConflict, AnalysisInbox

DEFAULT_MAX_BODY = 1024 * 1024


class FeedbackService:
    def __init__(
        self,
        inbox: AnalysisInbox,
        *,
        webhook_secret: str,
        bearer_token: str | None = None,
        expected_entity: str | None = None,
        expected_project: str | None = None,
        max_body: int = DEFAULT_MAX_BODY,
    ):
        if not webhook_secret:
            raise ValueError("webhook_secret is required")
        self.inbox = inbox
        self.webhook_secret = webhook_secret
        self.bearer_token = bearer_token
        self.expected_entity = expected_entity
        self.expected_project = expected_project
        self.max_body = max_body

    def process(
        self, *, path: str, headers: Mapping[str, str], body: bytes
    ) -> tuple[int, dict[str, object]]:
        if path != "/v1/wandb/aria":
            return 404, {"error": "not_found"}
        normalized = {str(k).lower(): str(v) for k, v in headers.items()}
        if len(body) > self.max_body:
            return 413, {"error": "payload_too_large"}
        content_type = normalized.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            return 415, {"error": "application_json_required"}
        if not verify_bearer_token(normalized.get("authorization"), self.bearer_token):
            return 401, {"error": "invalid_authorization"}
        if not verify_wandb_signature(
            body, normalized.get("x-wandb-signature"), self.webhook_secret
        ):
            return 401, {"error": "invalid_signature"}
        try:
            payload = json.loads(body.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ContractError("payload must be an object")
            analysis = parse_external_analysis(payload)
            if self.expected_entity is not None and analysis.entity != self.expected_entity:
                return 403, {"error": "source_scope_mismatch"}
            if self.expected_project is not None and analysis.project != self.expected_project:
                return 403, {"error": "source_scope_mismatch"}
            receipt = self.inbox.ingest(analysis)
        except (UnicodeDecodeError, json.JSONDecodeError, ContractError) as exc:
            return 400, {"error": "invalid_payload", "detail": str(exc)}
        except AnalysisConflict as exc:
            return 409, {"error": "analysis_id_conflict", "detail": str(exc)}
        status = 200 if receipt.status == "duplicate" else 202
        return status, receipt.as_dict()


def make_handler(service: FeedbackService):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, status: int, payload: Mapping[str, object]) -> None:
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/healthz":
                self._json(200, {"status": "ok", "authority": "advisory_only"})
            else:
                self._json(404, {"error": "not_found"})

        def do_POST(self) -> None:  # noqa: N802
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                self._json(400, {"error": "invalid_content_length"})
                return
            if length < 0 or length > service.max_body:
                self._json(413, {"error": "payload_too_large"})
                return
            body = self.rfile.read(length)
            status, payload = service.process(path=self.path, headers=self.headers, body=body)
            self._json(status, payload)

        def log_message(self, fmt: str, *args: object) -> None:
            # Deliberately avoid logging request headers or signed bodies.
            super().log_message(fmt, *args)

    return Handler


def _loopback_host(host: str) -> bool:
    return host in {"127.0.0.1", "::1", "localhost"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Receive signed W&B/ARIA advisory analyses")
    parser.add_argument("--host", default=os.getenv("AUTORESEARCH_FEEDBACK_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("AUTORESEARCH_FEEDBACK_PORT", "8787")))
    parser.add_argument(
        "--inbox",
        type=Path,
        default=Path(os.getenv("AUTORESEARCH_ARIA_INBOX", ".autoresearch/aria-inbox")),
    )
    parser.add_argument(
        "--allow-non-loopback",
        action="store_true",
        help="explicitly allow binding beyond loopback; use only behind an authenticated relay/proxy",
    )
    args = parser.parse_args()
    if not _loopback_host(args.host) and not args.allow_non_loopback:
        parser.error("non-loopback bind requires --allow-non-loopback")
    secret = os.getenv("WANDB_WEBHOOK_SECRET")
    if not secret:
        parser.error("WANDB_WEBHOOK_SECRET is required")
    service = FeedbackService(
        AnalysisInbox(args.inbox),
        webhook_secret=secret,
        bearer_token=os.getenv("WANDB_WEBHOOK_BEARER"),
        expected_entity=os.getenv("AUTORESEARCH_WANDB_ENTITY"),
        expected_project=os.getenv("AUTORESEARCH_WANDB_PROJECT"),
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(service))
    print(f"autoresearch feedback receiver listening on {args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
