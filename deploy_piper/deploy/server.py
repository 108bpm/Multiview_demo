"""Policy inference server for Piper deployment.

stdlib + numpy only, so it runs inside ANY conda env — launch it from the env
the policy needs and point the client at it.

Usage:
    python -m deploy.server --config=home

or the low-level form:
    python -m deploy.server --adapter=lerobot --port=8080 \
        --checkpoint=outputs/train/.../pretrained_model --device=cuda --fps=30

Flags other than --adapter/--host/--port are forwarded to the adapter
constructor as string keyword arguments.

Endpoints:
    GET  /info     -> adapter.info() as JSON
    POST /predict  -> body: protocol.encode_observation(...); reply: encode_chunk(...)
    POST /reset    -> clears adapter episode state
Any adapter exception -> HTTP 500 with the traceback as text; the server stays up.
"""
from __future__ import annotations

import concurrent.futures
import json
import sys
import traceback
from argparse import ArgumentParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deploy import protocol
from deploy.adapters import make_adapter
from deploy.config import bootstrap


def make_handler(adapter):
    # torch.compile(mode="max-autotune") cudagraph-tree state is thread-local;
    # a graph captured on one thread crashes if replayed/recaptured from
    # another. Route every adapter call through one persistent worker thread
    # so all captures+replays stay pinned to it (also serializes GPU work).
    worker = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, code: int, body: bytes, content_type="application/octet-stream"):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            try:
                if self.path == "/info":
                    info = {"protocol_version": protocol.PROTOCOL_VERSION, **adapter.info()}
                    self._send(200, json.dumps(info).encode(), "application/json")
                else:
                    self._send(404, b"not found", "text/plain")
            except Exception:
                self._send(500, traceback.format_exc().encode(), "text/plain")

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                if self.path == "/predict":
                    images, state, task, meta = protocol.decode_observation(body)
                    chunk = worker.submit(
                        adapter.predict_chunk, images, state, task,
                        consumed=meta["consumed"], delay_ticks=meta["delay_ticks"],
                    ).result()
                    self._send(200, protocol.encode_chunk(chunk))
                elif self.path == "/reset":
                    worker.submit(adapter.reset).result()
                    self._send(200, b"", "text/plain")
                else:
                    self._send(404, b"not found", "text/plain")
            except Exception:
                self._send(500, traceback.format_exc().encode(), "text/plain")

        def log_message(self, fmt, *args):
            pass  # keep per-request noise out of the console

    return Handler


def create_server(adapter, host: str = "127.0.0.1", port: int = 8080) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(adapter))


def parse_args(argv=None):
    argv = bootstrap("server", argv)
    parser = ArgumentParser(description="Piper deploy policy server")
    parser.add_argument("--adapter", required=True, help="adapter name, e.g. lerobot")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args, extra = parser.parse_known_args(argv)
    kwargs = {}
    for item in extra:
        if not (item.startswith("--") and "=" in item):
            parser.error(f"adapter flags must look like --key=value, got: {item}")
        key, value = item[2:].split("=", 1)
        kwargs[key] = value
    return args, kwargs


def main(argv=None):
    args, kwargs = parse_args(argv)
    adapter = make_adapter(args.adapter, **kwargs)
    print(f"[deploy.server] adapter={args.adapter} info={adapter.info()}")
    server = create_server(adapter, args.host, args.port)
    print(f"[deploy.server] listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[deploy.server] shutting down")


if __name__ == "__main__":
    main()
