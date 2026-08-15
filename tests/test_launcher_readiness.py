"""A server that is listening is not necessarily a server that can answer.

llama-server binds its HTTP port before the weights are loaded and answers /props with
503 Service Unavailable until they are. The readiness check treated any HTTP response as
ready - "answering at all means the server is up" - so `llmbench launch` returned while
the model was still loading, and detection then ran against a server that could not yet
describe itself.

Observed 2026-08-05 on build b10144-d73c1d6b2 with an 11.8 GB model: `llmbench run
--server heretic-12b` failed at detection every time.
"""
from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from llmbench.launcher import _is_ready


class _Server:
    """A real HTTP server that answers with the status codes given, in order.

    Real rather than mocked because the thing under test speaks HTTP through urllib,
    and a fake would only prove that the fake agrees with itself.
    """

    def __init__(self, statuses: list[int]):
        self.statuses = list(statuses)
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                code = outer.statuses.pop(0) if outer.statuses else 200
                body = b'{"ok":true}' if code == 200 else b'{"error":"loading"}'
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self._httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


def test_a_server_still_loading_its_model_is_not_ready():
    """The regression. 503 is llama-server's answer while the weights are loading."""
    with _Server([503]) as s:
        assert _is_ready(s.port) is False


def test_it_becomes_ready_once_the_model_is_loaded():
    """The success condition: the check must not simply always say no."""
    with _Server([503, 503, 200]) as s:
        assert _is_ready(s.port) is False
        assert _is_ready(s.port) is False
        assert _is_ready(s.port) is True


def test_a_server_that_merely_dislikes_the_request_is_ready():
    """404 means the endpoint is absent and the server is up - which is ready.

    Only 503 means 'up but cannot serve yet', so only 503 keeps the launcher waiting.
    """
    with _Server([404]) as s:
        assert _is_ready(s.port) is True


def test_nothing_listening_is_not_ready():
    assert _is_ready(1) is False
