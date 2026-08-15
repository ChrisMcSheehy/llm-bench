"""End-to-end smoke test: every test module against a fake model server.

This wraps the long-standing `selftest.py` script so it runs under pytest like
everything else. The script drives a mock backend that answers correctly where it
can, which proves the graders accept correct answers rather than merely running.
"""
from __future__ import annotations

import selftest


def test_full_suite_against_mock_backend():
    # selftest.main() asserts internally and raises on any failure.
    selftest.main()
