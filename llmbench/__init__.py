"""llmbench — a local-LLM quality test bench.

Detect what's deployed (model, quant, KV-cache, MTP/spec, llama.cpp commit),
run pluggable evaluators against it (needle, coding, ...), store results in
SQLite, and view them in a dashboard.
"""
__version__ = "0.1.0"
