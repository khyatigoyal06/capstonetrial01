"""Test-wide setup: never write to the real analysis database."""

import os
import tempfile

os.environ.setdefault("ANALYSIS_DB_PATH", os.path.join(tempfile.mkdtemp(prefix="pes_tests_"), "analyses.sqlite3"))
os.environ.setdefault("METRICS_PORT", "0")
