from prometheus_client import Counter, Histogram, start_http_server
import threading
import time

# Counters and histograms for basic observability
inference_requests = Counter(
    "capstone_inference_requests_total", "Total number of poster inference requests"
)

gaze_analysis_requests = Counter(
    "capstone_gaze_analysis_requests_total", "Total number of gaze analysis requests"
)

processing_seconds = Histogram(
    "capstone_processing_seconds", "Processing time for inference and gaze analysis"
)


_started = False


def start_metrics_server(port: int | None = None) -> bool:
    """Start the Prometheus endpoint once per process.

    Returns False (instead of crashing a background thread) when disabled with
    METRICS_PORT=0 or when the port is taken, e.g. by a second worker or by
    uvicorn's --reload process.
    """
    global _started
    if _started:
        return True
    if port is None:
        from config.settings import METRICS_PORT

        port = METRICS_PORT
    if not port:
        return False
    try:
        start_http_server(port)  # serves from its own daemon thread
    except OSError:
        return False
    _started = True
    return True


# Convenience context manager for timing
class timeit:
    def __init__(self, histogram: Histogram):
        self.histogram = histogram

    def __enter__(self):
        self._start = time.time()

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.time() - self._start
        try:
            self.histogram.observe(elapsed)
        except Exception:
            pass
