"""Pipeline module - Speculative execution orchestration."""

# Lazy import coordinator to avoid whisper dependency at import time
from .metrics import LatencyMetrics, MetricsTracker

def get_pipeline_coordinator():
    """Get PipelineCoordinator class (requires whisper for ASR)."""
    from .coordinator import PipelineCoordinator
    return PipelineCoordinator

def get_pipeline_result():
    """Get PipelineResult class."""
    from .coordinator import PipelineResult
    return PipelineResult

__all__ = ["LatencyMetrics", "MetricsTracker", "get_pipeline_coordinator", "get_pipeline_result"]
