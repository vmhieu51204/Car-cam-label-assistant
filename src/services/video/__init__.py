"""Video services with lazy exports, allowing CPU-only contract/review tools."""
from importlib import import_module

_EXPORTS = {
    "FFmpegVideoExtractor": "extractor", "VideoExtractionError": "extractor",
    "CvatSeedReader": "seeds", "SeedContractError": "seeds",
    "VideoJobService": "service", "VideoUploadError": "service",
    "VideoWorker": "worker", "VideoWorkspace": "workspace",
    "VideoPropagationError": "propagation", "PropagationWorker": "propagation_worker",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value
