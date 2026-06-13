"""Module face restoration."""
from .codeformer_wrapper import CodeFormerWrapper
from .config import FaceRestorationConfig
from .contracts import FaceRestorationResult
from .pipeline import restore_faces

__all__ = [
    "CodeFormerWrapper",
    "FaceRestorationConfig",
    "FaceRestorationResult",
    "restore_faces",
]
