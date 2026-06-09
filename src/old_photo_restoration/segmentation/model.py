from __future__ import annotations


class SegmentationModelPlaceholder:
    """Placeholder cho model segmentation r013."""

    def __init__(self) -> None:
        self.is_loaded = False

    def load(self, checkpoint_path: str) -> None:
        self.is_loaded = True
        self.checkpoint_path = checkpoint_path
