"""Video Export — camera tracking, composite view, annotations.

Produces MP4 demos of Genesis scenes by driving an offscreen engine camera
across the scene lifecycle. See :class:`VideoRecorder`.
"""

from moon_rover.visualization.video_export.recorder import (
    RecorderConfig,
    VideoRecorder,
)

__all__ = ["RecorderConfig", "VideoRecorder"]
