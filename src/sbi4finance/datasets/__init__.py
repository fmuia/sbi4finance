"""Small local observation fixtures; simulation banks live in their own workflow."""

from .local import ArtifactError, LocalArtifact, load_artifact, write_artifact
from .returns import prepare_return_window

__all__ = [
    "ArtifactError", "LocalArtifact", "load_artifact", "write_artifact",
    "prepare_return_window",
]
