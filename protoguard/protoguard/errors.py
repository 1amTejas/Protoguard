"""Exception types used across ProtoGuard."""


class ProtoGuardError(Exception):
    """Base class for all ProtoGuard errors."""


class CaptureError(ProtoGuardError):
    """A packet capture could not be read or is not in a supported format."""


class ModelError(ProtoGuardError):
    """A model could not be trained, saved, loaded, or applied."""
