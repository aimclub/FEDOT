class HasherNotFoundError(ValueError):
    """Raised when no hashing function matches the provided data."""


class SaverNotFoundError(ValueError):
    """Raised when no saver function matches the provided data."""


class LoaderNotFoundError(ValueError):
    """Raised when no loader function matches the provided data."""
