from pathlib import Path


def project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent
