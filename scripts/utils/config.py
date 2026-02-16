from pathlib import Path


def ensure_within_root(path: Path, root: Path, label: str) -> None:
    """Ensure path resolves within the expected root directory."""
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must be under {root}") from exc
