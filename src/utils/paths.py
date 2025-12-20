# src/utils/paths.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def get_repo_root() -> Path:
    """
    Returns repository root, assuming this file lives at:
      repo/src/utils/paths.py  -> parents[2] == repo
    """
    return Path(__file__).resolve().parents[2]


def default_dataset_dir(dataset_name: str = "Amazon_products") -> Path:
    """
    Default dataset directory under repo/project_release/{dataset_name}.
    This is a *default*; prefer passing --dataset_dir in CLI for server paths.
    """
    repo = get_repo_root()
    return repo / "project_release" / dataset_name


def default_data_dir() -> Path:
    """Default intermediate data directory under repo/data."""
    return get_repo_root() / "data"


def default_outputs_dir() -> Path:
    """Default outputs directory under repo/outputs."""
    return get_repo_root() / "outputs"


@dataclass(frozen=True)
class ProjectPaths:
    dataset_dir: Path
    data_dir: Path
    outputs_dir: Path

    @property
    def submissions_dir(self) -> Path:
        return self.outputs_dir / "submissions"

    @property
    def models_dir(self) -> Path:
        return self.outputs_dir / "models"

    @property
    def logs_dir(self) -> Path:
        return self.outputs_dir / "logs"


def resolve_paths(
    dataset_dir: Optional[str | Path] = None,
    data_dir: Optional[str | Path] = None,
    outputs_dir: Optional[str | Path] = None,
    dataset_name: str = "Amazon_products",
) -> ProjectPaths:
    """
    Resolve key directories with sensible defaults.
    - dataset_dir: if None, repo/project_release/{dataset_name}
    - data_dir: if None, repo/data
    - outputs_dir: if None, repo/outputs
    """
    ds = Path(dataset_dir).expanduser().resolve() if dataset_dir else default_dataset_dir(dataset_name)
    dd = Path(data_dir).expanduser().resolve() if data_dir else default_data_dir()
    od = Path(outputs_dir).expanduser().resolve() if outputs_dir else default_outputs_dir()
    return ProjectPaths(dataset_dir=ds, data_dir=dd, outputs_dir=od)
