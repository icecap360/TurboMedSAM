import importlib.util
from os import PathLike
from typing import Optional

from git.repo import Repo
from rich.console import Console

CONSOLE = Console(width=120)


def is_package_available(package_name: str) -> bool:
    """Checks if package is available.

    Args:
        package_name: Name of package to check.

    Returns:
        True if package is installed, False otherwise.
    """
    spec = importlib.util.find_spec(package_name)
    return False if spec is None else True


def get_repo_working_dir(path_to_repo: Optional[str] = ".") -> PathLike:
    """Returns the absolute path to the git repository root directory.

    Args:
        path_to_repo: The path to the repository or a directory within the repository.
            Defaults to the current directory (".").

    Returns:
        The absolute path to the root directory of the repository.
    """
    repo = Repo(path_to_repo, search_parent_directories=True)
    return repo.working_dir
