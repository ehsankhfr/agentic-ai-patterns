"""Discover Python tests anywhere in the repository; isolate each test folder."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import venv
from collections import defaultdict


ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ("requirements.txt", "requirements-dev.txt", "requirements-test.txt")


def discover(root):
    """Use Git's file list so ignored virtualenvs/build output are not collected."""
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
    )
    folders = defaultdict(list)
    for name in sorted(set(os.fsdecode(output).split("\0"))):
        path = root / name
        if (path.is_file() and path.suffix == ".py"
                and (path.name.startswith("test") or path.name.endswith("_test.py"))):
            folders[path.parent].append(path)
    return folders


def project_root(folder, root):
    """Find the nearest pattern's requirements or project configuration."""
    markers = (*REQUIREMENTS, "pyproject.toml", "setup.cfg", "setup.py")
    for directory in (folder, *folder.parents):
        if any((directory / marker).is_file() for marker in markers):
            return directory
        if directory == root:
            break
    return root


def prepare_python(project, destination):
    """Install only the dependencies of a project containing tests."""
    venv.EnvBuilder(with_pip=True).create(destination)
    python = destination / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    command = [str(python), "-m", "pip", "install", "pytest"]
    for filename in REQUIREMENTS:
        if (project / filename).is_file():
            command.extend(["-r", filename])
    subprocess.run(command, cwd=project, check=True)
    return python


def run(root, install=False):
    folders = discover(root)
    if not folders:
        print("No Python test files found; nothing to run.")
        return 0

    print(f"Discovered {sum(map(len, folders.values()))} test file(s) in {len(folders)} folder(s).", flush=True)
    failures = []
    interpreters = {}
    with tempfile.TemporaryDirectory(prefix="pattern-ci-") as temp:
        for folder, files in sorted(folders.items()):
            project = project_root(folder, root)
            label = str(folder.relative_to(root))
            print(f"\n=== {label} ===", flush=True)
            if project not in interpreters:
                try:
                    interpreters[project] = (prepare_python(project, Path(temp) / str(len(interpreters)))
                                             if install else Path(sys.executable))
                except subprocess.CalledProcessError:
                    interpreters[project] = None
            python = interpreters[project]
            if python is None:
                failures.append(label + " (dependency installation failed)")
                continue
            command = [str(python), "-m", "pytest", "-v", "-o",
                       "python_files=test*.py *_test.py", *map(str, files)]
            # Fresh process per folder avoids collisions between independent main.py files.
            result = subprocess.run(command, cwd=project)
            if result.returncode:
                failures.append(label)
    if failures:
        print("\nFailed test folders: " + ", ".join(failures), flush=True)
        return 1
    print("\nAll discovered test folders passed.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true",
                        help="Create isolated virtualenvs and install each project's requirements plus pytest")
    args = parser.parse_args()
    return run(ROOT, args.install)


if __name__ == "__main__":
    sys.exit(main())
