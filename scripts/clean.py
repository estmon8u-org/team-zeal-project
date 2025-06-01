# scripts/clean.py
import fnmatch  # For more flexible pattern matching
import os
import shutil


def clean_project(project_root="."):
    """
    Cleans Python cache files, build artifacts, and other generated clutter
    from the project directory.
    """
    print(f"Starting cleanup in: {os.path.abspath(project_root)}")

    # Patterns for files to delete (using fnmatch for glob-like patterns)
    file_patterns_to_delete = [
        "*.pyc",
        "*.pyo",
        "*.egg-info",  # Remnants from setuptools/pip
        ".coverage",  # Coverage reports
        "*.log",  # General log files (be careful if you have important logs not in .gitignore)
        "Thumbs.db",  # Windows thumbnail cache
        ".DS_Store",  # macOS custom attributes
        ".*pth",  # Any models
    ]

    # Directories to delete recursively
    dir_patterns_to_delete = [
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        "build",  # Common build output directory
        "dist",  # Common distribution directory
        "htmlcov",  # HTML coverage reports
        "site",  # MkDocs build output (if 'docs/site/' is the pattern)
        "outputs",  # Custom output directories, adjust as needed
        "pytorch_profiler_logs",  # Example for PyTorch profiler logs
        "wandb",  # Weights & Biases logs
        # Add other DVC cache or temp directories if they are not handled by 'dvc gc'
        # and you want 'make clean' to remove them (use with caution for DVC internals).
    ]

    # Files or directories to explicitly keep (even if they match a pattern)
    # For example, if you have a specific .log file you want to keep.
    # This is an advanced feature, can be omitted for simplicity.
    # keep_list = [
    #     os.path.join(project_root, "important.log")
    # ]

    for root, dirs, files in os.walk(project_root, topdown=True):
        # Skip .git and .venv directories early to avoid issues and speed up
        if ".git" in dirs:
            dirs.remove(".git")
        if ".venv" in dirs:  # Or your virtual environment directory name
            dirs.remove(".venv")
        if ".dvc" in dirs:  # Usually DVC manages its own cache, but good to skip
            dirs.remove(".dvc")
        if "data" in dirs:  # Be cautious about cleaning data dirs unless intended
            # Potentially add logic here if you have temp files in 'data/interim' for example
            # For now, we'll skip cleaning inside 'data' by default.
            # If you want to clean specific subfolders of data, handle it carefully.
            dirs.remove("data")

        # Delete files matching patterns
        for pattern in file_patterns_to_delete:
            for filename in fnmatch.filter(files, pattern):
                filepath = os.path.join(root, filename)
                try:
                    os.remove(filepath)
                    print(f"Removed file: {filepath}")
                except OSError as e:
                    print(f"Error removing file {filepath}: {e.strerror}")

        # Delete directories matching patterns
        # Iterate over a copy of dirs because we are modifying it
        for dirname in list(dirs):  # Iterate over a copy
            if dirname in dir_patterns_to_delete:
                dirpath = os.path.join(root, dirname)
                try:
                    shutil.rmtree(dirpath)
                    print(f"Removed directory: {dirpath}")
                    dirs.remove(dirname)  # Remove from list of dirs to descend into
                except OSError as e:
                    print(f"Error removing directory {dirpath}: {e.strerror}")

    print("Cleanup complete.")


if __name__ == "__main__":
    # By default, cleans from the directory where the script is located,
    # or more robustly, from the project root if the script is in a subdir.
    # Assuming 'scripts/clean.py', project root is one level up.
    project_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    clean_project(project_directory)
