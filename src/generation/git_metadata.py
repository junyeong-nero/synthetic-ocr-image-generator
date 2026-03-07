import subprocess
from pathlib import Path


def run_git_command(args: list[str]) -> str:
    repo_root = Path(__file__).resolve().parent.parent.parent
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def normalize_github_url(raw_url: str) -> str:
    url = raw_url.strip()
    if url.startswith("git@github.com:"):
        url = "https://github.com/" + url.split("git@github.com:", 1)[1]
    elif url.startswith("ssh://git@github.com/"):
        url = "https://github.com/" + url.split("ssh://git@github.com/", 1)[1]
    elif url.startswith("http://github.com/"):
        url = "https://github.com/" + url.split("http://github.com/", 1)[1]

    if url.endswith(".git"):
        url = url[:-4]
    return url.rstrip("/")


def resolve_git_metadata() -> dict[str, str]:
    remote_url = run_git_command(["remote", "get-url", "origin"])
    return {
        "github_url": normalize_github_url(remote_url) if remote_url else "",
        "commit": run_git_command(["rev-parse", "HEAD"]),
        "branch": run_git_command(["rev-parse", "--abbrev-ref", "HEAD"]),
    }
