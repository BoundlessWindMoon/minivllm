#!/usr/bin/env python3
"""
Pack or unpack the mini-vllm repo using git bundle.

Usage:
    python scripts/bundle_sync.py pack [output.bundle]
    python scripts/bundle_sync.py unpack <bundle-file> [target-dir]

Pack mode:  bundles current branch + tags into a single file.
            Use --all to include all local branches.
Unpack mode: clones the bundle, restores the original remote URL,
             and sets up branch tracking so the repo works normally.
             Refuses to unpack inside an existing git repo for safety.
"""

import argparse
import os
import subprocess
import sys


def run(cmd, cwd=None, check=True):
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"[ERROR] Command failed: {cmd}")
        print(result.stderr.strip())
        sys.exit(1)
    return result.stdout.strip()


def get_repo_root():
    return run("git rev-parse --show-toplevel")


def get_origin_url():
    try:
        return run("git remote get-url origin")
    except SystemExit:
        return ""


def get_current_branch():
    return run("git branch --show-current")


def is_inside_git_repo(path: str) -> bool:
    """Check whether the given path is inside an existing git repository."""
    try:
        run(f"git -C '{path}' rev-parse --git-dir", check=True)
        return True
    except SystemExit:
        return False


def do_pack(output_path: str, include_all: bool):
    repo_root = get_repo_root()
    os.chdir(repo_root)

    origin_url = get_origin_url()
    current_branch = get_current_branch()

    # Write remote URL into bundle description so unpack can restore it
    run(f"git config bundle.originUrl '{origin_url}'")

    if include_all:
        spec = "--all"
        scope = "all branches + tags"
    else:
        spec = f"HEAD {current_branch}"
        scope = f"current branch ({current_branch}) + HEAD"

    cmd = f"git bundle create '{output_path}' {spec}"
    print(f"[PACK] Running: {cmd}")
    print(f"[PACK] Scope: {scope}")
    run(cmd)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[PACK] Bundle created: {output_path} ({size_mb:.1f} MB)")
    if origin_url:
        print(f"[PACK] Remote URL stored: {origin_url}")
    print(
        "\nNext step: copy the bundle file to your target server, then run:\n"
        f"  python scripts/bundle_sync.py unpack {os.path.basename(output_path)}"
    )


def do_unpack(bundle_path: str, target_dir: str):
    bundle_path = os.path.abspath(bundle_path)
    target_abs = os.path.abspath(target_dir)

    if not os.path.isfile(bundle_path):
        print(f"[ERROR] Bundle file not found: {bundle_path}")
        sys.exit(1)

    # Safety 1: refuse to unpack inside an existing git repo
    if is_inside_git_repo(target_abs) or is_inside_git_repo(os.path.dirname(target_abs)):
        # Exception: allow creating a sibling directory of the current repo
        pass

    parent_dir = os.path.dirname(target_abs)
    if is_inside_git_repo(parent_dir):
        # Check if target is the repo root itself or a subdirectory of it
        try:
            repo_root = run(f"git -C '{parent_dir}' rev-parse --show-toplevel")
            if target_abs.startswith(repo_root):
                print(
                    f"[ERROR] Refusing to unpack inside an existing git repository:\n"
                    f"       detected repo root: {repo_root}\n"
                    f"       target path:        {target_abs}\n"
                    "       Choose a target directory outside any git repo."
                )
                sys.exit(1)
        except SystemExit:
            pass

    # Safety 2: refuse to overwrite non-empty directory
    if os.path.exists(target_abs) and os.listdir(target_abs):
        print(f"[ERROR] Target directory exists and is not empty: {target_abs}")
        sys.exit(1)

    # Clone from bundle
    os.makedirs(parent_dir, exist_ok=True)
    target_name = os.path.basename(target_abs)

    cmd = f"git clone '{bundle_path}' '{target_name}'"
    print(f"[UNPACK] Running: {cmd}")
    run(cmd, cwd=parent_dir)

    # Try to read stored remote URL from bundle config
    stored_url = ""
    try:
        stored_url = run("git config --local bundle.originUrl", cwd=target_abs, check=False)
    except Exception:
        pass

    # Fallback: use the original GitHub URL if available
    if not stored_url:
        stored_url = "https://github.com/BoundlessWindMoon/minivllm.git"

    # Restore remote
    run("git remote rename origin bundle-origin", cwd=target_abs, check=False)
    run(f"git remote add origin '{stored_url}'", cwd=target_abs)

    # Determine default branch inside the cloned repo
    default_branch = run("git branch --show-current", cwd=target_abs)

    # Set upstream tracking if the branch exists in the bundle
    branches = run("git branch -a", cwd=target_abs)
    if f"remotes/bundle-origin/{default_branch}" in branches:
        run(
            f"git branch --set-upstream-to=bundle-origin/{default_branch} {default_branch}",
            cwd=target_abs,
        )

    print(f"[UNPACK] Repo restored to: {target_abs}")
    print(f"[UNPACK] Remote origin set to: {stored_url}")
    print(f"[UNPACK] Current branch: {default_branch}")
    print(
        "\nYou can now work normally. When network is available, run:\n"
        "  git fetch origin\n"
        "  git push origin <branch>"
    )


def main():
    parser = argparse.ArgumentParser(description="Git bundle pack/unpack for mini-vllm")
    sub = parser.add_subparsers(dest="command", required=True)

    pack_parser = sub.add_parser("pack", help="Pack repo into a git bundle")
    pack_parser.add_argument(
        "output",
        nargs="?",
        default="mini-vllm.bundle",
        help="Output bundle file path (default: mini-vllm.bundle)",
    )
    pack_parser.add_argument(
        "--all",
        action="store_true",
        help="Include all local branches and tags (default: current branch only)",
    )

    unpack_parser = sub.add_parser("unpack", help="Unpack a git bundle into a repo")
    unpack_parser.add_argument("bundle", help="Path to the bundle file")
    unpack_parser.add_argument(
        "target",
        nargs="?",
        default="mini-vllm",
        help="Target directory name (default: mini-vllm)",
    )

    args = parser.parse_args()

    if args.command == "pack":
        do_pack(args.output, args.all)
    elif args.command == "unpack":
        do_unpack(args.bundle, args.target)


if __name__ == "__main__":
    main()
