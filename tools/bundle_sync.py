#!/usr/bin/env python3
"""
Pack or unpack the mini-vllm repo using git bundle.

Usage:
    # On source machine ────────────────────────────────────────────
    python tools/bundle_sync.py pack [output.bundle]

    # On target machine (after copying the bundle file) ────────────
    # Option A: manual clone, then run setup-remote inside repo
    git clone mini-vllm.bundle mini-vllm
    cd mini-vllm && python tools/bundle_sync.py setup-remote

    # Option B: copy both bundle + this script to target, then run
    python bundle_sync.py unpack mini-vllm.bundle [target-dir]

Pack mode:  bundles current branch + tags into a single file.
            Use --all to include all local branches.

setup-remote: restores origin URL and branch tracking inside a repo
              that was already cloned from a bundle.

unpack:      full clone + setup-remote in one step.
             Requires this script file to exist on the target machine.
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

    # Generate a self-contained shell unpack script alongside the bundle
    bundle_name = os.path.basename(output_path)
    script_name = bundle_name.rsplit(".", 1)[0] + "-unpack.sh"
    script_path = os.path.join(os.path.dirname(os.path.abspath(output_path)), script_name)

    shell_script = f'''#!/bin/bash
set -e
BUNDLE_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET_DIR="mini-vllm"

echo "[UNPACK] Cloning from bundle..."
git clone "$BUNDLE_DIR/{bundle_name}" "$TARGET_DIR"
cd "$TARGET_DIR"

# Restore remote URL from bundle config
ORIGIN_URL=$(git config --local bundle.originUrl 2>/dev/null || echo "")
if [ -z "$ORIGIN_URL" ]; then
    ORIGIN_URL="https://github.com/BoundlessWindMoon/minivllm.git"
fi

git remote rename origin bundle-origin 2>/dev/null || true
git remote add origin "$ORIGIN_URL"

BRANCH=$(git branch --show-current)
if git branch -a | grep -q "remotes/bundle-origin/$BRANCH"; then
    git branch --set-upstream-to="bundle-origin/$BRANCH" "$BRANCH"
fi

echo "[UNPACK] Done. Remote origin restored to: $ORIGIN_URL"
echo "[UNPACK] Current branch: $BRANCH"
echo ""
echo "You can now work normally. When network is available, run:"
echo "  cd $TARGET_DIR && git fetch origin"
'''

    with open(script_path, "w") as f:
        f.write(shell_script)
    os.chmod(script_path, 0o755)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[PACK] Bundle created: {output_path} ({size_mb:.1f} MB)")
    print(f"[PACK] Unpack script created: {script_path}")
    if origin_url:
        print(f"[PACK] Remote URL stored: {origin_url}")
    print(
        "\n=== Next steps on target machine ===\n"
        "Copy these two files to the target machine:\n"
        f"  {bundle_name}\n"
        f"  {script_name}\n"
        "Then run:\n"
        f"  bash {script_name}\n"
    )


def do_setup_remote(cwd: str):
    """Restore origin URL and branch tracking inside an already-cloned repo."""
    if not os.path.isdir(os.path.join(cwd, ".git")):
        print(f"[ERROR] Not a git repository: {cwd}")
        sys.exit(1)

    # Try to read stored remote URL from bundle config
    stored_url = ""
    try:
        stored_url = run("git config --local bundle.originUrl", cwd=cwd, check=False)
    except Exception:
        pass

    # Fallback
    if not stored_url:
        stored_url = "https://github.com/BoundlessWindMoon/minivllm.git"

    # Rename the bundle-origin remote if it exists, then add real origin
    run("git remote rename origin bundle-origin", cwd=cwd, check=False)
    run(f"git remote add origin '{stored_url}'", cwd=cwd)

    default_branch = run("git branch --show-current", cwd=cwd)

    branches = run("git branch -a", cwd=cwd)
    if f"remotes/bundle-origin/{default_branch}" in branches:
        run(
            f"git branch --set-upstream-to=bundle-origin/{default_branch} {default_branch}",
            cwd=cwd,
        )

    print(f"[SETUP] Remote origin restored to: {stored_url}")
    print(f"[SETUP] Current branch: {default_branch}")
    print(
        "\nYou can now work normally. When network is available, run:\n"
        "  git fetch origin\n"
        "  git push origin <branch>"
    )


def do_unpack(bundle_path: str, target_dir: str):
    bundle_path = os.path.abspath(bundle_path)
    target_abs = os.path.abspath(target_dir)

    if not os.path.isfile(bundle_path):
        print(f"[ERROR] Bundle file not found: {bundle_path}")
        sys.exit(1)

    # Safety: refuse to unpack inside an existing git repo
    parent_dir = os.path.dirname(target_abs)
    if is_inside_git_repo(parent_dir):
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

    # Safety: refuse to overwrite non-empty directory
    if os.path.exists(target_abs) and os.listdir(target_abs):
        print(f"[ERROR] Target directory exists and is not empty: {target_abs}")
        sys.exit(1)

    parent_dir = os.path.dirname(target_abs)
    os.makedirs(parent_dir, exist_ok=True)
    target_name = os.path.basename(target_abs)

    cmd = f"git clone '{bundle_path}' '{target_name}'"
    print(f"[UNPACK] Running: {cmd}")
    run(cmd, cwd=parent_dir)

    do_setup_remote(target_abs)


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

    setup_parser = sub.add_parser(
        "setup-remote",
        help="Restore origin remote inside a repo cloned from a bundle",
    )
    setup_parser.add_argument(
        "--cwd",
        default=".",
        help="Path to the cloned repo (default: current directory)",
    )

    unpack_parser = sub.add_parser("unpack", help="Clone bundle and restore remote")
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
    elif args.command == "setup-remote":
        do_setup_remote(args.cwd)
    elif args.command == "unpack":
        do_unpack(args.bundle, args.target)


if __name__ == "__main__":
    main()
