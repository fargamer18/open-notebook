#!/usr/bin/env bash
# Build the Linux release artifacts inside Ubuntu 22.04 (glibc 2.35 baseline).
#
#   packaging/linux/build-in-ubuntu-2204.sh [WORKDIR]
#
# Builds on a developer machine produce packages that need that machine's
# glibc (an Arch build needs glibc >= 2.38). Release builds come from this
# script (or the CI workflow's ubuntu-22.04 job), so they run on Ubuntu
# 22.04+, Debian 12+, Fedora 36+ and similar.
#
# Needs podman or docker. The source tree is copied into WORKDIR (default
# ~/.cache/litenotebook-ubuntu-build); nothing in the repository is written.
# Results: WORKDIR/src/litenotebook/dist/litenotebook/
set -euo pipefail

ENGINE=${CONTAINER_ENGINE:-$(command -v podman || command -v docker || true)}
[ -n "$ENGINE" ] || { echo "podman or docker is required" >&2; exit 1; }
IMAGE="docker.io/library/ubuntu:22.04@sha256:5ec03bb3441e8b0bf3b4f9cd4629a1ae763010dc3035bb8da3ae6cf026486401"
UV_VERSION=0.12.21
UV_SHA256=d69d543a55ec9cdf9d3d9f2648b0a161847e3dbddc477e3be6b5813a6d46f639

HERE=$(cd "$(dirname "$0")/../.." && pwd)   # the litenotebook/ folder
WORK=${1:-$HOME/.cache/litenotebook-ubuntu-build}
mkdir -p "$WORK/src" "$WORK/cache"
rsync -a --delete \
  --exclude .venv --exclude data --exclude build --exclude dist \
  --exclude __pycache__ --exclude .pytest_cache --exclude .ruff_cache \
  --exclude .mypy_cache \
  "$HERE/" "$WORK/src/litenotebook/"
if [ -f "$HERE/../LICENSE" ]; then cp "$HERE/../LICENSE" "$WORK/src/LICENSE"; fi
COMMIT=$(git -C "$HERE" rev-parse HEAD 2>/dev/null || echo unknown)
if [ -n "$(git -C "$HERE" status --porcelain -- . 2>/dev/null)" ]; then
  COMMIT="$COMMIT+dirty"
fi

"$ENGINE" run --rm \
  -v "$WORK:/work" -w /work/src/litenotebook \
  -e LN_GIT_COMMIT="$COMMIT" \
  -e LN_BUILD_RUNNER="ubuntu:22.04 container" \
  -e UV_CACHE_DIR=/work/cache/uv \
  -e UV_PYTHON_INSTALL_DIR=/work/cache/python \
  -e UV_VERSION="$UV_VERSION" -e UV_SHA256="$UV_SHA256" \
  -e RUN_TESTS="${RUN_TESTS:-0}" \
  "$IMAGE" bash -euo pipefail -c '
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq --no-install-recommends ca-certificates curl binutils file >/dev/null
    UVDIR=/work/cache/uv-$UV_VERSION
    if [ ! -x "$UVDIR/uv" ]; then
      mkdir -p "$UVDIR"
      curl -sSLf -o /tmp/uv.tgz \
        "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-x86_64-unknown-linux-musl.tar.gz"
      echo "$UV_SHA256  /tmp/uv.tgz" | sha256sum -c -
      tar -xzf /tmp/uv.tgz -C "$UVDIR" --strip-components=1
    fi
    export PATH="$UVDIR:$PATH"
    ldd --version 2>&1 | sed -n 1p || true  # (no head: SIGPIPE + pipefail)
    uv sync --frozen --group packaging --python 3.12
    if [ "$RUN_TESTS" = "1" ]; then uv run --frozen pytest -q -p no:cacheprovider; fi
    uv run --frozen --group packaging python tools/build.py
  '
echo "Artifacts: $WORK/src/litenotebook/dist/litenotebook/"
