#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export APEX_GIT_COMMIT="${APEX_GIT_COMMIT:-$(git -C "${SCRIPT_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)}"
export APEX_GIT_DIRTY="${APEX_GIT_DIRTY:-$(git -C "${SCRIPT_DIR}" status --porcelain 2>/dev/null | wc -l | tr -d ' ')}"

COMPOSE_ARGS=(up --no-build "$@")
if [[ "${APEX_SKIP_BUILD:-0}" != "1" ]]; then
  COMPOSE_ARGS=(up --build "$@")
fi

docker compose -f "${SCRIPT_DIR}/docker/docker-compose.yml" "${COMPOSE_ARGS[@]}"
