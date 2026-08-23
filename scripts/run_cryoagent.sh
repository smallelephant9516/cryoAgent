#!/usr/bin/env bash
# Recommended entry point for CryoAgent — runs preflight checks, then forwards
# all remaining arguments to cryoagent_workflow.py from the repository root.
#
# Preflight checks:
#   1. cryoagent conda environment
#   2. LLM API connection
#   3. CryoSPARC connection
#   4. CryoSift env + weights + evaluator script
#
# Skip checks with:  bash scripts/run_cryoagent.sh --skip-checks ...
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SKIP_CHECKS=0
ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--skip-checks" ]]; then
        SKIP_CHECKS=1
    else
        ARGS+=("$arg")
    fi
done

if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda not found. Install Anaconda/Miniconda and create the cryoagent environment first." >&2
    echo "  bash install.sh" >&2
    exit 1
fi

# shellcheck disable=SC1091
eval "$(conda shell.bash hook)"

if ! conda activate cryoagent 2>/dev/null; then
    echo "Error: could not activate conda environment 'cryoagent'." >&2
    echo "  Run: bash install.sh" >&2
    exit 1
fi

cd "${REPO_ROOT}"

if [[ "${SKIP_CHECKS}" -eq 0 ]]; then
    echo "========================================"
    echo "CryoAgent preflight checks"
    echo "========================================"
    echo ""

    echo "[1/4] cryoagent conda environment"
    echo "   ✅ Active: cryoagent"
    echo ""

    echo "[2/4] LLM connection"
    python check_LLM_connection.py
    echo ""

    echo "[3/4] CryoSPARC connection"
    python check_cryosparc_connection.py
    echo ""

    echo "[4/4] CryoSift connection"
    python check_cryosift_connection.py
    echo ""

    echo "========================================"
    echo "All preflight checks passed"
    echo "========================================"
    echo ""
else
    echo "⚠️  Skipping preflight checks (--skip-checks)"
    echo ""
fi

exec python cryoagent_workflow.py "${ARGS[@]}"
