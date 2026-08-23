#!/usr/bin/env bash
# Recommended entry point for CryoAgent.
#
# Pipelines (--pipeline):
#   full         Guided workflow, then hypothesis-driven exploration (default)
#   guided       Rigid/guided multi-stage workflow only
#   exploration  Hypothesis-driven improvement on a prior run (--improve) only
#
# Preflight checks (skip with --skip-checks):
#   1. cryoagent conda environment
#   2. LLM API connection
#   3. CryoSPARC connection
#   4. CryoSift env + weights + evaluator script
#
# Examples:
#   bash scripts/run_cryoagent.sh
#   bash scripts/run_cryoagent.sh --pipeline guided
#   bash scripts/run_cryoagent.sh --pipeline exploration --outputs-dir outputs
#   bash scripts/run_cryoagent.sh --skip-checks --workflow test
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PIPELINE="full"
SKIP_CHECKS=0
OUTPUTS_DIR="outputs"
GOAL=""
WORKFLOW_ARGS=()

usage() {
    cat <<'EOF'
Usage: bash scripts/run_cryoagent.sh [OPTIONS] [WORKFLOW_ARGS...]

Run CryoAgent with preflight checks. Remaining arguments are passed to
cryoagent_workflow.py (e.g. --workflow test, --dry-run).

Options:
  --pipeline PIPELINE   Pipeline to run (default: full)
                          full         — guided workflow, then hypothesis-driven exploration
                          guided       — rigid/guided multi-stage workflow only
                          exploration  — hypothesis-driven --improve on prior outputs only
  --outputs-dir DIR     Output directory (default: outputs)
  --goal TEXT           Goal passed to the workflow / improvement agent
  --skip-checks         Skip LLM / CryoSPARC / CryoSift preflight checks
  -h, --help            Show this help

Examples:
  bash scripts/run_cryoagent.sh
  bash scripts/run_cryoagent.sh --pipeline guided
  bash scripts/run_cryoagent.sh --pipeline exploration
  bash scripts/run_cryoagent.sh --goal "Reach 3 Å if data supports it"
  bash scripts/run_cryoagent.sh --skip-checks --workflow custom --stages preprocessing,particle_picking
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pipeline)
            PIPELINE="${2:?--pipeline requires guided, exploration, or full}"
            shift 2
            ;;
        --outputs-dir)
            OUTPUTS_DIR="${2:?--outputs-dir requires a path}"
            shift 2
            ;;
        --goal)
            GOAL="${2:?--goal requires text}"
            shift 2
            ;;
        --skip-checks)
            SKIP_CHECKS=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            WORKFLOW_ARGS+=("$1")
            shift
            ;;
    esac
done

case "${PIPELINE}" in
    guided|exploration|full) ;;
    *)
        echo "Error: unknown --pipeline '${PIPELINE}' (use guided, exploration, or full)" >&2
        exit 1
        ;;
esac

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
    echo "   Active: cryoagent"
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
    echo "Skipping preflight checks (--skip-checks)"
    echo ""
fi

build_workflow_cmd() {
    local -n _out=$1
    shift
    _out=(python cryoagent_workflow.py --outputs-dir "${OUTPUTS_DIR}")
    if [[ -n "${GOAL}" ]]; then
        _out+=(--goal "${GOAL}")
    fi
    _out+=("$@")
    if [[ ${#WORKFLOW_ARGS[@]} -gt 0 ]]; then
        _out+=("${WORKFLOW_ARGS[@]}")
    fi
}

run_workflow() {
    local -a cmd=()
    build_workflow_cmd cmd "$@"
    echo ">>> ${cmd[*]}"
    echo ""
    "${cmd[@]}"
}

case "${PIPELINE}" in
    guided)
        echo "Pipeline: guided (rigid multi-stage workflow)"
        echo ""
        run_workflow --workflow complete --mode guided
        ;;

    exploration)
        echo "Pipeline: exploration (hypothesis-driven improvement)"
        echo "Reading prior outputs from: ${OUTPUTS_DIR}"
        echo ""
        run_workflow --workflow complete --improve
        ;;

    full)
        echo "Pipeline: full (guided, then exploration)"
        echo ""
        echo "=== Stage 1/2: guided workflow ==="
        echo ""
        if run_workflow --workflow complete --mode guided; then
            echo ""
            echo "=== Stage 2/2: exploration (improvement) ==="
            echo ""
            run_workflow --workflow complete --improve
        else
            echo ""
            echo "Guided workflow failed; skipping exploration stage." >&2
            exit 1
        fi
        ;;
esac
