#!/bin/bash
# Regenerate workflow_state.json and vis_report.json for all workflows
# with full CryoSPARC enrichment and LLM summaries

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}CryoAgent Workflow Visualization Setup${NC}"
echo -e "${BLUE}======================================${NC}"
echo

# Check CryoSPARC license
if [ -z "$CRYOSPARC_LICENSE_ID" ]; then
    echo -e "${YELLOW}⚠️  Warning: CRYOSPARC_LICENSE_ID not set${NC}"
    echo "   Set it with: export CRYOSPARC_LICENSE_ID=\"your-license-id\""
    echo "   Continuing without CryoSPARC enrichment..."
    echo
    ENRICH_FLAG=""
else
    echo -e "${GREEN}✓ CryoSPARC license found${NC}"
    ENRICH_FLAG="--enrich"
fi

# Check master_config.json for LLM
if [ -f "master_config.json" ]; then
    echo -e "${GREEN}✓ master_config.json found${NC}"
    LLM_FLAG="--llm-summary"
else
    echo -e "${YELLOW}⚠️  Warning: master_config.json not found${NC}"
    echo "   Continuing without LLM summaries..."
    LLM_FLAG=""
fi

echo

# Default to all dynamic_mode workflows if no argument provided
TARGET_DIR="${1:-outputs/dynamic_mode}"

if [ ! -d "$TARGET_DIR" ]; then
    echo -e "${YELLOW}Error: Directory not found: $TARGET_DIR${NC}"
    exit 1
fi

echo -e "${BLUE}Processing workflows in: $TARGET_DIR${NC}"
echo

# Run the script
python3 scripts/create_workflow_visualization.py \
    $ENRICH_FLAG \
    $LLM_FLAG \
    --force \
    "$TARGET_DIR"

echo
echo -e "${GREEN}✨ Done!${NC}"
echo
echo -e "${BLUE}Next steps:${NC}"
echo "  1. Start the visualization servers:"
echo "     cd cryoagent_viz && ./start.sh"
echo
echo "  2. Open http://localhost:3000 in your browser"
echo
echo "  3. Click 'Add folder' and scan: $TARGET_DIR"
echo
