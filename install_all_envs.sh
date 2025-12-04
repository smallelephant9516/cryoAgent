#!/usr/bin/env bash
set -euo pipefail

#############################################
# Multi Conda Environment Installer
# - cryoagent        (from Gitee, environment.yml)
# - helicon          (from GitHub, pip install)
# - magellon2DAssess (for Magellon / CryoSift 2D assess)
#############################################

# ---- 0. Check conda and initialize shell ----
if ! command -v conda &> /dev/null; then
    echo "⛔ ERROR: 'conda' command not found. Please install Anaconda/Miniconda first."
    exit 1
fi

# Initialize conda in this script
eval "$(conda shell.bash hook)"

# Work in the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "📂 Working directory: $SCRIPT_DIR"
echo

# Helper: check if env exists
conda_env_exists () {
    local env_name="$1"
    conda env list | awk '{print $1}' | grep -qx "$env_name"
}

# Helper: update JSON config file using jq
# Usage: update_json_config <jq_expression>
#   or: update_json_config_with_arg <arg_name> <arg_value> <jq_expression_using_$arg_name>
update_json_config () {
    local config_file="$SCRIPT_DIR/configs/master_config.json"
    local jq_expr="$1"
    
    if ! command -v jq &> /dev/null; then
        echo "⚠️  WARNING: 'jq' command not found. Cannot update config file automatically."
        echo "   Please install jq: sudo apt-get install jq  (or equivalent for your system)"
        return 1
    fi
    
    if [ ! -f "$config_file" ]; then
        echo "⚠️  WARNING: Config file not found at $config_file"
        return 1
    fi
    
    # Create a temporary file for the update
    local temp_file=$(mktemp)
    jq "$jq_expr" "$config_file" > "$temp_file" && mv "$temp_file" "$config_file"
    if [ $? -eq 0 ]; then
        echo "✅ Updated config file: $config_file"
        return 0
    else
        echo "⚠️  WARNING: Failed to update config file"
        rm -f "$temp_file"
        return 1
    fi
}

# Helper: update JSON config with argument (safer for paths with special characters)
update_json_config_with_arg () {
    local config_file="$SCRIPT_DIR/configs/master_config.json"
    local arg_name="$1"
    local arg_value="$2"
    local jq_expr="$3"
    
    if ! command -v jq &> /dev/null; then
        echo "⚠️  WARNING: 'jq' command not found. Cannot update config file automatically."
        echo "   Please install jq: sudo apt-get install jq  (or equivalent for your system)"
        return 1
    fi
    
    if [ ! -f "$config_file" ]; then
        echo "⚠️  WARNING: Config file not found at $config_file"
        return 1
    fi
    
    # Create a temporary file for the update
    local temp_file=$(mktemp)
    jq --arg "$arg_name" "$arg_value" "$jq_expr" "$config_file" > "$temp_file" && mv "$temp_file" "$config_file"
    if [ $? -eq 0 ]; then
        echo "✅ Updated config file: $config_file"
        return 0
    else
        echo "⚠️  WARNING: Failed to update config file"
        rm -f "$temp_file"
        return 1
    fi
}

#############################################
# 1. cryoagent environment
#############################################

echo "============================="
echo "1️⃣  Setting up cryoagent env"
echo "============================="

CRYOAGENT_DIR="$SCRIPT_DIR"
CRYOAGENT_ENV_NAME="cryoagent"

# Assume cryoagent repository is already cloned (script is in the repo)
if [ ! -f "$CRYOAGENT_DIR/environment.yml" ]; then
    echo "⛔ ERROR: environment.yml not found in $CRYOAGENT_DIR"
    echo "   Make sure you're running this script from the cryoagent repository root."
    exit 1
fi

echo "📦 Creating/updating conda env: $CRYOAGENT_ENV_NAME from environment.yml"
if conda_env_exists "$CRYOAGENT_ENV_NAME"; then
    echo "   Environment '$CRYOAGENT_ENV_NAME' already exists. Updating..."
    conda env update -n "$CRYOAGENT_ENV_NAME" -f "$CRYOAGENT_DIR/environment.yml"
else
    conda env create -n "$CRYOAGENT_ENV_NAME" -f "$CRYOAGENT_DIR/environment.yml"
fi

echo "✅ cryoagent environment ready."
echo

#############################################
# 2. helicon environment
#############################################

echo "============================="
echo "2️⃣  Setting up helicon env"
echo "============================="

HELICON_ENV_NAME="helicon"

if conda_env_exists "$HELICON_ENV_NAME"; then
    echo "🔎 Environment '$HELICON_ENV_NAME' already exists. Skipping creation."
else
    echo "📦 Creating conda env: $HELICON_ENV_NAME (python=3.10)"
    conda create -y -n "$HELICON_ENV_NAME" python=3.10
fi

echo "📥 Installing helicon[all] from GitHub into '$HELICON_ENV_NAME'"
conda activate "$HELICON_ENV_NAME"
pip install "helicon[all] @ git+https://github.com/jianglab/helicon"
conda deactivate

echo "📝 Updating master_config.json with helicon environment name..."
update_json_config '.transition.micrograph_conversion.helicon.conda_env = "helicon" | .transition.particle_conversion.helicon.conda_env = "helicon"'

echo "✅ helicon environment ready."
echo

#############################################
# 3. Magellon / magellon2DAssess environment
#############################################

echo "============================="
echo "3️⃣  Setting up magellon2DAssess env (Magellon/CryoSift)"
echo "============================="

MAGELLON_DIR="$SCRIPT_DIR/Magellon"
MAGELLON_REPO="https://github.com/sstagg/Magellon"
MAGELLON_ENV_NAME="magellon2DAssess"
MAGELLON_REQ_REL="Sandbox/2dclass_evaluator/requirements.txt"
MAGELLON_REQ_PATH="$MAGELLON_DIR/$MAGELLON_REQ_REL"

if [ ! -d "$MAGELLON_DIR" ]; then
    echo "📥 Cloning Magellon from $MAGELLON_REPO"
    git clone "$MAGELLON_REPO" "$MAGELLON_DIR"
else
    echo "🔄 Magellon directory already exists, pulling latest changes..."
    git -C "$MAGELLON_DIR" pull
fi

if conda_env_exists "$MAGELLON_ENV_NAME"; then
    echo "🔎 Environment '$MAGELLON_ENV_NAME' already exists. Skipping creation."
else
    echo "📦 Creating conda env: $MAGELLON_ENV_NAME (python=3.12)"
    conda create -y -n "$MAGELLON_ENV_NAME" python=3.12
fi

if [ ! -f "$MAGELLON_REQ_PATH" ]; then
    echo "⛔ ERROR: requirements file not found at:"
    echo "   $MAGELLON_REQ_PATH"
    exit 1
fi

echo "📥 Installing Magellon 2D evaluator requirements into '$MAGELLON_ENV_NAME'"
conda activate "$MAGELLON_ENV_NAME"
pip install -r "$MAGELLON_REQ_PATH"
conda deactivate

# Update config with cryosift weights path and environment name
MAGELLON_WEIGHTS_PATH="$MAGELLON_DIR/Sandbox/2dclass_evaluator/CNNTraining/final_model/final_model_cont.pth"
if [ -f "$MAGELLON_WEIGHTS_PATH" ]; then
    echo "📝 Updating master_config.json with cryosift weights path and environment name..."
    # Use jq with --arg for safe path handling
    update_json_config_with_arg "weights_path" "$MAGELLON_WEIGHTS_PATH" ".cryosift.cryosift_weights_path = \$weights_path | .cryosift.cryosift_env = \"$MAGELLON_ENV_NAME\""
else
    echo "⚠️  WARNING: CryoSift weights file not found at: $MAGELLON_WEIGHTS_PATH"
    echo "   Updating only the environment name in config..."
    update_json_config ".cryosift.cryosift_env = \"$MAGELLON_ENV_NAME\""
fi

echo "✅ magellon2DAssess environment ready."
echo

#############################################
# 4. Activate cryoagent environment at the end
#############################################

echo "==============================================="
echo "🎉 All environments installed successfully!"
echo "   - cryoagent"
echo "   - helicon"
echo "   - magellon2DAssess"
echo
echo "🚀 Activating cryoagent environment now..."
echo "==============================================="

conda activate "$CRYOAGENT_ENV_NAME"

echo "🟢 cryoagent environment activated."
echo "You are now ready to use CryoAgent!"
