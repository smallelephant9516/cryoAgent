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
# Temporarily disable unbound variable check for conda initialization
set +u
eval "$(conda shell.bash hook)"
set -u

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
# Usage: update_json_config_with_arg <arg_name> <arg_value> <jq_expr> [config_file_path]
update_json_config_with_arg () {
    local arg_name="$1"
    local arg_value="$2"
    local jq_expr="$3"
    local config_file="${4:-$SCRIPT_DIR/configs/master_config.json}"
    
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

CRYOAGENT_ENV_NAME="cryoagent"

# Check if we're already in the cryoagent repository (by checking for environment.yml in current dir)
if [ -f "environment.yml" ]; then
    echo "✅ Already in cryoagent repository. Using current directory."
    CRYOAGENT_DIR="."
else
    # Fallback: check for subdirectory or clone (for users who don't have it yet)
    CRYOAGENT_REPO="https://gitee.com/fei_sun_lab/cryoagent.git"
    if [ -d "./cryoagent" ]; then
        echo "🔄 cryoagent subdirectory already exists, pulling latest changes..."
        CRYOAGENT_DIR="./cryoagent"
        git -C "$CRYOAGENT_DIR" pull
    else
        echo "📥 Cloning cryoagent from $CRYOAGENT_REPO"
        CRYOAGENT_DIR="./cryoagent"
        git clone "$CRYOAGENT_REPO" "$CRYOAGENT_DIR"
    fi
fi

if [ ! -f "$CRYOAGENT_DIR/environment.yml" ]; then
    echo "⛔ ERROR: environment.yml not found in $CRYOAGENT_DIR"
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
set +u  # Temporarily disable unbound variable check for conda activation
conda activate "$HELICON_ENV_NAME"
set -u  # Re-enable unbound variable check
pip install "helicon[all] @ git+https://github.com/jianglab/helicon"
set +u  # Temporarily disable for conda deactivate
conda deactivate
set -u  # Re-enable

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
set +u  # Temporarily disable unbound variable check for conda activation
conda activate "$MAGELLON_ENV_NAME"
set -u  # Re-enable unbound variable check
pip install -r "$MAGELLON_REQ_PATH"
set +u  # Temporarily disable for conda deactivate
conda deactivate
set -u  # Re-enable

# Update config with cryosift weights path, evaluator script path, and environment name
MAGELLON_WEIGHTS_PATH="$MAGELLON_DIR/Sandbox/2dclass_evaluator/CNNTraining/final_model/final_model_cont.pth"
MAGELLON_EVALUATOR_SCRIPT_PATH="$MAGELLON_DIR/Sandbox/2dclass_evaluator/CNNTraining/output_class_list.py"

if [ -f "$MAGELLON_WEIGHTS_PATH" ] && [ -f "$MAGELLON_EVALUATOR_SCRIPT_PATH" ]; then
    echo "📝 Updating master_config.json with cryosift weights path, evaluator script path, and environment name..."
    # Use jq with --arg for safe path handling
    update_json_config_with_arg "weights_path" "$MAGELLON_WEIGHTS_PATH" ".cryosift.cryosift_weights_path = \$weights_path | .cryosift.cryosift_env = \"$MAGELLON_ENV_NAME\""
    update_json_config_with_arg "evaluator_script_path" "$MAGELLON_EVALUATOR_SCRIPT_PATH" ".cryosift.cryosift_evaluator_script_path = \$evaluator_script_path"
    
    # Also update stage-specific config files
    echo "📝 Updating stage-specific config files with cryosift paths..."
    
    # Update optimization_2d_config.json
    OPT_2D_CONFIG="$SCRIPT_DIR/configs/cryosparc/optimization_2d_config.json"
    if [ -f "$OPT_2D_CONFIG" ]; then
        update_json_config_with_arg "weights_path" "$MAGELLON_WEIGHTS_PATH" ".workflow.\"2d_optimization\".select_2d_classes.cryosift_weights_path = \$weights_path | .workflow.\"2d_optimization\".select_2d_classes.cryosift_env = \"$MAGELLON_ENV_NAME\"" "$OPT_2D_CONFIG"
        update_json_config_with_arg "evaluator_script_path" "$MAGELLON_EVALUATOR_SCRIPT_PATH" ".workflow.\"2d_optimization\".select_2d_classes.cryosift_evaluator_script_path = \$evaluator_script_path" "$OPT_2D_CONFIG"
    fi
    
    # Update particle_picking_config.json
    PICKING_CONFIG="$SCRIPT_DIR/configs/cryosparc/particle_picking_config.json"
    if [ -f "$PICKING_CONFIG" ]; then
        update_json_config_with_arg "weights_path" "$MAGELLON_WEIGHTS_PATH" ".workflow.select_2d_classes.cryosift_weights_path = \$weights_path | .workflow.select_2d_classes.cryosift_env = \"$MAGELLON_ENV_NAME\"" "$PICKING_CONFIG"
        update_json_config_with_arg "evaluator_script_path" "$MAGELLON_EVALUATOR_SCRIPT_PATH" ".workflow.select_2d_classes.cryosift_evaluator_script_path = \$evaluator_script_path" "$PICKING_CONFIG"
    fi
elif [ -f "$MAGELLON_WEIGHTS_PATH" ]; then
    echo "⚠️  WARNING: CryoSift evaluator script not found at: $MAGELLON_EVALUATOR_SCRIPT_PATH"
    echo "📝 Updating master_config.json with cryosift weights path and environment name..."
    update_json_config_with_arg "weights_path" "$MAGELLON_WEIGHTS_PATH" ".cryosift.cryosift_weights_path = \$weights_path | .cryosift.cryosift_env = \"$MAGELLON_ENV_NAME\""
elif [ -f "$MAGELLON_EVALUATOR_SCRIPT_PATH" ]; then
    echo "⚠️  WARNING: CryoSift weights file not found at: $MAGELLON_WEIGHTS_PATH"
    echo "📝 Updating master_config.json with evaluator script path and environment name..."
    update_json_config_with_arg "evaluator_script_path" "$MAGELLON_EVALUATOR_SCRIPT_PATH" ".cryosift.cryosift_evaluator_script_path = \$evaluator_script_path | .cryosift.cryosift_env = \"$MAGELLON_ENV_NAME\""
else
    echo "⚠️  WARNING: CryoSift weights file not found at: $MAGELLON_WEIGHTS_PATH"
    echo "⚠️  WARNING: CryoSift evaluator script not found at: $MAGELLON_EVALUATOR_SCRIPT_PATH"
    echo "   Updating only the environment name in config..."
    update_json_config ".cryosift.cryosift_env = \"$MAGELLON_ENV_NAME\""
fi

echo "✅ magellon2DAssess environment ready."
echo



#############################################
# 4. Configure API keys and license
#############################################

echo "==============================================="
echo "🔑 Configuring API Keys and License"
echo "==============================================="
echo
echo "Please enter your API keys and license information."
echo "You can press Enter to skip any field (you can add them later)."
echo

BASHRC_FILE="$HOME/.bashrc"

# Helper: add or update environment variable in .bashrc
add_to_bashrc () {
    local var_name="$1"
    local var_value="$2"
    local bashrc_file="$3"
    
    # Create .bashrc if it doesn't exist
    if [ ! -f "$bashrc_file" ]; then
        touch "$bashrc_file"
        echo "   📝 Created ~/.bashrc"
    fi
    
    # Escape special characters in the value for bash
    local escaped_value=$(printf '%s\n' "$var_value" | sed "s/'/'\\\\''/g")
    
    # Check if the variable already exists in .bashrc
    if grep -q "^export ${var_name}=" "$bashrc_file" 2>/dev/null; then
        # Update existing entry
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS uses BSD sed
            sed -i '' "s|^export ${var_name}=.*|export ${var_name}='${escaped_value}'|" "$bashrc_file"
        else
            # Linux uses GNU sed
            sed -i "s|^export ${var_name}=.*|export ${var_name}='${escaped_value}'|" "$bashrc_file"
        fi
        echo "   ✅ Updated ${var_name} in ~/.bashrc"
    else
        # Add new entry
        echo "export ${var_name}='${escaped_value}'" >> "$bashrc_file"
        echo "   ✅ Added ${var_name} to ~/.bashrc"
    fi
}

# Prompt for DEEPSEEK_API_KEY
read -p "Enter DEEPSEEK_API_KEY (or press Enter to skip): " DEEPSEEK_API_KEY_INPUT
if [ -n "$DEEPSEEK_API_KEY_INPUT" ]; then
    add_to_bashrc "DEEPSEEK_API_KEY" "$DEEPSEEK_API_KEY_INPUT" "$BASHRC_FILE"
    export DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY_INPUT"
fi

# Prompt for OPENAI_API_KEY
read -p "Enter OPENAI_API_KEY (or press Enter to skip): " OPENAI_API_KEY_INPUT
if [ -n "$OPENAI_API_KEY_INPUT" ]; then
    add_to_bashrc "OPENAI_API_KEY" "$OPENAI_API_KEY_INPUT" "$BASHRC_FILE"
    export OPENAI_API_KEY="$OPENAI_API_KEY_INPUT"
fi

# Prompt for PANSHI_API_KEY
read -p "Enter PANSHI_API_KEY (or press Enter to skip): " PANSHI_API_KEY_INPUT
if [ -n "$PANSHI_API_KEY_INPUT" ]; then
    add_to_bashrc "PANSHI_API_KEY" "$PANSHI_API_KEY_INPUT" "$BASHRC_FILE"
    export PANSHI_API_KEY="$PANSHI_API_KEY_INPUT"
fi

# Prompt for LICENSE_ID
read -p "Enter LICENSE_ID (or press Enter to skip): " LICENSE_ID_INPUT
if [ -n "$LICENSE_ID_INPUT" ]; then
    add_to_bashrc "LICENSE_ID" "$LICENSE_ID_INPUT" "$BASHRC_FILE"
    export LICENSE_ID="$LICENSE_ID_INPUT"
    
    # Update master_config.json with LICENSE_ID
    echo "📝 Updating master_config.json with LICENSE_ID..."
    update_json_config_with_arg "license_id" "$LICENSE_ID_INPUT" ".cryosparc.license_id = \$license_id"
fi

echo
echo "🔄 Loading updated environment variables from ~/.bashrc..."
# Source .bashrc to load the new environment variables
# Use a subshell approach to avoid issues with set -e
if [ -f "$BASHRC_FILE" ]; then
    set +e  # Temporarily disable exit on error for sourcing
    source "$BASHRC_FILE" 2>/dev/null || true
    set -e  # Re-enable exit on error
    echo "✅ Environment variables loaded."
else
    echo "⚠️  WARNING: ~/.bashrc not found, skipping source."
fi

echo
echo "==============================================="
echo "✅ Configuration complete!"
echo "==============================================="
echo
echo "🟢 cryoagent environment activated."
echo "You are now ready to use CryoAgent!"
echo
echo "💡 Note: The API keys and LICENSE_ID have been added to ~/.bashrc"
echo "   They will be automatically loaded in new terminal sessions."
echo "   For the current session, they are already exported."

#############################################
# 5. Activate cryoagent environment at the end
#############################################

echo "==============================================="
echo "🎉 All environments installed successfully!"
echo "   - cryoagent"
echo "   - helicon"
echo "   - magellon2DAssess"
echo
echo "🚀 Activating cryoagent environment now..."
echo "==============================================="

set +u  # Temporarily disable unbound variable check for conda activation
conda activate "$CRYOAGENT_ENV_NAME"
set -u  # Re-enable unbound variable check

echo "🟢 cryoagent environment activated."
echo