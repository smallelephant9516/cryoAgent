#!/usr/bin/env bash
set -uo pipefail  # Note: removed -e to allow steps to continue on failure

#############################################
# Multi Conda Environment Installer
# - cryoagent        (from Gitee, environment.yml)
# - helicon          (from GitHub, pip install)
# - magellon2DAssess (for Magellon / CryoSift 2D assess)
#############################################

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS] [STEP_NUMBERS...]

Install conda environments for CryoAgent.

OPTIONS:
    -h, --help          Show this help message
    --steps STEP1 ...   Specify which steps to run (1-5)
                        Can specify multiple steps: --steps 1 3 5
                        Or use positional arguments: $0 1 3 5

STEPS:
    1   cryoagent environment
    2   helicon environment
    3   magellon2DAssess environment
    4   Configure API keys and license
    5   Activate cryoagent environment

EXAMPLES:
    $0                  # Run all steps
    $0 3                # Run only step 3
    $0 1 3 5            # Run steps 1, 3, and 5
    $0 --steps 2 4      # Run steps 2 and 4

EOF
}

# Parse command-line arguments
RUN_STEPS=()
USE_STEPS_FLAG=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        --steps)
            USE_STEPS_FLAG=true
            shift
            # Collect all step numbers after --steps
            while [[ $# -gt 0 ]] && [[ "$1" =~ ^[0-9]+$ ]]; do
                RUN_STEPS+=("$1")
                shift
            done
            ;;
        [0-9]*)
            # Positional argument - step number
            if [[ "$1" =~ ^[1-5]$ ]]; then
                RUN_STEPS+=("$1")
            else
                echo "⛔ ERROR: Invalid step number: $1 (must be 1-5)"
                exit 1
            fi
            shift
            ;;
        *)
            echo "⛔ ERROR: Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Function to check if a step should run
should_run_step() {
    local step_num=$1
    
    # If no steps specified, run all steps
    if [ ${#RUN_STEPS[@]} -eq 0 ]; then
        return 0
    fi
    
    # Check if step is in the list
    for step in "${RUN_STEPS[@]}"; do
        if [ "$step" = "$step_num" ]; then
            return 0
        fi
    done
    
    return 1
}

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

# Initialize step success variables (in case steps are skipped)
STEP1_SUCCESS=false
STEP2_SUCCESS=false
STEP3_SUCCESS=false
STEP4_SUCCESS=false
STEP5_SUCCESS=false

echo "📂 Working directory: $SCRIPT_DIR"
if [ ${#RUN_STEPS[@]} -gt 0 ]; then
    echo "📋 Running steps: ${RUN_STEPS[*]}"
fi
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

if should_run_step 1; then
    echo "============================="
    echo "1️⃣  Setting up cryoagent env"
    echo "============================="

    STEP1_SUCCESS=false
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
            git -C "$CRYOAGENT_DIR" pull || echo "⚠️  WARNING: Failed to pull cryoagent repository"
        else
            echo "📥 Cloning cryoagent from $CRYOAGENT_REPO"
            CRYOAGENT_DIR="./cryoagent"
            if ! git clone "$CRYOAGENT_REPO" "$CRYOAGENT_DIR"; then
                echo "⛔ ERROR: Failed to clone cryoagent repository"
                echo "⏭️  Skipping step 1 (cryoagent environment)"
                echo
            fi
        fi
    fi

    if [ -f "$CRYOAGENT_DIR/environment.yml" ]; then
        echo "📦 Creating/updating conda env: $CRYOAGENT_ENV_NAME from environment.yml"
        if conda_env_exists "$CRYOAGENT_ENV_NAME"; then
            echo "   Environment '$CRYOAGENT_ENV_NAME' already exists. Updating..."
            if conda env update -n "$CRYOAGENT_ENV_NAME" -f "$CRYOAGENT_DIR/environment.yml"; then
                echo "✅ cryoagent environment ready."
                STEP1_SUCCESS=true
            else
                echo "⛔ ERROR: Failed to update cryoagent environment"
                echo "⏭️  Skipping step 1 (cryoagent environment)"
            fi
        else
            if conda env create -n "$CRYOAGENT_ENV_NAME" -f "$CRYOAGENT_DIR/environment.yml"; then
                echo "✅ cryoagent environment ready."
                STEP1_SUCCESS=true
            else
                echo "⛔ ERROR: Failed to create cryoagent environment"
                echo "⏭️  Skipping step 1 (cryoagent environment)"
            fi
        fi
    else
        echo "⛔ ERROR: environment.yml not found in $CRYOAGENT_DIR"
        echo "⏭️  Skipping step 1 (cryoagent environment)"
    fi
    echo
else
    echo "⏭️  Skipping step 1 (cryoagent environment) - not specified"
    STEP1_SUCCESS=false
    echo
fi

#############################################
# 2. helicon environment
#############################################

if should_run_step 2; then
    echo "============================="
    echo "2️⃣  Setting up helicon env"
    echo "============================="

    STEP2_SUCCESS=false
    HELICON_ENV_NAME="helicon"

    if conda_env_exists "$HELICON_ENV_NAME"; then
        echo "🔎 Environment '$HELICON_ENV_NAME' already exists. Skipping creation."
    else
        echo "📦 Creating conda env: $HELICON_ENV_NAME (python=3.10)"
        if ! conda create -y -n "$HELICON_ENV_NAME" python=3.10; then
            echo "⛔ ERROR: Failed to create helicon environment"
            echo "⏭️  Skipping step 2 (helicon environment)"
            echo
        fi
    fi

    if conda_env_exists "$HELICON_ENV_NAME"; then
    echo "📥 Installing helicon[all] from GitHub into '$HELICON_ENV_NAME'"
    set +u  # Temporarily disable unbound variable check for conda activation
    conda activate "$HELICON_ENV_NAME"
    set -u  # Re-enable unbound variable check

    # Try pip install directly from git first
    if pip install "helicon[all] @ git+https://github.com/jianglab/helicon"; then
        echo "✅ Successfully installed helicon[all] from GitHub"
        STEP2_SUCCESS=true
    else
        echo "⚠️  Direct pip install failed. Cloning repository and installing from local copy..."
        HELICON_REPO="https://github.com/jianglab/helicon.git"
        HELICON_CLONE_DIR="$SCRIPT_DIR/helicon_clone"
        
        # Remove existing clone directory if it exists
        if [ -d "$HELICON_CLONE_DIR" ]; then
            rm -rf "$HELICON_CLONE_DIR"
        fi
        
        # Clone the repository
        if git clone "$HELICON_REPO" "$HELICON_CLONE_DIR"; then
            echo "✅ Successfully cloned helicon repository"
            # Install from local clone
            if pip install "$HELICON_CLONE_DIR[all]"; then
                echo "✅ Successfully installed helicon[all] from local clone"
                STEP2_SUCCESS=true
                # Optionally clean up the clone directory after installation
                # rm -rf "$HELICON_CLONE_DIR"
            else
                echo "⛔ ERROR: Failed to install helicon[all] from local clone"
                echo "⏭️  Skipping step 2 (helicon environment)"
            fi
        else
            echo "⛔ ERROR: Failed to clone helicon repository"
            echo "⏭️  Skipping step 2 (helicon environment)"
        fi
    fi

    set +u  # Temporarily disable for conda deactivate
    conda deactivate
    set -u  # Re-enable

        if [ "$STEP2_SUCCESS" = true ]; then
            echo "📝 Updating master_config.json with helicon environment name..."
            update_json_config '.transition.micrograph_conversion.helicon.conda_env = "helicon" | .transition.particle_conversion.helicon.conda_env = "helicon"'
            echo "✅ helicon environment ready."
        fi
    fi
    echo
else
    echo "⏭️  Skipping step 2 (helicon environment) - not specified"
    STEP2_SUCCESS=false
    echo
fi

#############################################
# 3. Magellon / magellon2DAssess environment
#############################################

if should_run_step 3; then
    echo "============================="
    echo "3️⃣  Setting up magellon2DAssess env (Magellon/CryoSift)"
    echo "============================="

    STEP3_SUCCESS=false
    MAGELLON_DIR="$SCRIPT_DIR/Magellon"
    MAGELLON_REPO="https://github.com/smallelephant9516/Magellon"
    MAGELLON_ENV_NAME="magellon2DAssess"
    MAGELLON_REQ_REL="Sandbox/2dclass_evaluator/requirements.txt"
    MAGELLON_REQ_PATH="$MAGELLON_DIR/$MAGELLON_REQ_REL"

    if [ ! -d "$MAGELLON_DIR" ]; then
    echo "📥 Cloning Magellon from $MAGELLON_REPO"
    if ! git clone "$MAGELLON_REPO" "$MAGELLON_DIR"; then
            echo "⛔ ERROR: Failed to clone Magellon repository"
            echo "⏭️  Skipping step 3 (magellon2DAssess environment)"
            echo
        fi
    else
        echo "🔄 Magellon directory already exists, pulling latest changes..."
        git -C "$MAGELLON_DIR" pull || echo "⚠️  WARNING: Failed to pull Magellon repository"
    fi

    if [ -d "$MAGELLON_DIR" ]; then
        if conda_env_exists "$MAGELLON_ENV_NAME"; then
        echo "🔎 Environment '$MAGELLON_ENV_NAME' already exists. Skipping creation."
    else
        echo "📦 Creating conda env: $MAGELLON_ENV_NAME (python=3.12)"
        if ! conda create -y -n "$MAGELLON_ENV_NAME" python=3.12; then
            echo "⛔ ERROR: Failed to create magellon2DAssess environment"
            echo "⏭️  Skipping step 3 (magellon2DAssess environment)"
            echo
        fi
    fi

        if [ -f "$MAGELLON_REQ_PATH" ] && conda_env_exists "$MAGELLON_ENV_NAME"; then
            echo "📥 Installing Magellon 2D evaluator requirements into '$MAGELLON_ENV_NAME'"
            set +u  # Temporarily disable unbound variable check for conda activation
            conda activate "$MAGELLON_ENV_NAME"
            set -u  # Re-enable unbound variable check
            if pip install -r "$MAGELLON_REQ_PATH"; then
                STEP3_SUCCESS=true
            else
                echo "⛔ ERROR: Failed to install Magellon requirements"
                echo "⏭️  Skipping step 3 (magellon2DAssess environment)"
            fi
            set +u  # Temporarily disable for conda deactivate
            conda deactivate
            set -u  # Re-enable
        elif [ ! -f "$MAGELLON_REQ_PATH" ]; then
            echo "⛔ ERROR: requirements file not found at:"
            echo "   $MAGELLON_REQ_PATH"
            echo "⏭️  Skipping step 3 (magellon2DAssess environment)"
        fi
    fi
    fi

    # Update config with cryosift weights path, evaluator script path, and environment name
    if [ "$STEP3_SUCCESS" = true ]; then
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
else
    echo "⏭️  Skipping step 3 (magellon2DAssess environment) - not specified"
    STEP3_SUCCESS=false
    echo
fi

#############################################
# 4. Configure API keys and license
#############################################

if should_run_step 4; then
    echo "==============================================="
    echo "4️⃣  Configuring API Keys and License"
    echo "==============================================="
    echo
    echo "Please enter your API keys and license information."
    echo "You can press Enter to skip any field (you can add them later)."
    echo

    STEP4_SUCCESS=true  # This step is interactive and can't really "fail"

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
    echo "💡 Note: The API keys and LICENSE_ID have been added to ~/.bashrc"
    echo "   They will be automatically loaded in new terminal sessions."
    echo "   For the current session, they are already exported."
    echo
else
    echo "⏭️  Skipping step 4 (API keys and license configuration) - not specified"
    STEP4_SUCCESS=false
    echo
fi

#############################################
# 5. Activate cryoagent environment at the end
#############################################

if should_run_step 5; then
    echo "==============================================="
    echo "5️⃣  Activating cryoagent environment"
    echo "==============================================="

    STEP5_SUCCESS=false
    if [ "$STEP1_SUCCESS" = true ]; then
        echo "🚀 Activating cryoagent environment now..."
        set +u  # Temporarily disable unbound variable check for conda activation
        if conda activate "$CRYOAGENT_ENV_NAME"; then
            set -u  # Re-enable unbound variable check
            echo "🟢 cryoagent environment activated."
            STEP5_SUCCESS=true
        else
            set -u  # Re-enable unbound variable check
            echo "⛔ ERROR: Failed to activate cryoagent environment"
            echo "⏭️  Skipping step 5 (activate cryoagent environment)"
        fi
    else
        echo "⚠️  WARNING: cryoagent environment was not successfully installed in step 1"
        echo "⏭️  Skipping step 5 (activate cryoagent environment)"
    fi
    echo
else
    echo "⏭️  Skipping step 5 (activate cryoagent environment) - not specified"
    STEP5_SUCCESS=false
    echo
fi

#############################################
# Summary
#############################################

echo "==============================================="
echo "📊 Installation Summary"
echo "==============================================="
echo

# Helper function to print step status
print_step_status() {
    local step_num=$1
    local step_name=$2
    local success=$3
    
    if should_run_step "$step_num"; then
        if [ "$success" = true ]; then
            echo "✅ Step $step_num: $step_name - SUCCESS"
        else
            echo "❌ Step $step_num: $step_name - FAILED"
        fi
    else
        echo "⏭️  Step $step_num: $step_name - SKIPPED"
    fi
}

print_step_status 1 "cryoagent environment" "$STEP1_SUCCESS"
print_step_status 2 "helicon environment" "$STEP2_SUCCESS"
print_step_status 3 "magellon2DAssess environment" "$STEP3_SUCCESS"

if should_run_step 4; then
    echo "✅ Step 4: API keys and license configuration - COMPLETED"
else
    echo "⏭️  Step 4: API keys and license configuration - SKIPPED"
fi

print_step_status 5 "Activate cryoagent environment" "$STEP5_SUCCESS"

echo
echo "==============================================="
if [ ${#RUN_STEPS[@]} -eq 0 ] || should_run_step 1 || should_run_step 2 || should_run_step 3; then
    if [ "$STEP1_SUCCESS" = true ] || [ "$STEP2_SUCCESS" = true ] || [ "$STEP3_SUCCESS" = true ]; then
        echo "🎉 Installation completed with some successes!"
        echo "   You can retry failed steps manually if needed."
    else
        echo "⚠️  Installation completed but all environment steps failed."
        echo "   Please check the errors above and retry."
    fi
else
    echo "✅ Selected steps completed!"
fi
echo "==============================================="
echo