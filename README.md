# CryoAgent

CryoAgent is an intelligent, agentic workflow framework for cryoEM/cryoET image processing using CryoSPARC and LangChain. It implements the **ReAct (Reasoning + Acting)** framework for transparent and reliable automated cryoEM data processing workflows.

## 🧠 ReAct Framework

CryoAgent uses the ReAct framework, which combines **Reasoning** and **Acting** in structured cycles:

1. **Reasoning**: The agent thinks through the problem step by step
2. **Acting**: The agent executes specific tools based on its reasoning  
3. **Observing**: The agent analyzes results and updates its understanding

This approach provides transparent, reliable workflow execution with intelligent error handling and dependency management.

## ✨ Features

- **🧠 ReAct Intelligence**: Transparent reasoning and acting cycles
- **⚙️ Comprehensive Configuration**: JSON-based configuration management
- **🔗 CryoSPARC Integration**: Seamless integration with CryoSPARC for professional cryoEM processing
- **🛠️ Modular Design**: Extensible framework for adding new processing steps
- **🔄 Error Handling**: Robust error handling and job status monitoring
- **📊 Workflow Monitoring**: Real-time workflow state tracking and reasoning history
- **🎯 Unified Interface**: Single script for all workflow types
- **📁 Output Path Tracking**: Automatic absolute path resolution for final picking results (see [docs/particle_picking_output_format.md](docs/particle_picking_output_format.md))

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-username/cryoagent.git
cd cryoagent

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `config.json` file in your project root:

```json
{
  "cryosparc": {
    "host": "localhost",
    "base_port": 61000,
    "username": "your-username",
    "password": "your-password",
    "license_id": "your-cryosparc-license-id-here"
  },
  "agent": {
    "model_name": "deepseek-chat",
    "temperature": 0.1,
    "max_iterations": 15,
    "verbose": true,
    "api_key": "your-api-key",
    "base_url": "https://api.deepseek.com",
    "memory_control": {
      "clear_memory_on_new_conversation": true,
      "maintain_context_between_interactions": false
    }
  },
  "workflow": {
    "project_uid": "P1",
    "workspace_uid": "W1",
    "movies_path": "/path/to/your/movies",
    "pixel_size": 1.0,
    "voltage": 300.0,
    "cs_mm": 2.7,
    "dose": 1.0,
    "motion_correction_binning": 1,
    "motion_correction_patch_size": 5,
    "ctf_min_res": 30.0,
    "ctf_max_res": 4.0
  },
  "job_management": {
    "default_timeout": 3600,
    "status_check_interval": 10
  }
}
```

### 3. Basic Usage

**Using the Unified Workflow Script (Recommended):**

```bash
# Test DeepSeek API connection first
python test_deepseek_connection.py

# Test CryoSPARC connection
python cryoagent_workflow.py --workflow test

# Run the complete workflow
python cryoagent_workflow.py

# Run custom workflow
python cryoagent_workflow.py --workflow custom --steps import_movies,motion_correction

# Single step execution
python cryoagent_workflow.py --workflow single --steps "Import movies and wait for completion"

# Dry run (show what would be done)
python cryoagent_workflow.py --dry-run

# Verbose output
python cryoagent_workflow.py --verbose

# Get help
python cryoagent_workflow.py --help
```

**Using the Python API:**

```python
from cryoagent import (
    ReActCryoEMAgent, 
    ReActCryoEMWorkflow, 
    CryoSPARCTools,
    ConfigLoader
)

# Load configuration
config_loader = ConfigLoader("config.json")
config = config_loader.load_config()

# Initialize components
cryosparc_tools = CryoSPARCTools(config.cryosparc)
agent = ReActCryoEMAgent(cryosparc_tools=cryosparc_tools, config=config)
workflow = ReActCryoEMWorkflow(agent=agent, config=config)

# Run the workflow
results = workflow.run_basic_workflow()

# Check results
for result in results:
    print(f"Step: {result.step.value}, Success: {result.success}")
```

## 🔧 Workflow Types

### 1. Basic Workflow
Executes the complete cryoEM processing pipeline:
- Import Movies → Motion Correction → CTF Estimation

```bash
python cryoagent_workflow.py --workflow basic
```

### 2. Custom Workflow
Execute specific steps in order:

```bash
# Only import movies and motion correction
python cryoagent_workflow.py --workflow custom --steps import_movies,motion_correction

# Only CTF estimation (requires previous steps completed)
python cryoagent_workflow.py --workflow custom --steps ctf_estimation
```

**Valid Steps:**
- `import_movies`: Import movie files into CryoSPARC
- `motion_correction`: Perform motion correction on imported movies
- `ctf_estimation`: Estimate CTF parameters for micrographs

### 3. Single Step
Execute a single step with custom description:

```bash
python cryoagent_workflow.py --workflow single --steps "Import movies from the configured path and wait for completion"
```

### 4. Connection Test
Test CryoSPARC connection and configuration:

```bash
python cryoagent_workflow.py --workflow test
```

## 🧠 ReAct Process Example

```
Thought: I need to start the cryoEM workflow. The first step is to import movies 
from the specified path. I should check if the movies path exists and then 
start the import process with the configured parameters.

Action: import_movies
Parameters: movies_path=/path/to/movies, pixel_size=1.0, voltage=300.0, 
cs_mm=2.7, dose=1.0, project_uid=P1, workspace_uid=W1

Observation: Successfully queued import movies job: J123. The job is now 
running. I need to wait for this job to complete before proceeding to 
motion correction.

Thought: The import job J123 is running. I need to wait for it to complete 
before I can start motion correction, as motion correction depends on the 
imported movies.

Action: wait_for_job
Parameters: job_uid=J123

Observation: Job J123 completed successfully. Now I can proceed with motion 
correction using the imported movies.
```

## 🏗️ Architecture

### Core Components

- **ReActCryoEMAgent**: ReAct-based agent for intelligent workflow orchestration
- **ReActCryoEMWorkflow**: Workflow orchestrator using ReAct methodology
- **CryoSPARCTools**: Direct interface to CryoSPARC operations
- **ConfigLoader**: JSON-based configuration management with validation

### Modular Stage Agents

CryoAgent includes specialized modular agents for each stage of the cryoEM workflow:

- **PreprocessingAgent**: Handles movie import, motion correction, CTF estimation, and micrograph selection
- **PickingAgent**: Manages particle detection using blob picker and template-based picking with 2D classification
- **ReconstructionAgent**: Generates initial 3D models using ab initio reconstruction (see [3D Reconstruction Agent Documentation](docs/3D_RECONSTRUCTION_AGENT.md))

Each modular agent follows the same architectural pattern with:
- ReAct framework implementation
- Dedicated tools and workflow orchestration
- JSON-based configuration
- Comprehensive documentation

### ReAct Workflow Process

1. **Reasoning Phase**: Agent analyzes current state and determines next actions
2. **Acting Phase**: Agent executes specific tools with appropriate parameters
3. **Observation Phase**: Agent analyzes results and updates understanding
4. **Iteration**: Process repeats until workflow completion

## 🔍 Key Benefits

### ReAct Advantages
- **Transparent Reasoning**: See exactly how the agent thinks through problems
- **Better Error Handling**: More intelligent error recovery and retry strategies
- **Dependency Management**: Automatic handling of workflow dependencies
- **Self-Reflection**: Agent can analyze its own performance and adjust

### Configuration Benefits
- **Centralized Settings**: All parameters in one JSON file
- **Environment Flexibility**: Easy switching between different configurations
- **Validation**: Pydantic-based configuration validation
- **Type Safety**: Strong typing for all configuration parameters

## 🚨 Error Handling

The ReAct agent includes sophisticated error handling:

1. **Automatic Retries**: Configurable retry strategies for failed jobs
2. **Fallback Strategies**: Multiple approaches when primary methods fail
3. **Graceful Degradation**: Continue workflow when possible, even with partial failures
4. **Detailed Logging**: Comprehensive logging of all reasoning and actions

## 🧠 Memory Control

CryoAgent includes sophisticated memory control features to manage LLM conversation history and context.

### Memory Control Parameters

Configure memory behavior in your `config.json`:

```json
{
  "agent": {
    "memory_control": {
      "clear_memory_on_new_conversation": true,
      "maintain_context_between_interactions": false
    }
  }
}
```

### Memory Control Options

- **`clear_memory_on_new_conversation`**: Whether to clear conversation history when starting a new conversation
- **`maintain_context_between_interactions`**: Whether to maintain context between different interactions

### Usage Examples

```python
# Run workflow with conversation ID for memory management
result = agent.run_react_workflow(
    "Process my cryoEM data", 
    conversation_id="session_1"
)

# Check memory status
memory_status = agent.get_memory_status()
print(f"Conversation count: {memory_status['conversation_count']}")

# Dynamically change memory control settings
agent.set_memory_control(
    clear_on_new_conversation=False,
    maintain_context=True
)

# Force clear memory when needed
agent.force_clear_memory()
```

### Use Cases

- **Debugging**: Set `clear_memory_on_new_conversation=true` for fresh starts each time
- **Continuous Workflows**: Set `maintain_context_between_interactions=true` for ongoing sessions
- **Session Management**: Use `conversation_id` to group related interactions

## 📊 Monitoring and Debugging

### Connection Testing

Before running workflows, test your connections:

```bash
# Test DeepSeek API connection
python test_deepseek_connection.py

# Test CryoSPARC connection
python test_cryosparc_connection.py
```

### Real-time Status Updates

The workflow provides comprehensive status monitoring:

```
📊 Basic Workflow Results:
1. import_movies: ✅ SUCCESS
   Job UID: J81
   Message: Step import_movies completed successfully

2. motion_correction: ✅ SUCCESS
   Job UID: J82
   Message: Step motion_correction completed successfully

3. ctf_estimation: ✅ SUCCESS
   Job UID: J83
   Message: Step ctf_estimation completed successfully

📈 Workflow Summary:
   Total Steps: 3
   Successful: 3
   Failed: 0
   Execution Time: 1250.45 seconds

🧠 ReAct Reasoning History:
   1. I need to start by importing movies from the configured path
   2. The import job J81 has started, now I need to wait for completion
   3. Import job completed successfully, now I can start motion correction
   ...
```

### Logging and Debugging

```python
# Get the agent's reasoning history
reasoning_history = agent.get_reasoning_history()
for reasoning in reasoning_history:
    print(f"Reasoning: {reasoning}")

# Get current workflow state
current_state = workflow.get_current_state()
print(f"Current state: {current_state}")

# Get workflow summary
summary = workflow.get_workflow_summary()
print(f"Summary: {summary}")
```

## ⚠️ Troubleshooting

### Common Issues

1. **Connection Failed**
   ```
   ❌ Failed to connect to CryoSPARC
   ```
   - Check CryoSPARC is running
   - Verify host, port, and credentials in config
   - Test connection: `python cryoagent_workflow.py --workflow test`

2. **Configuration Error**
   ```
   ❌ Configuration file not found
   ```
   - Ensure `config.json` exists
   - Use `--config` to specify different config file

3. **Job Timeout**
   ```
   ⏰ Job J81 timed out after 3600 seconds
   ```
   - Increase timeout: `--timeout 7200`
   - Check CryoSPARC job queue for issues

### Debug Mode

For detailed debugging:

```bash
python cryoagent_workflow.py --verbose --workflow test
```

## 🎯 Best Practices

1. **Always test connection first**: `python cryoagent_workflow.py --workflow test`
2. **Use dry run for new configurations**: `python cryoagent_workflow.py --dry-run`
3. **Monitor with verbose output**: `python cryoagent_workflow.py --verbose`
4. **Set appropriate timeouts**: `--timeout 7200` for large datasets
5. **Check CryoSPARC resources** before running large workflows
6. **Keep configuration files secure** (don't commit credentials)

## 🔌 API Connection Testing

Before running workflows, test your API connections:

### Comprehensive API Test
```bash
# Test all API connections (CryoSPARC + LLM + Integration)
python test_api_connections.py
```

This script tests:
- ✅ Configuration loading and validation
- ✅ CryoSPARC connection and basic operations
- ✅ LLM API connection (DeepSeek/OpenAI)
- ✅ Component integration
- ✅ Workflow readiness

### CryoSPARC-Only Test
```bash
# Test only CryoSPARC connection with performance metrics
python test_cryosparc_connection.py
```

This script tests:
- ✅ CryoSPARC connection
- ✅ Project and workspace access
- ✅ Connection performance
- ✅ Basic operations

### Quick Connection Test
```bash
# Test connection using the workflow script
python cryoagent_workflow.py --workflow test
```

## 📚 Examples

### Complete Workflow
```bash
# Run the full pipeline
python cryoagent_workflow.py
```

### Test Connection
```bash
# Verify setup
python cryoagent_workflow.py --workflow test
```

### Custom Pipeline
```bash
# Only import and motion correction
python cryoagent_workflow.py --workflow custom --steps import_movies,motion_correction
```

### Development/Testing
```bash
# Dry run to see what would happen
python cryoagent_workflow.py --dry-run

# Verbose output for debugging
python cryoagent_workflow.py --verbose --workflow test
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- CryoSPARC team for the excellent cryoEM processing platform
- LangChain team for the agentic framework
- The cryoEM community for feedback and contributions