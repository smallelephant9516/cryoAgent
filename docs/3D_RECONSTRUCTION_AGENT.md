# 3D Reconstruction Agent Documentation

## Overview

The 3D Reconstruction Agent is a modular, ReAct-based agent for generating and refining 3D structures from 2D particle images in CryoEM data processing. It follows the same architectural pattern as the preprocessing and picking agents.

## Features

- **Ab Initio Reconstruction**: Generate initial 3D models de novo without reference structures
- **Homogeneous Refinement**: Refine single 3D structures for higher resolution
- **Heterogeneous Refinement**: Simultaneously refine multiple structures with classification
- **ReAct Framework**: Transparent reasoning and acting cycles
- **Configurable Parameters**: Flexible configuration through JSON files
- **Automatic Path Resolution**: Outputs include absolute paths to reconstructed volumes

## Architecture

The reconstruction agent follows the modular pattern with three main components:

### 1. ReconstructionAgent (`reconstruction_agent.py`)
- Implements the ReAct framework for 3D reconstruction
- Provides tool implementations for CryoSPARC reconstruction jobs
- Manages job execution and status monitoring

### 2. ReconstructionWorkflow (`reconstruction_workflow.py`)
- Orchestrates the reconstruction workflow
- Parses configuration parameters
- Manages workflow state and results

### 3. ReconstructionTools (`reconstruction_tools.py`)
- Factory for creating LangChain tools
- Defines tool descriptions and parameters

## Workflow Steps

### Phase 1: Ab Initio Reconstruction (Required)

Generates initial 3D model(s) from 2D particles without requiring a reference structure:

- **Input**: Particles from 2D class selection or extraction
- **Parameters**:
  - `num_classes`: Number of 3D classes (1 for homogeneous, 2-4 for heterogeneous)
  - `initial_resolution`: Starting resolution in Å (default: 20.0)
  - `final_resolution`: Target resolution in Å (default: 10.0)
  - `max_iterations`: Maximum iterations (default: 50)
  - `symmetry`: Symmetry group (default: C1)
- **Output**: Initial 3D volume(s)

### Phase 2: Refinement (Optional)

#### Homogeneous Refinement
- Use when all particles represent the same structure
- Refines a single 3D model for higher resolution
- **Input**: Particles + single volume from ab initio
- **Output**: Refined high-resolution volume

#### Heterogeneous Refinement
- Use when structural heterogeneity is present
- Simultaneously refines multiple structures and classifies particles
- **Input**: Particles + multiple volumes from ab initio
- **Output**: Multiple refined volumes with particle classifications

## Configuration

### Configuration File Format

`configs/3d_reconstruction_config.json`:

```json
{
  "stage": "3d_reconstruction",
  "workflow": {
    "ab_initio": {
      "num_classes": 1,
      "initial_resolution": 20.0,
      "final_resolution": 10.0,
      "max_iterations": 50,
      "symmetry": "C1"
    },
    "refinement": {
      "type": "none",
      "resolution": null
    }
  }
}
```

### Parameter Guidelines

#### Number of Classes (`num_classes`)
- **1**: Homogeneous dataset (all particles same structure)
- **2-3**: Mild heterogeneity
- **3-4**: Significant structural variation
- **Note**: More classes = longer computation time

#### Resolution Settings
- **initial_resolution**: 20-30 Å (stable convergence)
- **final_resolution**: 8-12 Å for ab initio (don't go < 8 Å)
- **refinement_resolution**: 3-5 Å during refinement
- **Note**: Better resolution comes from refinement, not initial ab initio

#### Symmetry Options
- **C1**: No symmetry (safest default)
- **CN**: Cyclic symmetry (C2, C3, C5, etc.)
- **DN**: Dihedral symmetry (D2, D7, etc.)
- **T, O, I**: Tetrahedral, Octahedral, Icosahedral
- **Warning**: Wrong symmetry can cause artifacts!

#### Iterations
- **50**: Standard for most cases
- **100**: For difficult cases with poor initial models
- **Note**: Job may finish earlier if converged

## Usage Examples

### Example 1: Basic Ab Initio Reconstruction

```python
from cryoagent.config.config_loader import ConfigLoader
from cryoagent.tools.cryosparc_tools import CryoSPARCTools
from cryoagent.core.cryosparc_reconstruction import ReconstructionAgent, ReconstructionWorkflow

# Load configuration
config_loader = ConfigLoader("configs/master_config.json")
config = config_loader.load_config()

# Initialize tools and agent
cryosparc_tools = CryoSPARCTools(config.cryosparc)
reconstruction_agent = ReconstructionAgent(cryosparc_tools, config)
reconstruction_workflow = ReconstructionWorkflow(
    reconstruction_agent,
    config,
    stage_config_path="configs/3d_reconstruction_config.json"
)

# Run ab initio reconstruction
results = reconstruction_workflow.run(
    particles_job_uid="J112",  # From particle picking
    conversation_id="reconstruction_1",
    run_refinement=False
)

# Check results
for result in results:
    print(f"{result.step.value}: {result.success}")
    if result.success:
        print(f"  Job UID: {result.job_uid}")
```

### Example 2: Using the Example Script

```bash
# Run the example script
python examples/reconstruction_example.py
```

The script will:
1. Load the latest particle picking results
2. Run ab initio reconstruction
3. Display results and output paths

### Example 3: Multiple Classes for Heterogeneity

Modify `configs/3d_reconstruction_config.json`:

```json
{
  "workflow": {
    "ab_initio": {
      "num_classes": 3,
      "initial_resolution": 20.0,
      "final_resolution": 10.0,
      "max_iterations": 50,
      "symmetry": "C1"
    }
  }
}
```

### Example 4: With Known Symmetry

```json
{
  "workflow": {
    "ab_initio": {
      "num_classes": 1,
      "initial_resolution": 20.0,
      "final_resolution": 10.0,
      "max_iterations": 50,
      "symmetry": "C2"
    },
    "refinement": {
      "type": "homogeneous",
      "resolution": null
    }
  }
}
```

## Output Format

The reconstruction workflow generates a JSON output file in the `outputs/` directory:

```json
{
  "stage": "3d_reconstruction",
  "status": "completed",
  "timestamp": "20251012_180000",
  "project_uid": "P1",
  "workspace_uid": "W1",
  "input_particles_job_uid": "J112",
  "reconstruction_type": "ab_initio",
  "job_uids": {
    "ab_initio": "J113",
    "homogeneous_refinement": null,
    "heterogeneous_refinement": null,
    "final_volume": "J113"
  },
  "outputs": {
    "final_volume_job_uid": "J113",
    "volume_location": "/path/to/cryosparc/P1/J113",
    "final_volume_absolute_path": "/path/to/cryosparc/P1/J113"
  },
  "usage_notes": {
    "next_stage": "refinement_or_analysis",
    "volume_usage": "Use the final_volume_job_uid for further refinement or analysis",
    "final_volume_path": "The final_volume_absolute_path field contains the absolute path to the job directory with the reconstructed volume"
  }
}
```

## Integration with Master Orchestrator

The reconstruction agent is fully integrated with the master orchestrator:

```python
from cryoagent.core.master_orchestrator import MasterOrchestrator

# Initialize orchestrator
orchestrator = MasterOrchestrator("configs/master_config.json")

# Run complete workflow including reconstruction
orchestrator.run_complete_workflow(
    stages=["preprocessing", "particle_picking", "reconstruction"]
)
```

## CryoSPARC Tools

The following methods were added to `CryoSPARCTools`:

### `ab_initio_reconstruction()`
Generate initial 3D models de novo from 2D particles.

**Parameters:**
- `project_uid`: CryoSPARC project UID
- `workspace_uid`: CryoSPARC workspace UID
- `particles_job_uid`: Input particles job UID
- `num_classes`: Number of 3D classes (default: 1)
- `initial_resolution`: Starting resolution in Å (default: 20.0)
- `final_resolution`: Target resolution in Å (default: 10.0)
- `max_iterations`: Maximum iterations (default: 50)
- `symmetry`: Symmetry group (default: C1)

### `homogeneous_refinement()`
Refine a single 3D structure.

**Parameters:**
- `project_uid`: CryoSPARC project UID
- `workspace_uid`: CryoSPARC workspace UID
- `particles_job_uid`: Input particles job UID
- `volume_job_uid`: Input volume job UID from ab initio
- `refinement_resolution`: Target resolution (optional)
- `symmetry`: Symmetry group (default: C1)

### `heterogeneous_refinement()`
Refine multiple 3D structures simultaneously.

**Parameters:**
- `project_uid`: CryoSPARC project UID
- `workspace_uid`: CryoSPARC workspace UID
- `particles_job_uid`: Input particles job UID
- `volume_job_uids`: List of volume job UIDs from ab initio
- `num_classes`: Number of classes (default: length of volume_job_uids)

## Best Practices

### 1. Start Conservative
- Begin with `num_classes=1` unless heterogeneity is certain
- Use initial_resolution=20-25 Å and final_resolution=10-12 Å
- Only use known symmetry if you're certain

### 2. Validate Ab Initio Results
- Check FSC curves in CryoSPARC
- Inspect the 3D volume visually
- Verify resolution estimates are reasonable

### 3. Iterate if Needed
- Re-run with more classes if single class is poor
- Try different symmetries if structure suggests it
- Increase iterations for difficult cases

### 4. Use Refinement Wisely
- Only refine structures that look promising from ab initio
- Homogeneous refinement is faster for single structures
- Heterogeneous refinement helps resolve conformational variation

### 5. Monitor Resources
- Ab initio is computationally intensive
- More classes = more time and resources
- Expect ab initio to take minutes to hours

## Troubleshooting

### Problem: Ab initio fails to converge
**Solutions:**
- Increase `initial_resolution` (start at 25-30 Å)
- Reduce `num_classes` (try 1 or 2)
- Increase `max_iterations`
- Check particle quality and number (need 10k-50k particles minimum)

### Problem: Multiple classes all look the same
**Solutions:**
- Dataset may be homogeneous - use `num_classes=1`
- Try homogeneous refinement instead
- Check if particles are too similar

### Problem: Volume has artifacts or unusual features
**Solutions:**
- Wrong symmetry - try C1 (no symmetry)
- Not enough particles - collect more data
- Poor particle alignment - revisit 2D classification

### Problem: Low resolution
**Solutions:**
- This is expected for ab initio (8-12 Å is typical)
- Run refinement for higher resolution
- Check FSC curves for actual resolution
- More particles and better quality help

## File Structure

```
cryoagent/core/cryosparc_reconstruction/
├── __init__.py                    # Package exports
├── reconstruction_agent.py        # Main agent implementation
├── reconstruction_workflow.py     # Workflow orchestrator
└── reconstruction_tools.py        # Tool definitions

configs/
└── 3d_reconstruction_config.json  # Configuration file

examples/
└── reconstruction_example.py      # Usage example script

outputs/
└── 3d_reconstruction_results_*.json  # Output files
```

## Dependencies

The reconstruction agent requires:
- CryoSPARC with ab initio reconstruction support
- LangChain for ReAct framework
- CryoSPARC tools Python API
- Sufficient computational resources (GPU recommended)

## Next Steps

After ab initio reconstruction:

1. **Assess Quality**: Review 3D volume and FSC curves in CryoSPARC
2. **Refinement**: Run homogeneous or heterogeneous refinement
3. **Classification**: If heterogeneous, classify particles into conformational states
4. **High-Resolution Refinement**: Iteratively refine for publication-quality structures
5. **Model Building**: Use volume for atomic model building

## References

- CryoSPARC Documentation: https://guide.cryosparc.com/
- Ab Initio Reconstruction: CryoSPARC Guide section on initial model generation
- ReAct Framework: Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models"

## Support

For issues or questions:
1. Check CryoSPARC job logs in the web interface
2. Review configuration parameters
3. Verify input particles quality and quantity
4. Consult CryoSPARC documentation for job-specific issues

