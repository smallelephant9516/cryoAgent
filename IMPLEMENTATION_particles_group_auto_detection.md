# Implementation: Automatic Particles Group Name Detection

## Overview

The optimizer 2D agent now automatically determines which `particles_group_name` to use based on workflow context, eliminating the need for the LLM to explicitly specify it in most cases.

## Workflow Scenarios

### 1. First Round (Step A) - Initial Classification
- **Input**: Original particles from picking/extraction job (e.g., J83)
- **Detection**: Compares `particles_job_uid` with `original_input_job_uid` from workflow defaults
- **Result**: Returns `None` → Uses default connection logic
- **Default Logic**: Tries output labels in order: `("particles", "particles_all", "extracted_particles", "particles_selected", "particles_excluded")`
- **Why**: Extraction/picking jobs typically have `"particles"` or `"particles_all"` output groups

### 2. Rescue (Step B) - Function 2
- **Input**: First `select_2d` job from Step A (e.g., J116)
- **Detection Logic**:
  - Checks if `particles_job_uid` matches a `select_2d` job in execution log
  - Verifies it's the first `select_2d` job (index 0)
  - Confirms only 1 `class_2d` job has run (Step A's initial classification)
  - Checks if Function 2 (Rescue) is enabled
- **Result**: Returns `"particles_excluded"`
- **Why**: Rescue workflow needs to classify the excluded particles from Step A

### 3. Iterative (Step C) - Function 1
- **Input**: `select_2d` job from previous round
- **Detection Logic**:
  - Checks if `particles_job_uid` matches a `select_2d` job
  - If it's NOT the first select_2d job OR more than 1 class_2d has run
- **Result**: Returns `"particles_selected"`
- **Why**: Iterative refinement uses the selected (good) particles from the previous round

## Implementation Details

### Method: `_determine_particles_group_name(particles_job_uid: str)`

```python
def _determine_particles_group_name(self, particles_job_uid: str) -> Optional[str]:
    """
    Intelligently determine which particles group name to use based on workflow context.
    
    Returns:
        - None: For original input jobs (use default connection logic)
        - "particles_excluded": For rescue workflow (Step B)
        - "particles_selected": For iterative workflow (Step C)
    """
```

### Detection Algorithm

1. **Check if original input**:
   ```python
   if particles_job_uid == original_input_job_uid:
       return None  # Use default connection logic
   ```

2. **Check if select_2d job**:
   ```python
   # Search execution log for select_2d_classes jobs
   # Find if particles_job_uid matches any select_2d job
   ```

3. **Determine workflow stage**:
   ```python
   # Count class_2d and select_2d_classes jobs
   # If first select_2d (index 0) AND 1 class_2d AND rescue enabled:
   #   → Rescue (Step B) → return "particles_excluded"
   # Otherwise:
   #   → Iterative (Step C) → return "particles_selected"
   ```

## Benefits

1. **Automatic Detection**: No need for LLM to specify `particles_group_name` in most cases
2. **Prevents Errors**: Eliminates connection failures from wrong group selection
3. **Context-Aware**: Uses execution log to understand workflow state
4. **Backward Compatible**: Still allows explicit `particles_group_name` override
5. **Logging**: Logs which group was auto-selected for debugging

## Edge Cases Handled

1. **Rescue Disabled**: If Function 2 is disabled, first select_2d job is treated as iterative
2. **Multiple Rounds**: Correctly identifies iterative rounds after rescue
3. **Explicit Override**: If `particles_group_name` is explicitly provided, uses that instead
4. **Non-select_2d Jobs**: Falls back to default connection logic for unknown job types

## Example Workflow

```
Step A: class_2d(J83) 
  → Auto-detects: None (original input)
  → Connects to: J83.particles (or J83.particles_all)

Step A: select_2d_classes(J115)
  → Creates: J116 with particles_selected and particles_excluded

Step B: class_2d(J116)
  → Auto-detects: "particles_excluded" (first select_2d, rescue enabled)
  → Connects to: J116.particles_excluded ✅

Step B: select_2d_classes(J120)
  → Creates: J121 with particles_selected and particles_excluded

Step C: class_2d(J121)
  → Auto-detects: "particles_selected" (iterative round)
  → Connects to: J121.particles_selected ✅

Step C: select_2d_classes(J125)
  → Creates: J126 with particles_selected and particles_excluded

Step C (round 2): class_2d(J126)
  → Auto-detects: "particles_selected" (iterative round)
  → Connects to: J126.particles_selected ✅
```

## Verification

The implementation has been verified against the analysis document requirements:
- ✅ First round always uses default (particles_selected for extraction jobs)
- ✅ Iterative rounds use particles_selected
- ✅ Rescue uses particles_excluded
- ✅ Automatic detection based on workflow context
- ✅ Prevents connection failures

