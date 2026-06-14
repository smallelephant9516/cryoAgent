Execute the complete ADVANCED particle picking workflow with 9 steps (template-based refinement):

**Input**: Micrographs from job {{micrographs_job_uid}}

**Task**: Use blob picker for initial detection, then refine with template-based picking for high-quality particles

**Project**: {{project_uid}} | **Workspace**: {{workspace_uid}}

**Workflow Steps** (execute in order):

═══ PHASE 1: Initial Blob-Based Picking ═══

1. **Blob Picker** - Initial particle detection
   - Tool: blob_picker
   - Parameters: micrographs_job_uid={{micrographs_job_uid}}, particle_diameter={{particle_diameter}}, diameter_max={{diameter_max}}
   - Wait for completion and record job UID

2. **Extract Particles (Round 1)** - Extract blob-picked particles
   - Tool: extract_particles
   - Parameters: particles_job_uid=[from step 1], micrographs_job_uid={{micrographs_job_uid}}, box_size_pix={{box_size_pix}}
   - Wait for completion and record job UID

3. **2D Classification (Round 1)** - Classify initial particles
   - Tool: class_2d
   - Parameters: particles_job_uid=[from step 2], num_classes={{num_classes}}
   - Wait for completion and record job UID

═══ PHASE 2: Template-Based Refinement ═══

4. {{select_step_title}}
   - Tool: select_2d_classes
   - Parameters: {{select_params}}
   - Wait for completion and record job UID

5. **Template Picker** - Re-pick particles using class averages as templates
   - Tool: template_picker
   - Parameters: micrographs_job_uid={{micrographs_job_uid}}, template_job_uid=[from step 4], lowpass_resolution={{lowpass_resolution}}
   - More accurate than blob picker - uses actual particle images
   - Wait for completion and record job UID

6. **Extract Particles (Round 2)** - Extract template-picked particles
   - Tool: extract_particles
   - Parameters: particles_job_uid=[from step 5], micrographs_job_uid={{micrographs_job_uid}}, box_size_pix={{box_size_pix}} (same as round 1)
   - Wait for completion and record job UID

7. **2D Classification (Round 2)** - Classify refined particles
   - Tool: class_2d
   - Parameters: particles_job_uid=[from step 6], num_classes={{num_classes}} (same as round 1)
   - Wait for completion and record job UID

═══ PHASE 3: Final Selection ═══

8. {{final_select_step_title}}
   - Tool: select_2d_classes
   - Parameters: {{final_select_params}}
   - These are the highest quality particles
   - Wait for completion and record job UID

9. **Final Particles** - Particles ready for 3D reconstruction
   - No additional tool needed - step 8 output contains final selected particles
   - Report the select job UID from step 8 as final output

**Critical Instructions**:
- Execute ALL 9 steps in order - do not skip any steps
- Each step MUST complete successfully before proceeding
- Always wait_for_job after each CryoSPARC job
- Track all job UIDs - each step depends on previous outputs
- Template picking (step 5) requires BOTH micrographs AND templates
- Both extractions (steps 2 & 6) require BOTH particles AND micrographs

**Expected Outcome**:
- High-quality particles from 2 rounds of picking and classification
- Template-based refinement improves particle quality significantly
- Final selected particles ready for 3D reconstruction

Begin by executing step 1 (blob_picker) and proceed sequentially through all 9 steps.
