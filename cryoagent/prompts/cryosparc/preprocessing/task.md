Execute the complete cryoEM preprocessing workflow. Choose the appropriate path based on your input data:

**Option A: If you have raw movie files:**
1. **Import Movies**: Import movie files from {{movies_path}}
   - Pixel size: {{pixel_size}} Å
   - Voltage: {{voltage}} kV
   - CS: {{cs_mm}} mm
   - Dose: {{dose}} e-/Å²
   - Project: {{project_uid}}
   - Workspace: {{workspace_uid}}
   - If movies_path is a list, import all paths in one import_movies call

2. **Motion Correction**: Correct motion in the imported movies
   - Connect all import job UIDs to a single motion correction job when multiple sets were imported
   - Binning: {{motion_binning}}
   - Patch size: {{motion_patch_size}}

3. **CTF Estimation**: Estimate CTF parameters for micrographs
   - Min resolution: {{ctf_min_res}} Å
   - Max resolution: {{ctf_max_res}} Å

**Option B: If you have already motion-corrected micrographs:**
1. **Import Micrographs**: Import micrograph files directly from {{micrographs_or_movies_path}}
   - Pixel size: {{pixel_size}} Å
   - Voltage: {{voltage}} kV
   - CS: {{cs_mm}} mm
   - Dose: {{dose}} e-/Å²
   - Project: {{project_uid}}
   - Workspace: {{workspace_uid}}
   - **CRITICAL**: Skip motion correction and proceed directly to CTF estimation

2. **CTF Estimation**: Estimate CTF parameters for micrographs
   - Min resolution: {{ctf_min_res}} Å
   - Max resolution: {{ctf_max_res}} Å

**Common Final Step:**
4. **Micrograph Selection**: Select micrographs with resolution better than {{min_resolution}} Å
   - Min resolution threshold: {{min_resolution}} Å
   - Filters out low-quality micrographs

**Important**:
- Each step must complete successfully before the next begins
- If using import_micrographs, DO NOT run motion_correction
- Always check job status and wait for completion
- Handle any errors gracefully
- Provide clear status updates throughout the process

Start by reasoning about the workflow state and then proceed step by step.
