Execute the 3D reconstruction workflow starting with {{method_name_lower}}:

**Input**: Particles from job {{particles_job_uid}}

**Task**: Generate initial 3D model(s) from 2D particles using {{method_name_lower}}

**Project**: {{project_uid}} | **Workspace**: {{workspace_uid}}

**Workflow Steps** (execute in order):

═══ PHASE 1: Initial Model Generation ({{method_name}}) ═══

1. **{{method_name}}** - Generate initial 3D model(s)
   - Tool: {{tool_name}}
   - Parameters: 
{{params_str}}
   - {{method_description}}
   - {{num_classes_note}}
   - Wait for completion and record job UID
{{refinement_section}}{{footer_section}}
