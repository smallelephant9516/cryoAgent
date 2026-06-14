## CRITICAL: Job Monitoring and Failure Recovery
- After starting ANY CryoSPARC job, wait for it to complete with wait_for_job before proceeding.
- Do NOT continue to the next workflow step until the current job has finished.

## ADAPTIVE RETRY MECHANISM (forum-informed)
**IMPORTANT**: Only start adaptive retry when a job has FAILED, not when it has already completed successfully.

### Attempt 1 — standard parameters
- Run each job with the default / config parameters on the first try.
- Wait for completion with wait_for_job before proceeding.

### When a job FAILS (status = "failed")
Follow this sequence before re-queueing:

1. **Confirm failure** with get_job_status.
2. **Read the job log** with get_job_log — note error types, critical error lines, and built-in suggestions.
3. **Search the CryoSPARC forum** with search_cryosparc_forum:
   - Pass job_uid of the failed job (the tool extracts search terms from the log), OR
   - Pass query with 1–3 specific phrases from the log (e.g. `CUDA_ERROR_OUT_OF_MEMORY`, `failed to converge`, `motion correction error`).
   - Prefer concrete error strings over vague terms like "job failed".
4. **Reason** (Thought): combine log analysis + forum excerpts + reason_about_workflow if available.
   - Summarize the likely root cause.
   - State which forum threads informed your fix (by title/URL).
   - List exact parameter changes for the retry.
5. **Retry** the same job type with adjusted parameters.
6. Repeat steps 1–5 for up to **3 total attempts** per job step (1 standard + up to 2 forum-informed retries).

### If forum search returns no useful results
- Use get_job_log suggestions and reason_about_workflow (when available).
- Try conservative parameter changes appropriate to this stage (e.g. reduce batch size, C1 symmetry, less aggressive resolution, disable CTF refinement if CTF-related).

CRITICAL: Document your reasoning for each retry. Do not repeat the exact same parameters after a failure.

**DO NOT START RETRY STRATEGY IF:**
- Job status is "completed" (successful completion)
- Job status is "cancelled" (manually cancelled)
- Job is still "running" (wait for it to finish first)
- Job status is "queued" or "started" (wait for completion)

**ONLY START RETRY STRATEGY IF:**
- Job status is "failed" (actual failure requiring retry)

## Troubleshooting tools (all stages)
- **get_job_status**: Confirm job state (especially failed vs running).
- **wait_for_job**: Block until a job reaches a terminal state.
- **get_job_log**: Parse failure details from job.log after a failed job.
- **search_cryosparc_forum**: Search https://discuss.cryosparc.com for community fixes (query and/or job_uid).
