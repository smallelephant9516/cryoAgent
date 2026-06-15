"""Shared LangChain tool factories used across CryoSPARC stage agents."""

from langchain.tools import Tool


class CryoSPARCCommonTools:
    """Common diagnostic and troubleshooting tools for CryoSPARC ReAct agents."""

    @staticmethod
    def create_get_job_log_tool(agent) -> Tool:
        """Create tool for reading job logs and analyzing errors."""
        return Tool(
            name="get_job_log",
            description="Read and analyze the log file of a CryoSPARC job to understand failures and get suggestions. "
                       "Required parameters: job_uid. "
                       "Optional parameters: project_uid, workspace_uid. "
                       "Use after get_job_status confirms status = failed.",
            func=agent._get_job_log_tool,
        )

    @staticmethod
    def create_search_cryosparc_forum_tool(agent) -> Tool:
        """Create tool for searching CryoSPARC Discuss when a job fails."""
        return Tool(
            name="search_cryosparc_forum",
            description="Search https://discuss.cryosparc.com for troubleshooting threads related to a CryoSPARC error. "
                       "Use after a job fails to find community solutions before retrying. "
                       "Required: query (error keywords from the job log) and/or job_uid (failed job; log is parsed for search terms). "
                       "Optional: max_results (default 5), project_uid, workspace_uid.",
            func=agent._search_cryosparc_forum_tool,
        )

    @staticmethod
    def create_describe_job_params_tool(agent) -> Tool:
        """Create tool exposing the full CryoSPARC parameter spec for a job type."""
        return Tool(
            name="describe_job_params",
            description="Look up the full parameter specification (keys, types, defaults) for a "
                       "CryoSPARC job type, so you can set ANY parameter via a job tool's 'params' dict. "
                       "Required: job_type (a friendly name like 'motion_correction', 'ctf_estimation', "
                       "'class_2d', 'ab_initio_reconstruction', or a raw CryoSPARC id like "
                       "'patch_motion_correction_multi'). Optional: include_hidden. "
                       "Call this before submitting a job when you need a parameter that is not one of the "
                       "tool's friendly named arguments.",
            func=agent._describe_job_params_tool,
        )
