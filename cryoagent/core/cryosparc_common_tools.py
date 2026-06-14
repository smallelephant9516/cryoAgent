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
