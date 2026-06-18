"""Tools for CryoSPARC heterogeneity depth analysis operations.

Most heterogeneity-depth tools are now defined in the unified registry
(cryoagent/core/cryosparc_tool_registry.py) and bound to atomic agent methods.
The only factory retained here is the resolution-aware density comparison tool,
which the agent appends with config-bound construction.
"""

from langchain.tools import Tool

from ...tools.flexible_tool import make_flexible_tool


class HeterogeneityDepthTools:
    """Factory for the heterogeneity-depth density comparison tool."""

    @staticmethod
    def create_compare_all_densities_tool(agent) -> Tool:
        """Create resolution-aware density comparison tool for depth analysis."""
        return make_flexible_tool(
            "compare_all_densities",
            "Compare density maps in a folder and filter clusters by resolution. "
            "Normally auto-run inside run_heterogeneous_refinement — call manually only to re-analyze an older job UID. "
            "Required: folder (path to hetero job directory). "
            "Do NOT pass the full get_hetero_class_resolutions JSON — only folder + optional class_resolutions. "
            "KEPT clusters continue; FILTERED OUT (BAD) clusters must be thrown away with no further processing.",
            agent._compare_all_densities_tool,
        )
