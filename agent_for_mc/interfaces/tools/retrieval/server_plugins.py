from langchain_core.tools import tool

from agent_for_mc.application.deepagent_state import get_turn_context


@tool("get_server_plugins_list")
def get_server_plugins_list() -> list[str]:
    """Return the list of server plugins available to the agent."""
    context = get_turn_context()
    if context is None:
        return []
    return list(context.server_plugins)
