from langchain_core.tools import tool

from agent_for_mc.application.deepagent_state import record_server_plugins


@tool("get_server_plugins_list")
def get_server_plugins_list() -> list[str]:
    """Return the list of server plugins available to the agent."""
    plugins = ["authme", "viaversion"]
    record_server_plugins(plugins)
    return plugins
