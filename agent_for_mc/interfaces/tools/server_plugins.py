from langchain_core.tools import tool


@tool("get_server_plugins_list")
def get_server_plugins_list() -> list[str]:
    """Return the list of server plugins available to the agent."""
    return ["authme", "viaversion"]
