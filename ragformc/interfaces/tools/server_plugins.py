from langchain_core.tools import tool


@tool("get_server_plugins_list")
def get_server_plugins_list() -> list[str]:
    return ["authme", "viaversion"]
