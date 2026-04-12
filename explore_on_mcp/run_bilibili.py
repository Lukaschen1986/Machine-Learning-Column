"""
MCP Sever
"""
import tools_bilibili
from mcp.server.fastmcp import FastMCP


mcp = FastMCP(name="bilibiliSearch")
mcp.add_tool(fn=tools_bilibili.general_search)

def main():
    print("Hello there!")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
