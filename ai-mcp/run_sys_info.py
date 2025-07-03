"""
MCP Sever
"""
import tools_sys_info
from mcp.server.fastmcp import FastMCP


mcp = FastMCP(name="sysInfo")
mcp.add_tool(fn=tools_sys_info.get_host_info)
# mcp.add_tool(fn=...)  # 可以添加多个工具函数

# 等价于add_tool
# @mcp.tool(name="foo")
# def foo():
#     return ""

def main():
    print("Hello there!")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
