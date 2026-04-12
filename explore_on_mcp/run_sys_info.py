"""
MCP Sever
https://github.com/jlowin/fastmcp
uv add fastmcp
"""
import tools_sys_info
from fastmcp import FastMCP


mcp = FastMCP(name="sysInfo", host="127.0.0.1", port=8080)
mcp.tool(name_or_fn=tools_sys_info.get_host_info)
# mcp.tool(name_or_fn=...)  # 可以添加多个工具函数

# 等价于tool
# @mcp.tool(name="foo")
# def foo():
#     return ""

def main():
    mcp.run(transport="stdio")
    # mcp.run(transport="sse", path="/my_sse")  # http://127.0.0.1:8080/my_sse
    # mcp.run(transport="streamable-http", path="/my_mcp")  # http://127.0.0.1:8080/my_mcp


if __name__ == "__main__":
    print("Hello there!")
    main()