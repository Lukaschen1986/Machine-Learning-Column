"""
MCP Sever
"""
import tools
from mcp.server.fastmcp import FastMCP


mcp = FastMCP(name="ai-mcp")
mcp.add_tool(fn=tools.get_host_info)
mcp.add_tool(fn=tools.get_district_geocode)
mcp.add_tool(fn=tools.read_pdf_file)

# 等价于add_tool
# @mcp.tool(name="foo")
# def foo():
#     return ""

def main():
    print("Hello from ai-mcp!")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
