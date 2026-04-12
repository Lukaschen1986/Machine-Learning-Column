"""
MCP Sever
"""
import tools_pdf_file
from mcp.server.fastmcp import FastMCP


mcp = FastMCP(name="pdfReader")
mcp.add_tool(fn=tools_pdf_file.read_pdf_file)

def main():
    print("Hello there!")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
