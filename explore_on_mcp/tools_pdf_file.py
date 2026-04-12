import os
import pandas as pd
import pdfplumber

from typing import Dict


_path = os.path.dirname(__file__)

def read_pdf_file(fileName: str) -> str:
    """根据文件名读取PDF文档
    
    Args:
        fileName (str): 文件名称
    
    Returns:
        str: PDF文档
    """
    text = ""
    with pdfplumber.open(os.path.join(_path, f"data/{fileName}.pdf")) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
            text += "\n\n------分隔页------\n\n"
    return text
