# -*- coding: utf-8 -*-
"""
https://python.langchain.com/docs/modules/data_connection/document_loaders/
"""
from langchain.document_loaders import TextLoader

loader = TextLoader("./index.md")
loader.load()

