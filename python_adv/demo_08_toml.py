# -*- coding: utf-8 -*-
'''
vscode插件：Even Better TOML
pip install tomllib
'''
import os
import tomllib
from pprint import pp


_path = os.path.dirname(__file__)
print(_path)

with open(os.path.join(_path, "config.toml"), "br") as f:
    config = tomllib.load(f)

pp(config)