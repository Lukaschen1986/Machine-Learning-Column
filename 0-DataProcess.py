# -*- coding: utf-8 -*-
import os
import tarfile
import shutil
import pandas as pd

path = os.path.dirname(__file__)

# 载入 excel
with pd.ExcelFile(os.path.join(path, "inputs.xlsx")) as reader:
    df_1 = reader.parse(sheet_name=reader.sheet_names[0])
    df_2 = reader.parse(sheet_name=reader.sheet_names[1])

# 写入 excel
with pd.ExcelWriter(os.path.join(path, "outputs.xlsx")) as writer:
    df_1.to_excel(writer, sheet_name="page_1", index=False)
    df_2.to_excel(writer, sheet_name="page_2", index=False)


lst_dir = os.listdir(path)
# 打包
with tarfile.open(os.path.join(path, "tar_file.tar"), "w") as tar:
    for d in lst_dir:
        tar.add(os.path.join(path, d))

# 删除原文件
for d in lst_dir:
    if d.endswith(".tar"):
        continue
    else:
        shutil.rmtree(os.path.join(path, d))


# 解包
with tarfile.open(os.path.join(path, "tar_file.tar"), "r") as tar:
    lst_dir = tar.getnames()
    for d in lst_dir:
        tar.extract(d, path)
