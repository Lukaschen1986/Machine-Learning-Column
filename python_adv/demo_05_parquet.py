# -*- coding: utf-8 -*-
"""
pip install pyarrow fastparquet
"""
import os
import sys
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import random as rd


# ----------------------------------------------------------------------------------------------------------------
path_project = os.getcwd()
path_data = os.path.join(os.path.dirname(path_project), "data")
path_model = os.path.join(os.path.dirname(path_project), "model")
path_output = os.path.join(os.path.dirname(path_project), "output")

# ----------------------------------------------------------------------------------------------------------------
df = pd.DataFrame({
    "dates": pd.date_range(start="2023-01-01", periods=100000, freq="T"),
    "cates": rd.choices(population=["A", "B", "C", "D"], k=100000)
    })
df["cates"] = df["cates"].astype("category")
df.info()

# ----------------------------------------------------------------------------------------------------------------
df.to_csv("df_tmp.csv", index=False)  # 2247KB
df.to_parquet("df_tmp.parquet", index=False)  # 920KB

df_csv = pd.read_csv("df_tmp.csv")
df_parquet = pd.read_parquet("df_tmp.parquet")

df_csv.info()
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 100000 entries, 0 to 99999
Data columns (total 2 columns):
 #   Column  Non-Null Count   Dtype 
---  ------  --------------   ----- 
 0   dates   100000 non-null  object
 1   cates   100000 non-null  object
dtypes: object(2)
memory usage: 1.5+ MB
"""

df_parquet.info()
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 100000 entries, 0 to 99999
Data columns (total 2 columns):
 #   Column  Non-Null Count   Dtype         
---  ------  --------------   -----         
 0   dates   100000 non-null  datetime64[ns]
 1   cates   100000 non-null  category      
dtypes: category(1), datetime64[ns](1)
memory usage: 879.2 KB
"""




