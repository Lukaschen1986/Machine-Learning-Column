import os
import sys
import time
import joblib
import pickle
import operator as op
import numpy as np
import pandas as pd
import tushare as ts
import akshare as ak

from dotenv import load_dotenv
from tqdm import (trange, tqdm)


# ----------------------------------------------------------------------------------------------------------------
# 设置环境变量并获取 tushare token
load_dotenv(dotenv_path="stock.env")
token = os.getenv("TU_SHARE_KEY")
ts.set_token(token)
pro = ts.pro_api()

# ----------------------------------------------------------------------------------------------------------------
# 获取当前脚本所在目录路径
path = os.path.dirname(__file__)

# ----------------------------------------------------------------------------------------------------------------
def update_index_dailybasic(start_days: int = 10) -> pd.DataFrame:
    """更新大盘指数每日指标数据
    
    Args:
        start_days (int): 从当前日期往前推多少天的数据需要更新，默认为10天
    
    Returns:
        pd.DataFrame: 更新后的大盘指数每日指标数据
    """
    fields = """ts_code, trade_date, total_mv, 
    float_mv, total_share, float_share, 
    free_share, turnover_rate, turnover_rate_f, 
    pe, pe_ttm, pb
    """.replace("\n", "")
    
    curr_date = pd.Timestamp.now()
    curr_date = str(curr_date)[0:10].replace("-", "")
    
    start_date = pd.Timestamp.now() - pd.Timedelta(days=start_days)
    start_date = str(start_date)[0:10].replace("-", "")
    
    date_range = pd.date_range(start=start_date, end=curr_date, freq="D")
    date_range = date_range.strftime("%Y%m%d")
    
    df_index_dailybasic = pd.DataFrame()
    
    for date in tqdm(date_range):
        flag = True
        while flag:
            try:
                df = pro.index_dailybasic(ts_code="000300.SH", trade_date=date, fields=fields)
            except Exception as e:
                print(e)
                time.sleep(10)
            else:
                df_index_dailybasic = pd.concat([df_index_dailybasic, df], axis=0, ignore_index=True)
                flag = False
    
    df_index_dailybasic_local = joblib.load(os.path.join(path, "df_index_dailybasic.pkl"))
    df_index_dailybasic = pd.concat([df_index_dailybasic_local, df_index_dailybasic], axis=0, ignore_index=True)
    df_index_dailybasic = df_index_dailybasic.drop_duplicates().reset_index(drop=True)
    joblib.dump(df_index_dailybasic, filename=os.path.join(path, "df_index_dailybasic.pkl"), compress=9)
    print(f"✅ 大盘指数每日指标数据已更新，当前数据行数: {len(df_index_dailybasic)}")
    return df_index_dailybasic


def update_index_daily() -> pd.DataFrame:
    """更新历史行情数据
    
    Args:
        None
    
    Returns:
        pd.DataFrame: 更新后的历史行情数据
    """
    df_index_daily = ak.stock_zh_index_daily(symbol="sh000300")  # 沪深300
    df_index_daily_local = joblib.load(os.path.join(path, "df_index_daily.pkl"))
    df_index_daily = pd.concat([df_index_daily_local, df_index_daily], axis=0, ignore_index=True)
    df_index_daily = df_index_daily.drop_duplicates().reset_index(drop=True)
    joblib.dump(df_index_daily, filename=os.path.join(path, "df_index_daily.pkl"), compress=9)
    print(f"✅ 大盘指数历史行情数据已更新，当前数据行数: {len(df_index_daily)}")
    return df_index_daily


if __name__ == "__main__":
    if len(sys.argv) > 1:
        start_days = int(sys.argv[1])
        df_index_dailybasic = update_index_dailybasic(start_days)
        df_index_daily = update_index_daily()
    else:
        print("请提供往前推多少天的数据需要更新作为参数，例如：python update.py 10")
