import os
import platform
import psutil
import subprocess
import json
import pandas as pd


_path = os.path.dirname(__file__)

def get_district_geocode(district: str) -> str:
    """根据城市或区县名称查询对应的行政区划编码

    Args:
        district (str): 城市或区县名称

    Returns:
        str: district_geocode行政区划编码
    """
    with pd.ExcelFile(os.path.join(_path, "weather_district_id.xlsx")) as reader:
        df = reader.parse(sheet_name=reader.sheet_names[0], encoding="utf-8-sig")
    
    district_geocode = str(df.loc[df.district == district, "district_geocode"].values[0])
    info = {"district_geocode": district_geocode}
    return json.dumps(info, indent=4, ensure_ascii=False)


def get_host_info() -> str:
    """get host information
    Returns:
        str: the host information in JSON string
    """
    info: dict[str, str] = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "memory_gb": str(round(psutil.virtual_memory().total / (1024**3), 2)),
    }

    cpu_count = psutil.cpu_count(logical=True)
    if cpu_count is None:
        info["cpu_count"] = "-1"
    else:
        info["cpu_count"] = str(cpu_count)
    
    try:
        cpu_model = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"]
        ).decode().strip()
        info["cpu_model"] = cpu_model
    except Exception:
        info["cpu_model"] = "Unknown"

    return json.dumps(info, indent=4)



if __name__ == '__main__':
    print(get_district_geocode(district="北京"))