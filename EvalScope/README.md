# It is recommended to use Python 3.10
conda create -n evalscope python=3.10

# Activate the conda environment
conda activate evalscope

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple evalscope --user

# Additional options
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "evalscope[opencompass]"
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "evalscope[vlmeval]"
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "evalscope[rag]"
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "evalscope[perf]"
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "evalscope[app]"

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "evalscope[all]"