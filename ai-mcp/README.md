https://github.com/astral-sh/uv  

# On Windows.  
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"  

# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or 
pip install uv
uv --version  

# 安装 python
uv python list
uv python install python-3.11
uv run -p 3.11 python

# 用 uv 创建项目  
uv init dir_name -p 3.11

cd dir_name
uv init
uv venv

# 用 uv 管理环境  
uv add jupyter
uv add ipykernel
python -m ipykernel install --user --name unsloth --display-name "Python unsloth"
uv add unsloth unsloth_zoo

uv add mcp
uv pip install -i https://pypi.tuna.tsinghua.edu.cn/simple mcp

uv run main.py

# 打印依赖树
uv tree
uv sync  # 复刻虚拟环境  
uv init --script main.py  # 创建临时虚拟环境，把三方库添加到 dependencies 中，运行python脚本  

# 用 uv 管理命令行工具  
uv add ruff --dev  # 避免ruff被打包
uv remove ruff --dev

uv tool install ruff 
uv tool uninstall ruff
uv tool list

# 打包
-- 进入 pyproject.toml
[project.scripts]
pacakage_name = "module_name:function_name"

-- 回到命令行，打包生成whl文件
uv build

-- 安装打包
uv tool install pacakage_name.whl