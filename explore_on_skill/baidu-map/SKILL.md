---
name: baidu-map
description: 百度地图是一个基于百度的开放平台，用于提供地图、导航、位置信息、搜索等功能。如地图、经纬度、地址解析、周边搜索、驾车、骑行、步行和公交四种交通方式的距离(米)和预计时间（分钟）。
---

# 角色定义
百度地图是一个基于百度的开放平台，用于提供地图、导航、位置信息、搜索等功能。如地图、经纬度、地址解析、周边搜索、驾车、骑行、步行和公交四种交通方式的距离(米)和预计时间（分钟）。

# 技能调用条件
当用户需要查询地图、导航、位置信息、周边搜索等功能时，调用百度地图的技能。

# 技能调用方式
1. 当用户提及从某地到某地需要多少时间或距离时，调用 ./baidu-map/scripts/get_distance_tool.py 脚本，完成距离和预计时间的查询和返回。
```python
python ./baidu-map/scripts/get_distance_tool.py 城市名称 出发地名称 目的地名称
```
2. 当用户提及周边搜索、查询周边信息时，调用 ./baidu-map/scripts/get_surround_tool.py 脚本，完成周边搜索和返回。
```python
python ./baidu-map/scripts/get_surround_tool.py 城市名称 地点 周边类型关键字
```

# 注意事项
1. get_distance_tool.py 脚本和 get_surround_tool.py 脚本中已经实现了具体的逻辑，你只需要执行，不需要做其他多余的操作；
2. 城市名称为必填参数，如果用户没有提供城市名称，则根据当前ip地址所在地判断城市。

# 输出格式
以 Markdown 格式输出结果，合理布局。
