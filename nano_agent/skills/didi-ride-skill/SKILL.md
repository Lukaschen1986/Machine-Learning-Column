---
name: didi-ride-skill
description: 中国城市出行服务。当用户表达任何交通出行需求时使用此技能——包括打车/叫车/网约车、查价格、路线规划（公交/驾车/步行/骑行）、周边搜索、查询订单/司机位置/取消订单。关键词："打车"、"叫车"、"去[地点]"、"回家"、"上班"、"下班"、"查价格"、"多少钱"、"路线"、"怎么走"、"步行到"、"附近"、"周边"、"司机"、"订单"、"查询订单"。注意：即使用户未明确说"打车"，只要涉及从A地到B地、通勤、或交通方式选择，都应触发。不触发场景：开发打车应用、使用其他导航app、订外卖、查公交时刻表、股票/财报查询。
homepage: https://mcp.didichuxing.com
---

# 滴滴出行服务

通过 MCP 工具（maps_* / taxi_*）提供打车、路线规划、周边搜索、订单管理等出行能力。

> **工具由 nano_agent 通过 AgentScope function calling 自动注册，直接调用即可，无需 shell 命令。**

---

## 1. 用户指南

支持以下操作：

- **打车**：直接说"打车去[地点]"、"回家"、"上班"
- **查价**：查一下从 A 到 B 多少钱
- **查询订单**：输入「查询订单」了解当前订单状态
- **司机位置**：司机在哪里、多久到
- **路线规划**：驾车/公交/步行/骑行路线
- **取消订单**：取消当前订单

---

## 2. Agent 执行指令

### 2.1 文件地图

| 文件 | 用途 | 何时读取 |
|------|------|----------|
| `SKILL.md` | 触发条件、主流程、状态码、规则 | 每次触发必读 |
| `references/workflow.md` | 分阶段详细流程 | 需要实现细节时读 |
| `references/api_references.md` | 工具签名与参数定义 | 每次调用工具前**必须**核对 |
| `references/error_handling.md` | 统一错误码、参数错误排查 | 任何调用失败时读取 |
| `assets/PREFERENCE.md` | 地址别名/车型/手机号偏好 | 用户提到别名或未给出起终点时**必须**读取 |

### 2.2 工具调用原则

所有工具通过 AgentScope function calling 调用，无需 shell 命令。

- **参数名核对：** 每次调用前核对 `references/api_references.md`。常见错误：`keyword` → 应为 `keywords`；`region` → 应为 `city`；只用四字段坐标 → 打车预估需要六字段（`from_lat/from_lng/from_name` + `to_lat/to_lng/to_name`）
- **参数值用字符串：** 经纬度、品类代码等全部用字符串格式（如 `"118.732"` 而非 `118.732`）
- **先预估再下单：** `taxi_create_order` 依赖 `taxi_estimate` 返回的 `traceId`。traceId 有时效性，过期（-32021）需重新预估
- **坐标来源：** 坐标必须来自 `maps_textsearch`，不要凭空编造。**禁止用对话历史记忆补充起终点**——用户可能已换了地方

### 2.3 起终点处理

**缺失补全**（按优先级）：
1. 读 `assets/PREFERENCE.md`，有地址别名且值非空则按场景推断（早晨→起点"家"、下班→起点"公司"）
2. 无可用别名则直接询问用户

**别名匹配：** 精确优先——"家"只匹配"家"，不匹配"妈妈家"。读取时**扫描整张表格**到下一个 `##` 为止。

**确认规则：** 推断的起终点或 `maps_textsearch` 返回多个候选时，向用户确认。用户明确指定且精确匹配时无需确认。

### 2.4 用户确认策略

| 场景 | 规则 |
|------|------|
| 打车 | 推断地址或搜索返回多个候选时必须确认起终点 |
| 取消订单 | 即使用户说了"取消订单"，仍必须先问"确认取消吗？" |

---

## 3. 主流程

### Step 1 — 地址解析

调用 `maps_textsearch`。必要时结合 `assets/PREFERENCE.md`。

```
maps_textsearch(city="南京市", keywords="福润雅居")
```

### Step 2 — 确认起终点

- **单一精确匹配** → 直接使用
- **多个候选** → 列出前 3 个供用户选择
- **别名推断** → 向用户确认 + 告知来源（如"按偏好里「家」推断终点是福润雅居，对吗？"）

### Step 3 — 价格预估

调用 `taxi_estimate`，记录返回的 `traceId`。

```
taxi_estimate(
  from_lat="31.948873", from_lng="118.732288", from_name="福润雅居",
  to_lat="31.985028", to_lng="118.765948", to_name="华为云楼"
)
```

### Step 4 — 车型决策

优先级：**当前消息 > 偏好 > 询问用户**

- 用户说了"叫快车" → 精确匹配 `productCategory`（快车=1），覆盖一切偏好
- 用户未指定 → 使用 `assets/PREFERENCE.md` 中场景车型偏好
- 偏好也未配置 → 向用户询问，不要自行推荐
- ⚠️ 快车（1）和特惠快车（201）是不同服务等级，不可自动替换
- 可用车型以 API 实际返回为准。偏好车型不可用时说明原因并让用户重新选择

### Step 5 — 创建订单

调用 `taxi_create_order`（使用最新 `traceId`）。

```
taxi_create_order(estimate_trace_id="...", product_category="1")
```

- 只接受三个字段：`estimate_trace_id`、`product_category`、`caller_car_phone`（可选）
- 不要把预估的坐标/名称字段带入
- 手机号从 PREFERENCE.md 读取，没有就不传——不要反复索要
- 若返回 `Streamable HTTP error: Unexpected content type: text/plain`，停止流程，输出：

> 未开通 DiDi MCP 免密支付，请到 DiDi MCP 官网开通。审核完成后即可使用。

### Step 6 — 结果输出

```
✅ 订单已创建！
🚖 订单号: [orderId]
📍 [起点] → [终点]
🚗 车型: [车型名称]
💰 预估: 约 [价格] 元
💡 发送「查询订单」可了解当前订单状态
```

---

## 4. 查询订单

触发词：`查询订单` / `查询订单 <orderId>`

订单号来源：用户消息 > 上下文最近一次 > 询问用户

调用 `taxi_query_order`：

```
taxi_query_order(order_id="ORDER_ID")
```

### 状态码与输出

| code | 含义 | 输出 |
|:----:|------|------|
| 0 | 匹配中 | ⏳ 正在为您匹配司机，请稍候 |
| 1 | 司机已接单 | 司机姓名、车型、车牌、电话、距上车点距离、预计到达时间 |
| 2 | 司机已到达 | 🔔 司机已到达上车点，请前往上车 |
| 4 | 行程进行中 | 🚗 行程已开始 |
| 5 | 订单完成 | ✅ 行程结束，展示费用 |
| 6 | 系统取消 | ❌ 订单已被系统取消 |
| 7 | 已取消 | ❌ 订单已取消 |
| 3/8-12 | 其他终态 | 显示对应状态描述 |

---

## 5. 偏好设置更新

用户说"记住/设置/保存"地址、车型或手机号时，**必须写入 `assets/PREFERENCE.md`**，口头承诺无效。

执行步骤：
1. 读取 `assets/PREFERENCE.md`
2. 对地址别名：先调 `maps_textsearch` 获取坐标
3. 定位表格行，用 `edit_file` 或 `write_file` 更新
4. 回读验证

- **地址别名：** 先解析坐标再更新表格行，新别名追加新行
- **场景车型：** 更新对应行。品类代码：快车=1，特惠快车=201，专车=8，豪华车=17
- **叫车手机号：** 更新「默认偏好」表

---

## 6. 路线规划

直接调用对应的路线工具，无需预估或下单。

| 工具 | 用途 | 关键参数 |
|------|------|----------|
| `maps_direction_driving` | 驾车 | origin, destination（经纬度字符串） |
| `maps_direction_transit` | 公交/地铁 | city（完整名如"南京市"）, origin, destination |
| `maps_direction_walking` | 步行 | origin, destination |
| `maps_direction_bicycling` | 骑行 | origin, destination |

---

## 7. 错误码速查

| 错误码 | 含义 | 处理 |
|:------:|------|------|
| -32001 | 限流 | 等待后重试 |
| -32002 | 鉴权失败 | Key 无效，提示用户重新获取 |
| -32010 | 参数验证失败 | 检查参数格式 |
| -32011 | 订单不存在 | 确认订单 ID |
| -32021 | 预估过期 | 重新调用 `taxi_estimate` |
| -32040 | 已取消过 | 无需重复操作 |
| -32041 | 无法取消 | 司机已接单或订单已完成 |

---

## 8. 工具清单

| 领域 | 工具 |
|------|------|
| 地图 | `maps_textsearch`, `maps_regeocode` |
| 路线 | `maps_direction_driving`, `maps_direction_transit`, `maps_direction_walking`, `maps_direction_bicycling` |
| 周边 | `maps_place_around` |
| 打车 | `taxi_estimate`, `taxi_create_order`, `taxi_query_order`, `taxi_cancel_order`, `taxi_get_driver_location`, `taxi_generate_ride_app_link` |
