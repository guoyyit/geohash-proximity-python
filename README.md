# geohash-proximity-python

本地经纬度邻近查询服务：维护带唯一标识与属性的经纬度点位，支持
**半径查询**、**最近邻查询**，在点位持续增删改期间保证结果一致，
并正确处理国际日期变更线与极点附近的地理边界。

- 纯 Python 标准库实现（无第三方地理库依赖）
- 多精度 geohash 网格 + 最佳优先前缀树最近邻搜索，查询延迟亚线性
- `threading.RLock` + 不可变值对象，查询与增删改可并发
- 点位数量、半径、最近邻数量、扫描格子数均可配置且超限显式报错
- 可选 FastAPI HTTP 层

## 快速开始

```python
from geohash_proximity import ProximityIndex, IndexConfig, Point

idx = ProximityIndex(IndexConfig(max_points=1_000_000))

# 写入
idx.insert("a", 39.9042, 116.4074, {"name": "Beijing"})
idx.upsert("b", 31.2304, 121.4737, {"name": "Shanghai"})
idx.bulk_load(
    [Point("c", 22.5431, 114.0579, {"name": "Shenzhen"})],
    on_duplicate="replace",  # replace | skip | raise
)

# 半径查询（返回距离 <= 半径的全部点，按 (distance, id) 升序）
for sp in idx.radius_query(34.5, 108.9, radius_m=1_500_000):
    print(sp.point.id, round(sp.distance), "m")

# 最近邻（按 (distance, id) 升序的前 limit 个）
nearest = idx.nearest(34.5, 108.9, limit=5)
```

距离单位为**米**；口径为地球平均半径 `R = 6 371 000 m` 的球面
**haversine 大圆距离**，该口径在测试中固定。完整说明见
[`docs/design.md`](docs/design.md)。

## 写入与重复标识

| 操作 | 语义 |
| --- | --- |
| `insert` | 新增；id 已存在抛 `DuplicatePointError` |
| `upsert` | 新增或确定性整体替换 |
| `update` | 部分字段更新；id 不存在抛 `PointNotFoundError` |
| `delete` | 删除；id 不存在抛 `PointNotFoundError` |
| `bulk_load` | 原子批量，`on_duplicate` 取 `replace` / `skip` / `raise` |

非法坐标（纬度超出 ±90、经度超出 ±180、NaN/Inf、非数值）与非法参数
（负半径、非正 limit 等）一律抛 `InvalidCoordinateError` /
`InvalidParameterError`。

## 排序规则

半径查询与最近邻的返回列表统一按：

```
(distance 升序, point.id 的 Unicode 码位序)
```

距离相同时 id 给出唯一稳定次序，与点位到达顺序无关。最近邻在点数不足
`limit` 时返回全部点，空索引返回 `[]`。

## 配置项

```python
IndexConfig(
    max_points=1_000_000,      # 点位数量（内存）上限
    max_radius_m=5_000_000.0,  # 半径查询最大半径（米）
    max_limit=10_000,          # 最近邻最大 limit
    max_scanned_cells=100_000, # 单查询最多探查的 geohash 格子数
    geohash_precision=9,       # 最细 geohash 长度（1..12）
)
```

任何上限被突破都会抛 `LimitExceededError`，服务不会静默丢弃或截断结果。

## HTTP 服务

```bash
uv run uvicorn main:app --reload
```

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 健康检查与当前点数 |
| PUT | `/points/{id}` | 新增（重复返回 409） |
| POST | `/points/{id}` | upsert |
| PATCH | `/points/{id}` | 部分更新 |
| DELETE | `/points/{id}` | 删除 |
| GET | `/points/{id}` | 获取单点 |
| POST | `/bulk` | 批量载入 |
| GET | `/query/radius?latitude&longitude&radius_m` | 半径查询 |
| GET | `/query/nearest?latitude&longitude&limit` | 最近邻 |

状态码：`422` 非法坐标/参数，`409` 重复 id，`404` 点不存在，
`413` 超出配置上限。交互式文档见 `/docs`。

## 测试

```bash
uv run pytest
```

覆盖距离口径、半径与最近邻、变更线/极点边界（与线性穷举对账）、
并发一致性、重复标识、非法参数、规模与上限、HTTP 端到端。
