# 设计说明：本地经纬度邻近查询服务

## 1. 概览

`geohash_proximity.ProximityIndex` 是一个线程安全的纯内存经纬度邻近查询服务：

- 写入：`insert` / `upsert` / `update` / `delete` / `bulk_load`；
- 查询：`radius_query`（半径内全部点）与 `nearest`（最近的 k 个点）；
- 每个点有唯一字符串 `id`、经纬度与任意属性字典。

核心数据结构：

```
_points : dict[id -> Point]                    # 点位主存储（不可变值对象）
_grids  : list[dict[geohash_cell -> set[id]]]  # 每个精度一张倒排网格（精度 1..P）
_prefix_refcount : dict[cell_prefix -> int]    # 非空前缀引用计数，用于最近邻剪枝
```

点在每个 geohash 精度（1..P）上各入一份网格，代价是每点 P 次哈希写入，
换来半径查询与最近邻查询都能选择最合适的粒度。

## 2. 距离口径（固定、受测试锁定）

- 球面模型，地球平均半径 **R = 6 371 000 米**（`geo.EARTH_RADIUS_METRES`）；
- 使用 **haversine 公式**计算大圆距离，单位米；
- 距离对称、非负、与参数顺序无关；
- 半径查询为**闭区间**：`distance <= radius + 1e-9 m`（`_DISTANCE_EPSILON_M`
  仅用于吸收浮点误差，1 纳米量级，不改变业务语义）。

锁定的参考值（见 `tests/test_geo.py`）：

- 赤道上经度 1°：`111194.92664455874 m`；
- 北极到赤道（1/4 子午线）：`10007543.398010284 m`；
- 对跖点：`πR ≈ 20015086.8 m`。

## 3. Geohash 网格与边界正确性

- 标准 Niemeyer base32 字母表 `0123456789bcdefghjkmnpqrstuvwxyz`；
- 经度 180/纬度 90 边界归入最后一个格子；
- 圆盘 -> 包围盒：纬度直接加减角半径；经度跨度
  `dlon = asin(sin r / cos lat)`，触极判定
  `r >= π/2 - |lat|`（触极则覆盖全部经度）；
- 国际日期变更线：经度**不做钳制**，格子列号用取模枚举，
  保证跨界圆盘两侧的点都被且仅被枚举一次；
- 极点：覆盖整行经度列（p1 北极带为 `b c f g u v y z` 共 8 列）；
- 格子集合是圆盘的**保守超集**，候选点再用 haversine 精确过滤，
  因此不会丢点；每个格子只出现一次，因此不会重复返回。

`tests/test_boundaries.py` 在 5° 步长的全球网格上，对变更线、两极、
普通区域共 11 个中心 × 3 个半径做穷举线性扫描对账。

## 4. 半径查询

1. 校验坐标与半径（`0 <= radius <= max_radius_m`）；
2. 从精度 P 向下挑选**最细且扩展格子数 ≤ `max_scanned_cells`** 的精度；
   连精度 1 都超限时抛 `LimitExceededError`，**绝不静默截断**；
3. 枚举包围盒覆盖的格子，在锁内对候选点去重快照；
4. 锁外计算 haversine 距离，过滤 `<= radius`，按 `(distance, id)` 升序返回。

## 5. 最近邻查询（最佳优先前缀树搜索）

不再使用“半径倍增 + bbox 候选数”的做法（bbox 是保守超集，
在稀疏数据上会过早停止且反复构造大量格子字符串）。现在：

1. 从 32 个 p1 根格中，只把“当前有点”的根放入最小堆，键为
   查询点到该格子包围盒的**距离下界**
   （`geo.minimum_distance_to_cell_m`，对变更线感知）；
2. 每次弹出下界最小的格子：
   - 在最深层叶子（depth=P-1）才收集点（点在每层都有记录，
     中间层仅用于引导下行，避免同一点被重复计入）；
   - 维护长度为 `limit` 的有序 top-k；
   - 非叶子只展开“前缀下有点”且“下界不劣于当前第 limit 名”的 32 个子格；
3. 当堆顶下界已严格大于当前第 `limit` 名距离时，剩余子树不可能改进，停止；
4. 已收集满 `limit` 个之前**不允许**按距离剪枝（否则会漏掉填空位的远点）。

复杂度只与“为给出 k 个答案必须探查的非空子树”相关，
20k 点上 k=5 的最近邻 P95 为亚毫秒级，且延迟不随总点数线性增长。

## 6. 排序与并列规则

所有返回列表按 **`(distance 升序, point.id 的 Unicode 码位序)`** 排序。
距离相同时 id 决定唯一、稳定的次序，与插入/到达顺序无关
（`tests/test_index.py::test_nearest_tie_broken_by_id_stably` 正反序验证）。

最近邻数量语义：

| 情况 | 行为 |
| --- | --- |
| `limit < 1` / 非整数 / 布尔 | `InvalidParameterError`（HTTP 422） |
| `limit > max_limit` | `LimitExceededError`（HTTP 413） |
| 点数不足 | 返回全部点，顺序仍为 `(distance, id)` |
| 索引为空 | 返回 `[]` |

## 7. 重复标识与写入语义

| 操作 | 已存在同 id | 行为 |
| --- | --- | --- |
| `insert` | 拒绝，`DuplicatePointError`（HTTP 409），原点不变 |
| `upsert` | 确定性整体替换（坐标 + 属性） |
| `update` | 部分字段更新；未知 id 抛 `PointNotFoundError`（404） |
| `delete` | 删除；未知 id 抛 `PointNotFoundError` |
| `bulk_load(on_duplicate=...)` | 见下 |

`bulk_load` 在触碰索引前完成整批校验与批内去重，随后在**单个临界区**
内完成容量检查与写入，因此要么全部生效要么全部不生效（原子）：

- `"replace"`（默认）：批内后者覆盖前者，且覆盖库中已有同 id；
- `"skip"`：库中已有优先，其次批内最先出现者；
- `"raise"`：批内重复或与库中冲突即 `DuplicatePointError`，整批不写入。

## 8. 并发一致性

- 单一 `threading.RLock` 串行化所有写操作与“候选收集”阶段；
- `Point` / `ScoredPoint` 为不可变值对象，更新采用**替换**而非原地修改；
- 查询在锁内取得候选点的不可变快照（并复制返回属性），
  haversine 计算与排序在锁外进行；因此单次查询只能看到某一个一致的
  点位集合，不会出现半新半旧、重复 id 或访问到已删除对象；
- 最近邻的“堆初始化 -> 遍历 -> 收集 id”全过程在同一个临界区内，
  避免两次加锁之间被批量写入腾空已选根格；
- 线程模型：适合线程级并发；无全局解释器争用之外的异步优化需求。

## 9. 可配置上限（超限显式报错，不静默截断）

`IndexConfig`：

| 配置 | 默认 | 含义 |
| --- | --- | --- |
| `max_points` | 1 000 000 | 点位数量（内存）上限 |
| `max_radius_m` | 5 000 000 | 半径查询允许的最大半径（米） |
| `max_limit` | 10 000 | 最近邻允许的最大 `limit` |
| `max_scanned_cells` | 100 000 | 单查询允许探查的 geohash 格子数 |
| `geohash_precision` | 9 | 索引使用的最细 geohash 长度（1..12） |

所有配置在构造时校验，非法值抛 `InvalidParameterError`。

## 10. HTTP 层

`geohash_proximity.api.create_app(config)` 提供 FastAPI 封装，
`uvicorn main:app` 即可启动。错误映射：422（非法坐标/参数）、
409（重复 id）、404（点不存在）、413（超出配置上限）。

## 11. 测试矩阵

- `tests/test_geo.py`：距离口径、坐标校验、geohash 编解码、圆盘格子枚举；
- `tests/test_index.py`：CRUD、批量、半径/最近邻、并列排序、上限；
- `tests/test_boundaries.py`：变更线/两极/全球 vs 穷举对账、顺序无关；
- `tests/test_concurrency.py`：并发读写快照一致性、批量原子性、竞争写入；
- `tests/test_scale.py`：亚线性延迟、各上限显式报错、精度密度；
- `tests/test_api.py`：HTTP 端到端与状态码。
