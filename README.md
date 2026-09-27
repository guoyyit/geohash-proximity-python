# geohash-proximity-python

本地内存型**经纬度邻近查询服务**：维护一批带唯一标识与属性的经纬度点位，
支持球面半径查询、最近邻查询，并在点位持续增删改与并发访问下保证结果
自洽。查询使用自适应多层经纬网格索引，局部查询的延迟显著优于全量线性
扫描。

* 纯 Python 标准库实现（几何、索引、读写锁均无第三方依赖）
* HTTP 层基于 FastAPI
* 线程安全：多读者 / 单写者（写者优先），查询只看到某个一致的点位快照

---

## 1. 距离口径（被测试固定）

* 地球建模为半径 **`R = 6 371 008.8 m`** 的球体（IUGG 平均地球半径
  `(2a+b)/3`，取 WGS-84 轴）。常量：`geo_proximity.geo.EARTH_RADIUS_M`。
* 距离采用 **haversine 大圆距离，单位米**，值域 `[0, πR]`。
  对跖点距离（半球周长）`πR ≈ 20 015 114.4 m`：
  `geo_proximity.geo.HALF_CIRCUMFERENCE_M`。
* **半径查询边界包含等号**：返回所有 `distance <= radius` 的点。
* 经度按模 360° 解释并归一化到 `[-180, 180)`：写入 `181` 等价于 `-179`，
  `±180°` 是同一条换日线；纬度不做折叠，超出 `[-90, 90]` 一律拒绝。

固定这些数值的测试见 `tests/test_geo.py`；改动常量需同步更新文档。

## 2. 排序与确定性规则

* 半径查询与最近邻查询的结果都按
  **`(distance_m 升序, point.id 升序)`** 排序。
* **等距用 `point.id` 字典序打破平局**，因此排序是全序、与点位到达顺序
  无关。
* 最近邻查询在点位不足时**返回全部已有点位（不报错）**；空库返回 `[]`。
  只有非法参数（`limit < 1`、半径为负/NaN、坐标越界等）才报错。
* 重复 id 写入策略由 `OnDuplicate` 显式给出：
  `error`（默认，抛 `DuplicateIdError`）、`replace`（覆盖，即 upsert）、
  `skip`（保留旧值，返回"未写入"）。行为确定且可解释。

## 3. 配置项（超限一律显式报错，绝不静默截断）

`ServiceConfig`（`geo_proximity/models.py`）：

| 配置 | 默认 | 含义 / 上限行为 |
| --- | --- | --- |
| `max_points` | `1_000_000` | 存储最大点位数；插入/批量载入会超出时抛 `LimitExceededError` |
| `max_query_radius_m` | `20_015_114.4`（对跖距离） | 半径查询/`max_distance_m` 的上限；超过抛 `LimitExceededError`（HTTP 413） |
| `max_neighbors` | `10_000` | 最近邻 `limit` 与半径查询可选 `limit` 的上限 |
| `index_levels` | `16`，范围 `[1, 20]` | 网格层级深度 `L`；层级 `l` 纬度分 `2**l` 带、经度分 `2**(l+1)` 带 |
| `max_batch_size` | `100_000` | 单次 `bulk_load` 最大条数 |

`index_levels` 越大，最细格越小（第 16 级赤道格边约 305 m），小半径
查询更精准，但每个点在索引中的成员关系更多（每级一个 set 槽位，内存随
`L` 线性增长）。点位密集且查询半径很小时可调大；点位稀疏或内存受限时
调小。

半径查询的可选 `limit` 是**硬上限**：命中数超过它时抛
`LimitExceededError`，而不是悄悄只返回一部分（避免调用方误以为拿到了
全部结果）。

## 4. 索引与性能

* 层级 `l`（`1..L`）：纬度 `2**l` 带 × 经度 `2**(l+1)` 带（赤道附近近似
  方格；第 1 级赤道格边约 5 004 km）。
* 每个点在**每一层各入一个格**（最细层唯一存储点对象，其余层只存 id
  成员集合，不复制点、也不会产生重复结果）。
* 查询按半径选"格边最接近半径"的一层，只读该层与球面盖相交的少量格，
  再对候选点做精确 haversine 过滤。扫描格数近似 `O((r/edge)^2)` 的小
  常数，与总点位数无关。
* 球面盖候选格用解析几何精确构造（每纬度带求盖的最大经度跨度，含内部
  极值 `sin φ* = sin φ₀/cos c`），保证候选集是结果的**安全超集**。

性能基准（Apple Silicon，10 万全球均匀点，`index_levels=16`，数量级
参考）：

| 查询 | 中位延迟 |
| --- | --- |
| 100 m – 50 km 半径 | 约 0.006–0.007 ms |
| 500 km 半径 | 约 0.03 ms |
| 5 000 km 半径 | 约 8 ms |
| 全球（对跖半径） | 约 25 ms（本质上要触及全部点） |
| 插入 | 约 21 µs/点 |

`tests/test_scale.py` 以"比线性扫描快至少 10×"与"候选剪枝比例"做相对
断言，避免依赖具体机器的绝对时间。

## 5. 换日线与极点

* 经度区间在**展开（unwrapped）坐标**下计算，再按经度带数取模，跨
  `±180°` 的盖会同时读到缝两侧的格且不重复（用 set 去重）。
* 纬度带在两极 clamp；当球面盖包含某极点时，盖在该极侧的高纬平行线上
  是**整环覆盖**，整环阈值纬度为：
  * 北极侧：`φ ≥ π − c − φ₀`
  * 南极侧：`φ ≤ c − π − φ₀`
  这些纬度带按完整经度环扫描，其余部分按有限经度区间扫描。
* 盖心恰在极点且半径为 0 时，该极点上**任意经度**的点距离都是 0
  （极点处经度无意义），会扫描整个极纬度带。
* 正确性由 `tests/test_index.py` 与 `tests/test_service.py` 中数千个
  随机 + 边界案例的"暴力全量对照"固定：不漏点、不重复、与顺序无关。

## 6. 并发语义

* 所有写操作持有独占写锁；所有查询持有共享读锁
  （`geo_proximity/rwlock.py`，写者优先，防止写饥饿）。
* 一次查询从头到尾看到的是**同一个不可变点位集合**：不会看到半新半旧
  的更新；一个点从一个格移动到另一个格也绝不会被返回两次。
* `bulk_load` 是**原子**的：先在锁外完成全部坐标/批量大小校验，再在单
  个写锁内应用；批次内重复 id 与库内已存在 id 都按同一 `on_duplicate`
  策略处理。校验失败时存储完全不变。
* 并发测试（`tests/test_concurrency.py`）在持续 update/insert/delete 下
  校验：结果 id 唯一、距离有序且真实满足半径、库存数量与索引一致、
  批量操作只见"前/后"两种规模。

## 7. Python API

```python
from geo_proximity import ProximityService, ServiceConfig, OnDuplicate

svc = ProximityService(ServiceConfig(index_levels=16))

svc.insert(("a", 39.90, 116.40, {"name": "x"}))          # 重复 id 默认报错
svc.insert(("a", 39.91, 116.41), on_duplicate=OnDuplicate.REPLACE)
svc.upsert(("b", 31.23, 121.47))
svc.update("b", latitude=31.24, merge_attributes=False,
           attributes={"k": "v"})
svc.delete("b")
n = svc.bulk_load([("c", 0.0, 0.0), ("d", 1.0, 1.0)],
                  on_duplicate=OnDuplicate.SKIP)

for s in svc.radius_query(39.90, 116.40, 5000.0):   # distance <= 5 km
    print(s.point.id, s.distance_m)

for s in svc.nearest(39.90, 116.40, limit=10, max_distance_m=50_000.0):
    print(s.point.id, s.distance_m)                 # (距离, id) 升序
```

点可以用 `Point`、dict（`id/latitude/longitude/attributes`）或元组
`(id, lat, lon[, attrs])` 传入。

异常（均继承 `ProximityError`）：
`InvalidCoordinatesError`、`InvalidParameterError`、`DuplicateIdError`、
`PointNotFoundError`、`LimitExceededError`。

## 8. HTTP API

启动：`uv run uvicorn main:app`（或 `uvicorn main:app`）。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 健康检查与当前点位数 |
| GET | `/config` | 当前距离口径、排序规则与配置项 |
| POST | `/points?on_duplicate=error|replace|skip` | 插入一个点 |
| PUT | `/points/{id}` | upsert（body.id 须与路径一致） |
| PATCH | `/points/{id}` | 改坐标/属性（`merge_attributes` 控制合并） |
| DELETE | `/points/{id}` | 删除（返回 `deleted` 布尔） |
| GET | `/points/{id}` | 查询单个点 |
| POST | `/points/bulk` | 批量载入（`{"points":[...],"on_duplicate":...}`） |
| GET | `/query/radius?latitude&longitude&radius_m&limit` | 半径查询 |
| GET | `/query/nearest?latitude&longitude&limit&max_distance_m` | 最近邻 |

错误码：`400` 非法坐标/参数、`404` 点位不存在、`409` 重复 id、
`413` 触发配置上限、`422` 请求体/查询参数校验失败。

```bash
curl -s "http://127.0.0.1:8000/query/nearest?latitude=39.9&longitude=116.4&limit=3"
```

## 9. 开发与测试

```bash
uv sync
uv run pytest            # 全部测试（几何/索引/服务/并发/HTTP/规模）
uv run uvicorn main:app  # 启动服务
```

测试模块：

* `tests/test_geo.py`：距离常量、haversine、换日线/极点、盖/环阈值
* `tests/test_index.py`：候选超集、无重复、顺序无关、边界（参数化多层级）
* `tests/test_service.py`：CRUD、重复 id、批量原子性、半径/最近邻、上限、
  换日线/极点、数千次暴力对照
* `tests/test_concurrency.py`：并发一致性
* `tests/test_api.py`：HTTP 状态码与错误映射
* `tests/test_scale.py`：候选剪枝与相对线性扫描的加速比、内存结构
