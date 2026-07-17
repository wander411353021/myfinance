# eltdx 接口总览与使用说明

> 来源：<https://github.com/hugo2046/eltdx>（pip 包名 `eltdx`，PyPI 最新 1.0.2，需 Python ≥ 3.10）
> 适用：通达信（TDX）7709 行情协议 + 7615 F10 资料网关的 Python 客户端，官方内置 MCP 服务。

`eltdx` 是一个**统一的 A 股行情协议 Python 库**，覆盖：行情快照、分时、逐笔成交、K 线、集合竞价、股本变迁/除权除息、财务基础信息、F10 资料、题材板块等。返回结构统一的 dataclass，并支持 MCP 工具给 Agent 调用。

---

## 1. 安装

```bash
pip install eltdx                 # 基础库
pip install "eltdx[mcp]"         # 额外带 MCP 服务依赖（stdio / HTTP）
```

源码安装：

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e .
```

安装后可用命令行：`eltdx-smoke`（连通性自检）、`eltdx-f10-smoke`、`eltdx-mcp`（stdio）、`eltdx-mcp-http`（HTTP）。

> ⚠️ 许可证：ELTDX Research-Only License（早期版本为 MIT）。**仅限个人学习、协议研究与非商业研究使用**，禁止商业、付费、生产、转售等用途。

---

## 2. 总入口 `TdxClient`

所有接口通过 `TdxClient` 的统一入口访问，既可用顶层便捷方法，也可访问分组子模块（`client.codes` / `client.quotes` / `client.bars` …）。

```python
from eltdx import TdxClient

# 1) 默认连接包内 7709 主站
with TdxClient(timeout=3) as client:
    q = client.get_quote(["sz000001", "sh600000"])

# 2) 指定单个主站
with TdxClient(host="116.205.183.150:7709", timeout=3) as client:
    ...

# 3) 连接池 + 启动主站测速
with TdxClient.from_hosts(pool_size=2, probe_hosts=True, timeout=3) as client:
    ...

# 4) 离线/单元测试内存客户端
with TdxClient.in_memory() as client:
    ...
```

### 连接构造参数（含 `from_hosts`）

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `host` | `None` | 指定单个 7709 主站（如 `"116.205.183.150:7709"`） |
| `hosts` | `None` | 指定多个 7709 主站 |
| `timeout` | `8.0` | 单次 socket 请求等待秒数 |
| `pool_size` | `1` | 连接池连接数 |
| `batch_size` | `80` | `get_quote()` 自动拆批大小 |
| `probe_hosts` | `False` | 启动时是否先 TCP 测速并排序（默认关，避免启动等待） |
| `heartbeat_interval` | `30.0` | 后台心跳秒数；`None` 或 ≤0 关闭 |

> 不传 `host`/`hosts` 时，读取包内 `tdx_server.json` 默认主站；缺失则回退内置列表。
> 真实 socket 默认每 30s 发 `0x0004` 心跳维持空闲连接（短脚本可忽略；`heartbeat_interval=None` 关闭）。

---

## 3. 顶层便捷方法（常用入口）

这组方法是对分组 API 的外层包装，日常使用优先用它们。

### 行情快照

| 方法 | 说明 / 底层 | 参数含义 |
| --- | --- | --- |
| `get_quote(codes)` | 批量行情快照（自动按 80 拆批，并补齐五档盘口） | `codes`：代码列表，如 `["sz000001","sh600000"]` |
| `get_quote_depth(codes)` | 直接取五档盘口 | `codes`：代码列表 |

### 代码表

| 方法 | 说明 | 参数含义 |
| --- | --- | --- |
| `get_count(market)` | 某市场代码数量 | `market`：`"sz"` / `"sh"` / `"bj"`（≠股票数量，是代码表条目数） |
| `get_codes(market, start=0, limit=1600)` | 分页代码表 | `market` 同上；`start` 起始行；`limit` 每页条数 |
| `get_codes_all(market)` | 全量代码表 | `market` 同上 |
| `get_a_share_codes_all()` | 全量 A 股代码（按 category 过滤） | — |
| `get_stock_codes_all()` | 全量股票代码 | — |
| `get_etf_codes_all()` | 全量 ETF 代码 | — |
| `get_index_codes_all()` | 全量指数代码 | — |

> 代码建议带市场前缀：`sz000001`、`sh600000`、`bj920001`。

### K 线

```python
client.get_kline("day", "sz000001", count=30)
client.get_kline("sz000001", "day", count=30)          # 两种参数顺序都支持
client.get_kline_all("day", "sz000001")                # 全量历史（分页）
client.get_adjusted_kline("day", "sz000001", adjust="qfq")
client.get_adjusted_kline("day", "sz000001", adjust="fixed_qfq", anchor_date="2024-06-03")
client.get_local_adjusted_kline_all("day", "sz000001", adjust="qfq")  # 本地复权校验
client.get_kline("1m", "sz000001", count=240)          # 分钟线
```

| 方法 | 参数 | 含义 |
| --- | --- | --- |
| `get_kline(period, code, count=30, adjust=None, anchor_date=None)` | `period` 周期；`code` 代码；`count` 根数；`adjust` 复权；`anchor_date` 定点复权基准日 | 单页 K 线 |
| `get_kline_all(period, code, adjust=None)` | 同上（不含 count） | 全量分页拉取 |
| `get_adjusted_kline(period, code, adjust, anchor_date=None)` | `adjust`/`anchor_date` 见下表 | 服务端复权 K 线 |

**周期 `period` 取值：**

| 值 | 含义 |
| --- | --- |
| `1m` `5m` `15m` `30m` `60m` | 分钟 K 线 |
| `day` | 日 K |
| `week` | 周 K |
| `month` | 月 K |
| `quarter` | 季 K |
| `year` | 年 K |
| `10m` `2d` `5s` | 协议层支持的自定义（分钟/N 日/N 秒），实际以服务端返回为准 |

**复权 `adjust` 取值：**

| 值 | 含义 |
| --- | --- |
| `None` / `none` | 不复权 |
| `qfq` / `front` | 前复权 |
| `hfq` / `back` | 后复权 |
| `fixed_qfq` / `fixed_front` | 定点前复权（需 `anchor_date`） |
| `fixed_hfq` / `fixed_back` | 定点后复权（需 `anchor_date`） |

### 分时 / 成交明细

```python
client.get_minute("sz000001")                  # 当日分时
client.get_history_minute("sz000001", "2026-05-20")   # 指定日期历史分时
client.get_trades("sz000001")                  # 当日成交明细
client.get_trades("sz000001", "2026-05-20")   # 指定日期历史成交明细
client.get_trades_all("sz000001", "2026-05-20")  # 全量历史成交明细
```

> 成交明细别名保留：`get_trade` / `get_trade_all` / `get_history_trade` / `get_history_trade_day`（与上面语义相同）。

### 集合竞价

| 方法 | 说明 | 参数 |
| --- | --- | --- |
| `get_call_auction(code)` | 当前交易日集合竞价明细（0x056a） | `code` 代码 |
| `get_auction_0925(code, date)` | 历史某日 09:25 竞价成交快照 | `code`；`date` 如 `"2026-05-20"` |

### 股本 / 除权除息 / 复权因子

| 方法 | 说明 | 参数 |
| --- | --- | --- |
| `get_gbbq(code)` | 股本变迁（旧名，底层 0x000f） | `code` |
| `get_xdxr(code)` | 除权除息记录整理 | `code` |
| `get_equity_changes(code)` | 股本变化历史 | `code` |
| `get_equity(code, date)` | 指定日期股本 | `code`；`date` |
| `get_factors(code)` | 本地复权因子序列 | `code` |
| `get_turnover(code, volume, unit="hand")` | 换手率 = 成交股数 / 流通股本 ×100 | `volume` 成交量；`unit="hand"`（手）或 `"share"`（股） |

### 低频缓存 & 调试

```python
client.get_count("sz", refresh=True)            # 强制刷新（低频缓存：代码数量/代码表/股本/财务）
client.get_codes_all("sz", refresh=True)
client.get_gbbq("sz000001", refresh=True)
client.get_finance_batch(["sz000001"], refresh=True)
client.clear_cache()                            # 清空全部低频缓存

client.get_gbbq("sz000001", include_raw=True)  # 保留原始十六进制 payload，便于协议排错
client.get_kline("day", "sz000001", include_raw=True)

from eltdx import to_json, to_jsonable
text = to_json(to_jsonable(client.get_quote("sz000001")), indent=2)
```

> 低频数据（代码数量/全量代码表/股本变迁/财务基础）自动内存缓存；实时行情/分时/成交/K 线不缓存。

---

## 4. 分组子模块 API

### `client.session` — 连接会话

| 方法 | 对应命令 | 说明 |
| --- | --- | --- |
| `handshake()` | `0x000d` | 握手，返回服务端日期时间、交易时段、主站名 |
| `heartbeat()` | `0x0004` | 心跳保活 |

### `client.codes` — 代码表（0x044e / 0x044d）

| 方法 | 参数 | 说明 |
| --- | --- | --- |
| `count(market)` | `market`：`"sz"/"sh"/"bj"` | 代码数量 |
| `list(market, start=0, limit=1600)` | `market`；`start`；`limit` | 分页代码表 |
| `all(market)` | `market` | 全量代码表 |

### `client.quotes` — 行情（0x054c / 0x0547 / 0x054b）

| 方法 | 参数 | 说明 |
| --- | --- | --- |
| `get_snapshots(codes)` 别名 `get` | `codes` 列表 | 批量快照（实盘仅稳定确认买一/卖一） |
| `list_by_category(category, sort_by=None, start=0, count=80, ascending=False)` | 分类名；排序字段；分页 | 分类行情列表 |
| `refresh(codes=None, cursors=None)` | 代码；游标字典 | 增量刷新请求（0x0547） |
| `get_depth(codes)` | 代码列表 | 直接取五档盘口 |
| `poll_push(timeout=0.0, parse=False)` | 超时秒；是否解析 | 读取一个推送帧 |
| `drain_pushes(parse=False)` | — | 取出队列内全部推送帧 |

### `client.bars` — K 线（0x052d）

| 方法 | 参数 | 说明 |
| --- | --- | --- |
| `get(code, period="day", start=0, count=800, adjust=None, anchor_date=None, include_raw=False)` | 代码；周期；起始；根数；复权；定点日 | 单页 K 线，返回 `KlineSeries` |
| `all(code, period="day", adjust=None, page_size=800, max_pages=200, include_raw=False)` | 同上；`page_size` 每页；`max_pages` 上限防死循环 | 分页全量 |

`KlineSeries` 主要字段：`period_name`（周期名）、`adjust_mode`（复权模式）、`bars[]`（记录列表，含 `time`/`open`/`high`/`low`/`close`/`volume_lots`(手)/`amount`(成交额)）。

### `client.minutes` — 分时

| 方法 | 对应命令 | 参数 | 说明 |
| --- | --- | --- | --- |
| `today(code, include_raw=False)` | `0x0537` | 代码 | 当日分时 |
| `history(code, trading_date, include_raw=False)` | `0x0fb4` | 代码；交易日 | 指定日期历史分时 |
| `recent(code, trading_date, include_raw=False)` | `0x0feb` | 代码；交易日 | 近期历史分时 |
| `aux(code, kind, include_raw=False)` | `0x051b` | 代码；`kind` 如 `"buy_sell_strength"`/`"volume_comparison"` | 分时副图 |
| `sparkline(code, selector=1, window=20, include_raw=False)` | `0x0fd1` | 代码；选择器；窗口 | 小走势图 |

### `client.trades` — 成交明细

| 方法 | 对应命令 | 参数 | 说明 |
| --- | --- | --- | --- |
| `today(code, start=0, count=1800, include_raw=False)` | `0x0fc5` | 代码；起始；条数 | 当日成交明细 |
| `history(code, trading_date, start=0, count=2000, include_raw=False)` | `0x0fc6` | 代码；交易日；分页 | 历史成交明细 |

### `client.auctions` — 集合竞价

| 方法 | 对应命令 | 参数 | 说明 |
| --- | --- | --- | --- |
| `series(code, include_raw=False)` | `0x056a` | 代码 | 当前交易日集合竞价明细 |

### `client.corporate` — 公司基础（0x000f / 0x0010）

| 方法 | 参数 | 说明 |
| --- | --- | --- |
| `capital_changes(code, include_raw=False)` | `code` | 股本变迁/除权相关数据 |
| `finance_batch(codes, fields=None, include_raw=False)` | `codes` 列表；`fields` 仅本地过滤返回字段 | 批量财务基础信息（`fields` 如 `["流通股本","total_shares"]`） |

### `client.limits` — 涨跌停（0x0452）

| 方法 | 参数 | 说明 |
| --- | --- | --- |
| `special(start_index=0)` | 表内行号 | 特殊品种涨跌停限制（分页） |
| `scan_special()` | — | 扫描建本地索引，便于按代码查 |

### `client.f10` / `F10Client` — F10 资料（7615/TQLEX HTTP 网关，独立于 7709）

走 HTTP 网关，无需 7709 握手。所有方法返回 `F10Response`，数据通常在 `response.rows` 或 `response.tables`。

```python
client.f10.company_profile("000034")      # 公司概况
client.f10.hot_topics("000034")           # 热点题材
client.f10.announcements("000034")        # 公告
client.f10.finance_report("000034")       # 财务报表
client.f10.valuation("000034")            # 估值
client.f10.business_composition()         # 主营构成
client.f10.stock_score()                  # 个股总评
client.f10.theme_market()                 # 题材概念行情
client.f10.northbound_holding()           # 沪深股通持仓
client.f10.call("CWServ.tdxf10_gg_gsgk", params=["8","000034",""])  # 通用 Entry 调用
```

> 轻量独立客户端：`from eltdx import F10Client; f10 = F10Client(timeout=3)` 直接调用上述方法。
> 完整 F10 方法表见官方 `docs/F10_7615.md`。

### `client.helpers` — 常用场景组合

| 方法 | 参数 | 说明 |
| --- | --- | --- |
| `stock_profile_table(codes)` | 代码列表 | 股票信息汇总表（合并行情+代码表+财务基础） |
| `stock_topics(code)` | 代码 | 个股全部概念板块 |
| `topic_stocks(seed_code, topic_name=None)` | 种子代码；板块名 | 某概念板块成分股 |
| `auction_data(code, date)` | 代码；日期 | 竞价数据（返回 `open_price`/`open_change_pct`/`open_amount` 等） |

```python
with TdxClient(timeout=3) as client:
    table = client.helpers.stock_profile_table(["sz000001","sh600000"])
    topics = client.helpers.stock_topics("000034")
    stocks = client.helpers.topic_stocks("000034", topic_name="存储芯片")
    auction = client.helpers.auction_data("sz000001", "2026-05-20")
```

### 工具类

- `WorkdayService`：交易日工具（判断/列举交易日）。
- `to_json()` / `to_jsonable()`：返回模型转 JSON。
- 低频缓存见 §3；`include_raw=True` 见 §3。

---

## 5. 返回结构与字段口径

- 统一 dataclass 返回，时间字段为原生 `date`/`datetime`；价格同时保留浮点值与 `*_milli` 整数值。
- K 线 `volume_lots` 单位为**手**，`amount` 为成交额。
- 完整字段含义与口径见官方 `docs/FIELD_REFERENCE.md`；底层命令号/payload 见 `docs/COMMANDS_7709.md`。

---

## 6. MCP 服务（给 Agent 用）

`eltdx` 官方内置 MCP 服务，两种运行方式：

- **stdio**（`eltdx-mcp`）：本地使用，默认免鉴权。
- **HTTP**（`eltdx-mcp-http` / FastAPI 部署）：远程使用，支持静态 token 鉴权，按 JSONL 记录 `client_id` 与来源 IP。

### 启动

```bash
pip install "eltdx[mcp]"
eltdx-mcp                          # stdio，占用当前终端

# HTTP（开发/简单部署）
eltdx-mcp-http                    # 默认 127.0.0.1:8000
# 正式部署
uvicorn eltdx.mcp:create_app --factory --host 0.0.0.0 --port 8000   # 端点挂在 /mcp
```

### HTTP 鉴权与环境变量

| 环境变量 | 说明 | 默认 |
| --- | --- | --- |
| `ELTDX_MCP_TOKENS` | 静态 token 配置（JSON：`{token: client_id}` 或 `{token: {client_id, scopes}}`）；**HTTP 入口必填，未配置拒绝启动** | 无 |
| `ELTDX_MCP_ALLOWED_HOSTS` | 允许调用方自定义 7709 主站白名单（逗号分隔）；未配则禁止自定义 host | 无 |
| `ELTDX_MCP_TRUST_PROXY` | `1`/`true` 时信任 `X-Forwarded-For`（仅可信代理后） | 关闭 |
| `ELTDX_MCP_HOST` | 监听地址 | `127.0.0.1` |
| `ELTDX_MCP_PORT` | 监听端口 | `8000` |
| `ELTDX_MCP_LOG` | JSONL 访问日志路径 | `logs/mcp_access.jsonl` |

### 官方 MCP 工具列表（9 个）

| 工具名 | 作用 | 主要参数 |
| --- | --- | --- |
| `eltdx_quote` | 查询一个或多个股票行情快照 | `codes`(列表), `timeout`(默认 8.0) |
| `eltdx_kline` | 查询 K 线/周期线（支持复权） | `code`, `period`, `count`, `adjust`, `timeout` |
| `eltdx_stock_profile` | 汇总股票表头（行情+代码表+财务） | `codes`, `timeout` |
| `eltdx_stock_topics` | 查询某股全部题材/概念板块 | `code`, `timeout` |
| `eltdx_topic_stocks` | 查询某题材/概念板块的成分股 | `seed_code`, `topic_name`, `sort_by`(如 `zdf`) |
| `eltdx_company_profile` | F10 公司概况 | `code`, `timeout` |
| `eltdx_hot_topics` | F10 热点题材明细 | `code`, `timeout` |
| `eltdx_auction_0925` | 指定日期 09:25 竞价成交快照 | `code`, `date`, `timeout` |
| `eltdx_docs_index` | 返回项目主要文档入口 | — |

**调用示例（JSON）：**

```json
{ "codes": ["sz000001","sh600000"], "timeout": 3 }
{ "code": "sz000001", "period": "day", "count": 120, "adjust": "qfq" }
{ "code": "000034", "timeout": 3 }
{ "seed_code": "000034", "topic_name": "存储芯片", "sort_by": "zdf" }
```

> MCP 工具统一连接参数：`timeout`（默认 8.0s）、`host`（可选，指定单个 7709 主站）。F10 工具走 7615/TQLEX HTTP 网关，不需要 7709 握手。
> ⚠️ 安全：HTTP 必须置于 HTTPS 之后（静态 token 是明文 Bearer）；`host` 经 HTTP 暴露有 SSRF 风险，默认禁止自定义，放开请用 `ELTDX_MCP_ALLOWED_HOSTS` 白名单。

### 补充：第三方 tdx-mcp 插件（非官方）

社区另有基于 `eltdx` 封装的 `tdx-mcp` 插件（如 lobehub 上的 `neoooo0909-tdx-mcp`），对外暴露 **23 个 MCP 工具**，在原 18 个 eltdx 原生工具基础上补充了 `tdx_get_finance`（财务数据）、`tdx_get_index_quote`（指数行情）、`tdx_get_futures_quote`（期货行情）、`tdx_get_block_info`（板块成分）、`tdx_get_company_info_categories`（F10 历史期间目录）等。若需在 Agent 里拿更全的现成工具，可一并参考；其底层仍依赖 `eltdx` + `TDXDataFetcher`。

---

## 7. 快速上手示例

```python
from eltdx import TdxClient

with TdxClient(timeout=3) as client:
    quote = client.get_quote(["sz000001", "sh600000"])
    bars  = client.get_kline("day", "sz000001", count=30)
    minute = client.get_minute("sz000001")
    ticks  = client.get_history_trade_day("sz000001", "2026-05-20")

print(quote[0])          # 快照对象
print(bars.bars[-1])     # 最后一根 K 线
```

```python
# F10 资料
from eltdx import TdxClient
client = TdxClient(timeout=3)
profile = client.f10.company_profile("000034")
topics  = client.f10.hot_topics("000034")
print(profile.rows[0])
print(topics.rows[:3])
```

---

## 8. 官方文档索引

| 文档 | 内容 |
| --- | --- |
| `docs/API_REFERENCE.md` | API 参数与返回值（本文主要来源） |
| `docs/METHOD_REFERENCE.md` | 逐方法参数与解析字段 |
| `docs/FIELD_REFERENCE.md` | 字段含义与口径 |
| `docs/F10_7615.md` | F10 资料方法表 |
| `docs/MCP.md` | MCP 工具与部署 |
| `docs/EXAMPLES.md` | 可直接复制示例 |
| `docs/DEBUG_GUIDE.md` | 连接/服务器/raw 数据排查 |
| `docs/ARCHITECTURE.md` | 架构说明 |
| `docs/methods/*.md` | 37 个底层命令逐条说明（含 0x 命令号） |
