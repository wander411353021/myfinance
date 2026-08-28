# AGENTS.md — 工作规则（每次任务必须严格执行）

本文件定义了本仓库的固定工作环境与强制工作流，任何 agent / 助手在此仓库内执行任务时必须遵守。

## 0. 未来函数红线（最高优先级，任何情况下不可触碰）

- **实盘信号 / 回测买入逻辑严禁任何形式未来函数（lookahead bias）**。这是全仓库最高红线，
  优先级高于一切策略、胜率、收益、效率考量。宁可少赚钱，不可用未来信息。
- **判定标准**：信号在产生时点（黄金坑=出坑日 lch 收盘）必须能用「当日及以前」数据完整复现。
  截断一致性测试（把 K线截断到信号日重跑检测，信号仍成立）必须 **100% 通过**。
- **自检工具**：`golden_pit_lookahead_check.py`（截断到出坑日重跑 detect_golden_pit）。
  任何新信号逻辑 / 任何 AI（含 reasonix / 豆包 / 其他）产出的实盘信号，上线前必须过此自检。
- **典型未来函数形态（一律禁止）**：
  1. 用未来信息选股、却从过去时点算收益（例：用"出坑后7天放量堆"选股、从出坑日算 r20——已证伪，2026-08-28）
  2. pass2 类"重算/回溯"机制把信号要素（坑底/出坑日）定义在"确认数据出现"之前（幽灵信号）
  3. 用 t+1..t+k 的信息决定 t 时刻是否买入 / 用整段历史事后标注当作实时信号
  4. 未排序/顺序混乱的数据源导致索引错位（如 tdx all_pages 乱序）
- **惩罚（对任何来源一视同仁，reasonix / 第三方 / AI 均适用）**：
  含未来函数的信号 → 判定无效、**立即废弃并禁止采用**；相关结论/文档/回测数字一并作废重算；
  违规项必须记录进「已证伪路径清单」与违规档案，并作为后续交叉验证的对照负例。
  **reasonix 输出的任何胜率/结论，默认先怀疑、必须过截断自检才可采信。**
- 实盘可执行信号 = 只有通过 100% 截断一致性自检的信号。

## 1. 工作目录（唯一工作区）

- **所有开发、回测、分析、文件操作默认在本仓库目录内进行**，不要将工作产物散落到 `/tmp` 或其他无关目录：
  `/home/user/.super_doubao/super-doubao-runtime/workspace/pressure-level-algorithm`
- 本目录即 Gitee 仓库 `polo4111/pressure-level-algorithm` 的本地副本，**任何代码/文件改动后必须同步回 Gitee**（见第 2 节）。

## 2. Git 工作流（修改前强制同步 + 任务收尾推送，硬性动作）

0. **修改前先同步（强制前置动作）**：每次开始任何修改/开发/分析前，必须先确认本地与远端一致：
   - `git status` 检查本地是否有未提交改动
   - `git fetch origin` 后对比本地 `master` 与 `origin/master`
   - **本地落后远端** → 先 `git pull --rebase origin master` 同步到最新，再开始修改
   - **本地有未提交改动** → 先 `git stash`（或先 commit 保存），同步完成后 `git stash pop` 恢复，再继续
   - 确认本地与 origin/master 一致、工作区干净，才允许动手修改

1. **工作目录**：`cd /home/user/.super_doubao/super-doubao-runtime/workspace/pressure-level-algorithm`
2. **Git 身份**：若 `.git/config` 丢失身份，必须重设：
   - `git config user.name "polo4111"`
   - `git config user.email "18326161185@noreply.gitee.com"`
3. **推送方式**：SSH（本机已配置 polo4111 密钥，remote 为 `git@gitee.com:polo4111/pressure-level-algorithm.git`）。
   推送时若默认 SSH 不生效，使用：
   `export GIT_SSH_COMMAND="ssh -i /home/user/.ssh/id_ed25519 -o StrictHostKeyChecking=no"`
4. **每次任务/阶段完成后必须**：`git add -A` → `git commit`（中文描述本次改动）→ `git push origin master`。
5. **提交信息规范**：中文，一句话概括改动（如 `feat: ...` / `fix: ...` / `chore: ...`）。
6. **禁止入库**运行期垃圾文件（锁文件、Chromium 缓存、临时 json、`__pycache__`、截图缓存等），规则已在 `.gitignore` 中；新发现的运行期产物一律追加到 `.gitignore`，而不是强行提交或删除。
7. 提交前先 `git status` 检查，确认没有误带入无关文件。

## 3. 黄金坑策略核心资产（防止口径漂移）

| 资产 | 路径 |
|---|---|
| 最终策略（信号扫描+6维评分+20天持有） | `golden_pit_final_strategy.py` |
| 核心算法模块（数据获取+信号检测） | `golden_pit_v2_backtest.py` |
| 股票池 300 只 / 1000 只 | `stock_pool_300.txt` / `stock_pool_1000.txt` |
| **skill 统一权威目录** | `.reasonix/skills/golden-pit-strategy/`（git 管理，所有改动在此进行） |
| 系统加载副本 | `workspace/.user_skills/golden-pit-strategy/`（勿直接改，跑 `sync_skills.sh` 同步） |
| 同步脚本 | `sync_skills.sh`（仓库权威 → 系统副本，改完 skill 必跑） |

**策略核心口径**（与 `golden-pit-strategy` skill 保持一致，改动需经用户确认并同步回 skill）：
- 信号【规则F，2026-08-28 定版】：250日对数回归 z< -1.5（60日 std）+ 快启动≤5天 + 出坑确认
  （**坑长≥8 约束已移除**——短坑V反胜率更高；坑长<8 是漏报元凶，详见 skill）
- 执行：出坑日收盘买入，**持有 20 天固定**（60 天持有弱市胜率崩塌，禁用）
- **放量堆加仓确认 = 已废弃**（未来函数，真实胜率 56.8% < 不买 82.4%，仅作事后标注）
- 6 维评分 = 不采用（本地数据无排序能力）
- 实盘前必过 `golden_pit_lookahead_check.py` 截断一致性自检（见第 0 节红线）

## 3.5 skill 统一管理（2026-08-28 polo4111 定规）

- **唯一权威 = 仓库 `.reasonix/skills/`**（git 管理，带版本历史）。系统加载副本 `workspace/.user_skills/` 只是部署副本，**不直接改**。
- **修改 skill 流程**：改仓库 `.reasonix/skills/` 下文件 → `bash sync_skills.sh` 同步到系统副本 → git 提交推送。
- 系统级 `.user_skills/golden-pit-strategy.bak_v1` 为合并前的旧版备份（只读存档，勿动）。
- 当前统一 skill：`golden-pit-strategy`（规则F 口径 + 未来函数红线 + 自检记录，见文件内）。

## 4. 数据与运行

- 数据源：新浪财经日K（`fetch_kline_sina`，自动清除代理环境变量）；大盘沪深300 相对强度数据自动获取并缓存。
- 依赖：按 `requirements.txt` 安装；首次使用需 `pip install -r requirements.txt`。
- 股票池抽样：1000 只为全市场分层抽样，覆盖次新股、不同市值，可扩容。
