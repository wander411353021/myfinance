# AGENTS.md — 工作规则（每次任务必须严格执行）

本文件定义了本仓库的固定工作环境与强制工作流，任何 agent / 助手在此仓库内执行任务时必须遵守。

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
| reasonix skill | `.reasonix/skills/golden-pit-strategy/` |
| 本机 skill 副本 | `.user_skills/golden-pit-strategy/` |

**策略核心口径**（与 `golden-pit-strategy` skill 保持一致，改动需经用户确认并同步回 skill）：
- 信号：250日对数回归 z< -1.5（60日 std）+ 快启动≤5天 + 坑长≥8 + 出坑确认
- 6 维评分：缩量挖坑25 / 放量出坑25 / 坑深适度15 / 相对强度15 / 回归斜率10 / 坑底缩量10
- 执行：评分≥52 放弃、42-51 加仓、30-41 正常、<30 观望
- **持有期 20 天固定**（60 天持有弱市胜率崩塌至 35%-44%，禁用）

## 4. 数据与运行

- 数据源：新浪财经日K（`fetch_kline_sina`，自动清除代理环境变量）；大盘沪深300 相对强度数据自动获取并缓存。
- 依赖：按 `requirements.txt` 安装；首次使用需 `pip install -r requirements.txt`。
- 股票池抽样：1000 只为全市场分层抽样，覆盖次新股、不同市值，可扩容。
