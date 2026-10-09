# -*- coding: utf-8 -*-
---
name: git-dual-push
description: 强制多仓上传（gitee + ModelScope + GitHub）。每当对 pressure-level-algorithm 仓库代码/skills 做了修改并需要提交上传时，除 push gitee(origin) 外，还**必须**同时同步推送到 ModelScope BigRabbit 创空间仓库和 GitHub myfinance 镜像仓库；本 skill 记录三仓 remote 配置、一键同步脚本用法、排除项与 SSH key/环境重置恢复步骤。违反"只传部分仓不传全"视为未完成上传。
---

# 强制多仓上传 git-dual-push (2026-09-30 polo4111 定版, 2026-09-30 升级为三仓)

## 为什么存在
用户要求: 以后每次上传代码/skills, **gitee + ModelScope + GitHub 三个仓库都要推**, 只推一个/两个不算完成。
- gitee: 主开发仓库(SSH, origin) `git@gitee.com:polo4111/pressure-level-algorithm.git`
- ModelScope: 创空间部署仓库(HTTPS+token, modelscope), 保留 app.py/.streamlit/runtime.txt 等部署文件
- GitHub: 镜像仓库(SSH, github) `git@github.com:wander411353021/myfinance.git`, **以 gitee 为准覆盖(force push)**

## 权威仓库
```
/home/user/.super_doubao/super-doubao-runtime/workspace/pressure-level-algorithm
(等价 /sandboxdata/workspace/file/pressure-level-algorithm)
```

## 三个 remote(主仓库已配置, 环境重置后需重建)
```
origin     = git@gitee.com:polo4111/pressure-level-algorithm.git        (SSH, gitee)
modelscope = https://oauth2:<TOKEN>@www.modelscope.cn/studios/polo411/BigRabbit.git  (HTTPS+token)
github     = git@github.com:wander411353021/myfinance.git               (SSH key: ~/.ssh/github_ed25519)
```
- modelscope token 属敏感凭据: **严禁明文写进任何入库文件**(SKILL/脚本/提交信息);
  只存在主仓库 `.git/config` 的 remote 配置里(不入库)。环境重置后 token 丢失,
  需向用户索取新 token 重建 remote, 或从安全备份恢复。
- GitHub key(私钥): 备份在仓库 `.ssh_backup/github_ed25519`(已被 .gitignore 排除, 不入库);
  `~/.ssh/config` 中已配置 `Host github.com → IdentityFile ~/.ssh/github_ed25519`。

## 强制上传流程(每次修改后必须走完)
```bash
cd <仓库>
git status -sb          # 1. 先看本地与远端是否一致, 不一致先 fetch/同步再改(用户红线)
git add <改动文件>       # 2. 只 add 代码/skills, 不 add 中间产物
git commit -m "<详细提交信息>"
git push origin master  # 3. gitee 先推
bash sync_modelscope.sh # 4. ModelScope 再推(脚本见下), 输出含 "✅" 才算成功
bash sync_github.sh     # 5. GitHub 再推(脚本见下), 输出含 "✅" 才算成功
```
- modelscope 与本地历史**不同源**: 严禁在主仓库直接 `git push modelscope master`(会被 unrelated histories 拒绝
  或需 force 造成部署文件丢失); 必须走 sync_modelscope.sh 的 clone+rsync+push 通道。
- GitHub 以 gitee 为准: sync_github.sh 内部 rsync 后 commit + `git push --force`(覆盖 GitHub master,
  reasonix 在 GitHub 上的独立提交会被取代——用户已确认此策略)。
- gitee SSH key 环境重置后先恢复(见 golden-pit-strategy 环境手册: .ssh_backup/ 恢复, 同一条命令内立即 push)。

## sync_modelscope.sh(仓库根目录, 一键同步)
逻辑: 主仓库 rsync -> /tmp/bigrabbit_ms 克隆(排除 .git/.ssh_backup/缓存/result/图集) -> commit -> push。
- 保留远端部署文件(app.py/.streamlit/runtime.txt/README.md), 只覆盖代码/skills 同名文件+新增文件。
- token 从主仓库 `.git/config` 的 modelscope remote 自动提取(不入库), 无需手动输入。
- 完成后验证: `git log --oneline -1` 为最新同步提交, `git status -sb` 干净。

## sync_github.sh(仓库根目录, 一键同步)
逻辑: 主仓库 rsync -> /tmp/myfinance_gh 克隆(排除同上) -> commit -> `git push --force origin master`。
- GitHub master 始终以 gitee 为唯一权威, force push 保证两端完全一致。
- 依赖 ~/.ssh/github_ed25519(备份于 .ssh_backup/, 环境重置后先恢复)。

## ⚠ 教训 1: GitHub 默认分支是 main, 必须双分支同步(2026-09-30 实测)
- **现象**: GitHub 仓库 `wander411353021/myfinance` 的**默认分支是 `main`**(不是 master),
  网页端打开仓库默认看到 main。此前一直只同步 master → 用户网页上看到旧代码
  (streamlit 无 show_chip 修复), 误以为"GitHub 没更新"。
- **修复**: 每次 sync_github.sh 除 force push master 外, **还必须 force push main**(同一内容):
  ```bash
  git push --force origin master
  git push --force origin main
  ```
- **验证**: 远端直查 `git ls-remote git@github.com:wander411353021/myfinance.git`
  需同时看到 main/master 两个 refs 均为最新。同步脚本后续迭代时把 main 一并纳入。

## ⚠ 教训 2: 创空间缺依赖时异常被吞、页面无提示(2026-09-30 实测)
- **现象**: ModelScope 创空间重启后 COST10-90 筹码淡蓝带不显示, 代码已是新版、页面无任何报错。
- **根因**: 筹码计算 `chip_panel.cost_series` 依赖 **fengwo** 库(`import fengwo`),
  但创空间 `requirements.txt` 缺该包 → 云端 `ModuleNotFoundError` 被代码里
  `except Exception: print(...)` 吞掉(只打印在服务端日志, 用户页面不可见) → 图照常出但无筹码带。
- **修复**: requirements.txt 补 `fengwo==0.0.7`; 创空间重启后自动安装。
- **教训**: ① 新增第三方库必须同步更新 requirements.txt 并三仓推送;
  ② 云端"图能出但某个图层缺失+无报错"= 先查被 except 吞掉的依赖/数据链路,
     再看服务端日志, 不要只怀疑部署/缓存; ③ 提交前本地模拟创空间依赖清单跑一遍。

## 排除项(严禁上传)
- `.ssh_backup/`(SSH 私钥, 最高优先级禁止; 已在 .gitignore 第43行)
- `result/`、`*.png` 图集、`hsperfdata_root/`、`node-compile-cache/`、`playwright-artifacts-*/`、`persistent-sync/`
- `__pycache__/`、`*.pyc`、`.git/`

## 红线(与 golden-pit-strategy 一致, 不可触碰)
- 实盘信号/画图绝不允许未来函数(第i天只用≤i数据); 违反者狠狠惩罚。
- 换手率/筹码运算必须用逐日历史流通股本(capital_changes 重建), 严禁当前快照。
- 上传前必须检查: 本地与 gitee origin 一致(先同步再改); **三仓都推送成功**才算完成。

## 环境重置恢复手册(2026-09-30 验证)
1. eltdx: `pip3 install eltdx -q --timeout 180 --retries 5 --default-timeout 180`
2. gitee SSH: 从仓库 `.ssh_backup/` 恢复 id_ed25519 后**同一条命令内立即 push**
3. GitHub SSH: 从仓库 `.ssh_backup/` 恢复 github_ed25519 + github_ed25519.pub 到 ~/.ssh/,
   并重建 `~/.ssh/config` 的 github.com 段(IdentityFile ~/.ssh/github_ed25519, IdentitiesOnly yes)
4. modelscope remote: 向用户索取新 token, `git remote add modelscope https://oauth2:<TOKEN>@www.modelscope.cn/studios/polo411/BigRabbit.git`
5. 先 `git fetch origin` + `git status -sb` 确认一致再开工

## GitHub 推送回退法 + 实测环境(2026-10-09 reasonix)
**本机现状**: `~/.ssh/` 只有 id_rsa(无 github_ed25519), 且仓库 `.ssh_backup/` **不存在** → `sync_github.sh` 的 SSH 路径必然失败。
**网络实测**: github.com:443 极不稳定 —— 连续 4 次 `Failed to connect ... after 21xxx ms: Timed out`, 或 `OpenSSL SSL_read: Connection was reset, errno 10054`; 偶发成功。
本地代理探测(7890/7897/10809/1080/8889/33210)全部未开放, 无 http_proxy 环境变量。

**可用回退(HTTPS remote + 强制覆盖 + 多次重试)**:
```bash
cd <仓库>
# 主仓库 github remote 已是 HTTPS: https://github.com/wander411353021/myfinance.git
for i in 1 2 3 4; do
  GIT_TERMINAL_PROMPT=0 timeout 150 git -c http.version=HTTP/1.1 push --force github master && break
  sleep 10
done
GIT_TERMINAL_PROMPT=0 timeout 150 git -c http.version=HTTP/1.1 push --force github master:main
# 验证
timeout 60 git -c http.version=HTTP/1.1 ls-remote github master main
```
**要点**:
- GitHub 端是"镜像历史"(每次 sync 生成新 commit, 与 gitee **不同源**) → 必须 `--force`
- 网络不稳时**必须循环重试**(单次成功率低); 一次成功即 break
- 无法确认远端最新时: `git log -1 github/master`(本地追踪引用) + `.git/refs/remotes/github/` 的 mtime 可给出"最后已知状态"
- GIT_TERMINAL_PROMPT=0 防止 HTTPS 要凭据时挂起; timeout 防止 443 长等待
