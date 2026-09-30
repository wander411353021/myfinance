#!/bin/bash
# ============================================================
# GitHub myfinance 镜像一键同步脚本 (polo4111)
# 逻辑: 主仓库 rsync -> /tmp/myfinance_gh 克隆 -> commit -> force push
# 策略: GitHub master 以 gitee 为唯一权威, force push 保证两端完全一致
# 依赖: ~/.ssh/github_ed25519 (备份在仓库 .ssh_backup/, 环境重置后先恢复)
# 用法: bash sync_github.sh
# ============================================================
set -e
REPO="$(cd "$(dirname "$0")" && pwd)"
CLONE="/tmp/myfinance_gh"
GH_URL="git@github.com:wander411353021/myfinance.git"

# 1. 重建克隆(保证与远端最新一致, 明确检出 master 分支)
rm -rf "$CLONE"
echo "[1/4] clone github 远端(master) ..."
timeout 180 git clone --branch master --single-branch "$GH_URL" "$CLONE"

# 2. rsync 同步代码/skills(排除私钥/缓存/图集)
echo "[2/4] rsync 本地代码 -> 克隆 ..."
rsync -av --exclude='.git' --exclude='.ssh_backup' --exclude='__pycache__' \
      --exclude='*.pyc' --exclude='result/' --exclude='*.png' \
      --exclude='hsperfdata_root' --exclude='node-compile-cache' \
      --exclude='playwright-artifacts-*' --exclude='persistent-sync' \
      "$REPO/" "$CLONE/" | tail -3

# 3. 清理残留缓存目录 + commit
cd "$CLONE"
rm -rf hsperfdata_root node-compile-cache playwright-artifacts-* persistent-sync
git config user.name "polo4111"
git config user.email "18326161185@163.com"
git add -A
if git diff --cached --quiet; then
  echo "无改动, 跳过 commit"
else
  git commit -m "sync: 镜像同步(以gitee为准) $(date +%Y-%m-%d)"
fi

# 4. force push(GitHub 以 gitee 为准, 双分支 master+main 都要同步:
#    main 是 GitHub 默认分支, 用户网页端默认看 main, 只推 master 会看到旧代码)
echo "[4/4] force push github (master + main) ..."
timeout 180 git push --force origin master
timeout 180 git push --force origin master:main
echo "✅ 已同步: gitee -> GitHub myfinance (master + main)"
git log --oneline -1
