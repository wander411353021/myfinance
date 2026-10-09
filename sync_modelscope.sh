#!/bin/bash
# ============================================================
# ModelScope BigRabbit 一键同步脚本 (polo4111)
# 逻辑: 主仓库 rsync -> /tmp/bigrabbit_ms 克隆 -> commit -> push
# 用途: git-dual-push skill 的强制第二步(每次上传 gitee 后必须运行)
# 用法: bash sync_modelscope.sh
# ============================================================
set -e
REPO="$(cd "$(dirname "$0")" && pwd)"
CLONE="/tmp/bigrabbit_ms"

# token 来源: 优先主仓库 .git/config 的 modelscope remote(不入库), 其次环境变量
CFG_URL="$(git -C "$REPO" config --get remote.modelscope.url 2>/dev/null || true)"
if [ -n "$CFG_URL" ]; then
  MS_URL="$CFG_URL"
elif [ -n "$MODELSCOPE_TOKEN" ]; then
  MS_URL="https://oauth2:${MODELSCOPE_TOKEN}@www.modelscope.cn/studios/polo411/BigRabbit.git"
else
  echo "❌ 找不到 modelscope 凭据: 请先 git remote add modelscope 或 export MODELSCOPE_TOKEN" >&2
  exit 1
fi

# 1. 重建克隆(保证与远端最新一致)
rm -rf "$CLONE"
echo "[1/4] clone modelscope 远端 ..."
timeout 300 git clone "$MS_URL" "$CLONE"

# 2. rsync 同步代码/skills(排除私钥/缓存/图集)
echo "[2/4] rsync 本地代码 -> 克隆 ..."
rsync -av --exclude='.git' --exclude='.ssh_backup' --exclude='__pycache__' \
      --exclude='*.pyc' --exclude='result/' --exclude='*.png' \
      --exclude='hsperfdata_root' --exclude='node-compile-cache' \
      --exclude='playwright-artifacts-*' --exclude='persistent-sync' \
      "$REPO/" "$CLONE/" | tail -5

# 3. 清理残留缓存目录
cd "$CLONE"
rm -rf hsperfdata_root node-compile-cache playwright-artifacts-* persistent-sync

# 4. commit + push
echo "[3/4] commit ..."
git config user.name "polo4111"
git config user.email "18326161185@163.com"
git add -A
if git diff --cached --quiet; then
  echo "无改动, 跳过 commit"
else
  git commit -m "sync: 双仓同步(代码+skills) $(date +%Y-%m-%d)"
fi
echo "[4/4] push modelscope ..."
timeout 300 git push origin master 2>&1 | tail -3
echo "✅ 已同步: gitee -> ModelScope BigRabbit"
git log --oneline -1
