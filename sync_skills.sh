#!/bin/bash
# ============================================================
# skill 统一同步脚本 (polo4111)
# 权威目录: 仓库 .reasonix/skills/  (git 管理, 所有改动在此进行)
# 系统副本: workspace/.user_skills/ (豆包系统 skill root, 加载用)
# 修改仓库 skill 后必须运行本脚本, 保持两端一致。
# 用法: bash sync_skills.sh
# ============================================================
set -e
REPO_SKILL="$(cd "$(dirname "$0")" && pwd)/.reasonix/skills"
SYS_SKILL="/home/user/.super_doubao/super-doubao-runtime/workspace/.user_skills"
[ -d "$SYS_SKILL" ] || mkdir -p "$SYS_SKILL"

for sk in golden-pit-strategy; do
  if [ -d "$REPO_SKILL/$sk" ]; then
    echo "syncing $sk ..."
    rsync -a --delete "$REPO_SKILL/$sk/" "$SYS_SKILL/$sk/"
  fi
done
echo "✅ skill 已同步: 仓库统一版 -> 系统加载副本"
echo "   权威: $REPO_SKILL"
echo "   副本: $SYS_SKILL"
