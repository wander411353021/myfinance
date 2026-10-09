# -*- coding: utf-8 -*-
"""rsync 的 Python 替代(Windows 无 rsync) —— 供 sync_modelscope.sh / sync_github.sh 调用。

用法: python sync_rsync.py <源目录> <目标目录>

行为等价于旧命令:
    rsync -av --exclude='.git' --exclude='.ssh_backup' --exclude='__pycache__'
          --exclude='*.pyc' --exclude='result/' --exclude='*.png' ... <源>/ <目标>/
- 只复制/覆盖, **不删除**目标端多余文件(ModelScope 的部署文件 app.py/.streamlit/runtime.txt 必须保留)
- 逐文件比较大小+mtime, 相同则跳过(近似 rsync -av 的增量语义)
- 任何异常以非 0 退出码结束, 便于调用方判断(修掉旧脚本的"假成功"问题)
"""
import os
import shutil
import sys

EXCLUDE_DIRS = {'.git', '.ssh_backup', '__pycache__', 'result',
                'node-compile-cache', 'hsperfdata_root', '.pytest_cache',
                '.mypy_cache', 'persistent-sync'}
EXCLUDE_EXT = {'.pyc', '.png', '.log'}
EXCLUDE_PREFIX = ('playwright-artifacts-',)


def skip_dir(name):
    return name in EXCLUDE_DIRS or any(name.startswith(p) for p in EXCLUDE_PREFIX)


def main():
    if len(sys.argv) < 3:
        print('用法: python sync_rsync.py <源目录> <目标目录>')
        return 2
    src_root, dst_root = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    if not os.path.isdir(src_root):
        print('源目录不存在: %s' % src_root)
        return 2
    os.makedirs(dst_root, exist_ok=True)
    copied = skipped = 0
    for root, dirs, files in os.walk(src_root):
        dirs[:] = [d for d in dirs if not skip_dir(d)]
        rel = os.path.relpath(root, src_root)
        dst_dir = dst_root if rel == '.' else os.path.join(dst_root, rel)
        os.makedirs(dst_dir, exist_ok=True)
        for f in files:
            if os.path.splitext(f)[1].lower() in EXCLUDE_EXT:
                continue
            s, d = os.path.join(root, f), os.path.join(dst_dir, f)
            try:
                if os.path.exists(d):
                    ss, ds = os.stat(s), os.stat(d)
                    if ss.st_size == ds.st_size and int(ss.st_mtime) == int(ds.st_mtime):
                        skipped += 1
                        continue
                shutil.copy2(s, d)
                copied += 1
            except Exception as e:
                print('复制失败 %s: %s' % (s, e))
                return 1
    print('sync_rsync: 复制 %d, 跳过(相同) %d' % (copied, skipped))
    return 0


if __name__ == '__main__':
    sys.exit(main())
