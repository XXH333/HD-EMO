#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/workspace/project-cpfs-1000109/niesihang/_workspace/hdemo-codec"
REMOTE_URL="git@github.com:XXH333/HD-EMO.git"
BRANCH="main"

cd "$PROJECT_DIR"

echo "==> 当前目录：$(pwd)"

# 初始化 Git 仓库（如果尚未初始化）
if [ ! -d ".git" ]; then
    echo "==> 初始化 Git 仓库..."
    git init
fi

# 设置主分支为 main
git branch -M "$BRANCH"

# 忽略预训练模型文件夹
touch .gitignore
if ! grep -qxF "pretrained_models/" .gitignore; then
    echo "pretrained_models/" >> .gitignore
fi

# 如果此前曾被 Git 跟踪，取消跟踪，但保留本地文件
git rm -r --cached pretrained_models/ 2>/dev/null || true

# 设置/更新远程仓库地址
if git remote get-url origin >/dev/null 2>&1; then
    git remote set-url origin "$REMOTE_URL"
else
    git remote add origin "$REMOTE_URL"
fi

echo "==> 远程仓库：$(git remote get-url origin)"

# 暂存所有未被 .gitignore 排除的文件
git add .

# 输出确认信息：这里不应出现 pretrained_models/
echo "==> Git 状态："
git status --short

# 有变更才创建提交
if ! git diff --cached --quiet; then
    git commit -m "Update HD-EMO code"
else
    echo "==> 没有新的文件变更，跳过 commit。"
fi

# 强制以本地 main 覆盖远程 main
echo "==> 正在强制推送到 GitHub..."
git push -u origin "$BRANCH" --force

echo "==> 推送完成！"
echo "==> 仓库地址：https://github.com/XXH333/HD-EMO"
