# Git LFS 状态报告

## ⚠️ 当前问题

Git LFS 拉取失败，错误信息：
```
This repository exceeded its LFS budget. The account responsible for the budget should increase it to restore access.
```

## 📊 LFS 文件统计

- **总 LFS 文件数**: 304 个
- **已下载**: 0 个
- **未下载**: 304 个

## 📁 LFS 文件类型

主要包括：
1. **机器人网格文件** (`.obj`, `.stl`, `.ply`):
   - Allegro 手部网格文件
   - Franka 机械臂网格文件
   - Shadow Hand 相关文件
   - 其他机器人模型文件

2. **图片文件** (`.gif`):
   - `images/robot_demo.gif`
   - `images/rrt_compare.gif`

## 🔍 文件位置

LFS 文件主要位于：
- `src/curobo/content/assets/robot/*/meshes/` - 机器人网格文件
- `images/` - 演示图片

## 💡 解决方案

### 方案 1: 增加 GitHub LFS 配额（推荐）

1. 访问 GitHub 仓库设置
2. 增加 LFS 存储配额
3. 重新运行 `git lfs pull`

### 方案 2: 分批拉取文件

尝试分批拉取特定目录的文件：

```bash
# 只拉取 Shadow Hand 相关文件（如果存在）
git lfs pull --include="src/curobo/content/assets/robot/*shadow*"

# 或者拉取特定机器人
git lfs pull --include="src/curobo/content/assets/robot/allegro_description/**"
```

### 方案 3: 使用其他下载方式

如果 GitHub LFS 配额无法增加，可以考虑：
- 从其他镜像源下载
- 手动下载必要的文件
- 使用其他 Git 托管服务

### 方案 4: 检查文件是否必需

对于程序运行，某些文件可能不是必需的：
- 图片文件（`.gif`）仅用于演示，不影响功能
- 某些机器人的网格文件可能只在特定配置中使用

## 🧪 测试程序运行

即使没有所有 LFS 文件，程序可能仍能运行（取决于使用的配置）。可以尝试：

```bash
# 测试运行（可能会缺少某些机器人模型）
./run_bodex.sh -c sim_shadow/fc.yml -w 1
```

如果程序报错缺少特定文件，再针对性地拉取那些文件。

## 📝 当前状态

- ✅ Git LFS 已安装并配置
- ❌ LFS 文件拉取失败（配额超限）
- ⚠️ 程序可能可以运行，但某些功能可能受限

## 🔄 后续操作

1. 联系仓库维护者增加 LFS 配额
2. 或者等待配额恢复后重试
3. 检查程序运行是否需要这些文件

