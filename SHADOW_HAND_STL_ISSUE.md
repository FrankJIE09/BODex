# Shadow Hand STL 文件缺失问题

## ⚠️ 问题描述

程序运行时出现错误：
```
ValueError: Could not load resource /home/lenovo/Frank/code/BODex/src/curobo/content/assets/robot/shadow_hand/stl/f_distal_pst.stl
Unable to open file "/home/lenovo/Frank/code/BODex/src/curobo/content/assets/robot/shadow_hand/stl/f_distal_pst.stl".
```

## 🔍 原因

Shadow Hand 的 STL 文件存储在 Git LFS 中，但由于 GitHub LFS 配额超限，无法拉取这些文件。

## 📋 需要的 STL 文件列表

根据 URDF 文件，需要以下 16 个 STL 文件：

1. `f_distal_pst.stl` - 手指远端
2. `f_knuckle.stl` - 手指关节
3. `f_middle.stl` - 手指中间
4. `f_proximal.stl` - 手指近端
5. `forearm_0.stl` - 前臂部分 0
6. `forearm_1.stl` - 前臂部分 1
7. `forearm_collision.stl` - 前臂碰撞模型
8. `left_lf_metacarpal.stl` - 左手小指掌骨
9. `left_palm.stl` - 左手掌
10. `left_wrist.stl` - 左手腕
11. `right_lf_metacarpal.stl` - 右手小指掌骨
12. `right_palm.stl` - 右手掌
13. `right_wrist.stl` - 右手腕
14. `th_distal_pst.stl` - 拇指远端
15. `th_middle.stl` - 拇指中间
16. `th_proximal.stl` - 拇指近端

## 💡 解决方案

### 方案 1: 等待 Git LFS 配额恢复（推荐）

1. 联系仓库维护者（FrankJIE09）增加 GitHub LFS 配额
2. 等待配额恢复后，运行：
   ```bash
   git lfs pull
   ```

### 方案 2: 从其他来源获取文件

如果仓库维护者提供了其他下载方式（如 Google Drive、百度网盘等），可以从那里下载并放置到：
```
src/curobo/content/assets/robot/shadow_hand/stl/
```

### 方案 3: 使用其他机器人配置

如果暂时无法获取 Shadow Hand 文件，可以尝试使用其他已配置的机器人（如 Allegro Hand），但需要修改配置文件。

### 方案 4: 手动创建占位符文件（不推荐）

可以创建空的 STL 文件作为占位符，但这会导致碰撞检测不准确，可能影响抓取质量。

## 🔄 当前状态

- ✅ 数据集已下载（DGN_2k）
- ✅ 环境已配置
- ✅ 程序可以运行
- ❌ Shadow Hand STL 文件缺失（Git LFS 配额超限）

## 📝 下一步

1. 联系仓库维护者解决 Git LFS 配额问题
2. 或寻找替代的 Shadow Hand 模型文件来源
3. 或等待配额恢复后重新拉取

## 🔗 相关资源

- [Git LFS 状态报告](./GIT_LFS_STATUS.md)
- [环境搭建状态](./ENV_SETUP_STATUS.md)
- [数据集下载状态](./DATASET_DOWNLOAD_STATUS.md)

