# Git LFS 配额超限，无法拉取 Shadow Hand STL 文件

## 问题

运行程序时出现错误，无法加载 Shadow Hand 的 STL 文件：

```
ValueError: Could not load resource .../shadow_hand/stl/f_distal_pst.stl
Failed to determine STL storage representation
```

运行 `git lfs pull` 时报错：
```
This repository exceeded its LFS budget. The account responsible for the budget should increase it to restore access.
```

## 需要的文件

Shadow Hand 需要以下 16 个 STL 文件（位于 `src/curobo/content/assets/robot/shadow_hand/stl/`）：
- `f_distal_pst.stl`, `f_knuckle.stl`, `f_middle.stl`, `f_proximal.stl`
- `forearm_0.stl`, `forearm_1.stl`, `forearm_collision.stl`
- `right_palm.stl`, `right_wrist.stl`, `right_lf_metacarpal.stl`
- `left_palm.stl`, `left_wrist.stl`, `left_lf_metacarpal.stl`
- `th_distal_pst.stl`, `th_middle.stl`, `th_proximal.stl`

## 当前状态

- ✅ 环境已配置
- ✅ 数据集已下载
- ❌ Git LFS 文件无法拉取（配额超限）
- ❌ 所有 STL 文件都是 Git LFS 指针，不是实际文件

## 建议

1. 增加 GitHub LFS 配额
2. 或提供替代下载方式（如 Google Drive、百度网盘等）

## 环境

- OS: Linux
- Git LFS: 3.0.2
- Python: 3.10.19 (conda bodex)
- PyTorch: 2.2.2+cu121
