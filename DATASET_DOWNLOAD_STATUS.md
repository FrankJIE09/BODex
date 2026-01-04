# 数据集下载状态

## ✅ 下载完成

已成功从 [Hugging Face](https://huggingface.co/datasets/JiayiChenPKU/BODex) 下载并解压 `DGN_2k_processed.tar.gz` 数据集。

## 📁 目录结构

```
src/curobo/content/assets/object/DGN_2k
├── processed_data/     (2.1 GB, 2398 个对象)
├── scene_cfg/          (278 MB, 2398 个场景配置)
└── valid_split/        (236 KB, 包含 all.json, test.json, train.json)
```

## 📊 数据集统计

- **对象数量**: 2398 个
- **processed_data**: 2.1 GB
- **scene_cfg**: 278 MB
- **valid_split**: 包含训练/测试/全部数据集的 JSON 文件

## ✅ 验证结果

程序现在可以正确识别数据集：
- ✓ 找到 100 个 scene cfgs（之前是 0 个）
- ✓ 目录结构符合 README 要求
- ✓ 程序可以正常运行

## 📝 使用说明

数据集已准备就绪，可以开始使用：

```bash
# 运行抓取合成
./run_bodex.sh -c sim_shadow/fc.yml -w 40

# 或使用环境设置脚本
source setup_env.sh
python example_grasp/plan_batch_env.py -c sim_shadow/fc.yml -w 40
```

## 🔗 数据来源

- **数据集**: [JiayiChenPKU/BODex on Hugging Face](https://huggingface.co/datasets/JiayiChenPKU/BODex)
- **文件**: `object_assets/DGN_2k_processed.tar.gz`
- **大小**: 约 523 MB (压缩), 2.4 GB (解压后)

