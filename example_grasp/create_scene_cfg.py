#!/usr/bin/env python3
"""
为 STL 文件创建 scene_cfg 文件的脚本
用于配置新的被抓取对象

使用方法:
    python example_grasp/create_scene_cfg.py <stl_file> [选项]

示例:
    # STL 文件在 example_grasp 目录下
    python example_grasp/create_scene_cfg.py tmr_100w_01_c.stl --scene-id tmr_100w_01_c
    
    # 或使用相对路径
    python example_grasp/create_scene_cfg.py example_grasp/tmr_100w_01_c.stl --scene-id tmr_100w_01_c
"""
import numpy as np
import os
import sys
from pathlib import Path

# 添加项目根目录到路径，以便导入 trimesh（如果可用）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def create_scene_cfg(stl_file_path, output_dir=None, scene_id=None, scale=[1.0, 1.0, 1.0]):
    """
    为 STL 文件创建 scene_cfg 文件
    
    参数:
        stl_file_path: STL 文件的完整路径或相对路径
        output_dir: 输出目录（如果不指定，会在 assets/object 下创建）
        scene_id: 场景ID（如果不指定，使用 STL 文件名）
        scale: 缩放比例 [x, y, z]
    """
    stl_path = Path(stl_file_path)
    
    # 如果是相对路径，从项目根目录解析
    if not stl_path.is_absolute():
        # 先尝试从当前工作目录解析
        if stl_path.exists():
            stl_path = stl_path.resolve()
        # 如果不存在，尝试从项目根目录解析
        elif (project_root / stl_path).exists():
            stl_path = project_root / stl_path
        # 如果还是不存在，尝试从 example_grasp 目录解析
        elif (project_root / "example_grasp" / stl_path.name).exists():
            stl_path = project_root / "example_grasp" / stl_path.name
        else:
            stl_path = project_root / stl_path
    
    if not stl_path.exists():
        raise FileNotFoundError(f"STL 文件不存在: {stl_path}")
    
    # 确定 scene_id
    if scene_id is None:
        scene_id = stl_path.stem  # 使用文件名（不含扩展名）
    
    # 确定输出目录
    if output_dir is None:
        # 在 assets/object 下创建目录结构
        assets_dir = project_root / "src/curobo/content/assets/object"
        output_dir = assets_dir / scene_id / "floating"
    else:
        output_dir = Path(output_dir)
        if not output_dir.is_absolute():
            output_dir = project_root / output_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制或创建 STL 文件的相对路径
    stl_relative_path = stl_path.name  # 相对于 scene_cfg 文件的路径
    
    # STL 文件应该和 scene_cfg 文件在同一目录（floating/）下
    target_stl_path = output_dir / stl_relative_path
    if stl_path != target_stl_path:
        import shutil
        shutil.copy2(stl_path, target_stl_path)
        print(f"已复制 STL 文件到: {target_stl_path}")
    
    # 计算重心和 OBB（使用 trimesh，如果可用）
    try:
        import trimesh
        # 使用绝对路径加载 mesh
        mesh = trimesh.load(str(stl_path.resolve()))
        gravity_center = mesh.center_mass.tolist()  # 重心
        obb = mesh.bounding_box.extents.tolist()  # OBB 尺寸
        print(f"计算得到重心: {gravity_center}")
        print(f"计算得到 OBB: {obb}")
    except ImportError:
        print("警告: trimesh 未安装，无法自动计算重心和 OBB")
        print("请手动设置或安装 trimesh: pip install trimesh")
        gravity_center = [0.0, 0.0, 0.0]
        obb = [1.0, 1.0, 1.0]
    except Exception as e:
        print(f"警告: 无法计算重心和 OBB，使用默认值: {e}")
        gravity_center = [0.0, 0.0, 0.0]
        obb = [1.0, 1.0, 1.0]
    
    # 创建 JSON 信息文件（放在 floating 目录下，与 scene_cfg 文件同一目录）
    import json
    info_data = {
        "gravity_center": gravity_center,
        "obb": obb
    }
    # info 文件应该和 scene_cfg 文件在同一目录（floating/）下
    info_path = output_dir / f"{scene_id}_info.json"
    with open(info_path, 'w') as f:
        json.dump(info_data, f, indent=2)
    print(f"已创建信息文件: {info_path}")
    
    # 创建 scene_cfg 字典
    obj_name = scene_id  # 对象名称
    scene_cfg = {
        "scene_id": scene_id,
        "task": {
            "obj_name": obj_name
        },
        "scene": {
            obj_name: {
                "type": "rigid_object",
                "scale": scale,
                "pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # 位置和四元数 [x, y, z, qw, qx, qy, qz]
                "file_path": stl_relative_path,  # 相对于 scene_cfg 文件的路径
                "info_path": f"{scene_id}_info.json"  # 信息文件路径（相对于 scene_cfg 文件，在同一目录下）
            }
        }
    }
    
    # 保存为 .npy 文件
    scene_cfg_path = output_dir / "scale008.npy"  # 使用 scale008 作为默认名称
    np.save(scene_cfg_path, scene_cfg)
    print(f"已创建 scene_cfg 文件: {scene_cfg_path}")
    
    return scene_cfg_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="为 STL 文件创建 scene_cfg 文件，用于配置新的被抓取对象",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # STL 文件在 example_grasp 目录下（推荐）
  python example_grasp/create_scene_cfg.py example_grasp/tmr_100w_01_c.stl --scene-id tmr_100w_01_c
  
  # 或从项目根目录运行，使用相对路径
  cd /home/lenovo/Frank/code/BODex
  python example_grasp/create_scene_cfg.py example_grasp/tmr_100w_01_c.stl --scene-id tmr_100w_01_c
  
  # 指定缩放比例
  python example_grasp/create_scene_cfg.py example_grasp/tmr_100w_01_c.stl --scene-id tmr_100w_01_c --scale 1.0 1.0 1.0
        """
    )
    parser.add_argument("stl_file", type=str, help="STL 文件路径（相对或绝对路径）")
    parser.add_argument("--output-dir", type=str, default=None, 
                       help="输出目录（默认：src/curobo/content/assets/object/<scene_id>/floating/）")
    parser.add_argument("--scene-id", type=str, default=None, 
                       help="场景ID（默认：使用 STL 文件名）")
    parser.add_argument("--scale", type=float, nargs=3, default=[1.0, 1.0, 1.0], 
                       help="缩放比例 [x y z]（默认：1.0 1.0 1.0）")
    
    args = parser.parse_args()
    
    try:
        scene_cfg_path = create_scene_cfg(
            args.stl_file,
            args.output_dir,
            args.scene_id,
            args.scale
        )
        print(f"\n✅ 成功创建 scene_cfg 文件: {scene_cfg_path}")
        print(f"\n📝 接下来，在配置文件中使用以下路径：")
        print(f"   template_path: object/{args.scene_id or Path(args.stl_file).stem}/floating/scale008.npy")
        print(f"\n💡 提示：如果重心和 OBB 计算不正确，请手动编辑 {scene_cfg_path.parent / (args.scene_id or Path(args.stl_file).stem + '_info.json')}")
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

