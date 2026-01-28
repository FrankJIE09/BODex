#!/usr/bin/env python3
"""
从 URDF 的 mesh 文件自动生成碰撞球配置
解决 Isaac Sim 中 instanceable mesh 无法生成碰撞球的问题
"""

import os
import sys
import yaml
import trimesh
import argparse
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from curobo.geom.sphere_fit import fit_spheres_to_mesh, SphereFitType


def load_urdf_mesh(urdf_path, link_name, asset_root_path=None):
    """从 URDF 中提取指定 link 的 mesh 路径并加载"""
    import xml.etree.ElementTree as ET
    
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    # 查找指定的 link
    for link in root.findall('link'):
        if link.get('name') == link_name:
            # 查找 collision 或 visual 中的 mesh
            for elem_type in ['collision', 'visual']:
                for elem in link.findall(elem_type):
                    geometry = elem.find('geometry')
                    if geometry is not None:
                        mesh_elem = geometry.find('mesh')
                        if mesh_elem is not None:
                            filename = mesh_elem.get('filename')
                            if filename:
                                # 处理相对路径
                                if asset_root_path:
                                    mesh_path = os.path.join(asset_root_path, filename)
                                else:
                                    # 从 URDF 路径推断
                                    urdf_dir = os.path.dirname(urdf_path)
                                    mesh_path = os.path.join(urdf_dir, filename)
                                
                                # 尝试加载 mesh
                                if os.path.exists(mesh_path):
                                    return trimesh.load(mesh_path)
                                # 尝试其他可能的路径
                                for base_dir in [urdf_dir, os.path.dirname(urdf_dir)]:
                                    alt_path = os.path.join(base_dir, filename)
                                    if os.path.exists(alt_path):
                                        return trimesh.load(alt_path)
    return None


def generate_spheres_for_link(link_name, mesh_path=None, urdf_path=None, asset_root_path=None,
                              n_spheres=10, surface_radius=0.017, fit_type=SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE):
    """为单个 link 生成碰撞球"""
    
    # 加载 mesh
    if mesh_path and os.path.exists(mesh_path):
        mesh = trimesh.load(mesh_path)
    elif urdf_path:
        mesh = load_urdf_mesh(urdf_path, link_name, asset_root_path)
    else:
        raise ValueError(f"需要提供 mesh_path 或 urdf_path 来加载 {link_name} 的 mesh")
    
    if mesh is None:
        print(f"⚠️  警告: 无法加载 {link_name} 的 mesh，跳过")
        return None
    
    # 生成碰撞球
    print(f"正在为 {link_name} 生成 {n_spheres} 个碰撞球...")
    try:
        pts, radii = fit_spheres_to_mesh(
            mesh,
            n_spheres=n_spheres,
            surface_sphere_radius=surface_radius,
            fit_type=fit_type
        )
        
        if pts is None or len(pts) == 0:
            print(f"⚠️  警告: {link_name} 无法生成碰撞球")
            return None
        
        # 转换为配置格式
        spheres = [
            {"center": [float(x), float(y), float(z)], "radius": float(r)}
            for (x, y, z), r in zip(pts, radii)
        ]
        
        print(f"✅ {link_name}: 生成了 {len(spheres)} 个碰撞球")
        return spheres
        
    except Exception as e:
        print(f"❌ 错误: {link_name} 生成碰撞球失败: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='从 mesh 文件生成碰撞球配置')
    parser.add_argument('--urdf', type=str, help='URDF 文件路径')
    parser.add_argument('--link', type=str, required=True, help='要生成碰撞球的 link 名称')
    parser.add_argument('--mesh', type=str, help='直接指定 mesh 文件路径（可选）')
    parser.add_argument('--asset-root', type=str, help='资源根路径（相对于 URDF）')
    parser.add_argument('--n-spheres', type=int, default=10, help='碰撞球数量（默认：10）')
    parser.add_argument('--radius', type=float, default=0.017, help='表面球半径（默认：0.017m）')
    parser.add_argument('--output', type=str, help='输出 YAML 文件路径（可选）')
    
    args = parser.parse_args()
    
    # 生成碰撞球
    spheres = generate_spheres_for_link(
        args.link,
        mesh_path=args.mesh,
        urdf_path=args.urdf,
        asset_root_path=args.asset_root,
        n_spheres=args.n_spheres,
        surface_radius=args.radius
    )
    
    if spheres is None:
        return
    
    # 输出结果
    result = {
        "collision_spheres": {
            args.link: spheres
        }
    }
    
    if args.output:
        with open(args.output, 'w') as f:
            yaml.dump(result, f, default_flow_style=False, sort_keys=False)
        print(f"\n✅ 配置已保存到: {args.output}")
    else:
        print("\n生成的碰撞球配置:")
        print(yaml.dump(result, default_flow_style=False, sort_keys=False))


if __name__ == "__main__":
    main()

