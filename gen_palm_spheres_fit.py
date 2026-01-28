#!/usr/bin/env python3
"""
用 mesh + sphere_fit + 聚类，为 palm_link 生成贴合异构体的少量碰撞球。

示例：
  python gen_palm_spheres_fit.py \
    --mesh src/curobo/content/assets/robot/wujihand-urdf/meshes/right/palm_link.STL \
    --raw-n 60 --target-n 14 --base-radius 0.009
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import trimesh
from sklearn.cluster import KMeans


def fit_palm_spheres(
    mesh_path: Path,
    raw_n: int = 60,
    target_n: int = 14,
    base_radius: float = 0.009,
):
    """从 palm_link mesh 拟合并聚类生成少量碰撞球。"""

    # 确保可以 import curobo.geom.sphere_fit
    repo_root = Path(__file__).resolve().parent
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from curobo.geom.sphere_fit import fit_spheres_to_mesh, SphereFitType

    mesh = trimesh.load(mesh_path)

    # 1. 先拟合较多小球，表面+体积一起
    pts, radii = fit_spheres_to_mesh(
        mesh,
        n_spheres=raw_n,
        surface_sphere_radius=base_radius,
        fit_type=SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
    )

    pts = np.asarray(pts)
    radii = np.asarray(radii)

    if pts.shape[0] == 0:
        raise RuntimeError("fit_spheres_to_mesh 返回空结果，检查 mesh 是否有效。")

    # 2. KMeans 聚类压缩为少量代表球
    k = min(target_n, pts.shape[0])
    kmeans = KMeans(n_clusters=k, n_init=8, random_state=0)
    labels = kmeans.fit_predict(pts)
    centers = kmeans.cluster_centers_

    spheres = []
    for i in range(k):
        idx = np.where(labels == i)[0]
        if len(idx) == 0:
            continue
        cluster_pts = pts[idx]
        cluster_r = radii[idx]

        # 半径：中心到 cluster 内所有点的最大距离 + 对应原始半径的最大值
        d = np.linalg.norm(cluster_pts - centers[i], axis=1)
        r = max(d.max() + cluster_r.max(), base_radius)

        c = centers[i]
        spheres.append(
            {
                "center": [
                    round(float(c[0]), 5),
                    round(float(c[1]), 5),
                    round(float(c[2]), 5),
                ],
                "radius": round(float(r), 5),
            }
        )

    # 排序：按 y 再 x 再 z，方便阅读
    spheres.sort(key=lambda s: (s["center"][1], s["center"][0], s["center"][2]))
    return spheres


def main():
    parser = argparse.ArgumentParser(
        description="为 palm_link 生成基于 sphere_fit+聚类的碰撞球配置"
    )
    parser.add_argument("--mesh", type=str, required=True, help="palm_link mesh 路径")
    parser.add_argument(
        "--raw-n", type=int, default=60, help="sphere_fit 初始拟合的小球数量"
    )
    parser.add_argument(
        "--target-n", type=int, default=14, help="聚类后期望得到的球数量"
    )
    parser.add_argument(
        "--base-radius", type=float, default=0.009, help="基础半径下限（m）"
    )
    args = parser.parse_args()

    mesh_path = Path(args.mesh)
    if not mesh_path.is_file():
        raise FileNotFoundError(f"mesh not found: {mesh_path}")

    spheres = fit_palm_spheres(
        mesh_path=mesh_path,
        raw_n=args.raw_n,
        target_n=args.target_n,
        base_radius=args.base_radius,
    )

    # 打印成可直接粘贴到 YAML 的 palm_link 段落
    print("palm_link:")
    for s in spheres:
        c = s["center"]
        r = s["radius"]
        print(f"  - center: [{c[0]}, {c[1]}, {c[2]}]")
        print(f"    radius: {r}")


if __name__ == "__main__":
    main()

