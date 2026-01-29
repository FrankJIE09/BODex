# Isaac Sim Robot Description Editor 生成碰撞球指南

## 概述

Isaac Sim Robot Description Editor 是 NVIDIA Isaac Sim 中的一个工具，用于可视化编辑机器人配置，包括自动生成碰撞球。

## 官方文档

**注意**：原文档链接已失效。请访问以下链接查找最新文档：

- [Isaac Sim 官方文档主页](https://docs.omniverse.nvidia.com/app_isaacsim/)
- [Isaac Sim 教程和示例](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/tutorial_intro.html)
- [Isaac Sim Robotics 文档](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/tutorial_robots.html)

**搜索建议**：在 Isaac Sim 文档中搜索 "Robot Description Editor" 或 "collision spheres" 查找最新教程。

## 使用步骤

### 1. 启动 Isaac Sim 并打开 Robot Description Editor

1. 启动 Isaac Sim
2. 在菜单栏选择：**Window** → **Robotics** → **Robot Description Editor**
   - 或者使用快捷键（如果有的话）

### 2. 加载机器人 URDF

1. 在 Robot Description Editor 中，点击 **Import Robot** 或 **Load URDF**
2. 选择你的机器人 URDF 文件（例如：`right.urdf`）
3. 机器人模型会加载到场景中

### 3. 生成碰撞球

#### 方法 A：自动生成（如果支持）

1. 在 Robot Description Editor 中，找到 **Collision Spheres** 或 **Generate Collision Spheres** 选项
2. 选择要生成碰撞球的链接（link）
3. 设置参数：
   - **Number of spheres**: 每个链接的碰撞球数量
   - **Sphere radius**: 碰撞球半径（或自动计算）
   - **Fit method**: 拟合方法（如 VOXEL_VOLUME_SAMPLE_SURFACE）
4. 点击 **Generate** 或 **Auto-generate**

#### ⚠️ 常见错误：Instanceable Mesh 问题

如果遇到错误：**"Could not generate spheres for any meshes in link /palm_link. This is likely due to all meshes nested under /palm_link being instanceable"**

**原因**：链接下的 mesh 被标记为 "instanceable"（可实例化），Isaac Sim 无法直接访问其几何数据。

**解决方案**：

1. **方法 1：禁用 Instanceable 属性**
   - 在 Isaac Sim 的 Stage 窗口中，找到 `/palm_link` 下的 mesh 节点
   - 在属性面板中，找到 **Instanceable** 选项
   - 取消勾选 **Instanceable**，使其变为非实例化
   - 重新尝试生成碰撞球

2. **方法 2：使用 Python 脚本生成（推荐）**
   - 直接使用 BODex 中的 `sphere_fit.py` 工具
   - 从 URDF 的 mesh 文件直接生成碰撞球
   - 不受 instanceable 限制影响
   - 见下方"替代方法"部分

3. **方法 3：手动配置**
   - 参考项目中的示例配置（如 `right_wujihand_sim.yml`）
   - 根据 URDF 中的实际几何尺寸手动设置碰撞球
   - 这是最可靠的方法，不依赖 Isaac Sim 的自动生成功能

#### 方法 B：手动添加

1. 选择要添加碰撞球的链接
2. 在链接的属性面板中，找到 **Collision Spheres** 部分
3. 点击 **Add Sphere**
4. 在 3D 视图中调整球的位置和大小：
   - 拖动球体中心来调整位置
   - 调整半径滑块来改变大小
5. 重复添加多个球直到覆盖链接的几何形状

### 4. 导出碰撞球配置

1. 在 Robot Description Editor 中，找到 **Export** 或 **Save Configuration** 选项
2. 选择导出格式：
   - **YAML 格式**（推荐，用于 cuRobo/BODex）
   - **JSON 格式**
3. 保存文件到项目配置目录：
   ```
   src/curobo/content/configs/robot/spheres/your_robot_spheres.yml
   ```

### 5. 在 BODex 配置中使用

在你的机器人配置文件中引用生成的碰撞球：

```yaml
robot_cfg:
  kinematics:
    collision_spheres: "spheres/your_robot_spheres.yml"
    # 或者直接内联配置
    collision_spheres:
      link_name:
        - "center": [x, y, z]
          "radius": r
```

## 碰撞球配置格式

导出的 YAML 格式应该类似：

```yaml
collision_spheres:
  palm_link:
    - "center": [0.0, 0.0, 0.02]
      "radius": 0.018
    - "center": [0.005, 0.015, 0.025]
      "radius": 0.017
  finger1_link1:
    - "center": [0.0, 0.0, 0.007]
      "radius": 0.01
  # ... 更多链接
```

## 注意事项

1. **坐标系**：碰撞球的中心位置是相对于链接坐标系（link frame）的
2. **单位**：位置和半径单位是**米（m）**
3. **覆盖原则**：
   - 确保碰撞球充分覆盖链接的几何形状
   - 对于长链接，使用多个球体
   - 对于复杂形状，可能需要更多球体
4. **性能**：球的数量会影响碰撞检测性能，需要在精度和速度之间平衡

## 替代方法：使用 Python 脚本（推荐）

**推荐使用此方法**，特别是遇到 instanceable mesh 问题时。使用 BODex 中的 `sphere_fit.py` 工具：

```python
from curobo.geom.sphere_fit import fit_spheres_to_mesh, SphereFitType
import trimesh
import yaml

# 加载 mesh
mesh = trimesh.load("path/to/link_mesh.stl")

# 生成碰撞球
pts, radii = fit_spheres_to_mesh(
    mesh,
    n_spheres=10,
    surface_sphere_radius=0.01,
    fit_type=SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE
)

# 转换为 YAML 格式
collision_spheres = {
    "link_name": [
        {"center": [float(x), float(y), float(z)], "radius": float(r)}
        for (x, y, z), r in zip(pts, radii)
    ]
}

# 保存
with open("output.yml", "w") as f:
    yaml.dump({"collision_spheres": collision_spheres}, f)
```

## 验证碰撞球

生成碰撞球后，可以使用以下方法验证：

1. **在 Isaac Sim 中可视化**：
   - 在 Robot Description Editor 中查看碰撞球是否覆盖了链接
   - 检查是否有遗漏的区域

2. **使用 BODex 的验证脚本**：
   ```bash
   python examples/isaac_sim/load_all_robots.py --robot your_robot.yml --visualize_spheres
   ```

3. **运行抓取测试**：
   - 使用生成的配置运行抓取规划
   - 检查是否有碰撞检测错误或警告

## 参考资源

### 官方文档
- [Isaac Sim 官方文档主页](https://docs.omniverse.nvidia.com/app_isaacsim/)
- [Isaac Sim Robotics 教程](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/tutorial_robots.html)
- [cuRobo GitHub 仓库](https://github.com/NVlabs/cuRobo)
- [cuRobo 文档](https://nvlabs.github.io/curobo/)

### 项目示例配置
BODex 项目中的示例配置：
- `src/curobo/content/configs/robot/spheres/franka.yml`
- `src/curobo/content/configs/robot/spheres/allegro.yml`
- `src/curobo/content/configs/robot/right_wujihand_sim.yml`

### 相关工具
- **sphere_fit.py**: `src/curobo/geom/sphere_fit.py` - 从 mesh 自动生成碰撞球
- **验证脚本**: `examples/isaac_sim/load_all_robots.py` - 可视化验证碰撞球配置

## 重要提示

⚠️ **如果找不到 Isaac Sim Robot Description Editor**：
1. 该功能可能在不同版本的 Isaac Sim 中位置不同
2. 建议使用 **Python 脚本方法**（见"替代方法"部分）自动生成碰撞球
3. 可以手动编辑 YAML 配置文件，参考项目中的示例配置

