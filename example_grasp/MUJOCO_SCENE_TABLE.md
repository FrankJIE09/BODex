# MuJoCo 抓取场景 (grasp_eval) 结构表

## 1. 全局选项与默认

| 层级 | 属性 | 值 | 说明 |
|------|------|-----|------|
| **option** | timestep | 0.0005 | 仿真步长 (s) |
| | gravity | 0 0 -9.81 | 重力 (m/s²) |
| | cone | elliptic | 摩擦锥类型 |
| | impratio | 10 | 约束求解器参数 |
| **default/geom** | solimp | 0.9 0.95 0.001 0.5 2 | 接触阻抗默认值 |
| | solref | 0.05 1 | 接触刚度/阻尼默认值 |

---

## 2. 资源 (asset)

| 类型 | name | file / 参数 | 说明 |
|------|------|-------------|------|
| mesh | simplified | simplified.obj | scale="0.08 0.08 0.08" |

---

## 3. 世界体 (worldbody)

### 3.1 桌子 (table)

| 属性 | 值 | 说明 |
|------|-----|------|
| body pos | 0 0 -0.4 | 桌子中心位置 |
| geom name | table | |
| type | box | |
| size | 4 3 0.02 | 半轴 (x y z) |
| rgba | 0.6 0.5 0.4 1 | 颜色 |
| friction | 0.1 0.005 0.0001 | 摩擦系数 |

### 3.2 被抓物体 (object)

| 属性 | 值 | 说明 |
|------|-----|------|
| body pos | 0 0 0 | 初始位置 |
| body quat | 1 0 0 0 | 初始四元数 |
| joint | object_joint (free) | 自由刚体 |
| geom name | obj | |
| type | mesh | mesh="simplified" |
| mass | 0.03 | 质量 (kg) |
| friction | 0.1 0.005 0.0001 | |
| condim | 3 | 3D 摩擦锥 |
| contype | 1 | 碰撞类型 |
| conaffinity | 1 | 碰撞对象组 |
| solref | 0.02 0.9 | 接触刚度/阻尼 |
| solimp | 0.9 0.95 0.001 0.5 2 | 接触阻抗 |

### 3.3 手基座 (hand_base)

| 属性 | 值 | 说明 |
|------|-----|------|
| body pos | 0 0 0.5 | 手基座初始位置 |
| joint | hand_free (free) | 自由刚体 |
| inertial | mass=0.01, diaginertia=1e-5 | 惯性 |
| 子体 | hand_origin | 挂载灵巧手用 |

---

## 4. 汇总

| 元素 | 数量/内容 |
|------|-----------|
| option | 1 组 (步长、重力、摩擦锥、impratio) |
| default geom | 1 条 (solimp, solref) |
| asset mesh | 1 个 (物体 mesh) |
| worldbody 子体 | 3 个：table, object, hand_base |
| 自由关节 | 2 个：object_joint, hand_free |
