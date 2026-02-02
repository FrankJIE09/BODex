# MuJoCo 抓取评估：读入 *_grasp.npy，回放 robot_pose 轨迹并判定抓取成功（抬升+稳定）。
# 速度控制参数：大幅减慢手部趋近目标的速度
VEL_GAIN = 10
MAX_VEL_LIN = 1000
MAX_VEL_ANG = 1000
SECONDS_PER_WAYPOINT = .1

import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.join(_script_dir, "..", "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import glob
import re
import tempfile
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from curobo.util_file import (
    load_yaml,
    join_path,
    get_manip_configs_path,
    get_robot_configs_path,
    get_output_path,
    get_assets_path,
)


def find_grasp_npy_files(manip_cfg_file: str, exp_path: str) -> List[str]:
    """定位所有 *_grasp.npy。支持绝对路径或相对实验路径。"""
    path = os.path.expanduser(exp_path.strip())
    if not os.path.isabs(path) and (os.path.isfile(path) or os.path.isdir(path)):
        path = os.path.abspath(path)
    if os.path.isabs(path):
        search_root = os.path.dirname(path) if os.path.isfile(path) else (
            path.rstrip(os.sep) if os.path.isdir(path) else os.path.dirname(path))
        pattern_recursive = os.path.join(search_root, "**", "*_grasp.npy")
        pattern_same_dir = os.path.join(search_root, "*_grasp.npy")
        files = set(glob.glob(pattern_recursive, recursive=True)) | set(glob.glob(pattern_same_dir))
        return sorted(files)

    path = path.rstrip(os.sep)
    if "graspdata" not in path:
        path = os.path.join(path, "graspdata")
    manip_cfg_stem = os.path.splitext(manip_cfg_file)[0]
    save_folder = os.path.join(manip_cfg_stem, path)
    root = os.path.join(get_output_path(), save_folder)
    pattern_recursive = os.path.join(root, "**", "*_grasp.npy")
    pattern_same_dir = os.path.join(root, "*_grasp.npy")
    files = set(glob.glob(pattern_recursive, recursive=True))
    files.update(glob.glob(pattern_same_dir))
    return sorted(files)


def run_mujoco_viewer(
        npy_path: Optional[str] = None,
        traj: Optional[np.ndarray] = None,
        joint_names: Optional[List[str]] = None,
        timeout_s: float = 30.0,
        world_cfg: Any = None,
        robot_mjcf_path: str = "",
        friction: float = 1.0,
        object_deformable: Optional[bool] = None,
        save_xml_path: Optional[str] = None,
        solref: Optional[Tuple[float, float]] = None,
        solimp: Optional[Tuple[float, float, float, float, float]] = None,
        condim: int = 6,
        vel_gain: float = VEL_GAIN,
        max_vel_lin: float = MAX_VEL_LIN,
        max_vel_ang: float = MAX_VEL_ANG,
        timestep: float = 0.0005,
) -> bool:
    """启动 MuJoCo 画面并回放轨迹。"""
    import mujoco
    import mujoco.viewer

    print(f"[DEBUG] run_mujoco_viewer: 使用 timestep={timestep} s")

    scene_path = None
    if npy_path and os.path.isfile(npy_path):
        _d = np.load(npy_path, allow_pickle=True).item()
        if isinstance(_d, dict):
            scene_path = _d.get("scene_path", None)

    if robot_mjcf_path and os.path.isfile(robot_mjcf_path):
        scene = build_mujoco_scene_for_grasp(
            world_cfg,
            robot_mjcf_path=robot_mjcf_path,
            joint_names=joint_names or [],
            npy_path=npy_path or "",
            scene_path=scene_path,
            friction=friction,
            object_deformable=object_deformable,
            save_xml_path=save_xml_path,
            solref=solref,
            solimp=solimp,
            condim=condim,
            timestep=timestep,
        )
        if scene is not None:
            model, data = scene
        else:
            xml_fallback = _world_cfg_to_mjcf(
                world_cfg,
                include_hand_placeholder=False,
                npy_path=npy_path or "",
                scene_path=scene_path,
                friction=friction,
                object_deformable=object_deformable,
                condim=condim,
                hand_init_pose=None,
                timestep=timestep,
            )
            model = mujoco.MjModel.from_xml_string(xml_fallback)
            data = mujoco.MjData(model)
            print(f"[DEBUG] 回退模型创建后实际 timestep: {model.opt.timestep} s")
            print("[WARN] 灵巧手未加载，仅显示桌面与物体。请检查上方错误信息。")

    traj = np.asarray(traj) if traj is not None else None
    nq_traj = int(traj.shape[-1]) if traj is not None and traj.size > 0 else 0
    print(f"[DEBUG] Viewer模式: nq_traj={nq_traj}, joint_names数量={len(joint_names) if joint_names else 0}")
    if traj is not None and traj.size > 0 and traj.ndim >= 2:
        # 打印手部初始位置（第一个路点的前7个值）
        hand_init_pos = traj[0, :7] if traj.shape[0] > 0 else None
        if hand_init_pos is not None:
            print(
                f"[DEBUG] 手部初始位置: pos=({hand_init_pos[0]:.4f}, {hand_init_pos[1]:.4f}, {hand_init_pos[2]:.4f}), quat=({hand_init_pos[3]:.4f}, {hand_init_pos[4]:.4f}, {hand_init_pos[5]:.4f}, {hand_init_pos[6]:.4f})")
    mapping = _build_joint_qpos_map(model, joint_names or [], nq_traj) if nq_traj else []
    if not mapping and nq_traj >= 7:
        print("[DEBUG] 映射为空，尝试查找 hand_free 关节")
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hand_free")
        if jid >= 0:
            mapping = [(0, int(model.jnt_qposadr[jid]), 7, int(model.jnt_dofadr[jid]), 6)]
            print(f"[DEBUG] 找到 hand_free，添加 base 映射")
    print(f"[DEBUG] Viewer模式最终映射数量: {len(mapping)}")
    if mapping and traj is not None and traj.size > 0:
        _print_traj_vs_qpos(model, data, traj, joint_names or [], mapping, waypoint_idx=0,
                            label="Viewer: NPY_wp0 vs qpos_initial")

    num_waypoints = traj.shape[0] if traj is not None and traj.ndim >= 2 else 0
    steps_per_waypoint = max(1, int(SECONDS_PER_WAYPOINT / model.opt.timestep))

    def step_and_sync(playback_step_idx: int):
        if num_waypoints > 0 and traj is not None and mapping:
            wp = min(int(playback_step_idx) // steps_per_waypoint, num_waypoints - 1)
            _apply_velocity_to_target(model, data, mapping, traj[wp], kp=vel_gain, max_vel_lin=max_vel_lin,
                                      max_vel_ang=max_vel_ang)
        mujoco.mj_step(model, data)

    t0 = time.monotonic()
    playback_step_idx = 0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("[INFO] MuJoCo 窗口已打开（灵巧手场景）" if robot_mjcf_path else "[INFO] MuJoCo 窗口已打开（Demo）")
        print("[INFO] 运行最多 {:.0f} 秒。关闭窗口可提前结束。".format(timeout_s))

        # 打印 DOF 报错对应的关节名称
        # 假设报错通常在 DOF 9 或 26，这里通用化打印
        print(f"[DEBUG] Model NV (DOF): {model.nv}, NU (Ctrl): {model.nu}")
        if model.nv > 0:
            # 打印前几个 DOF 对应的关节名，辅助定位
            # 报错的 DOF index 是 0-based
            target_dofs = [9, 26]
            for d in target_dofs:
                if d < model.nv:
                    j_id = model.dof_jntid[d]
                    j_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j_id)
                    print(f"[DEBUG] DOF {d} -> Joint ID {j_id} -> Name: {j_name}")

        while viewer.is_running() and (time.monotonic() - t0) < timeout_s:
            step_and_sync(playback_step_idx)
            playback_step_idx += 1
            viewer.sync()

        # 轨迹播放结束后，继续保持仿真，直到手动关闭窗口
        print("[INFO] 轨迹播放结束，进入保持模式 (Hold)... 请手动关闭窗口。")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

        return True


def _urdf_package_root(urdf_path: str) -> str:
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    if os.path.isdir(os.path.join(urdf_dir, "meshes")):
        return urdf_dir
    parent = os.path.dirname(urdf_dir)
    if os.path.isdir(os.path.join(parent, "meshes")):
        return parent
    return urdf_dir


def _inject_urdf_inertials(urdf_path: str) -> Optional[str]:
    """为缺 inertial 的 link 注入最小惯性；全有则返回 None。"""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    needs_patch = False
    for link in root.findall(".//link"):
        if link.find("inertial") is None:
            needs_patch = True
            inertial = ET.SubElement(link, "inertial")
            ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
            ET.SubElement(inertial, "mass", value="0.05")
            ET.SubElement(
                inertial,
                "inertia",
                ixx="1e-4",
                ixy="0",
                ixz="0",
                iyy="1e-4",
                iyz="0",
                izz="1e-4",
            )
    if not needs_patch:
        return None
    pkg_root = _urdf_package_root(urdf_path)
    fd, path = tempfile.mkstemp(suffix=".urdf", prefix="hand_patched_", dir=pkg_root)
    os.close(fd)
    tree.write(
        path,
        encoding="utf-8",
        default_namespace="",
        xml_declaration=True,
        method="xml",
    )
    return path


def _get_robot_urdf_path(manip_cfg_file: str) -> str:
    manip_path = join_path(get_manip_configs_path(), manip_cfg_file)
    manip_cfg = load_yaml(manip_path)
    robot_file = manip_cfg.get("robot_file", "")
    if not robot_file:
        return ""
    robot_path = join_path(get_robot_configs_path(), robot_file)
    if not os.path.exists(robot_path):
        return ""
    robot_cfg = load_yaml(robot_path)
    urdf_rel = robot_cfg.get("robot_cfg", {}).get("kinematics", {}).get("urdf_path", "")
    if not urdf_rel:
        return ""
    return os.path.join(get_assets_path(), urdf_rel)


def _world_cfg_to_mjcf(
        world_cfg: Any,
        include_hand_placeholder: bool = True,
        include_hand_origin_body: bool = False,
        npy_path: str = "",
        scene_path: Any = None,
        friction: float = 1.0,
        object_deformable: Optional[bool] = None,
        condim: int = 6,
        hand_init_pose: Optional[np.ndarray] = None,
        timestep: float = 0.0005,
) -> str:
    """从 world_cfg 生成 MJCF（桌面+物体+可选手占位/hand_origin）。
    object_deformable: 若指定则覆盖 world_cfg 中 mesh 的 deformable；否则用 mesh 配置。
    condim: 接触维度，6 表示 3 平移+3 旋转（含扭转/滚动摩擦），手与物体均使用。
    """

    def _project_root() -> str:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _resolve_scene_cfg_dir(scene_path_value: Any) -> Optional[str]:
        if not scene_path_value:
            return None
        sp = (
            scene_path_value[0]
            if isinstance(scene_path_value, (list, tuple)) and len(scene_path_value) > 0
            else scene_path_value
        )
        if not isinstance(sp, str) or not sp.strip():
            return None
        sp = os.path.expanduser(sp.strip())
        candidates: List[str] = []
        if os.path.isabs(sp):
            candidates.append(sp)
        else:
            # 常见：scene_cfg 与 *_grasp.npy 在同一输出树内，因此先相对 npy 所在目录解析
            if npy_path:
                candidates.append(os.path.join(os.path.dirname(os.path.abspath(npy_path)), sp))
            candidates.append(os.path.join(_project_root(), sp))
        for c in candidates:
            if os.path.exists(c):
                return os.path.dirname(os.path.abspath(c))
        return None

    def _resolve_existing_path(path: Any) -> Optional[str]:
        if not path or not isinstance(path, str):
            return None
        p = os.path.expanduser(path.strip())
        if not p:
            return None
        base_dirs: List[str] = []
        scene_dir = _resolve_scene_cfg_dir(scene_path)
        if scene_dir:
            base_dirs.append(scene_dir)
        if npy_path:
            base_dirs.append(os.path.dirname(os.path.abspath(npy_path)))
        base_dirs.append(_project_root())
        base_dirs.append(get_assets_path())
        candidates = [p] if os.path.isabs(p) else []
        candidates += [os.path.join(b, p) for b in base_dirs if b]
        for c in candidates:
            if c and os.path.exists(c):
                return os.path.abspath(c)
        return None

    table_pos, table_size = "0 0 -0.4", "4 3 0.02"
    obj_pos, obj_quat = "0 0 0.46", "1 0 0 0"
    obj_mesh_file = obj_scale = None

    if not world_cfg:
        raise ValueError(
            f"world_cfg 为空，无法构建场景；npy_path={npy_path!r}, scene_path={scene_path!r}"
        )
    cfg = world_cfg[0] if isinstance(world_cfg, (list, tuple)) and len(world_cfg) > 0 else world_cfg
    if not isinstance(cfg, dict):
        raise ValueError(
            f"world_cfg 类型异常（期望 dict 或 [dict]），实际={type(cfg)}；npy_path={npy_path!r}, scene_path={scene_path!r}"
        )
    if "cuboid" in cfg and "table" in cfg["cuboid"]:
        t = cfg["cuboid"]["table"]
        p = t.get("pose", [0, 0, -0.1, 1, 0, 0, 0])
        d = t.get("dims", [0.8, 0.6, 0.04])
        table_pos = f"{p[0]} {p[1]} {p[2]}"
        table_size = f"{d[0] / 2} {d[1] / 2} {d[2] / 2}"
    if not ("mesh" in cfg and cfg["mesh"]):
        raise ValueError(
            f"world_cfg 缺少 mesh 配置，无法构建物体；npy_path={npy_path!r}, scene_path={scene_path!r}"
        )
    name = next(iter(cfg["mesh"]))
    m = cfg["mesh"][name]
    p = m.get("pose", [0, 0, 0.46, 1, 0, 0, 0])
    obj_pos = f"{p[0]} {p[1]} {p[2]}"
    if len(p) >= 7:
        obj_quat = f"{p[3]} {p[4]} {p[5]} {p[6]}"
    print(f"[DEBUG] 物体位置: pos={obj_pos}, quat={obj_quat} (来自 world_cfg.mesh['{name}'].pose)")
    file_path = m.get("file_path", "")
    if not file_path:
        raise ValueError(
            f"world_cfg.mesh['{name}'] 缺少 file_path；npy_path={npy_path!r}, scene_path={scene_path!r}"
        )
    resolved = _resolve_existing_path(file_path)
    if not resolved:
        raise FileNotFoundError(
            f"找不到物体 mesh 文件：file_path={file_path!r}（可能是相对 scene_cfg 的路径）; "
            f"npy_path={npy_path!r}, scene_path={scene_path!r}"
        )
    obj_mesh_file = resolved
    s = m.get("scale", [1.0, 1.0, 1.0])
    s = np.atleast_1d(np.asarray(s))
    obj_scale = f"{s[0]} {s[1]} {s[2]}"
    mesh_name = os.path.splitext(os.path.basename(obj_mesh_file))[0]
    mesh_path_abs = os.path.abspath(obj_mesh_file).replace("\\", "/")
    deformable = bool(object_deformable if object_deformable is not None else m.get("deformable", False))
    young = float(m.get("young", 3e4))
    poisson = float(m.get("poisson", 0.2))
    flex_damping = float(m.get("damping", 0.5))
    # flexcomp 类型：mesh=2D 三角面，gmsh=3D 四面体等（.msh 或 mesh.flex_type 指定）
    flex_type = m.get("flex_type", "").strip().lower() or (
        "gmsh" if (obj_mesh_file or "").lower().endswith(".msh") else "mesh")

    if deformable:
        # MuJoCo 3.0+ flexcomp：type=mesh 为 2D 可变形面，type=gmsh 为 3D 体网格（.msh）
        # 接触略软、弹性带阻尼，减少“爆开”与振荡
        obj_body_content = f"""      <flexcomp name="obj_flex" type="{flex_type}" file="{mesh_path_abs}" scale="{obj_scale}" mass="0.3" rgba="0.2 0.6 0.9 1" radius="0.002">
        <contact condim="{condim}" friction="{friction} 0.005 0.0001" solref="0.02 0.95" solimp="0.9 0.95 0.001 0.5 2" selfcollide="none"/>
        <edge damping="1"/>
        <elasticity young="{young}" poisson="{poisson}" damping="{flex_damping}"/>
      </flexcomp>"""
        mesh_asset = ""
    else:
        # 刚体：freejoint + mesh geom
        obj_body_content = f"""      <freejoint name="object_joint"/>
      <geom name="obj" type="mesh" mesh="{mesh_name}" rgba="0.2 0.6 0.9 1" mass="0.3" friction="1 0.005 0.0001" condim="{condim}" contype="1" conaffinity="1" solref="1 0.95" solimp="0.0 0 0.1 0.5 2"/>"""
        mesh_asset = f'    <mesh name="{mesh_name}" file="{mesh_path_abs}" scale="{obj_scale}"/>'

    hand_block = ""
    if include_hand_placeholder:
        hand_block = """
    <body name="hand_base" pos="0.2 0 0.5">
      <freejoint name="hand_free"/>
      <geom name="hand_vis" type="box" size="0.04 0.06 0.02" rgba="0.9 0.3 0.2 0.9"/>
    </body>"""
    elif include_hand_origin_body:
        # hand_base 位置设为 (0, 0, 0)，这样 freejoint 的位置就是绝对位置（与 robot_pose 中的位置一致）
        hand_block = """
    <body name="hand_base" pos="0 0 0">
      <freejoint name="hand_free"/>
      <inertial pos="0 0 0" mass="0.01" diaginertia="1e-5 1e-5 1e-5"/>
      <body name="hand_origin"/>
    </body>"""

    print(f"[DEBUG] _world_cfg_to_mjcf: 使用 timestep={timestep} s")
    return f"""<mujoco model="grasp_eval">
  <option timestep="{timestep}" gravity="0 0 0" cone="elliptic" impratio="10"/>
  <default>
    <geom condim="{condim}" solimp="0.009 0.0095 1 0.5 2" solref="0.05 1"/>
  </default>
  <asset>
{mesh_asset}
  </asset>
  <worldbody>
    <body name="table" pos="{table_pos}">
      <geom name="table" type="box" size="{table_size}" rgba="0.6 0.5 0.4 1" condim="{condim}" friction="{friction} 0.005 0.0001"/>
    </body>
    <body name="object" pos="{obj_pos}" quat="{obj_quat}">
{obj_body_content}
    </body>{hand_block}
  </worldbody>
</mujoco>
"""


def build_mujoco_scene_for_grasp(
        world_cfg: Any,
        robot_mjcf_path: str,
        joint_names: List[str],
        npy_path: str = "",
        scene_path: Any = None,
        friction: float = 1.0,
        object_deformable: Optional[bool] = None,
        save_xml_path: Optional[str] = None,
        solref: Optional[Tuple[float, float]] = None,
        solimp: Optional[Tuple[float, float, float, float, float]] = None,
        condim: int = 6,
        timestep: float = 0.0005,
) -> Optional[Tuple[Any, Any]]:
    """从 world_cfg 建 MuJoCo 场景；有 robot_mjcf_path 则 attach 灵巧手。
    condim=6 时接触含 3 平移+3 旋转（扭转/滚动摩擦），手与物体均使用。
    """
    import mujoco
    if not robot_mjcf_path or not robot_mjcf_path.strip():
        xml = _world_cfg_to_mjcf(
            world_cfg, npy_path=npy_path, scene_path=scene_path, friction=friction,
            object_deformable=object_deformable, condim=condim, hand_init_pose=None, timestep=timestep,
        )
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        print(f"[DEBUG] 模型创建后实际 timestep: {model.opt.timestep} s")
        return (model, data)

    # 解析 mjcf_path：如果传入的已经是绝对路径且存在，直接使用；否则尝试解析相对路径
    mjcf_path = robot_mjcf_path.strip()
    if os.path.isabs(mjcf_path) and os.path.isfile(mjcf_path):
        mjcf_path = os.path.abspath(mjcf_path)
    else:
        # 相对路径：相对于项目根目录（_script_dir 的父目录）
        if not os.path.isabs(mjcf_path):
            _project_root = os.path.dirname(_script_dir)
            mjcf_path = os.path.join(_project_root, mjcf_path)
        mjcf_path = os.path.abspath(mjcf_path)

    if not os.path.isfile(mjcf_path):
        raise FileNotFoundError(f"手部 MJCF 文件不存在: {mjcf_path}")

    print(f"[INFO] 使用手部 MJCF: {mjcf_path}")
    print(f"[DEBUG] build_mujoco_scene_for_grasp: 使用 timestep={timestep} s")

    # 获取手部初始位置（第一个路点的前7个值）用于调整物体位置
    hand_init_pose_for_obj = None
    if joint_names and len(joint_names) > 0:
        # 尝试从 npy_path 读取 robot_pose 获取手部初始位置
        try:
            npy_data = np.load(npy_path, allow_pickle=True).item() if npy_path and os.path.isfile(npy_path) else None
            if npy_data and "robot_pose" in npy_data:
                robot_pose = np.asarray(npy_data["robot_pose"])
                if robot_pose.ndim >= 2 and robot_pose.shape[-1] >= 7:
                    # 取第一个路点的前7个值（手部base位姿）
                    if robot_pose.ndim == 4:
                        hand_init_pose_for_obj = robot_pose[0, 0, 0, :7]
                    elif robot_pose.ndim == 3:
                        hand_init_pose_for_obj = robot_pose[0, 0, :7]
                    elif robot_pose.ndim == 2:
                        hand_init_pose_for_obj = robot_pose[0, :7]
        except Exception as e:
            print(f"[WARN] 无法从 npy 文件读取手部初始位置: {e}")

    scene_xml = _world_cfg_to_mjcf(
        world_cfg,
        include_hand_placeholder=False,
        include_hand_origin_body=True,
        npy_path=npy_path,
        scene_path=scene_path,
        friction=friction,
        object_deformable=object_deformable,
        condim=condim,
        hand_init_pose=hand_init_pose_for_obj,
        timestep=timestep,
    )
    scene_spec = mujoco.MjSpec.from_string(scene_xml)
    hand_origin_body = scene_spec.body("hand_origin")
    if hand_origin_body is None:
        raise RuntimeError("hand_origin body not found")
    hand_frame = hand_origin_body.add_frame()

    # 直接加载 MJCF 文件
    mjcf_dir = os.path.dirname(mjcf_path)
    original_cwd = os.getcwd()
    os.chdir(mjcf_dir)
    mjcf_filename = os.path.basename(mjcf_path)
    hand_model = mujoco.MjModel.from_xml_path(mjcf_filename)
    os.chdir(original_cwd)

    # 保存临时文件用于注入 friction
    hand_mjcf_path = tempfile.mktemp(suffix=".xml", prefix="hand_mjcf_", dir=mjcf_dir)
    mujoco.mj_saveLastXML(hand_mjcf_path, hand_model)
    # 为手部 MJCF 中所有 geom 注入 friction，使保存的合并 XML 中手部也有摩擦属性
    with open(hand_mjcf_path, "r", encoding="utf-8") as f:
        hand_xml_content = f.read()
    with open(hand_mjcf_path, "w", encoding="utf-8") as f:
        f.write(_inject_geom_friction_in_mjcf(hand_xml_content, friction, condim=condim))

    original_cwd = os.getcwd()
    os.chdir(mjcf_dir)
    hand_spec = mujoco.MjSpec.from_file(os.path.basename(hand_mjcf_path))
    scene_spec.attach(hand_spec, frame=hand_frame, prefix="hand_")
    model = scene_spec.compile()
    os.chdir(original_cwd)
    data = mujoco.MjData(model)
    print(f"[DEBUG] 模型创建后实际 timestep: {model.opt.timestep} s")
    # 统一摩擦与接触：手/物体/桌面 geom 使用与场景一致的 condim、friction、solref、solimp
    _apply_unified_friction(model, friction, solref=solref, solimp=solimp, condim=condim)
    if save_xml_path:
        mujoco.mj_saveLastXML(save_xml_path, model)
        # mj_saveLastXML 不会把 model.geom_friction/solref/solimp 写回 XML，故后处理：给所有 <geom> 注入
        with open(save_xml_path, "r", encoding="utf-8") as f:
            saved_content = f.read()
        with open(save_xml_path, "w", encoding="utf-8") as f:
            f.write(_inject_geom_contact_in_mjcf(
                saved_content, friction,
                solref=solref or DEFAULT_SOLREF,
                solimp=solimp or DEFAULT_SOLIMP,
                condim=condim,
            ))
        print(f"[INFO] 已保存合并后的场景 MJCF（含 friction/solref/solimp）: {save_xml_path}")
    if hand_mjcf_path and os.path.isfile(hand_mjcf_path):
        os.remove(hand_mjcf_path)
    return (model, data)


def _inject_geom_friction_in_mjcf(
        content: str, friction: float, condim: Optional[int] = None
) -> str:
    """在 MJCF 字符串中为没有 friction 的 <geom> 注入 friction（及可选的 condim），便于保存的 XML 中可见。"""
    return _inject_geom_contact_in_mjcf(content, friction, solref=None, solimp=None, condim=condim)


def _inject_geom_contact_in_mjcf(
        content: str,
        friction: float,
        solref: Optional[Tuple[float, float]] = None,
        solimp: Optional[Tuple[float, float, float, float, float]] = None,
        condim: Optional[int] = None,
) -> str:
    """在 MJCF 中为 <geom> 注入 friction（及可选的 solref、solimp、condim），便于保存的 XML 中可见。"""
    friction_attr = f' friction="{friction} 0.005 0.0001"'
    solref_attr = f' solref="{solref[0]} {solref[1]}"' if solref is not None else ""
    solimp_attr = f' solimp="{solimp[0]} {solimp[1]} {solimp[2]} {solimp[3]} {solimp[4]}"' if solimp is not None else ""
    condim_attr = f' condim="{condim}"' if condim is not None else ""

    def repl(m: re.Match) -> str:
        tag = m.group(0)
        add: List[str] = []
        if "friction=" not in tag:
            add.append(friction_attr)
        if solref_attr and "solref=" not in tag:
            add.append(solref_attr)
        if solimp_attr and "solimp=" not in tag:
            add.append(solimp_attr)
        if condim_attr and "condim=" not in tag:
            add.append(condim_attr)
        if not add:
            return tag
        extra = "".join(add)
        if tag.rstrip().endswith("/>"):
            return tag.replace("/>", extra + "/>")
        return tag.replace(">", extra + ">")

    return re.sub(r"<geom[^>]*>", repl, content)


# 灵巧手接触默认：与可变形体 contact 一致，便于抓取稳定
DEFAULT_SOLREF = (0.02, 0.95)
DEFAULT_SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)


def _apply_unified_friction(
        model: Any,
        friction: float,
        solref: Optional[Tuple[float, float]] = None,
        solimp: Optional[Tuple[float, float, float, float, float]] = None,
        condim: int = 6,
) -> None:
    """将手/物体/桌面的 geom 的 condim、摩擦与 solref/solimp 设为与场景统一（body 名 table、object 或 hand_ 前缀）。"""
    import mujoco
    solref = solref or DEFAULT_SOLREF
    solimp = solimp or DEFAULT_SOLIMP
    for i in range(model.ngeom):
        body_id = model.geom_bodyid[i]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        is_hand = body_name.startswith("hand_")
        is_table = body_name == "table"
        is_object = body_name == "object" or geom_name == "obj"
        if is_hand or is_table or is_object:
            model.geom_condim[i] = condim
            model.geom_friction[i, 0] = friction
            model.geom_friction[i, 1] = 0.005
            model.geom_friction[i, 2] = 0.0001
            model.geom_solref[i, 0] = solref[0]
            model.geom_solref[i, 1] = solref[1]
            model.geom_solimp[i, 0] = solimp[0]
            model.geom_solimp[i, 1] = solimp[1]
            model.geom_solimp[i, 2] = solimp[2]
            model.geom_solimp[i, 3] = solimp[3]
            model.geom_solimp[i, 4] = solimp[4]


def _build_joint_qpos_map(
        model: Any, joint_names: List[str], nq_traj: int
) -> List[Tuple[int, int, int, int, int]]:
    """(traj_start, qpos_start, ndof_q, dof_start, ndof_v) 映射，供写 data.qpos 或速度控制写 data.qvel。"""
    import mujoco
    mapping: List[Tuple[int, int, int, int, int]] = []
    has_hand_free = False
    for cand in ("hand_free", "hand_freejoint", "freejoint", "free"):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, cand)
        if jid < 0 or model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        qpos_adr = model.jnt_qposadr[jid]
        dof_adr = model.jnt_dofadr[jid]
        mapping.append((0, int(qpos_adr), 7, int(dof_adr), 6))
        has_hand_free = True
        break
    if not has_hand_free and nq_traj >= 7:
        for jid in range(model.njnt):
            if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
                continue
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
            bid = model.jnt_bodyid[jid]
            bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if "hand" in jname.lower() or "hand" in bname.lower():
                qpos_adr = model.jnt_qposadr[jid]
                dof_adr = model.jnt_dofadr[jid]
                mapping.append((0, int(qpos_adr), 7, int(dof_adr), 6))
                has_hand_free = True
                break
    n_finger = nq_traj - 7 if (has_hand_free or nq_traj > 7) else nq_traj
    traj_finger_start = 7 if (has_hand_free or nq_traj == 29) else 0
    if n_finger > 22:
        n_finger = 22
    # joint_names 通常只包含手指关节，应该按照顺序使用，而不是取最后几个
    # 如果 joint_names 的长度等于或大于 n_finger，使用前 n_finger 个（按顺序）
    # 这样可以确保轨迹中的索引和 joint_names 的索引一一对应
    if len(joint_names) >= n_finger:
        finger_names = joint_names[:n_finger]  # 使用前 n_finger 个，保持顺序一致
    else:
        finger_names = list(joint_names)  # 如果 joint_names 少于 n_finger，使用全部
        n_finger = len(finger_names)  # 更新 n_finger 为实际可用的关节数
    matched_count = 0
    matched_joints = []
    for i, jname in enumerate(finger_names):
        if i >= n_finger:
            break
        # 尝试多种名称格式：由于 attach 时使用了 prefix="hand_"，所有关节名称都会有 hand_ 前缀
        # 支持的格式：
        # 1. hand_<jname> - 标准格式（attach 后）：hand_rh_THJ5 或 hand_finger1_joint1
        # 2. <jname> - 原始名称：rh_THJ5 或 finger1_joint1（某些情况下可能没有前缀）
        name_candidates = [
            "hand_" + jname,  # 优先尝试：attach 后的标准格式
            jname,  # 备选：原始名称
        ]

        jid = -1
        matched_name = None
        for name_to_try in name_candidates:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name_to_try)
            if jid >= 0:
                matched_name = name_to_try
                matched_count += 1
                matched_joints.append((i, jname, matched_name))
                break

        if jid < 0:
            # 如果所有候选名称都失败，打印警告但继续处理其他关节
            print(f"[WARN] 无法找到关节 '{jname}' (索引 {i})，尝试的名称: {name_candidates}")
            continue

        qpos_adr = model.jnt_qposadr[jid]
        jtype = model.jnt_type[jid]
        ndof_q = 7 if jtype == mujoco.mjtJoint.mjJNT_FREE else 1
        ndof_v = 6 if jtype == mujoco.mjtJoint.mjJNT_FREE else 1
        dof_adr = model.jnt_dofadr[jid]
        mapping.append((traj_finger_start + i, int(qpos_adr), ndof_q, int(dof_adr), ndof_v))

    # 打印匹配结果摘要
    if matched_count > 0:
        print(f"[INFO] 关节匹配成功: {matched_count}/{n_finger} 个关节")
        if matched_count <= 5:
            for idx, orig_name, matched_name in matched_joints:
                print(f"[DEBUG]   [{idx}] {orig_name} -> {matched_name}")
        else:
            print(f"[DEBUG]   前3个: {[f'{orig}->{matched}' for _, orig, matched in matched_joints[:3]]}")
            print(f"[DEBUG]   后3个: {[f'{orig}->{matched}' for _, orig, matched in matched_joints[-3:]]}")

    if matched_count < n_finger:
        print(f"[WARN] 关节匹配不完整：期望 {n_finger} 个关节，实际匹配 {matched_count} 个")
        print(f"[DEBUG] joint_names 前几个: {joint_names[:min(5, len(joint_names))]}")
        # 打印模型中的所有手部关节名称（以 hand_ 开头）
        hand_joints = []
        for jid in range(model.njnt):
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
            if jname.startswith("hand_"):
                hand_joints.append(jname)
        print(f"[DEBUG] 模型中以 'hand_' 开头的关节 ({len(hand_joints)} 个): {hand_joints[:min(10, len(hand_joints))]}")

    print(f"[DEBUG] 最终映射数量: {len(mapping)} (期望: {1 + n_finger if has_hand_free else n_finger})")

    return mapping


def _print_traj_vs_qpos(
        model: Any,
        data: Any,
        traj: np.ndarray,
        joint_names: List[str],
        mapping: List[Tuple[int, int, int, int, int]],
        waypoint_idx: int = 0,
        label: str = "traj_vs_qpos",
) -> None:
    """打印 NPY 轨迹与 MuJoCo qpos 逐项对比，便于排查关节对应错误。"""
    import mujoco
    q = np.asarray(traj).flatten()
    if traj.ndim >= 2:
        q = np.asarray(traj[waypoint_idx]).flatten()
    nq = len(q)

    # 从 model 根据 qpos_start 反查关节名
    def _qpos_to_joint_name(qpos_start: int, ndof_q: int) -> str:
        for jid in range(model.njnt):
            if model.jnt_qposadr[jid] == qpos_start:
                return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"jnt_{jid}"
        return f"qpos_{qpos_start}"

    print(f"\n--- [{label}] NPY 轨迹 vs MuJoCo qpos (waypoint={waypoint_idx}, traj_dim={nq}) ---")
    for k, (traj_start, qpos_start, ndof_q, dof_start, ndof_v) in enumerate(mapping):
        end = min(traj_start + ndof_q, nq)
        n_compare = end - traj_start
        if n_compare <= 0:
            continue
        traj_vals = q[traj_start:end]
        qpos_vals = data.qpos[qpos_start: qpos_start + n_compare]
        if traj_start == 0:
            npy_name = "base(xyz,qw,qx,qy,qz)"
        else:
            idx_in_joint = traj_start - 7
            npy_name = joint_names[idx_in_joint] if idx_in_joint < len(joint_names) else f"traj[{traj_start}]"
        mj_name = _qpos_to_joint_name(qpos_start, ndof_q)
        diff = np.abs(np.asarray(traj_vals) - np.asarray(qpos_vals))
        max_diff = float(np.max(diff)) if diff.size else 0.0
        ok = "OK" if max_diff < 1e-5 else "DIFF"
        print(f"  [{k}] NPY {npy_name:32s} traj[{traj_start}:{end}] = {traj_vals}")
        print(
            f"      MJ  {mj_name:32s} qpos[{qpos_start}:{qpos_start + n_compare}] = {qpos_vals}  ({ok} max_diff={max_diff:.6f})")
    print("---\n")


def _apply_velocity_to_target(
        model: Any,
        data: Any,
        mapping: List[Tuple[int, int, int, int, int]],
        target_q: np.ndarray,
        kp: float = VEL_GAIN,
        max_vel_lin: float = MAX_VEL_LIN,
        max_vel_ang: float = MAX_VEL_ANG,
) -> None:
    """base 与手指均按轨迹做位置跟随（直接写 qpos）。原先手指用速度控制，但 mj_step 会覆盖 qvel 且无 actuator 时几乎不动，故改为直接设 qpos。"""
    import mujoco
    q = np.asarray(target_q).flatten()
    for traj_start, qpos_start, ndof_q, dof_start, ndof_v in mapping:
        end = min(traj_start + ndof_q, len(q))
        target = q[traj_start:end]
        n_write = end - traj_start
        if ndof_q == 7:
            # base（free joint）：位置跟随
            data.qpos[qpos_start: qpos_start + 7] = target
            data.qvel[dof_start: dof_start + 6] = 0.0
        else:
            # 手指等 1-dof 关节：直接位置跟随（与 NPY 一致），否则动力学会覆盖 qvel 导致手指不动
            data.qpos[qpos_start: qpos_start + n_write] = target
            data.qvel[dof_start: dof_start + ndof_v] = 0.0


def run_one_grasp_physics(
        model: Any,
        data: Any,
        traj: np.ndarray,
        joint_names: List[str],
        waypoint_duration_steps: int = 100,
        hold_steps: int = 50,
        vel_gain: float = VEL_GAIN,
        max_vel_lin: float = MAX_VEL_LIN,
        max_vel_ang: float = MAX_VEL_ANG,
) -> None:
    """回放轨迹（速度控制趋近路点）并保持 hold_steps 步。"""
    import mujoco
    traj = np.asarray(traj)
    if traj.size == 0:
        return
    if traj.ndim == 1:
        traj = traj.reshape(1, -1)
    nq_traj = traj.shape[-1]
    print(f"[DEBUG] Physics模式: nq_traj={nq_traj}, joint_names数量={len(joint_names) if joint_names else 0}")
    # 打印手部初始位置（第一个路点的前7个值）
    if traj.shape[0] > 0:
        hand_init_pos = traj[0, :7]
        print(
            f"[DEBUG] 手部初始位置: pos=({hand_init_pos[0]:.4f}, {hand_init_pos[1]:.4f}, {hand_init_pos[2]:.4f}), quat=({hand_init_pos[3]:.4f}, {hand_init_pos[4]:.4f}, {hand_init_pos[5]:.4f}, {hand_init_pos[6]:.4f})")
    mapping = _build_joint_qpos_map(model, joint_names or [], nq_traj)
    if not mapping:
        print("[DEBUG] 映射为空，尝试查找 hand_free 关节")
        if nq_traj >= 7:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hand_free")
            if jid >= 0:
                mapping = [(0, int(model.jnt_qposadr[jid]), 7, int(model.jnt_dofadr[jid]), 6)]
                print(f"[DEBUG] 找到 hand_free，添加 base 映射")
        if not mapping:
            print("[ERROR] 无法建立关节映射，退出物理仿真")
            return
    print(f"[DEBUG] Physics模式最终映射数量: {len(mapping)}")

    mujoco.mj_resetData(model, data)
    # 打印 NPY 与 MuJoCo qpos 对比（初始状态 vs 第 0 个路点目标）
    _print_traj_vs_qpos(model, data, traj, joint_names or [], mapping, waypoint_idx=0,
                        label="Physics: NPY_wp0 vs qpos_initial")
    num_waypoints = traj.shape[0]
    for wp in range(num_waypoints):
        for _ in range(waypoint_duration_steps):
            _apply_velocity_to_target(model, data, mapping, traj[wp], kp=vel_gain, max_vel_lin=max_vel_lin,
                                      max_vel_ang=max_vel_ang)
            mujoco.mj_step(model, data)
    for _ in range(hold_steps):
        mujoco.mj_step(model, data)
    # 回放结束：打印最后一帧 NPY 与当前 qpos 对比
    last_wp = num_waypoints - 1 if num_waypoints > 0 else 0
    _print_traj_vs_qpos(model, data, traj, joint_names or [], mapping, waypoint_idx=last_wp,
                        label="Physics: NPY_last vs qpos_final")


def run_mujoco_sim_stub(
        world_cfg: Any,
        joint_names: List[str],
        traj: np.ndarray,
        grasp_index: int,
        npy_path: str,
        robot_mjcf_path: str = "",
        scene_path: Any = None,
        friction: float = 1.0,
        object_deformable: Optional[bool] = None,
        save_xml_path: Optional[str] = None,
        solref: Optional[Tuple[float, float]] = None,
        solimp: Optional[Tuple[float, float, float, float, float]] = None,
        condim: int = 6,
        waypoint_duration_steps: int = 1500,
        hold_steps: int = 500,
        vel_gain: float = VEL_GAIN,
        max_vel_lin: float = MAX_VEL_LIN,
        max_vel_ang: float = MAX_VEL_ANG,
        timestep: float = 0.0005,
) -> None:
    """建场景并回放轨迹。"""
    print(f"[DEBUG] run_mujoco_sim_stub: 使用 timestep={timestep} s")
    scene = build_mujoco_scene_for_grasp(
        world_cfg,
        robot_mjcf_path=robot_mjcf_path,
        joint_names=joint_names,
        npy_path=npy_path,
        scene_path=scene_path,
        friction=friction,
        object_deformable=object_deformable,
        save_xml_path=save_xml_path,
        solref=solref,
        solimp=solimp,
        condim=condim,
        timestep=timestep,
    )
    if scene is None:
        return
    model, data = scene
    run_one_grasp_physics(
        model, data, traj, joint_names,
        waypoint_duration_steps=waypoint_duration_steps,
        hold_steps=hold_steps,
        vel_gain=vel_gain,
        max_vel_lin=max_vel_lin,
        max_vel_ang=max_vel_ang,
    )


def evaluate_grasps_for_file(
        npy_path: str,
        max_grasps: int = -1,
        grasp_index: int = 0,
        show_viewer: bool = False,
        viewer_timeout_s: float = 300.0,
        robot_mjcf_path: str = "",
        friction: float = 1.0,
        object_deformable: Optional[bool] = None,
        save_xml_path: Optional[str] = None,
        solref: Optional[Tuple[float, float]] = None,
        solimp: Optional[Tuple[float, float, float, float, float]] = None,
        condim: int = 6,
        waypoint_duration_steps: int = 1500,
        hold_steps: int = 500,
        vel_gain: float = VEL_GAIN,
        max_vel_lin: float = MAX_VEL_LIN,
        max_vel_ang: float = MAX_VEL_ANG,
        timestep: float = 0.0005,
) -> int:
    """对单个 npy 回放；支持 (1,20,3,27) 时只取 robot_pose[0, grasp_index] 即 (3,27)。"""
    data: Dict[str, Any] = np.load(npy_path, allow_pickle=True).item()

    if "robot_pose" not in data:
        print(f"[WARN] 文件中没有 robot_pose，跳过：{npy_path}")
        return 0

    robot_pose = np.asarray(data["robot_pose"])
    if robot_pose.ndim < 2:
        print(f"[WARN] robot_pose 维度异常 ({robot_pose.shape})，跳过：{npy_path}")
        return 0

    # (1, 20, 3, 27)：第 2 维是 20 种抓取，只取 grasp_index 对应的一条轨迹 (3, 27)
    if robot_pose.ndim == 4:
        n_candidates = robot_pose.shape[1]
        g = max(0, min(grasp_index, n_candidates - 1))
        traj = robot_pose[0, g]
        joint_names = data.get("joint_names", [])
        world_cfg = data.get("world_cfg", None)
        scene_path = data.get("scene_path", None)
        print(f"[INFO] 文件：{npy_path}, 抓取候选数={n_candidates}, 使用第 {g} 条, traj_shape={traj.shape}")
        if show_viewer:
            run_mujoco_viewer(
                npy_path=npy_path,
                traj=traj,
                joint_names=joint_names,
                timeout_s=viewer_timeout_s,
                world_cfg=world_cfg,
                robot_mjcf_path=robot_mjcf_path,
                friction=friction,
                object_deformable=object_deformable,
                save_xml_path=save_xml_path,
                solref=solref,
                solimp=solimp,
                condim=condim,
                vel_gain=vel_gain,
                max_vel_lin=max_vel_lin,
                max_vel_ang=max_vel_ang,
                timestep=timestep,
            )
        run_mujoco_sim_stub(
            world_cfg, joint_names, traj, g, npy_path,
            robot_mjcf_path=robot_mjcf_path,
            scene_path=scene_path,
            friction=friction,
            object_deformable=object_deformable,
            save_xml_path=save_xml_path,
            solref=solref,
            solimp=solimp,
            condim=condim,
            waypoint_duration_steps=waypoint_duration_steps,
            hold_steps=hold_steps,
            vel_gain=vel_gain,
            max_vel_lin=max_vel_lin,
            max_vel_ang=max_vel_ang,
            timestep=timestep,
        )
        return 1

    num_grasps = robot_pose.shape[0]
    if max_grasps > 0:
        num_grasps = min(num_grasps, max_grasps)

    joint_names = data.get("joint_names", [])
    world_cfg = data.get("world_cfg", None)
    scene_path = data.get("scene_path", None)

    print(f"[INFO] 文件：{npy_path}, 抓取数={num_grasps}, traj_shape={robot_pose[0].shape}")

    for g in range(num_grasps):
        traj = robot_pose[g]
        if show_viewer and g == 0:
            run_mujoco_viewer(
                npy_path=npy_path,
                traj=traj,
                joint_names=joint_names,
                timeout_s=viewer_timeout_s,
                world_cfg=world_cfg,
                robot_mjcf_path=robot_mjcf_path,
                friction=friction,
                object_deformable=object_deformable,
                save_xml_path=save_xml_path,
                solref=solref,
                solimp=solimp,
                condim=condim,
                vel_gain=vel_gain,
                max_vel_lin=max_vel_lin,
                max_vel_ang=max_vel_ang,
                timestep=timestep,
            )
        run_mujoco_sim_stub(
            world_cfg, joint_names, traj, g, npy_path,
            robot_mjcf_path=robot_mjcf_path,
            scene_path=scene_path,
            friction=friction,
            object_deformable=object_deformable,
            save_xml_path=save_xml_path,
            solref=solref,
            solimp=solimp,
            condim=condim,
            waypoint_duration_steps=waypoint_duration_steps,
            hold_steps=hold_steps,
            vel_gain=vel_gain,
            max_vel_lin=max_vel_lin,
            max_vel_ang=max_vel_ang,
            timestep=timestep,
        )

    return num_grasps


def _load_eval_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """加载 mujoco_eval_grasp_config.yaml；未指定路径时使用脚本同目录下的默认文件。"""
    if config_path is None:
        config_path = os.path.join(_script_dir, "mujoco_eval_grasp_config.yaml")
    config_path = os.path.abspath(config_path)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"评估配置不存在: {config_path}")
    cfg = load_yaml(config_path)
    return cfg if isinstance(cfg, dict) else {}


def main():
    config = _load_eval_config()
    path = config.get("path")
    if path is None or (isinstance(path, str) and not path.strip()):
        raise ValueError("请在 mujoco_eval_grasp_config.yaml 中设置 path（实验目录或 *_grasp.npy 路径）")
    path = path.strip() if isinstance(path, str) else str(path)

    manip_cfg_file = config.get("manip_cfg_file", "sim_shadow/fc.yml")
    manip_cfg_path = join_path(get_manip_configs_path(), manip_cfg_file)
    if not os.path.exists(manip_cfg_path):
        raise FileNotFoundError(f"manip_cfg_file 不存在: {manip_cfg_path}")
    manip_cfg = load_yaml(manip_cfg_path)

    # 从配置中读取 mjcf_path
    robot_mjcf_path = config.get("mjcf_path", "")
    if not robot_mjcf_path or not isinstance(robot_mjcf_path, str) or not robot_mjcf_path.strip():
        raise ValueError("请在 mujoco_eval_grasp_config.yaml 中设置 mjcf_path（手部 MJCF 文件路径）")
    robot_mjcf_path = robot_mjcf_path.strip()

    # 解析路径：支持绝对路径或相对项目根目录（_script_dir 的父目录）
    if not os.path.isabs(robot_mjcf_path):
        _project_root = os.path.dirname(_script_dir)
        robot_mjcf_path = os.path.join(_project_root, robot_mjcf_path)
    robot_mjcf_path = os.path.abspath(robot_mjcf_path)

    if not os.path.isfile(robot_mjcf_path):
        raise FileNotFoundError(f"手部 MJCF 文件不存在: {robot_mjcf_path}")

    _manip_display = os.path.abspath(os.path.normpath(manip_cfg_path))
    _mjcf_display = os.path.abspath(os.path.normpath(robot_mjcf_path))
    print(f"[INFO] manip 配置: {_manip_display}, 灵巧手 MJCF: {_mjcf_display}")

    friction = float(config.get("friction", 10.0))
    if config.get("use_friction_from_manip", True):
        if "grasp_cfg" in manip_cfg and "ge_param" in manip_cfg["grasp_cfg"]:
            miu_coef = manip_cfg["grasp_cfg"]["ge_param"].get("miu_coef", [1.0])
            if isinstance(miu_coef, list) and len(miu_coef) > 0:
                friction = float(miu_coef[0])
    print(f"[INFO] 设置摩擦系数: {friction}")

    solref: Optional[Tuple[float, float]] = None
    solimp: Optional[Tuple[float, float, float, float, float]] = None
    sr = config.get("solref")
    if sr is not None and (isinstance(sr, (list, tuple)) and len(sr) >= 2):
        solref = (float(sr[0]), float(sr[1]))
    si = config.get("solimp")
    if si is not None and (isinstance(si, (list, tuple)) and len(si) >= 5):
        solimp = (float(si[0]), float(si[1]), float(si[2]), float(si[3]), float(si[4]))

    condim = int(config.get("condim", 6))
    obj_deform = config.get("object_deformable")
    if obj_deform is not None:
        obj_deform = bool(obj_deform)
    save_xml_path = config.get("save_xml_path")
    if save_xml_path is not None and (not isinstance(save_xml_path, str) or not save_xml_path.strip()):
        save_xml_path = None

    npy_files = find_grasp_npy_files(manip_cfg_file, path)
    if not npy_files:
        path_exp = os.path.expanduser(path)
        if os.path.isabs(path_exp):
            search_root = path_exp if os.path.isdir(path_exp) else os.path.dirname(path_exp)
        else:
            p = path_exp if "graspdata" in path_exp else os.path.join(path_exp, "graspdata")
            search_root = os.path.join(get_output_path(), os.path.splitext(manip_cfg_file)[0], p)
        raise FileNotFoundError(f"未找到 *_grasp.npy，搜索根目录: {search_root}")

    max_grasps = int(config.get("max_grasps", 1))
    grasp_index = int(config.get("grasp_index", 0))
    show_viewer = bool(config.get("show_viewer", True))
    viewer_timeout_s = float(config.get("viewer_timeout_s", 30.0))
    waypoint_duration_steps = int(config.get("waypoint_duration_steps", 1500))
    hold_steps = int(config.get("hold_steps", 500))

    # 读取速度控制参数
    vel_gain = float(config.get("vel_gain", VEL_GAIN))
    max_vel_lin = float(config.get("max_vel_lin", MAX_VEL_LIN))
    max_vel_ang = float(config.get("max_vel_ang", MAX_VEL_ANG))
    print(f"[INFO] 速度控制参数: vel_gain={vel_gain}, max_vel_lin={max_vel_lin}, max_vel_ang={max_vel_ang}")

    # 读取 timestep 配置；过大会导致 QACC 爆炸、仿真不稳定，自动限制在安全范围
    timestep = float(config.get("timestep", 0.0005))
    TIMESTEP_MAX = 0.01  # 超过约 10ms 易不稳定
    if timestep > TIMESTEP_MAX:
        print(f"[WARN] timestep={timestep} s 过大，MuJoCo 易出现 NaN/Inf（QACC 爆炸），已限制为 {TIMESTEP_MAX} s")
        timestep = TIMESTEP_MAX
    print(f"[INFO] MuJoCo timestep: {timestep} s")

    total_grasps = 0
    for npy_path in npy_files:
        try:
            n_g = evaluate_grasps_for_file(
                npy_path,
                max_grasps=max_grasps,
                grasp_index=grasp_index,
                show_viewer=show_viewer,
                viewer_timeout_s=viewer_timeout_s,
                robot_mjcf_path=robot_mjcf_path,
                friction=friction,
                object_deformable=obj_deform,
                save_xml_path=save_xml_path,
                solref=solref,
                solimp=solimp,
                condim=condim,
                waypoint_duration_steps=waypoint_duration_steps,
                hold_steps=hold_steps,
                vel_gain=vel_gain,
                max_vel_lin=max_vel_lin,
                max_vel_ang=max_vel_ang,
                timestep=timestep,
            )
            total_grasps += n_g
            if show_viewer and n_g > 0:
                show_viewer = False
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            print(f"[WARN] 跳过 {npy_path}: {e}")

    print(f"[RESULT] 总回放抓取数={total_grasps}")


if __name__ == "__main__":
    main()
