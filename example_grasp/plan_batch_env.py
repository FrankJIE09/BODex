# ============================================================================
# 标准库导入
# ============================================================================
import time
from typing import Dict, List
import datetime
import os

# ============================================================================
# 第三方库导入
# ============================================================================
import torch
import numpy as np
import argparse

# ============================================================================
# CuRobo 库导入
# ============================================================================
from curobo.geom.sdf.world import WorldConfig
from curobo.wrap.reacher.grasp_solver import GraspSolver, GraspSolverConfig
from curobo.util.world_cfg_generator import get_world_config_dataloader
from curobo.util.logger import setup_logger, log_warn
from curobo.util.save_helper import SaveHelper
from curobo.util_file import (
    get_manip_configs_path,
    join_path,
    load_yaml,
)

# ============================================================================
# PyTorch 性能优化设置
# ============================================================================
torch.backends.cudnn.benchmark = True  # 启用 cuDNN 自动调优，加速卷积运算

torch.backends.cuda.matmul.allow_tf32 = True  # 允许使用 TF32 精度进行矩阵乘法（A100/H100）
torch.backends.cudnn.allow_tf32 = True  # 允许 cuDNN 使用 TF32 精度

# ============================================================================
# 随机种子设置（确保结果可复现）
# ============================================================================
import numpy as np
import random

seed = 123
np.random.seed(seed)  # NumPy 随机种子
torch.manual_seed(seed)  # PyTorch 随机种子
random.seed(seed)  # Python 随机种子


def process_grasp_result(result, save_debug, save_data, save_id):
    """
    处理抓取结果，根据参数选择保存的数据
    
    参数:
        result: 抓取求解器的结果对象
        save_debug: 是否保存调试信息（梯度、法线等）
        save_data: 保存哪些优化步骤的数据（'all', 'final', 'final_and_mid', 'init', 'select_N'）
        save_id: 保存哪些结果ID（None表示保存所有，List表示保存指定的ID）
    
    返回:
        save_traj: 处理后的机器人轨迹
        debug_info: 调试信息（如果save_debug=True）
    """
    # 获取优化过程中的所有轨迹步骤
    traj = result.debug_info["solver"]["steps"][0]
    all_traj = torch.cat(traj, dim=1)  # 拼接所有步骤: [batch*num_seeds, horizon, q_dim]
    batch, horizon = all_traj.shape[:2]  # batch: 批次大小, horizon: 优化步数

    # ========================================================================
    # 根据 save_data 参数选择要保存的优化步骤
    # ========================================================================
    if save_data == "all":
        # 保存所有优化步骤
        select_horizon_lst = list(range(0, horizon))
    elif "select_" in save_data:
        # 保存均匀分布的N个步骤，例如 select_5 表示保存5个步骤
        part_num = int(save_data.split("select_")[-1])
        select_horizon_lst = list(range(0, horizon, horizon // (part_num - 1)))
        select_horizon_lst[-1] = horizon - 1  # 确保包含最后一步
    elif save_data == "init":
        # 只保存初始状态（第一步）
        select_horizon_lst = [0]
    elif save_data == "final" or save_data == "final_and_mid":
        # 只保存最终状态（最后一步）
        select_horizon_lst = [-1]
    else:
        raise NotImplementedError(f"不支持的 save_data 参数: {save_data}")

    # ========================================================================
    # 根据 save_id 参数选择要保存的结果ID
    # ========================================================================
    if save_id is None:
        # 保存所有结果
        save_id_lst = list(range(0, batch))
    elif isinstance(save_id, List):
        # 只保存指定的结果ID
        save_id_lst = save_id
    else:
        raise NotImplementedError(f"save_id 必须是 None 或 List，当前类型: {type(save_id)}")

    # 根据选择的步骤和ID提取轨迹
    save_traj = all_traj[:, select_horizon_lst]  # 选择指定的优化步骤
    save_traj = save_traj[save_id_lst, :]  # 选择指定的结果ID

    # ========================================================================
    # 如果启用调试模式，提取调试信息
    # ========================================================================
    if save_debug:
        # 获取手部接触点数量和对象接触点数量
        n_num = torch.stack(result.debug_info["solver"]["hp"][0]).shape[-2]  # 手部接触点数量
        o_num = torch.stack(result.debug_info["solver"]["op"][0]).shape[-2]  # 对象接触点数量
        
        # 提取各种调试信息轨迹
        hp_traj = torch.stack(result.debug_info["solver"]["hp"][0], dim=1).view(-1, n_num, 3)  # 手部接触点位置
        grad_traj = torch.stack(result.debug_info["solver"]["grad"][0], dim=1).view(-1, n_num, 3)  # 梯度方向
        op_traj = torch.stack(result.debug_info["solver"]["op"][0], dim=1).view(-1, o_num, 3)  # 对象接触点位置
        posi_traj = torch.stack(result.debug_info["solver"]["debug_posi"][0], dim=1).view(
            -1, o_num, 3
        )  # 调试位置
        normal_traj = torch.stack(result.debug_info["solver"]["debug_normal"][0], dim=1).view(
            -1, o_num, 3
        )  # 接触法线方向

        # 组织调试信息字典
        debug_info = {
            "hp": hp_traj,  # 手部接触点 (hand points)
            "grad": grad_traj * 100,  # 梯度方向（放大100倍便于可视化）
            "op": op_traj,  # 对象接触点 (object points)
            "debug_posi": posi_traj,  # 调试位置
            "debug_normal": normal_traj,  # 接触法线
        }

        # 对每个调试信息进行筛选（只保留选择的步骤和ID）
        for k, v in debug_info.items():
            debug_info[k] = v.view((all_traj.shape[0], -1) + v.shape[1:])[:, select_horizon_lst]
            debug_info[k] = debug_info[k][save_id_lst, :]
            debug_info[k] = debug_info[k].view((-1,) + v.shape[1:])
    else:
        # 不保存调试信息
        debug_info = None
        # 如果保存最终和中间结果，需要添加中间结果
        if save_data == "final_and_mid":
            mid_robot_pose = torch.cat(result.debug_info["solver"]["mid_result"][0], dim=1)
            mid_robot_pose = mid_robot_pose[save_id_lst, :]
            save_traj = torch.cat([mid_robot_pose, save_traj], dim=-2)

    return save_traj, debug_info


if __name__ == "__main__":
    # ========================================================================
    # 命令行参数解析
    # ========================================================================
    parser = argparse.ArgumentParser(description="批量生成抓取姿态")

    parser.add_argument(
        "-c",
        "--manip_cfg_file",
        type=str,
        default="fc_leap.yml",
        help="配置文件路径，例如: sim_shadow/fc.yml",
    )

    parser.add_argument(
        "-f",
        "--save_folder",
        type=str,
        default=None,
        help="保存文件夹路径。如果为None，则使用 join_path(manip_cfg_file[:-4], $TIME) 作为保存文件夹",
    )

    parser.add_argument(
        "-m",
        "--save_mode",
        choices=["usd", "npy", "usd+npy", "none"],
        default="npy",
        help="保存模式: 'usd'=USD格式(可可视化), 'npy'=NumPy格式(数据文件), 'usd+npy'=两种格式都保存, 'none'=不保存",
    )

    parser.add_argument(
        "-d",
        "--save_data",
        # choices=['all', 'final', 'final_and_mid', 'init', 'select_{$INT}'],
        default="final_and_mid",
        help="保存哪些优化步骤的数据: 'all'=所有步骤, 'final'=最终结果, 'final_and_mid'=最终和中间结果, 'init'=初始状态, 'select_N'=均匀分布的N个步骤",
    )

    parser.add_argument(
        "-i",
        "--save_id",
        type=int,
        nargs="+",
        default=None,
        help="保存哪些结果ID。例如: -i 0 1 表示只保存第0和第1个结果。如果不指定，则保存所有结果",
    )

    parser.add_argument(
        "-debug",
        "--save_debug",
        action="store_true",
        help="启用调试模式，保存接触法线、梯度等调试信息（必须配合 -m usd 使用）",
    )

    parser.add_argument(
        "-w",
        "--parallel_world",
        type=int,
        default=20,
        help="并行处理的世界数量（同时处理的对象数量）。建议debug模式使用1，正常模式使用20-40",
    )

    parser.add_argument(
        "-k",
        "--skip",
        action="store_true",
        default=False,
        help="若指定则跳过已存在的 *_grasp.npy 文件；默认不跳过，会覆盖或新建",
    )

    # 设置日志级别为警告
    setup_logger("warn")

    # 解析命令行参数
    args = parser.parse_args()
    
    # ========================================================================
    # 加载配置文件
    # ========================================================================
    manip_config_data = load_yaml(join_path(get_manip_configs_path(), args.manip_cfg_file))

    # ========================================================================
    # 创建世界配置数据加载器（用于批量处理多个对象）
    # ========================================================================
    world_generator = get_world_config_dataloader(manip_config_data["world"], args.parallel_world)

    # ========================================================================
    # 确定保存文件夹路径
    # ========================================================================
    if args.save_folder is not None:
        # 如果用户指定了保存文件夹，直接使用
        save_folder = os.path.join(args.save_folder, "graspdata")
    elif manip_config_data["exp_name"] is not None:
        # 如果配置文件中指定了实验名称，使用它
        save_folder = os.path.join(
            args.manip_cfg_file[:-4], manip_config_data["exp_name"], "graspdata"
        )
    else:
        # 否则使用当前时间戳作为文件夹名
        save_folder = os.path.join(
            args.manip_cfg_file[:-4],
            datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
            "graspdata",
        )

    # ========================================================================
    # 创建保存助手对象
    # ========================================================================
    save_helper = SaveHelper(
        robot_file=manip_config_data["robot_file"],  # 机器人URDF文件路径
        save_folder=save_folder,  # 保存文件夹
        task_name="grasp",  # 任务名称
        mode=args.save_mode,  # 保存模式（usd/npy/usd+npy/none）
    )
    
    # 记录总开始时间
    tst = time.time()
    grasp_solver = None  # 抓取求解器（延迟初始化）
    
    # ========================================================================
    # 遍历所有对象，生成抓取姿态
    # ========================================================================
    for world_info_dict in world_generator:
        sst = time.time()  # 记录单个对象的开始时间
        
        # 如果文件已存在且启用了跳过功能，则跳过
        if args.skip and save_helper.exist_piece(world_info_dict["save_prefix"]):
            log_warn(f"skip {world_info_dict['save_prefix']}")
            continue

        # ====================================================================
        # 初始化或更新抓取求解器
        # ====================================================================
        if grasp_solver is None:
            # 第一次使用，创建抓取求解器
            grasp_config = GraspSolverConfig.load_from_robot_config(
                world_model=world_info_dict["world_cfg"],  # 世界模型配置
                manip_name_list=world_info_dict["manip_name"],  # 操作器名称列表
                manip_config_data=manip_config_data,  # 操作器配置数据
                obj_gravity_center=world_info_dict["obj_gravity_center"],  # 对象重心
                obj_obb_length=world_info_dict["obj_obb_length"],  # 对象OBB长度
                use_cuda_graph=False,  # 不使用CUDA图优化
                store_debug=args.save_debug,  # 是否存储调试信息
            )
            grasp_solver = GraspSolver(grasp_config)
            world_info_dict["world_model"] = grasp_solver.world_coll_checker.world_model
        else:
            # 后续使用，更新世界模型（复用求解器以提高效率）
            world_info_dict["world_model"] = world_model = [
                WorldConfig.from_dict(world_cfg) for world_cfg in world_info_dict["world_cfg"]
            ]
            grasp_solver.update_world(
                world_model,  # 新的世界模型
                world_info_dict["obj_gravity_center"],  # 对象重心
                world_info_dict["obj_obb_length"],  # 对象OBB长度
                world_info_dict["manip_name"],  # 操作器名称
            )

        # ====================================================================
        # 求解抓取姿态
        # ====================================================================
        result = grasp_solver.solve_batch_env(return_seeds=grasp_solver.num_seeds)

        # ====================================================================
        # 处理并保存结果
        # ====================================================================
        if args.save_debug:
            # 调试模式：保存详细的调试信息
            robot_pose, debug_info = process_grasp_result(
                result, args.save_debug, args.save_data, args.save_id
            )
            world_info_dict["debug_info"] = debug_info  # 调试信息（梯度、法线等）
            world_info_dict["robot_pose"] = robot_pose.reshape(
                (len(world_info_dict["world_model"]), -1) + robot_pose.shape[1:]
            )
        else:
            # 普通模式：只保存最终结果
            # 计算挤压姿态（squeeze pose）：手指进一步闭合
            squeeze_pose_qpos = torch.cat(
                [
                    result.solution[..., 1, :7],  # 保持根姿态不变
                    result.solution[..., 1, 7:] * 2 - result.solution[..., 0, 7:],  # 手指进一步闭合
                ],
                dim=-1,
            )
            # 拼接所有姿态：预抓取、抓取、挤压
            all_hand_pose_qpos = torch.cat(
                [result.solution, squeeze_pose_qpos.unsqueeze(-2)], dim=-2
            )
            world_info_dict["robot_pose"] = all_hand_pose_qpos
            world_info_dict["contact_point"] = result.contact_point  # 接触点位置
            world_info_dict["contact_frame"] = result.contact_frame  # 接触坐标系
            world_info_dict["contact_force"] = result.contact_force  # 接触力
            world_info_dict["grasp_error"] = result.grasp_error  # 抓取误差
            world_info_dict["dist_error"] = result.dist_error  # 距离误差
        
        # 记录单个对象的处理时间
        log_warn(f"Sinlge Time: {time.time()-sst}")
        
        # 保存结果到文件
        save_helper.save_piece(world_info_dict)

    # 记录总处理时间
    log_warn(f"Total Time: {time.time()-tst}")
