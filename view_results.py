#!/usr/bin/env python3
"""
查看 BODex 运行结果的工具脚本
支持查看 .npy 文件的内容和统计信息
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

def view_npy_file(file_path):
    """查看单个 .npy 文件的内容"""
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"📁 文件: {file_path}")
    print(f"{'='*60}\n")
    
    try:
        data = np.load(file_path, allow_pickle=True)
        
        # 如果是字典格式
        if isinstance(data, np.ndarray) and data.dtype == object:
            data = data.item()
        
        if isinstance(data, dict):
            print("📊 数据内容:")
            print(f"  键数量: {len(data.keys())}")
            print(f"  键列表: {list(data.keys())}\n")
            
            print("📏 数据形状和类型:")
            for key, value in data.items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                elif isinstance(value, (list, tuple)):
                    print(f"  {key}: {type(value).__name__}, length={len(value)}")
                else:
                    print(f"  {key}: {type(value).__name__}")
            
            # 显示一些示例数据
            print("\n📋 示例数据 (前几个值):")
            for key, value in list(data.items())[:5]:
                if hasattr(value, 'shape') and value.size > 0:
                    if value.ndim <= 2:
                        print(f"\n  {key}:")
                        print(f"    {value[:min(3, len(value))]}")
                    else:
                        print(f"\n  {key}: shape={value.shape}")
                elif isinstance(value, (list, tuple)) and len(value) > 0:
                    print(f"\n  {key}: {value[:min(3, len(value))]}")
        else:
            print(f"数据类型: {type(data)}")
            if hasattr(data, 'shape'):
                print(f"形状: {data.shape}")
                print(f"数据类型: {data.dtype}")
                if data.size < 100:
                    print(f"内容:\n{data}")
    
    except Exception as e:
        print(f"❌ 读取文件时出错: {e}")
        import traceback
        traceback.print_exc()

def list_results(directory):
    """列出目录中的所有结果文件"""
    if not os.path.exists(directory):
        print(f"❌ 目录不存在: {directory}")
        return
    
    print(f"\n{'='*60}")
    print(f"📂 目录: {directory}")
    print(f"{'='*60}\n")
    
    npy_files = list(Path(directory).rglob("*.npy"))
    
    if not npy_files:
        print("❌ 未找到 .npy 文件")
        return
    
    print(f"📊 找到 {len(npy_files)} 个结果文件\n")
    
    # 按对象分组
    objects = {}
    for file_path in npy_files:
        # 提取对象名称（从路径中）
        parts = file_path.parts
        if 'graspdata' in parts:
            idx = parts.index('graspdata')
            if idx + 1 < len(parts):
                obj_name = parts[idx + 1]
                if obj_name not in objects:
                    objects[obj_name] = []
                objects[obj_name].append(str(file_path))
    
    print(f"📦 对象数量: {len(objects)}\n")
    
    # 显示前10个对象
    for i, (obj_name, files) in enumerate(list(objects.items())[:10]):
        print(f"  {i+1}. {obj_name}: {len(files)} 个文件")
        for file_path in files[:3]:  # 只显示前3个文件
            rel_path = os.path.relpath(file_path, directory)
            file_size = os.path.getsize(file_path) / 1024  # KB
            print(f"     - {rel_path} ({file_size:.1f} KB)")
        if len(files) > 3:
            print(f"     ... 还有 {len(files) - 3} 个文件")
    
    if len(objects) > 10:
        print(f"\n  ... 还有 {len(objects) - 10} 个对象")

def get_statistics(directory):
    """获取结果统计信息"""
    if not os.path.exists(directory):
        print(f"❌ 目录不存在: {directory}")
        return
    
    npy_files = list(Path(directory).rglob("*_grasp.npy"))
    
    if not npy_files:
        print("❌ 未找到抓取结果文件")
        return
    
    print(f"\n{'='*60}")
    print(f"📊 统计信息")
    print(f"{'='*60}\n")
    
    total_size = sum(os.path.getsize(f) for f in npy_files)
    print(f"总文件数: {len(npy_files)}")
    print(f"总大小: {total_size / (1024**2):.2f} MB")
    print(f"平均文件大小: {total_size / len(npy_files) / 1024:.2f} KB")
    
    # 按对象统计
    objects = {}
    for file_path in npy_files:
        parts = file_path.parts
        if 'graspdata' in parts:
            idx = parts.index('graspdata')
            if idx + 1 < len(parts):
                obj_name = parts[idx + 1]
                objects[obj_name] = objects.get(obj_name, 0) + 1
    
    print(f"\n对象数量: {len(objects)}")
    print(f"平均每个对象的抓取数: {len(npy_files) / len(objects):.1f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="查看 BODex 运行结果")
    parser.add_argument(
        "-f", "--file",
        type=str,
        help="查看单个 .npy 文件"
    )
    parser.add_argument(
        "-d", "--directory",
        type=str,
        default="src/curobo/content/assets/output/sim_shadow/fc/debug/graspdata",
        help="列出目录中的所有结果文件"
    )
    parser.add_argument(
        "-s", "--stats",
        action="store_true",
        help="显示统计信息"
    )
    
    args = parser.parse_args()
    
    if args.file:
        view_npy_file(args.file)
    elif args.stats:
        get_statistics(args.directory)
    else:
        list_results(args.directory)

