#!/usr/bin/env python3
"""
快速预设位置测试脚本
提供常用的palm_lower位置预设，便于快速测试
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))
from urdf_palm_adjuster import URDFPalmAdjuster

def main():
    urdf_path = "/deltadisk/dachuang/colcon_ws/src/leaphandsim/assets/isaac.urdf"
    adjuster = URDFPalmAdjuster(urdf_path)
    adjuster.backup_urdf()
    
    presets = {
        "1": {
            "name": "默认位置",
            "x": -0.102595, "y": -0.100108, "z": -0.056522,
            "roll": 1.57, "pitch": 0, "yaw": 1.57
        },
        "2": {
            "name": "向前移动 5cm",
            "x": -0.052595, "y": -0.100108, "z": -0.056522,
            "roll": 1.57, "pitch": 0, "yaw": 1.57
        },
        "3": {
            "name": "向上移动 3cm",
            "x": -0.102595, "y": -0.100108, "z": -0.026522,
            "roll": 1.57, "pitch": 0, "yaw": 1.57
        },
        "4": {
            "name": "向右移动 2cm",
            "x": -0.102595, "y": -0.080108, "z": -0.056522,
            "roll": 1.57, "pitch": 0, "yaw": 1.57
        },
        "5": {
            "name": "旋转90度 (Roll)",
            "x": -0.102595, "y": -0.100108, "z": -0.056522,
            "roll": 0, "pitch": 0, "yaw": 1.57
        },
        "6": {
            "name": "旋转90度 (Yaw)",
            "x": -0.102595, "y": -0.100108, "z": -0.056522,
            "roll": 1.57, "pitch": 0, "yaw": 0
        },
        "r": {
            "name": "恢复备份",
            "restore": True
        }
    }
    
    print("\n=== Palm位置快速测试 ===")
    print("选择预设位置:")
    for key, preset in presets.items():
        if key != "r":
            print(f"  {key}: {preset['name']}")
        else:
            print(f"  {key}: {preset['name']}")
    print("-" * 30)
    
    choice = input("请选择 (1-6, r): ").strip()
    
    if choice in presets:
        preset = presets[choice]
        if "restore" in preset:
            adjuster.restore_backup()
        else:
            print(f"\n🔄 应用预设: {preset['name']}")
            adjuster.update_palm_position(
                x=preset["x"], y=preset["y"], z=preset["z"],
                roll=preset["roll"], pitch=preset["pitch"], yaw=preset["yaw"]
            )
            print("🚀 现在可以运行 python test_is.py 查看效果")
    else:
        print("❌ 无效选择")

if __name__ == "__main__":
    main()
