#!/usr/bin/env python3
"""
URDF Palm Position Adjustment Tool
用于快速调整 palm_lower link 的位置和姿态
"""

import re
import os

class URDFPalmAdjuster:
    def __init__(self, urdf_path):
        self.urdf_path = urdf_path
        self.backup_path = urdf_path + ".backup"
        
    def backup_urdf(self):
        """备份原始URDF文件"""
        if not os.path.exists(self.backup_path):
            with open(self.urdf_path, 'r') as f:
                content = f.read()
            with open(self.backup_path, 'w') as f:
                f.write(content)
            print(f"✅ 已备份原始文件到: {self.backup_path}")
    
    def update_palm_position(self, x=None, y=None, z=None, roll=None, pitch=None, yaw=None):
        """
        更新palm_lower的位置和姿态
        
        参数:
        x, y, z: 位置偏移 (米)
        roll, pitch, yaw: 姿态角度 (弧度)
        """
        with open(self.urdf_path, 'r') as f:
            content = f.read()
        
        # 找到palm_lower的visual origin行
        pattern = r'(<link name="palm_lower">.*?<visual>.*?<origin rpy=")([^"]+)(" xyz=")([^"]+)(".*?</visual>)'
        match = re.search(pattern, content, re.DOTALL)
        
        if not match:
            print("❌ 未找到palm_lower的visual origin")
            return False
        
        # 解析当前的rpy和xyz值
        current_rpy = [float(x) for x in match.group(2).split()]
        current_xyz = [float(x) for x in match.group(4).split()]
        
        # 更新数值
        new_rpy = current_rpy.copy()
        new_xyz = current_xyz.copy()
        
        if roll is not None: new_rpy[0] = roll
        if pitch is not None: new_rpy[1] = pitch  
        if yaw is not None: new_rpy[2] = yaw
        if x is not None: new_xyz[0] = x
        if y is not None: new_xyz[1] = y
        if z is not None: new_xyz[2] = z
        
        # 构建新的origin字符串
        new_rpy_str = f"{new_rpy[0]:.6f} {new_rpy[1]:.6f} {new_rpy[2]:.6f}"
        new_xyz_str = f"{new_xyz[0]:.6f} {new_xyz[1]:.6f} {new_xyz[2]:.6f}"
        
        # 替换visual部分
        new_visual = f'{match.group(1)}{new_rpy_str}{match.group(3)}{new_xyz_str}{match.group(5)}'
        content = content[:match.start()] + new_visual + content[match.end():]
        
        # 同样更新collision部分
        pattern_collision = r'(<collision>.*?<origin rpy=")([^"]+)(" xyz=")([^"]+)(".*?</collision>)'
        match_collision = re.search(pattern_collision, content, re.DOTALL)
        
        if match_collision:
            new_collision = f'{match_collision.group(1)}{new_rpy_str}{match_collision.group(3)}{new_xyz_str}{match_collision.group(5)}'
            content = content[:match_collision.start()] + new_collision + content[match_collision.end():]
        
        # 保存文件
        with open(self.urdf_path, 'w') as f:
            f.write(content)
        
        print(f"✅ 已更新palm_lower位置:")
        print(f"   RPY: {new_rpy_str}")
        print(f"   XYZ: {new_xyz_str}")
        return True
    
    def restore_backup(self):
        """恢复备份文件"""
        if os.path.exists(self.backup_path):
            with open(self.backup_path, 'r') as f:
                content = f.read()
            with open(self.urdf_path, 'w') as f:
                f.write(content)
            print("✅ 已恢复原始文件")
        else:
            print("❌ 备份文件不存在")

def main():
    urdf_path = "/deltadisk/dachuang/colcon_ws/src/leaphandsim/assets/isaac.urdf"
    adjuster = URDFPalmAdjuster(urdf_path)
    
    # 创建备份
    adjuster.backup_urdf()
    
    print("\n=== URDF Palm Position Adjuster ===")
    print("输入新的位置和姿态参数（直接回车跳过该参数）:")
    print("位置单位: 米，姿态单位: 弧度")
    print("-" * 40)
    
    try:
        # 获取用户输入
        x_input = input("X 位置 (当前: -0.102595): ").strip()
        y_input = input("Y 位置 (当前: -0.100108): ").strip()
        z_input = input("Z 位置 (当前: -0.056522): ").strip()
        roll_input = input("Roll 角度 (当前: 1.57): ").strip()
        pitch_input = input("Pitch 角度 (当前: 0): ").strip()
        yaw_input = input("Yaw 角度 (当前: 1.57): ").strip()
        
        # 转换输入
        x = float(x_input) if x_input else None
        y = float(y_input) if y_input else None
        z = float(z_input) if z_input else None
        roll = float(roll_input) if roll_input else None
        pitch = float(pitch_input) if pitch_input else None
        yaw = float(yaw_input) if yaw_input else None
        
        # 更新位置
        if any(v is not None for v in [x, y, z, roll, pitch, yaw]):
            adjuster.update_palm_position(x, y, z, roll, pitch, yaw)
            print("\n🚀 可以运行 python test_is.py 查看效果")
        else:
            print("未做任何修改")
            
    except ValueError:
        print("❌ 输入格式错误，请输入数字")
    except KeyboardInterrupt:
        print("\n👋 已取消操作")

if __name__ == "__main__":
    main()
