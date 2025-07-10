#!/usr/bin/env python3
"""
脚本功能：对cache文件夹下的.npy文件进行关节值插入和随机赋值
- 原始文件形状：1024 x 23
- 在第16个位置开始插入5个关节值
- 处理后形状：1024 x 28
- 按照指定范围进行随机赋值
"""

import numpy as np
import os
import glob

def main():
    # 源文件目录
    source_dir = "/deltadisk/dachuang/colcon_ws/src/LEAP_Hand_Sim-master/leapsim/cache"
    
    # 目标文件目录
    target_dir = "/deltadisk/dachuang/colcon_ws/src/handfather/leapsim/cache"
    
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)
 
    # 查找源目录中的所有.npy文件
    npy_files = glob.glob(os.path.join(source_dir, "*.npy"))
    
    if len(npy_files) == 0:
        print(f"在 {source_dir} 中未找到.npy文件")
        return
    
    print(f"在源目录中找到 {len(npy_files)} 个.npy文件")
    print(f"目标保存目录: {target_dir}")
    
    # 定义要插入的5个关节值范围（按插入顺序）
    joint_ranges = [
        (np.pi, np.pi),     # joint5 - 固定为π (插入到索引16)
        (-0.262, 0.262),    # joint4 (插入到索引17)
        (-0.262, 0.262),    # joint3 (插入到索引18)
        (-0.262, 0.262),    # joint2 (插入到索引19)
        (-3.14, 3.14),      # joint1 (插入到索引20)
    ]
    
    # 处理每个.npy文件
    for npy_file in npy_files:
        filename = os.path.basename(npy_file)
        print(f"\n处理文件: {filename}")
        
        # 加载数据
        try:
            data = np.load(npy_file)
            print(f"  原始形状: {data.shape}")
            
            if data.shape[0] != 1024:
                print(f"  警告: 行数不是1024，跳过此文件")
                continue
                
        except Exception as e:
            print(f"  错误: 无法加载文件 - {e}")
            continue
        
        final_data = None
        
        if data.shape == (1024, 23):
            print("  检测到(1024, 23)形状，执行插入操作...")
            
            # 创建5个新列的数据
            new_columns = []
            joint_names = ['joint5', 'joint4', 'joint3', 'joint2', 'joint1']  # 按插入顺序
            
            for i, (min_val, max_val) in enumerate(joint_ranges):
                joint_name = joint_names[i]
                insert_index = 16 + i  # 从索引16开始依次插入
                
                if min_val == max_val:  # joint5固定为π
                    new_col = np.full((data.shape[0], 1), min_val)
                    print(f"    {joint_name} (索引{insert_index}) 设置为固定值: {min_val:.6f}")
                else:
                    # 生成随机值
                    new_col = np.random.uniform(min_val, max_val, size=(data.shape[0], 1))
                    print(f"    {joint_name} (索引{insert_index}) 随机赋值范围: [{min_val:.3f}, {max_val:.3f}]")
                new_columns.append(new_col)
            
            # 将5个新列合并
            new_data = np.hstack(new_columns)
            print(f"    生成的新数据形状: {new_data.shape}")
            
            # 在第16个位置插入新数据（在第16和第18位置之间）
            # 分割原数据：前16列 + 新5列 + 后7列
            data_before = data[:, :16]      # 前16列 (索引0-15)
            data_after = data[:, 16:]       # 后7列 (索引16-22)
            
            # 合并数据
            final_data = np.hstack([data_before, new_data, data_after])
            print(f"    最终数据形状: {final_data.shape}")
            
        elif data.shape == (1024, 28):
            print("  检测到(1024, 28)形状，执行重新随机化操作...")
            
            # 复制原数据
            final_data = data.copy()
            
            # 对第17-21列（索引16-20）重新随机化
            joint_names = ['joint5', 'joint4', 'joint3', 'joint2', 'joint1']  # 按顺序
            
            for i, (min_val, max_val) in enumerate(joint_ranges):
                col_idx = 16 + i  # 从索引16开始
                joint_name = joint_names[i]
                
                if min_val == max_val:  # joint5固定为π
                    final_data[:, col_idx] = min_val
                    print(f"    {joint_name} (第{col_idx+1}列) 重新设置为固定值: {min_val:.6f}")
                else:
                    # 生成随机值
                    random_values = np.random.uniform(min_val, max_val, size=final_data.shape[0])
                    final_data[:, col_idx] = random_values
                    print(f"    {joint_name} (第{col_idx+1}列) 重新随机赋值范围: [{min_val:.3f}, {max_val:.3f}]")
            
        else:
            print(f"  警告: 文件形状不支持 {data.shape}，跳过此文件")
            continue
        
        # 验证最终形状
        if final_data is not None and final_data.shape == (1024, 28):
            # 保存到目标目录
            target_file = os.path.join(target_dir, filename)
            try:
                np.save(target_file, final_data)
                print(f"    成功保存到: {target_file}")
            except Exception as e:
                print(f"    错误: 无法保存文件 - {e}")
        else:
            print(f"    警告: 最终形状不正确，跳过保存")
    
    print(f"\n处理完成！共处理了 {len(npy_files)} 个文件")
    print(f"文件已保存到: {target_dir}")

if __name__ == "__main__":
    main()
