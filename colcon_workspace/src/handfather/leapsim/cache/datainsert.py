import numpy as np
import os
import glob

# 定义要添加的随机数据范围
RANGES = [
    (-3.14, 3.14),    # 第一个数据范围：joint1 (-180° to +180°)
    (-0.785, 0.785),  # 第二个数据范围：joint2 (-45° to +45°)
    (-0.785, 0.785),  # 第三个数据范围：joint3 (-45° to +45°)
    (-0.785, 0.785),  # 第四个数据范围：joint4 (-45° to +45°)
    (2.35619, 3.92699) # 第五个数据范围：joint5 (135° to 225°)
]

def process_single_file(file_path):
    """处理单个.npy文件"""
    try:
        # 加载原始数据
        data = np.load(file_path)
        print(f"\n处理文件: {os.path.basename(file_path)}")
        print(f"原始数据形状: {data.shape}")
        
        # 显示处理前的第一行数据样本
        print(f"处理前第一行数据: {data[0]}")
        
        # 创建新数组（在第16个数字后面插入5列）
        new_data = np.zeros((data.shape[0], data.shape[1] + len(RANGES)))
        
        for i in range(data.shape[0]):  # 遍历每一行
            # 生成5个随机值（使用预定义范围）
            rand_values = [np.random.uniform(low, high) for low, high in RANGES]
            
            # 复制前16个数字
            new_data[i, :16] = data[i, :16]
            
            # 在第16个数字后面插入5个随机值
            new_data[i, 16:16+len(RANGES)] = rand_values
            
            # 复制剩余的原始数据
            new_data[i, 16+len(RANGES):] = data[i, 16:]
        
        # 保存处理后的数据（覆盖原文件）
        np.save(file_path, new_data)
        print(f"✅ 处理完成！新形状: {new_data.shape}")
        print(f"处理后第一行数据: {new_data[0]}")
        
        return True
        
    except Exception as e:
        print(f"❌ 处理文件 {file_path} 时出错: {str(e)}")
        return False

def process_folder(folder_path):
    """批量处理文件夹中的所有.npy文件"""
    # 查找文件夹中的所有.npy文件
    npy_files = glob.glob(os.path.join(folder_path, "*.npy"))
    
    if not npy_files:
        print(f"在文件夹 {folder_path} 中未找到.npy文件")
        return
    
    # 按文件名排序，确保按顺序处理
    npy_files.sort()
    
    print(f"找到 {len(npy_files)} 个.npy文件:")
    for i, file_path in enumerate(npy_files, 1):
        print(f"{i}. {os.path.basename(file_path)}")
    
    print("\n开始批量处理...")
    
    success_count = 0
    total_count = len(npy_files)
    
    # 逐个处理文件
    for i, file_path in enumerate(npy_files, 1):
        print(f"\n{'='*50}")
        print(f"处理进度: {i}/{total_count}")
        
        if process_single_file(file_path):
            success_count += 1
    
    # 输出处理结果统计
    print(f"\n{'='*50}")
    print(f"批量处理完成！")
    print(f"成功处理: {success_count}/{total_count} 个文件")
    
    if success_count < total_count:
        print(f"失败文件数: {total_count - success_count}")

# 使用示例
if __name__ == "__main__":
    # 指定包含.npy文件的文件夹路径
    folder_path = "."  # 使用当前cache目录
    
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹路径 {folder_path} 不存在")
        print("请修改 folder_path 变量为正确的文件夹路径")
    else:
        process_folder(folder_path)