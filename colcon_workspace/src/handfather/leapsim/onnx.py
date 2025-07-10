import torch
import os
from leapsim.learning.amp_models import ModelAMPContinuous

def export_model_to_onnx(model_path, onnx_path, input_shape):
    """
    将 AMP 模型转换为 ONNX 格式。
    
    Args:
        model_path (str): PyTorch 模型的路径。
        onnx_path (str): 导出的 ONNX 文件路径。
        input_shape (tuple): 模型输入的形状。
    """
    # 加载模型
    model = ModelAMPContinuous(network=None)  # 初始化模型
    model.load_state_dict(torch.load(model_path))  # 加载权重
    model.eval()

    # 创建一个虚拟输入
    dummy_input = torch.randn(*input_shape)

    # 导出为 ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    print(f"模型已成功导出为 ONNX 格式: {'.'}")

if __name__ == "__main__":
    # 示例路径
    model_path = "./runs/LeapHand.pth"  # 替换为实际模型路径
    onnx_path = "."  # 替换为实际导出路径
    input_shape = (1, 132)  # 替换为实际输入形状

    export_model_to_onnx(model_path, onnx_path, input_shape)