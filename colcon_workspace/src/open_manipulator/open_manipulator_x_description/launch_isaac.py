from omni.isaac.lab.app import SimulationApp  # 新的应用入口

# 启动配置（注意新版默认使用RTX实时渲染）
simulation_app = SimulationApp({"headless": False})

from omni.isaac.lab.sim import SimulationContext  # 代替旧版World
from omni.isaac.lab.assets import Robot  # 新的机器人资产类
import omni.isaac.lab.sim as sim_utils  # 新的工具模块

# 1. 创建场景
simulation_context = SimulationContext()
stage = simulation_context.get_stage()

# 2. 添加默认地面
ground_plane = sim_utils.GroundPlane()
ground_plane.initialize()

# 3. 加载URDF机器人
urdf_path = "~/colcon_ws/src/open_manipulator/open_manipulator_x_description/urdf/open_manipulator_x_robot.urdf"  # 建议使用绝对路径

# 使用新的Robot类导入URDF
robot = Robot(
    prim_path="/World/Robot",
    name="open_manipulator_x",
    urdf_path=urdf_path,
    translation=(0, 0, 0.1),  # Z方向偏移避免穿透地面
    orientation=(1, 0, 0, 0),  # 四元数格式 (w, x, y, z)
    # 以下是可选参数：
    # physics_material=sim_utils.RigidBodyMaterial(...),
    # articulation_props=sim_utils.ArticulationRootProperties(...)
)

# 初始化物理和资产
simulation_context.initialize_physics()
simulation_context.initialize_assets_in_context()

# 4. 主仿真循环
while simulation_app.is_running():
    # 执行物理步进
    simulation_context.step()
    
    # 如果需要界面刷新
    simulation_app.update()

# 关闭应用
simulation_app.close()