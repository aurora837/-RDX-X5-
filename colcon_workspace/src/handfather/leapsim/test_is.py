from isaacgym import gymapi

def main():
    # 初始化Isaac Gym
    gym = gymapi.acquire_gym()

    # 创建模拟配置
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = 4
    sim_params.physx.use_gpu = True

    # 创建模拟
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        raise Exception("Failed to create simulation")

    # 创建查看器with改进的相机属性
    camera_props = gymapi.CameraProperties()
    camera_props.enable_tensors = True
    camera_props.horizontal_fov = 75.0  # 增大视野角度
    camera_props.width = 1920
    camera_props.height = 1080
    
    viewer = gym.create_viewer(sim, camera_props)
    if viewer is None:
        raise Exception("Failed to create viewer")

    # 加载URDF文件
    asset_root = "/deltadisk/dachuang/colcon_ws/src/handfather/assets"  # 替换为URDF文件所在目录
    urdf_file = "isaac.urdf"  # 替换为URDF文件名
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True  # 如果需要固定基座
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
    # 添加厚度和网格选项以改善显示
    asset_options.thickness = 0.001
    asset_options.armature = 0.0
    asset_options.density = 1000.0
    asset = gym.load_asset(sim, asset_root, urdf_file, asset_options)
    if asset is None:
        raise Exception(f"Failed to load URDF file: {urdf_file}")

    # 打印关节和链接信息
    num_links = gym.get_asset_rigid_body_count(asset)
    num_joints = gym.get_asset_joint_count(asset)

    print(f"Number of links: {num_links}")
    print(f"Number of joints: {num_joints}")

    for i in range(num_links):
        link_name = gym.get_asset_rigid_body_name(asset, i)
        print(f"Link {i}: {link_name}")

    for i in range(num_joints):
        joint_name = gym.get_asset_joint_name(asset, i)
        joint_type = gym.get_asset_joint_type(asset, i)
        print(f"Joint {i}: {joint_name}, Type: {joint_type}")

    # 创建环境并实例化资产
    # 增大环境边界，允许更好的视角控制
    env = gym.create_env(sim, gymapi.Vec3(-5.0, -5.0, 0.0), gymapi.Vec3(5.0, 5.0, 5.0), 1)
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0, 0, 0)
    pose.r = gymapi.Quat(0, 0, 0, 1)
    actor = gym.create_actor(env, asset, pose, "robot", 0, 1)

    # 设置查看器相机位置和目标
    cam_pos = gymapi.Vec3(0.3, 0.3, 0.3)  # 相机位置 - 更接近模型
    cam_target = gymapi.Vec3(0, 0, 0.1)   # 相机目标点 - 稍微向上
    gym.viewer_camera_look_at(viewer, env, cam_pos, cam_target)

    # 打印控制说明
    print("\n=== 控制说明 ===")
    print("鼠标右键拖拽: 旋转视角")
    print("鼠标中键拖拽: 平移视角")
    print("鼠标滚轮: 缩放")
    print("键盘 R: 重置视角")
    print("键盘 Q: 退出")
    print("按 ESC 或关闭窗口退出")
    print("================\n")

    # 主循环
    while not gym.query_viewer_has_closed(viewer):
        # 更新物理模拟
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # 渲染
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    # 清理资源
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

if __name__ == "__main__":
    main()
