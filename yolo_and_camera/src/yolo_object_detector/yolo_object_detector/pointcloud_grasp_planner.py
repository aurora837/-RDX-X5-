#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from ai_msgs.msg import PerceptionTargets
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2

class PointCloudGraspPlanner(Node):
    def __init__(self):
        super().__init__('pointcloud_grasp_planner')
        
        # 订阅点云数据
        self.pointcloud_sub = self.create_subscription(
            PointCloud2, '/camera/depth_registered/points',
            self.pointcloud_callback, 10)
        
        # 订阅检测结果
        self.detection_sub = self.create_subscription(
            PerceptionTargets, '/yolo_3d_detections',
            self.detection_callback, 10)
        
        # 订阅抓取请求
        self.grasp_request_sub = self.create_subscription(
            PoseStamped, '/grasp_target_pose',
            self.grasp_request_callback, 10)
        
        # 发布优化后的抓取姿态
        self.optimized_grasp_pub = self.create_publisher(
            PoseStamped, '/optimized_grasp_pose', 10)
        
        # 发布抓取分析结果
        self.grasp_analysis_pub = self.create_publisher(
            String, '/grasp_analysis', 10)
        
        # 存储最新数据
        self.latest_pointcloud = None
        self.latest_detections = None
        
        self.get_logger().info("点云抓取规划节点启动")

    def pointcloud_callback(self, msg):
        """处理点云数据"""
        self.latest_pointcloud = msg
        # self.get_logger().info(f"收到点云数据，点数: {msg.width * msg.height}")

    def detection_callback(self, msg):
        """处理检测结果"""
        self.latest_detections = msg

    def grasp_request_callback(self, msg):
        """处理抓取请求"""
        if not self.latest_pointcloud:
            self.get_logger().warn("没有点云数据，无法规划抓取")
            return
        
        target_position = msg.pose.position
        self.get_logger().info(f"收到抓取请求，目标位置: ({target_position.x:.2f}, {target_position.y:.2f}, {target_position.z:.2f})")
        
        # 基于点云优化抓取姿态
        optimized_pose = self.plan_grasp_with_pointcloud(msg)
        
        if optimized_pose:
            self.optimized_grasp_pub.publish(optimized_pose)
            
            # 发布分析结果
            analysis = {
                "success": True,
                "message": "抓取姿态已优化",
                "original_position": [target_position.x, target_position.y, target_position.z],
                "optimized_position": [
                    optimized_pose.pose.position.x,
                    optimized_pose.pose.position.y, 
                    optimized_pose.pose.position.z
                ]
            }
        else:
            analysis = {
                "success": False,
                "message": "无法找到合适的抓取姿态",
                "reason": "点云数据不足或目标区域无法到达"
            }
        
        analysis_msg = String()
        analysis_msg.data = str(analysis)
        self.grasp_analysis_pub.publish(analysis_msg)

    def plan_grasp_with_pointcloud(self, grasp_request):
        """基于点云规划抓取姿态"""
        try:
            # 提取目标区域的点云
            target_pos = grasp_request.pose.position
            region_points = self.extract_region_pointcloud(
                target_pos, radius=0.1)  # 10cm半径
            
            if len(region_points) < 100:  # 点数太少
                self.get_logger().warn("目标区域点云数据不足")
                return None
            
            # 计算表面法向量
            normal = self.estimate_surface_normal(region_points)
            
            # 优化抓取位置（避开遮挡，找到最佳接近角度）
            optimized_position = self.optimize_grasp_position(
                target_pos, region_points, normal)
            
            # 创建优化后的抓取姿态
            optimized_pose = PoseStamped()
            optimized_pose.header = grasp_request.header
            optimized_pose.pose.position = optimized_position
            
            # 根据表面法向量计算抓取姿态
            optimized_pose.pose.orientation = self.calculate_grasp_orientation(normal)
            
            return optimized_pose
            
        except Exception as e:
            self.get_logger().error(f"抓取规划失败: {e}")
            return None

    def extract_region_pointcloud(self, center, radius):
        """提取目标区域的点云"""
        points = []
        
        # 使用sensor_msgs_py解析点云
        for point in pc2.read_points(self.latest_pointcloud, skip_nans=True):
            x, y, z = point[0], point[1], point[2]
            
            # 检查点是否在目标区域内
            distance = np.sqrt(
                (x - center.x)**2 + 
                (y - center.y)**2 + 
                (z - center.z)**2
            )
            
            if distance <= radius:
                points.append([x, y, z])
        
        return np.array(points)

    def estimate_surface_normal(self, points):
        """估计表面法向量"""
        if len(points) < 3:
            return np.array([0, 0, 1])  # 默认向上
        
        # 使用PCA估计主平面
        centered_points = points - np.mean(points, axis=0)
        
        # 计算协方差矩阵
        cov_matrix = np.cov(centered_points.T)
        
        # 获取最小特征值对应的特征向量作为法向量
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        normal = eigenvectors[:, np.argmin(eigenvalues)]
        
        # 确保法向量指向相机方向
        if normal[2] > 0:
            normal = -normal
            
        return normal

    def optimize_grasp_position(self, original_pos, region_points, normal):
        """优化抓取位置"""
        from geometry_msgs.msg import Point
        
        # 简单优化：沿着法向量方向稍微偏移
        offset = 0.02  # 2cm偏移，避免碰撞
        
        optimized = Point()
        optimized.x = original_pos.x - normal[0] * offset
        optimized.y = original_pos.y - normal[1] * offset  
        optimized.z = original_pos.z - normal[2] * offset
        
        return optimized

    def calculate_grasp_orientation(self, normal):
        """根据表面法向量计算抓取姿态"""
        from geometry_msgs.msg import Quaternion
        import math
        
        # 简化的姿态计算
        # 这里可以根据你的机械臂类型做更复杂的计算
        
        # 默认垂直向下抓取
        orientation = Quaternion()
        orientation.x = 0.0
        orientation.y = 0.0
        orientation.z = 0.0
        orientation.w = 1.0
        
        return orientation


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = PointCloudGraspPlanner()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
