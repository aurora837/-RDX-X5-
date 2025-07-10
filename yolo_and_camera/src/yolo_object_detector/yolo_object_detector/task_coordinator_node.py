#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from ai_msgs.msg import PerceptionTargets
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Point
import json
import time

class TaskCoordinatorNode(Node):
    def __init__(self):
        super().__init__('task_coordinator_node')
        
        # 订阅语音识别结果（来自交互节点或WebSocket桥接节点）
        # 这里订阅 /speech_recognition_result 话题，该话题由 websocket_bridge_node 转发Web指令
        self.speech_sub = self.create_subscription(
            String, '/speech_recognition_result', 
            self.speech_callback, 10)
        
        # 订阅YOLO检测结果
        self.detection_sub = self.create_subscription(
            PerceptionTargets, '/yolo_3d_detections',
            self.detection_callback, 10)
        
        # 发布任务状态（WebSocket桥接节点会订阅此话题并转发给App）
        self.status_pub = self.create_publisher(
            String, '/task_status', 10)
        
        # 发布抓取任务（为将来的机械臂准备，目前仅为消息定义）
        self.grasp_task_pub = self.create_publisher(
            PoseStamped, '/grasp_target_pose', 10)
        
        # 存储当前检测结果
        self.current_detections = {}
        
        # --- 新增：声明和获取参数 ---
        # 声明YOLO相关的参数，以便在调度逻辑中使用
        # 保持与yolo节点相同的默认值，或者根据实际需求调整
        self.declare_parameter('conf_threshold', 0.25) 
        self.declare_parameter('nms_threshold', 0.7) # 尽管这个节点不直接用NMS，但为了完整性可以声明

        # 获取YOLO节点的置信度阈值，用于过滤检测结果
        self.yolo_conf_threshold = self.get_parameter('conf_threshold').value
        # --- 结束新增 ---

        # 中英文目标映射，用于将用户指令（中文）映射到YOLO识别的英文类别
        self.target_mapping = {
            "人": ["person"],
            "杯子": ["cup", "wine glass"], 
            "瓶子": ["bottle"],
            "碗": ["bowl"],
            "苹果": ["apple"],
            "香蕉": ["banana"],
            "手机": ["cell phone"],
            "书": ["book"],
            "椅子": ["chair"],
            "电脑": ["laptop"],
            "鼠标": ["mouse"],
            "键盘": ["keyboard"],
            "遥控器": ["remote"],
            "电视": ["tv"],
            "猫": ["cat"],
            "狗": ["dog"],
            "汽车": ["car"],
        }
        
        # 可抓取物体及其距离限制（基于英文名称）
        self.graspable_objects = {
            "cup": {"max_distance": 1.5, "min_distance": 0.3},
            "wine glass": {"max_distance": 1.5, "min_distance": 0.3},
            "bottle": {"max_distance": 1.5, "min_distance": 0.3},
            "apple": {"max_distance": 1.2, "min_distance": 0.3},
            "banana": {"max_distance": 1.2, "min_distance": 0.3},
            "cell phone": {"max_distance": 1.0, "min_distance": 0.3},
            "book": {"max_distance": 1.3, "min_distance": 0.3},
            "bowl": {"max_distance": 1.2, "min_distance": 0.3},
            "mouse": {"max_distance": 1.0, "min_distance": 0.3},
            "keyboard": {"max_distance": 1.3, "min_distance": 0.3},
        }
        
        self.get_logger().info("任务调度节点启动")

    def speech_callback(self, msg):
        """处理来自交互节点或WebSocket桥接节点的指令"""
        command = msg.data.strip()
        self.get_logger().info(f"收到指令: '{command}'")
        
        # 解析不同类型的指令
        if any(word in command for word in ["找", "看", "检测", "寻找"]):
            target_object_english_names = self.extract_target_object(command)
            # 提取原始指令中的中文目标词，用于更好的反馈
            target_chinese_name = self.extract_chinese_target_from_command(command)
            if target_object_english_names:
                self.find_object(target_object_english_names, target_chinese_name, command)
            else:
                self.publish_status("未识别到有效的目标物体。")
                
        elif any(word in command for word in ["抓取", "拿", "取", "抓", "夹"]):
            target_object_english_names = self.extract_target_object(command)
            target_chinese_name = self.extract_chinese_target_from_command(command)
            if target_object_english_names:
                self.execute_grasp_task(target_object_english_names, target_chinese_name, command)
            else:
                self.publish_status("未识别到要抓取的物体。")
        
        elif "列表" in command: # 新增：处理“列表”指令
            self.list_detected_objects()
            
        else:
            self.publish_status(f"未理解的指令: {command}")

    def extract_target_object(self, command):
        """从指令中提取目标物体的英文名称列表"""
        # 返回所有可能对应的英文名
        for chinese_name, english_names in self.target_mapping.items():
            if chinese_name in command:
                return english_names # 返回的是一个列表，如 ["cup", "wine glass"]
        return None
    
    def extract_chinese_target_from_command(self, command):
        """从原始指令中提取第一个匹配的中文目标词"""
        for chinese_name in self.target_mapping.keys():
            if chinese_name in command:
                return chinese_name
        return "未知物体" # 如果没有找到具体的中文名

    def detection_callback(self, msg):
        """更新检测结果
        这个回调函数会被 /yolo_3d_detections 话题触发，更新 self.current_detections 字典
        """
        self.current_detections = {} 
        current_time = time.time()
        
        for target in msg.targets:
            if not target.rois or not target.points:
                continue

            roi = target.rois[0] 
            object_name = roi.type
            confidence = roi.confidence
            
            center_3d = None
            for point in target.points:
                if point.type == "center_3d" and point.point:
                    center_3d = point.point[0] 
                    break
            
            # 使用 self.yolo_conf_threshold 进行过滤
            if center_3d and confidence > self.yolo_conf_threshold: 
                self.current_detections[object_name] = {
                    'confidence': confidence,
                    'position': center_3d, 
                    'timestamp': current_time
                }

    def find_object(self, target_english_names, target_chinese_name, original_command):
        """查找物体"""
        found_objects_info = []
        
        for english_name in target_english_names:
            if english_name in self.current_detections:
                detection = self.current_detections[english_name]
                # 再次确认置信度
                if detection['confidence'] > self.yolo_conf_threshold:
                    found_objects_info.append({
                        'english_name': english_name,
                        'detection': detection
                    })
        
        if found_objects_info:
            if len(found_objects_info) > 1:
                best_match = max(found_objects_info, key=lambda x: x['detection']['confidence'])
                obj_name_cn = self.get_chinese_name(best_match['english_name'])
                pos = best_match['detection']['position']
                conf = best_match['detection']['confidence']
                
                other_found_names_cn = [self.get_chinese_name(info['english_name']) for info in found_objects_info if info != best_match]
                # 使用 set 去重并转换为列表以方便 join
                other_info_display = ', '.join(sorted(list(set(other_found_names_cn))))
                other_info = f" (还检测到{len(set(other_found_names_cn))}个其他相关物体，如：{other_info_display})" if other_found_names_cn else ""

                response = (f"找到了{obj_name_cn}！"
                            f"位置：距离{pos.z:.2f}米，"
                            f"左右偏移{pos.x:.2f}米，"
                            f"上下偏移{pos.y:.2f}米，"
                            f"置信度{conf:.1%}{other_info}")
            else: 
                obj_info = found_objects_info[0]
                obj_name_cn = self.get_chinese_name(obj_info['english_name'])
                pos = obj_info['detection']['position']
                conf = obj_info['detection']['confidence']
                
                response = (f"找到了{obj_name_cn}！"
                            f"位置：距离{pos.z:.2f}米，"
                            f"左右偏移{pos.x:.2f}米，"
                            f"上下偏移{pos.y:.2f}米，"
                            f"置信度{conf:.1%}")
        else:
            available_objects_english = list(self.current_detections.keys())
            available_objects_chinese = sorted(list(set([self.get_chinese_name(obj) for obj in available_objects_english])))
            
            response = f"没有找到{target_chinese_name}。" 
            
            if available_objects_chinese:
                response += f" 当前可见：{', '.join(available_objects_chinese)}。"
            else:
                response += " 当前没有检测到任何物体。"
        
        self.publish_status(response)

    def execute_grasp_task(self, target_english_names, target_chinese_name, original_command):
        """执行抓取任务"""
        
        graspable_targets_candidates = [name for name in target_english_names if name in self.graspable_objects]
        
        if not graspable_targets_candidates:
            self.publish_status(f"{target_chinese_name} 不支持抓取操作。")
            return
        
        best_target_detection = None
        best_english_name = ""
        best_confidence = 0.0
        
        for english_name in graspable_targets_candidates:
            if english_name in self.current_detections:
                detection = self.current_detections[english_name]
                # 检查是否满足YOLO置信度阈值
                if detection['confidence'] > self.yolo_conf_threshold:
                    if detection['confidence'] > best_confidence:
                        best_confidence = detection['confidence']
                        best_target_detection = detection
                        best_english_name = english_name
        
        if best_target_detection:
            feasibility = self.assess_grasp_feasibility(best_english_name, best_target_detection)
            
            if not feasibility['feasible']:
                self.publish_status(f"无法抓取{self.get_chinese_name(best_english_name)}：{feasibility['reason']}。")
                return
            
            grasp_pose = PoseStamped()
            grasp_pose.header.stamp = self.get_clock().now().to_msg()
            grasp_pose.header.frame_id = "camera_color_optical_frame"
            
            target_pos = best_target_detection['position']
            grasp_pose.pose.position.x = target_pos.x
            grasp_pose.pose.position.y = target_pos.y
            grasp_pose.pose.position.z = target_pos.z
            
            grasp_pose.pose.orientation.x = 0.0
            grasp_pose.pose.orientation.y = 0.0
            grasp_pose.pose.orientation.z = 0.0
            grasp_pose.pose.orientation.w = 1.0
            
            self.grasp_task_pub.publish(grasp_pose)
            
            chinese_name = self.get_chinese_name(best_english_name)
            pos = best_target_detection['position']
            
            status_msg = (f"准备抓取{chinese_name}，"
                         f"目标位置：X={pos.x:.2f}m, Y={pos.y:.2f}m, Z={pos.z:.2f}m，"
                         f"置信度：{best_target_detection['confidence']:.1%}。")
            
            self.publish_status(status_msg)
            self.get_logger().info(f"发布抓取任务: {best_english_name}，位置：({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f})")
            
        else:
            self.publish_status(f"未找到要抓取的{target_chinese_name}。")
            available_objects_english = list(self.current_detections.keys())
            available_objects_chinese = sorted(list(set([self.get_chinese_name(obj) for obj in available_objects_english])))
            
            if available_objects_chinese:
                self.publish_status(f"当前可见：{', '.join(available_objects_chinese)}。")
            else:
                self.publish_status("当前没有检测到任何物体。")

    def assess_grasp_feasibility(self, target_english_name, detection):
        """评估抓取可行性（距离）"""
        result = {'feasible': True, 'reason': ''}
        
        if target_english_name not in self.graspable_objects:
            result['feasible'] = False
            result['reason'] = "该物体类型不支持抓取"
            return result
        
        distance = detection['position'].z 
        limits = self.graspable_objects[target_english_name]
        
        if distance < limits['min_distance']:
            result['feasible'] = False
            result['reason'] = f"目标距离太近 ({distance:.2f}m < {limits['min_distance']:.2f}m)"
        elif distance > limits['max_distance']:
            result['feasible'] = False
            result['reason'] = f"目标距离太远 ({distance:.2f}m > {limits['max_distance']:.2f}m)"
        
        return result

    def get_chinese_name(self, english_name):
        """将英文名称转换为中文"""
        for chinese, english_list in self.target_mapping.items():
            if english_name in english_list:
                return chinese
        return english_name

    def list_detected_objects(self):
        """列出当前检测到的所有物体及其信息"""
        if not self.current_detections:
            self.publish_status("当前没有检测到任何物体。")
            return

        detected_chinese_names = sorted(list(set([self.get_chinese_name(obj_name) for obj_name in self.current_detections.keys()])))

        if not detected_chinese_names:
            self.publish_status("当前没有检测到任何物体。") 
            return

        response_parts = ["当前检测到的物体有："]
        # 对检测到的物体按中文名进行排序，然后输出详细信息
        sorted_detections = sorted(self.current_detections.items(), key=lambda item: self.get_chinese_name(item[0]))
        for obj_name_english, details in sorted_detections:
            obj_name_chinese = self.get_chinese_name(obj_name_english)
            confidence = details['confidence']
            position = details['position']
            response_parts.append(
                f"- {obj_name_chinese} (英文名: {obj_name_english}, 置信度{confidence:.1%}, 距离{position.z:.2f}米)"
            )
        
        self.publish_status("\n".join(response_parts))
        self.get_logger().info(f"Published detected objects list.")


    def publish_status(self, status_text):
        """发布任务状态"""
        status_msg = String()
        status_msg.data = status_text
        self.status_pub.publish(status_msg)
        self.get_logger().info(f"状态: {status_text}")


def main(args=None):
    rclpy.init(args=args)
    node = TaskCoordinatorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("任务调度节点被中断。")
    except Exception as e:
        node.get_logger().error(f"任务调度节点发生错误: {e}")
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
