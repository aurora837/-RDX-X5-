#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from ai_msgs.msg import PerceptionTargets
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
import asyncio
import websockets
import json
import threading
import time
from websockets.server import serve

class WebSocketBridgeNode(Node):
    def __init__(self):
        super().__init__('websocket_bridge_node')
        
        # WebSocket连接管理
        self.connected_clients = set()
        
        # 数据存储
        self.latest_detections = []
        self.latest_status = ""
        
        # ROS订阅器
        self.detection_sub = self.create_subscription(
            PerceptionTargets, '/yolo_3d_detections',
            self.detection_callback, 10)
        
        self.status_sub = self.create_subscription(
            String, '/task_status',
            self.status_callback, 10)
        
        # ROS发布器
        self.command_pub = self.create_publisher(
            String, '/speech_recognition_result', 10)
        
        # 启动WebSocket服务器
        self.start_websocket_server()
        
        self.get_logger().info("WebSocket桥接节点启动 - 端口: 8765")

    def start_websocket_server(self):
        """启动WebSocket服务器"""
        def run_server():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def start_server():
                self.get_logger().info("启动WebSocket服务器，监听端口8765")
                async with serve(self.handle_websocket, "0.0.0.0", 8765):
                    await asyncio.Future()  # 保持服务器运行
            
            loop.run_until_complete(start_server())
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()

    async def handle_websocket(self, websocket, path):
        """处理WebSocket连接"""
        client_addr = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        self.get_logger().info(f"新客户端连接: {client_addr}")
        
        self.connected_clients.add(websocket)
        
        # 发送欢迎消息
        welcome_msg = {
            "type": "welcome",
            "message": "成功连接到机器人",
            "timestamp": time.time()
        }
        await websocket.send(json.dumps(welcome_msg))
        
        # 发送当前状态
        await self.send_current_state(websocket)
        
        try:
            async for message in websocket:
                await self.handle_client_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            self.get_logger().info(f"客户端断开连接: {client_addr}")
        except Exception as e:
            self.get_logger().error(f"WebSocket处理错误: {e}")
        finally:
            self.connected_clients.discard(websocket)

    async def handle_client_message(self, websocket, message):
        """处理客户端消息"""
        try:
            data = json.loads(message)
            msg_type = data.get('type', '')
            
            if msg_type == 'robot_command':
                command = data.get('command', '')
                self.get_logger().info(f"收到Web指令: {command}")
                
                # 转发到ROS系统
                ros_msg = String()
                ros_msg.data = command
                self.command_pub.publish(ros_msg)
                
                # 发送确认
                ack_msg = {
                    "type": "command_ack",
                    "message": f"指令已发送: {command}",
                    "timestamp": time.time()
                }
                await websocket.send(json.dumps(ack_msg))
                
            elif msg_type == 'request_status':
                await self.send_current_state(websocket)
                
        except json.JSONDecodeError as e:
            error_msg = {
                "type": "error",
                "message": "无效的JSON格式",
                "timestamp": time.time()
            }
            await websocket.send(json.dumps(error_msg))

    async def send_current_state(self, websocket):
        """发送当前状态"""
        state_msg = {
            "type": "robot_state",
            "detections": self.latest_detections,
            "status": self.latest_status,
            "timestamp": time.time()
        }
        
        try:
            await websocket.send(json.dumps(state_msg))
        except:
            pass

    def detection_callback(self, msg):
        """处理检测结果"""
        self.latest_detections = self.format_detections(msg)
        
        # 广播给所有Web客户端
        if self.connected_clients:
            asyncio.run_coroutine_threadsafe(
                self.broadcast_detections(),
                asyncio.get_event_loop()
            )

    def status_callback(self, msg):
        """处理状态更新"""
        self.latest_status = msg.data
        
        # 广播给所有Web客户端
        if self.connected_clients:
            asyncio.run_coroutine_threadsafe(
                self.broadcast_status(),
                asyncio.get_event_loop()
            )

    async def broadcast_detections(self):
        """广播检测结果"""
        if not self.connected_clients:
            return
        
        msg = {
            "type": "detections_update",
            "detections": self.latest_detections,
            "timestamp": time.time()
        }
        
        message = json.dumps(msg)
        disconnected = set()
        
        for client in self.connected_clients.copy():
            try:
                await client.send(message)
            except:
                disconnected.add(client)
        
        self.connected_clients -= disconnected

    async def broadcast_status(self):
        """广播状态更新"""
        if not self.connected_clients:
            return
        
        msg = {
            "type": "status_update", 
            "status": self.latest_status,
            "timestamp": time.time()
        }
        
        message = json.dumps(msg)
        disconnected = set()
        
        for client in self.connected_clients.copy():
            try:
                await client.send(message)
            except:
                disconnected.add(client)
        
        self.connected_clients -= disconnected

    def format_detections(self, msg):
        """格式化检测结果"""
        formatted = []
        
        for target in msg.targets:
            for roi in target.rois:
                if target.points:
                    for point in target.points:
                        if point.type == "center_3d" and point.point:
                            pos = point.point[0]
                            
                            detection = {
                                "name": roi.type,
                                "chinese_name": self.get_chinese_name(roi.type),
                                "confidence": round(roi.confidence * 100, 1),
                                "distance": round(pos.z, 2),
                                "position": {
                                    "x": round(pos.x, 2),
                                    "y": round(pos.y, 2), 
                                    "z": round(pos.z, 2)
                                }
                            }
                            formatted.append(detection)
        
        return formatted

    def get_chinese_name(self, english_name):
        """英文转中文"""
        mapping = {
            "person": "人",
            "cup": "杯子", 
            "bottle": "瓶子",
            "apple": "苹果",
            "banana": "香蕉",
            "cell phone": "手机",
            "book": "书",
            "chair": "椅子",
            "laptop": "电脑",
            "bowl": "碗"
        }
        return mapping.get(english_name, english_name)


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = WebSocketBridgeNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
