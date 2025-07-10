#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import message_filters
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from geometry_msgs.msg import Point32
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from scipy.special import softmax
import image_geometry
import sys
import time

try:
    from hobot_dnn import pyeasy_dnn as dnn
    from ai_msgs.msg import PerceptionTargets, Target, Roi, Point
except ImportError as e:
    print(f"ERROR: Could not import D-Robotics libraries: {e}")
    sys.exit(1)

# COCO类别名称
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", 
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", 
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", 
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", 
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

class YoloV11Helper:
    """
    针对yolov11n_640_nv12.bin模型的推理助手类
    """
    def __init__(self, model_path, conf_thres=0.25, nms_thres=0.7):
        # 加载模型
        try:
            self.quantize_model = dnn.load(model_path)
            self.model = self.quantize_model[0]
            print(f"Successfully loaded model: {model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise
        
        # 打印模型信息
        print("-> Model Input:")
        for i, inp in enumerate(self.model.inputs):
            print(f"  input[{i}]: {inp.name}, shape={inp.properties.shape}, type={inp.properties.tensor_type}")

        print("-> Model Outputs:")
        for i, out in enumerate(self.model.outputs):
            print(f"  output[{i}]: {out.name}, shape={out.properties.shape}, type={out.properties.tensor_type}")
        
        # 参数设置
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.num_classes = 80
        self.reg = 16  # DFL回归通道数
        self.conf_thres_raw = -np.log(1 / self.conf_thres - 1)
        
        # 获取模型输入尺寸
        self.input_h = self.model.inputs[0].properties.shape[2]
        self.input_w = self.model.inputs[0].properties.shape[3]
        print(f"Model input size: {self.input_w}x{self.input_h}")
        
        # DFL权重
        self.weights_static = np.array([i for i in range(self.reg)]).astype(np.float32)[np.newaxis, np.newaxis, :]
        
        # 初始化anchors
        self._init_anchors()
        
        # 预处理参数
        self.x_scale = 1.0
        self.y_scale = 1.0
        self.x_shift = 0
        self.y_shift = 0
        self.img_w = 0
        self.img_h = 0

    def _init_anchors(self):
        """初始化anchor points"""
        # 80x80 grid (stride=8)
        self.s_anchor = np.stack([
            np.tile(np.linspace(0.5, 79.5, 80), reps=80), 
            np.repeat(np.arange(0.5, 80.5, 1), 80)
        ], axis=0).transpose(1,0)
        
        # 40x40 grid (stride=16)  
        self.m_anchor = np.stack([
            np.tile(np.linspace(0.5, 39.5, 40), reps=40), 
            np.repeat(np.arange(0.5, 40.5, 1), 40)
        ], axis=0).transpose(1,0)
        
        # 20x20 grid (stride=32)
        self.l_anchor = np.stack([
            np.tile(np.linspace(0.5, 19.5, 20), reps=20), 
            np.repeat(np.arange(0.5, 20.5, 1), 20)
        ], axis=0).transpose(1,0)
        
        print(f"Anchors initialized: s={self.s_anchor.shape}, m={self.m_anchor.shape}, l={self.l_anchor.shape}")

    def bgr2nv12(self, bgr_img):
        """BGR转NV12格式"""
        height, width = bgr_img.shape[0], bgr_img.shape[1]
        area = height * width
        yuv420p = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
        y = yuv420p[:area]
        uv_planar = yuv420p[area:].reshape((2, area // 4))
        uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))
        nv12 = np.zeros_like(yuv420p)
        nv12[:height * width] = y
        nv12[height * width:] = uv_packed
        return nv12

    def preprocess_yuv420sp(self, img):
        """预处理函数，使用letterbox保持宽高比"""
        self.img_h, self.img_w = img.shape[0:2]
        print(f"Input image size: {self.img_w}x{self.img_h}")
        
        # Letterbox处理
        self.x_scale = min(1.0 * self.input_h / self.img_h, 1.0 * self.input_w / self.img_w)
        self.y_scale = self.x_scale
        
        if self.x_scale <= 0 or self.y_scale <= 0:
            raise ValueError("Invalid scale factor.")
        
        new_w = int(self.img_w * self.x_scale)
        self.x_shift = (self.input_w - new_w) // 2
        x_other = self.input_w - new_w - self.x_shift
        
        new_h = int(self.img_h * self.y_scale)
        self.y_shift = (self.input_h - new_h) // 2
        y_other = self.input_h - new_h - self.y_shift
        
        # Resize和padding
        input_tensor = cv2.resize(img, (new_w, new_h))
        input_tensor = cv2.copyMakeBorder(input_tensor, self.y_shift, y_other, self.x_shift, x_other, 
                                        cv2.BORDER_CONSTANT, value=[114, 114, 114])
        
        # 转换为NV12
        input_tensor = self.bgr2nv12(input_tensor)
        
        print(f"Preprocess: scale={self.x_scale:.3f}, shift=({self.x_shift}, {self.y_shift})")
        return input_tensor

    def forward(self, input_tensor):
        """BPU推理"""
        return self.model.forward(input_tensor)

    def c2numpy(self, outputs):
        """pyDNNTensor转numpy"""
        return [dnnTensor.buffer for dnnTensor in outputs]

    def postProcess(self, outputs_numpy):
        """后处理函数"""
        print(f"Post-processing {len(outputs_numpy)} outputs...")
        
        # 提取bbox和分类输出（F32格式，无需反量化）
        s_bboxes = outputs_numpy[0].reshape(80*80, 64)  # (6400, 64)
        m_bboxes = outputs_numpy[1].reshape(40*40, 64)  # (1600, 64)  
        l_bboxes = outputs_numpy[2].reshape(20*20, 64)  # (400, 64)
        
        s_clses = outputs_numpy[3].reshape(80*80, 80)   # (6400, 80)
        m_clses = outputs_numpy[4].reshape(40*40, 80)   # (1600, 80)
        l_clses = outputs_numpy[5].reshape(20*20, 80)   # (400, 80)

        print(f"Extracted features: s_bbox={s_bboxes.shape}, s_cls={s_clses.shape}")

        # 分类分支：置信度筛选
        s_max_scores = np.max(s_clses, axis=1)
        s_valid_indices = np.flatnonzero(s_max_scores >= self.conf_thres_raw)
        print(f"Small scale: {len(s_valid_indices)} candidates above threshold")

        m_max_scores = np.max(m_clses, axis=1)
        m_valid_indices = np.flatnonzero(m_max_scores >= self.conf_thres_raw)
        print(f"Medium scale: {len(m_valid_indices)} candidates above threshold")

        l_max_scores = np.max(l_clses, axis=1)
        l_valid_indices = np.flatnonzero(l_max_scores >= self.conf_thres_raw)
        print(f"Large scale: {len(l_valid_indices)} candidates above threshold")

        # 如果没有检测到任何目标
        if len(s_valid_indices) == 0 and len(m_valid_indices) == 0 and len(l_valid_indices) == 0:
            print("No detections above confidence threshold")
            return np.array([]), np.array([]), np.array([])

        # 处理检测结果
        results = []
        
        # 处理小尺度
        if len(s_valid_indices) > 0:
            s_ids = np.argmax(s_clses[s_valid_indices, :], axis=1)
            s_scores = 1 / (1 + np.exp(-s_max_scores[s_valid_indices]))
            s_bboxes_valid = s_bboxes[s_valid_indices, :]
            s_ltrb = np.sum(softmax(s_bboxes_valid.reshape(-1, 4, 16), axis=2) * self.weights_static, axis=2)
            s_anchor_valid = self.s_anchor[s_valid_indices, :]
            s_x1y1 = s_anchor_valid - s_ltrb[:, 0:2]
            s_x2y2 = s_anchor_valid + s_ltrb[:, 2:4]
            s_dbboxes = np.hstack([s_x1y1, s_x2y2]) * 8
            
            for i in range(len(s_valid_indices)):
                results.append((s_ids[i], s_scores[i], s_dbboxes[i]))

        # 处理中尺度
        if len(m_valid_indices) > 0:
            m_ids = np.argmax(m_clses[m_valid_indices, :], axis=1)
            m_scores = 1 / (1 + np.exp(-m_max_scores[m_valid_indices]))
            m_bboxes_valid = m_bboxes[m_valid_indices, :]
            m_ltrb = np.sum(softmax(m_bboxes_valid.reshape(-1, 4, 16), axis=2) * self.weights_static, axis=2)
            m_anchor_valid = self.m_anchor[m_valid_indices, :]
            m_x1y1 = m_anchor_valid - m_ltrb[:, 0:2]
            m_x2y2 = m_anchor_valid + m_ltrb[:, 2:4]
            m_dbboxes = np.hstack([m_x1y1, m_x2y2]) * 16
            
            for i in range(len(m_valid_indices)):
                results.append((m_ids[i], m_scores[i], m_dbboxes[i]))

        # 处理大尺度
        if len(l_valid_indices) > 0:
            l_ids = np.argmax(l_clses[l_valid_indices, :], axis=1)
            l_scores = 1 / (1 + np.exp(-l_max_scores[l_valid_indices]))
            l_bboxes_valid = l_bboxes[l_valid_indices, :]
            l_ltrb = np.sum(softmax(l_bboxes_valid.reshape(-1, 4, 16), axis=2) * self.weights_static, axis=2)
            l_anchor_valid = self.l_anchor[l_valid_indices, :]
            l_x1y1 = l_anchor_valid - l_ltrb[:, 0:2]
            l_x2y2 = l_anchor_valid + l_ltrb[:, 2:4]
            l_dbboxes = np.hstack([l_x1y1, l_x2y2]) * 32
            
            for i in range(len(l_valid_indices)):
                results.append((l_ids[i], l_scores[i], l_dbboxes[i]))

        print(f"Total candidates before NMS: {len(results)}")

        if not results:
            return np.array([]), np.array([]), np.array([])

        # 拼接所有结果
        all_ids = np.array([r[0] for r in results])
        all_scores = np.array([r[1] for r in results])
        all_bboxes = np.array([r[2] for r in results])

        # 转换为xywh格式用于NMS
        hw = all_bboxes[:, 2:4] - all_bboxes[:, 0:2]
        xywh = np.hstack([all_bboxes[:, 0:2], hw])

        # 分类别NMS
        final_results = []
        for class_id in range(self.num_classes):
            class_mask = all_ids == class_id
            if not np.any(class_mask):
                continue
            
            class_boxes = xywh[class_mask]
            class_scores = all_scores[class_mask]
            class_bboxes = all_bboxes[class_mask]
            
            indices = cv2.dnn.NMSBoxes(class_boxes.tolist(), class_scores.tolist(), 
                                     self.conf_thres, self.nms_thres)
            
            if len(indices) > 0:
                for idx in indices:
                    bbox = class_bboxes[idx]
                    score = class_scores[idx]
                    
                    # 坐标还原到原图
                    x1 = int((bbox[0] - self.x_shift) / self.x_scale)
                    y1 = int((bbox[1] - self.y_shift) / self.y_scale)
                    x2 = int((bbox[2] - self.x_shift) / self.x_scale)
                    y2 = int((bbox[3] - self.y_shift) / self.y_scale)

                    # 边界检查
                    x1 = max(0, min(x1, self.img_w))
                    x2 = max(0, min(x2, self.img_w))
                    y1 = max(0, min(y1, self.img_h))
                    y2 = max(0, min(y2, self.img_h))

                    final_results.append((class_id, score, x1, y1, x2, y2))

        print(f"Final detections after NMS: {len(final_results)}")

        if not final_results:
            return np.array([]), np.array([]), np.array([])

        # 解包结果
        final_ids = np.array([r[0] for r in final_results])
        final_scores = np.array([r[1] for r in final_results])
        final_bboxes = np.array([[r[2], r[3], r[4], r[5]] for r in final_results])
        
        return final_bboxes, final_scores, final_ids


class UltimateYoloNode(Node):
    def __init__(self):
        super().__init__('ultimate_yolo_node')
        
        # 参数声明
        self.declare_parameter('model_path', '/root/yolo_models/yolov11n_640_nv12.bin')
        self.declare_parameter('conf_threshold', 0.25)
        self.declare_parameter('nms_threshold', 0.7)
        
        # 获取参数
        model_path = self.get_parameter('model_path').value
        conf_thres = self.get_parameter('conf_threshold').value
        nms_thres = self.get_parameter('nms_threshold').value
        
        # 初始化YOLO助手
        try:
            self.yolo_helper = YoloV11Helper(model_path, conf_thres, nms_thres)
            self.get_logger().info(f"YOLO model loaded successfully from {model_path}")
        except Exception as e:
            self.get_logger().fatal(f"Failed to load YOLO model: {e}")
            raise
        
        # 初始化其他组件
        self.bridge = CvBridge()
        self.camera_model = None
        
        # 创建QoS配置文件
        qos_profile = QoSProfile(depth=5)
        qos_profile.reliability = QoSReliabilityPolicy.RELIABLE
        qos_profile.history = QoSHistoryPolicy.KEEP_LAST
        
        # 订阅器设置 - 使用message_filters进行时间同步
        self.color_sub = message_filters.Subscriber(
            self, CompressedImage, '/camera/color/image_raw/compressed')
        self.depth_sub = message_filters.Subscriber(
            self, Image, '/camera/depth/image_raw')
        
        # 时间同步器 - 增加容忍度
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub], queue_size=10, slop=0.5)
        self.ts.registerCallback(self.sync_callback)
        
        # 添加单独的订阅器用于调试
        self.color_debug_sub = self.create_subscription(
            CompressedImage, '/camera/color/image_raw/compressed', 
            self.color_debug_callback, 10)
        self.depth_debug_sub = self.create_subscription(
            Image, '/camera/depth/image_raw', 
            self.depth_debug_callback, 10)
        
        # 相机信息订阅
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/color/camera_info', self.camera_info_callback, qos_profile)
        
        # 发布器
        self.detection_pub = self.create_publisher(PerceptionTargets, '/yolo_3d_detections', 10)
        
        # 计数器
        self.color_count = 0
        self.depth_count = 0
        self.sync_count = 0
        
        self.get_logger().info("Ultimate YOLO Node initialized successfully!")

    def color_debug_callback(self, msg):
        """彩色图像调试回调"""
        self.color_count += 1
        if self.color_count % 30 == 0:  # 每30帧打印一次
            self.get_logger().info(f"Received color image #{self.color_count}")

    def depth_debug_callback(self, msg):
        """深度图像调试回调"""
        self.depth_count += 1
        if self.depth_count % 30 == 0:  # 每30帧打印一次
            self.get_logger().info(f"Received depth image #{self.depth_count}, encoding: {msg.encoding}")

    def camera_info_callback(self, msg):
        """相机信息回调，用于3D重建"""
        if self.camera_model is None:
            self.camera_model = image_geometry.PinholeCameraModel()
            self.camera_model.fromCameraInfo(msg)
            self.get_logger().info("Camera model initialized")

    def sync_callback(self, color_msg: CompressedImage, depth_msg: Image):
        """同步回调函数，处理彩色图和深度图"""
        self.sync_count += 1
        self.get_logger().info(f"Synchronized callback #{self.sync_count}")
        
        if self.camera_model is None:
            self.get_logger().warn("Camera model not ready, skipping frame")
            return
        
        try:
            # 解码JPEG压缩图像
            np_arr = np.frombuffer(color_msg.data, np.uint8)
            cv_color_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            self.get_logger().info(f"Color image decoded: {cv_color_image.shape}")
            
            # 修正：处理深度图像格式转换问题
            self.get_logger().info(f"Depth image encoding: {depth_msg.encoding}")
            
            # 根据实际编码格式进行转换
            if depth_msg.encoding == "16UC1":
                # 直接使用 passthrough 模式，避免格式转换错误
                cv_depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            elif depth_msg.encoding == "mono16":
                cv_depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "mono16")
            else:
                # 尝试自动检测
                try:
                    cv_depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
                except:
                    cv_depth_image = self.bridge.imgmsg_to_cv2(depth_msg)
            
            self.get_logger().info(f"Depth image converted: {cv_depth_image.shape}, dtype: {cv_depth_image.dtype}")
            
        except CvBridgeError as e:
            self.get_logger().error(f"Image conversion error: {e}")
            return
        except Exception as e:
            self.get_logger().error(f"Image processing error: {e}")
            return
        
        # YOLO推理
        start_time = time.time()
        self.get_logger().info("Starting YOLO inference...")
        
        # 预处理
        input_tensor = self.yolo_helper.preprocess_yuv420sp(cv_color_image)
        
        # 推理
        quantize_outputs = self.yolo_helper.forward(input_tensor)
        
        # 转换为numpy
        outputs_numpy = self.yolo_helper.c2numpy(quantize_outputs)
        
        # 后处理
        bboxes, scores, class_ids = self.yolo_helper.postProcess(outputs_numpy)
        
        inference_time = (time.time() - start_time) * 1000
        self.get_logger().info(f"Inference time: {inference_time:.2f}ms, Detections: {len(bboxes)}")
        
        if len(bboxes) == 0:
            self.get_logger().info("No objects detected")
            return
        
        # 创建检测结果消息
        detection_msg = PerceptionTargets()
        detection_msg.header = color_msg.header
        
        # 处理每个检测结果
        for i in range(len(bboxes)):
            bbox = bboxes[i]
            score = scores[i]
            class_id = int(class_ids[i])
            class_name = COCO_NAMES[class_id] if class_id < len(COCO_NAMES) else f"class_{class_id}"
            
            # 创建目标消息
            target_msg = Target()
            roi_msg = Roi()
            
            # 填充2D信息
            x1, y1, x2, y2 = bbox
            roi_msg.type = class_name
            roi_msg.confidence = float(score)
            roi_msg.rect.x_offset = int(x1)
            roi_msg.rect.y_offset = int(y1)
            roi_msg.rect.width = int(x2 - x1)
            roi_msg.rect.height = int(y2 - y1)
            
            target_msg.rois.append(roi_msg)
            
            # 3D坐标计算
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # 检查坐标是否在深度图范围内
            if (0 <= center_y < cv_depth_image.shape[0] and 
                0 <= center_x < cv_depth_image.shape[1]):
                
                # 获取深度值（使用小区域的中位数提高稳定性）
                patch_size = 5
                half_patch = patch_size // 2
                y_start = max(0, center_y - half_patch)
                y_end = min(cv_depth_image.shape[0], center_y + half_patch + 1)
                x_start = max(0, center_x - half_patch)
                x_end = min(cv_depth_image.shape[1], center_x + half_patch + 1)
                
                depth_patch = cv_depth_image[y_start:y_end, x_start:x_end]
                valid_depths = depth_patch[depth_patch > 0]
                
                if len(valid_depths) > 0:
                    # 深度值通常已经是毫米，转换为米
                    depth_mm = np.median(valid_depths)
                    depth_m = float(depth_mm) / 1000.0
                    
                    # 过滤不合理的深度值
                    if 0.1 < depth_m < 10.0:
                        # 3D坐标计算
                        ray = self.camera_model.projectPixelTo3dRay((center_x, center_y))
                        x_cam = ray[0] * depth_m
                        y_cam = ray[1] * depth_m
                        z_cam = ray[2] * depth_m
                        
                        self.get_logger().info(
                            f"{class_name}: 3D=({x_cam:.2f}, {y_cam:.2f}, {z_cam:.2f})m, "
                            f"2D=({center_x}, {center_y}), conf={score:.2f}")
                        
                        # 创建3D点消息
                        point_msg = Point()
                        point_msg.type = "center_3d"
                        point_3d = Point32()
                        point_3d.x = float(x_cam)
                        point_3d.y = float(y_cam)
                        point_3d.z = float(z_cam)
                        point_msg.point.append(point_3d)
                        target_msg.points.append(point_msg)
            
            detection_msg.targets.append(target_msg)
        
        # 发布检测结果
        if detection_msg.targets:
            self.detection_pub.publish(detection_msg)
            self.get_logger().info(f"Published {len(detection_msg.targets)} detections")


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = UltimateYoloNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
