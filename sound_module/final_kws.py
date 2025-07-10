#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import numpy as np
import pyaudio
import threading
import time
import os
import traceback
import librosa

try:
    import bpu_infer_lib
except ImportError:
    bpu_infer_lib = None
try:
    import paddle
    from paddleaudio.compliance.kaldi import fbank
except ImportError:
    paddle = fbank = None

class FinalKwsNode(Node): 
    def __init__(self):
        super().__init__('final_kws_node')
        self.get_logger().info("✅ Final KWS Node (FBank-Shape-Fix) starting...")

        # ... (所有初始化参数和函数保持不变，照抄我上一次的回复) ...
        if not all([bpu_infer_lib, paddle, fbank, librosa]):
            self.get_logger().fatal("Critical libraries missing. Shutting down.")
            self.create_timer(0.1, self.shutdown_gracefully)
            return
        self.declare_parameter('model_path', '/root/xiaojiqiren_kws_final.bin')
        self.declare_parameter('threshold', 0.8) 
        self.declare_parameter('cooldown_seconds', 2.0)
        self.declare_parameter('min_volume_threshold', 10.0) # 修正后的阈值
        self.declare_parameter('detection_interval_s', 0.2) 
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.threshold = self.get_parameter('threshold').get_parameter_value().double_value
        self.cooldown = self.get_parameter('cooldown_seconds').get_parameter_value().double_value
        self.min_volume_threshold = self.get_parameter('min_volume_threshold').get_parameter_value().double_value
        self.detection_interval_s = self.get_parameter('detection_interval_s').get_parameter_value().double_value
        self.last_detection_time = 0 
        self.last_inference_time = 0
        self.native_sample_rate = 44100
        self.target_sample_rate = 16000
        self.channels = 1
        self.bitdepth = pyaudio.paInt16 
        self.window_duration_s = 1.0 
        self.read_chunk_s = 0.1
        self.read_chunk_samples = int(self.native_sample_rate * self.read_chunk_s)
        self.feat_func = lambda waveform, sr: fbank(
            waveform=paddle.to_tensor(waveform), sr=sr,
            frame_shift=10, frame_length=25, n_mels=80,
        )
        self.kws_model = self.setup_model()
        if not self.kws_model: return
        self.audio = pyaudio.PyAudio()
        self.stream = self._initialize_audio_stream()
        if not self.stream: self.audio.terminate(); return
        self.trigger_pub = self.create_publisher(String, '/voice_trigger', 10)
        self.is_running = True
        self.audio_thread = threading.Thread(target=self.audio_loop, daemon=True)
        self.audio_thread.start()
        self.get_logger().info('✅ Node is fully initialized and listening...')
        
    def setup_model(self):
        try:
            model = bpu_infer_lib.Infer(False)
            if not os.path.exists(self.model_path):
                self.get_logger().fatal(f"❌ KWS model file not found: {self.model_path}")
                return None
            model.load_model(self.model_path)
            self.get_logger().info(f'✅ KWS model loaded: {self.model_path}')
            return model
        except Exception as e:
            self.get_logger().fatal(f'❌ Model loading failed: {e}\n{traceback.format_exc()}'); return None

    def _initialize_audio_stream(self): 
        try:
            stream = self.audio.open(
                format=self.bitdepth, 
                channels=self.channels, 
                rate=self.native_sample_rate,
                input=True, 
                input_device_index=0,
                frames_per_buffer=self.read_chunk_samples
            )
            self.get_logger().info(f'✅ Audio stream opened on device 0 with native rate {self.native_sample_rate}Hz.')
            return stream
        except Exception as e:
            self.get_logger().fatal(f'❌ Audio stream initialization failed: {e}.'); return None
    
    def audio_loop(self):
        num_chunks_in_window = int(self.window_duration_s / self.read_chunk_s)
        buffer_chunks = [] 

        self.get_logger().info(f"Initializing audio buffer with {self.window_duration_s}s of data...")
        for _ in range(num_chunks_in_window):
            if not (self.is_running and rclpy.ok()): return 
            try:
                data = self.stream.read(self.read_chunk_samples, exception_on_overflow=False)
                buffer_chunks.append(np.frombuffer(data, dtype=np.int16))
            except Exception as e:
                self.get_logger().error(f"Error filling initial buffer: {e}")
                return 

        self.get_logger().info("✅ Audio buffer initialized. Starting KWS detection loop.")

        while self.is_running and rclpy.ok():
            try:
                data = self.stream.read(self.read_chunk_samples, exception_on_overflow=False)
                new_samples = np.frombuffer(data, dtype=np.int16)
                
                buffer_chunks.pop(0)
                buffer_chunks.append(new_samples)
                
                audio_window_int16_native = np.concatenate(buffer_chunks)
                
                current_time = time.time()
                if current_time - self.last_inference_time >= self.detection_interval_s:
                    self.detect_keyword(audio_window_int16_native, current_time)
                    self.last_inference_time = current_time

            except Exception as e:
                self.get_logger().error(f'❌ Audio loop error: {e}\n{traceback.format_exc()}', throttle_duration_sec=5)

    def detect_keyword(self, audio_data_int16_native, current_time):
        try:
            if current_time - self.last_detection_time < self.cooldown: return

            audio_float32_native = audio_data_int16_native.astype(np.float32) / 32768.0
            audio_float32_target = librosa.resample(y=audio_float32_native, orig_sr=self.native_sample_rate, target_sr=self.target_sample_rate, res_type='kaiser_fast')
            
            rms = np.sqrt(np.mean(audio_float32_target**2)) * 1000 
            
            if rms < self.min_volume_threshold:
                self.get_logger().info(f'[KWS DEBUG] Vol: {rms:5.1f} | Too quiet. Skipping inference.')
                return

            # 1. 官方FBank特征提取: 输出形状为 (Time, N_mels)
            paddle_waveform = paddle.to_tensor(audio_float32_target[np.newaxis, :], dtype='float32')
            keyword_feat_numpy = self.feat_func(paddle_waveform, self.target_sample_rate).numpy() # -> (Time, 80)
            
            # 2. 【核心修正】: 直接送入prepare_model_input，它会处理所有的维度和填充/截断
            final_input = self.prepare_model_input(keyword_feat_numpy)
            
            if final_input is None: return 

            self.kws_model.read_input(final_input, 0)
            self.kws_model.forward(more=True)
            self.kws_model.get_output()
            out = self.kws_model.outputs[0].data

            exp_scores = np.exp(out - np.max(out, axis=1, keepdims=True))
            probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            wake_word_score = float(probabilities[0, 1])
            
            self.get_logger().info(f'[KWS DEBUG] Vol: {rms:5.1f} | Wake Score: {wake_word_score:.4f}')
            
            if wake_word_score > self.threshold:
                self.get_logger().info("🎉🎉🎉 'XIAOJIQIREN' WAKE WORD DETECTED! 🎉🎉🎉")
                msg = String(); msg.data = "wake_up"
                self.trigger_pub.publish(msg)
                self.last_detection_time = current_time

        except Exception as e:
            self.get_logger().error(f'❌ Keyword detection error: {e}\n{traceback.format_exc()}', throttle_duration_sec=5)

    def prepare_model_input(self, features_2d_numpy):
        """
        接收一个2D的(Time, 80)特征Numpy数组，输出一个4D的(1, 1, 373, 80)最终输入。
        """
        # 1. 验证输入形状的初步有效性
        if features_2d_numpy.ndim != 2 or features_2d_numpy.shape[1] != 80:
            self.get_logger().warn(f"Unexpected feature shape for prepare_model_input: {features_2d_numpy.shape}. Expected (Time, 80). Skipping.", throttle_duration_sec=1.0)
            return None

        # 2. 填充或截断时间维度，以匹配模型期望的373帧
        target_time_frames = 373
        current_time_frames = features_2d_numpy.shape[0]
        
        if current_time_frames > target_time_frames:
            padded_features = features_2d_numpy[-target_time_frames:, :]
        elif current_time_frames < target_time_frames:
            padding_size = target_time_frames - current_time_frames
            padding = np.zeros((padding_size, features_2d_numpy.shape[1]), dtype=np.float32)
            padded_features = np.vstack((padding, features_2d_numpy)) # 在垂直方向堆叠，在前面补零
        else:
            padded_features = features_2d_numpy

        # 3. 增加 Batch 和 Channel 维度，并确保最终类型为 float32
        # padded_features 形状是 (373, 80)
        # 最终形状为 (1, 1, 373, 80)
        final_input = np.expand_dims(padded_features, axis=0) # (1, 373, 80)
        final_input = np.expand_dims(final_input, axis=0) # (1, 1, 373, 80)
        
        return final_input.astype(np.float32)
            
    def destroy_node(self):
        """节点关闭时的清理工作"""
        self.is_running = False
        if hasattr(self, 'audio_thread') and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=1.0)
        if hasattr(self, 'stream') and self.stream.is_active():
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'audio'):
            self.audio.terminate()
        self.get_logger().info("✅ KWS node cleaned up.")
        super().destroy_node()
        
    def shutdown_gracefully(self):
        """用于处理初始化失败时的关闭"""
        self.get_logger().info("Shutting down due to initialization failure.")
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'audio') and self.audio:
            self.audio.terminate()
        rclpy.shutdown()

def main(args=None):
    """ROS2主入口点"""
    rclpy.init(args=args)
    node = FinalKwsNode()
    try:
        if hasattr(node, 'is_running') and node.is_running:
            rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nCtrl+C detected, shutting down.")
    except Exception as e:
        if node:
            node.get_logger().fatal(f"An unhandled exception occurred in spin: {e}\n{traceback.format_exc()}")
    finally:
        if node:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
