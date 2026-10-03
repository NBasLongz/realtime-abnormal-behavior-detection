# core/inference/pipeline.py
import cv2
import torch
import time
import csv
import numpy as np
import threading
from datetime import datetime
from collections import defaultdict, deque
from typing import List

from core.detection.pose_tracker import PoseTracker
from core.models.lstm_skeleton import SkeletonLSTM
from core.inference.preprocessor import SequencePreprocessor
from core.inference.postprocessor import BehaviorPostprocessor
from config.settings import settings

class RealtimeBehaviorRecognizer:
    def __init__(
        self,
        pose_weight: str,
        lstm_weight: str,
        classes: List[str] = ["normal", "fighting", "falling"],
        device: str = "cuda",
        seq_len: int = 30,
        loitering_threshold_sec: float = 10.0,
        immobility_time_sec: float = 2.0
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.classes = classes
        
        self.tracker = PoseTracker(pose_weight, self.device)
        self.preprocessor = SequencePreprocessor(seq_len=seq_len, device=self.device)
        # Khởi tạo postprocessor mới (đã có State Machine chống báo giả ngã)
        self.postprocessor = BehaviorPostprocessor(
            loitering_threshold_sec=loitering_threshold_sec,
            immobility_time_sec=immobility_time_sec
        )
        
        self.model = SkeletonLSTM(
            input_size=68, 
            hidden_size=128, 
            num_layers=2,
            num_classes=3
        ).to(self.device)
        
        try:
            self.model.load_state_dict(torch.load(lstm_weight, map_location=self.device, weights_only=False))
            print(f"Loaded LSTM weights from {lstm_weight}")
        except Exception as e:
            print(f"Failed to load LSTM weights: {e}")
            
        self.model.eval()

        self.track_skeletons = defaultdict(lambda: deque(maxlen=seq_len))
        self.track_start_times = {}
        self.track_recent_preds = defaultdict(lambda: deque(maxlen=7))

    def _setup_logger(self):
        log_file = settings.data_dir / "logs"/ f"realtime_log_{datetime.now().strftime('%Y%m%d')}.csv"
        if not log_file.exists():
            with open(log_file, mode='w', newline='') as f:
                csv.writer(f).writerow(["Timestamp", "Track_ID", "Behavior", "Confidence"])
        return log_file

    def recognize_from_video(self, source, display=True):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Lỗi: Không thể mở video {source}")
            return
            
        log_file = self._setup_logger()
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps != fps:
            fps = 30.0
        frame_delay = 1.0 / fps

        # --- KIẾN TRÚC TÁCH LUỒNG (DECOUPLED ARCHITECTURE) ---
        latest_frame = None
        latest_results = {}
        latest_skeletons = {}
        stop_event = threading.Event()
        
        # Luồng 1: Đọc Camera mượt mà 30 FPS
        def camera_worker():
            nonlocal latest_frame
            while not stop_event.is_set() and cap.isOpened():
                start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    break
                
                latest_frame = frame
                
                # Giả lập FPS thực tế nếu là video file (để không đọc quá nhanh)
                # Nếu là webcam (source == 0), cv2.read() tự block nên không sao
                if isinstance(source, str): 
                    elapsed = time.time() - start_time
                    time.sleep(max(0, frame_delay - elapsed))
            stop_event.set()

        # Luồng 2: AI Inference
        def ai_worker():
            nonlocal latest_results, latest_skeletons
            while not stop_event.is_set():
                if latest_frame is None:
                    time.sleep(0.01)
                    continue
                
                # Copy frame để AI xử lý không ảnh hưởng tới luồng hiển thị
                frame_to_process = latest_frame.copy()
                current_time = time.time()
                
                # 1. Pose Tracking
                persons = self.tracker.track_frame(frame_to_process)
                current_ids = [p["track_id"] for p in persons]
                
                # Xóa rác state machine
                self.postprocessor.cleanup_old_tracks(current_ids)
                for tid in list(self.track_start_times.keys()):
                    if tid not in current_ids:
                        del self.track_start_times[tid]
                        if tid in self.track_recent_preds: del self.track_recent_preds[tid]
                        if tid in self.track_skeletons: del self.track_skeletons[tid]

                current_preds = {}
                current_skeletons = {}
                
                # 2. Action Classification cho từng người
                for p in persons:
                    tid = p["track_id"]
                    bbox = p["bbox"]
                    keypoints = p["keypoints"]
                    current_skeletons[tid] = keypoints
                    
                    if tid not in self.track_start_times:
                        self.track_start_times[tid] = current_time
                    time_tracked = current_time - self.track_start_times[tid]

                    self.track_skeletons[tid].append(keypoints)
                    label, conf = "...", 0.0

                    if len(self.track_skeletons[tid]) >= 15:
                        x = self.preprocessor(list(self.track_skeletons[tid]))
                        
                        with torch.no_grad():
                            logits = self.model(x)
                            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                            cls_id = int(np.argmax(probs))
                            raw_conf = float(probs[cls_id])
                            raw_label = self.classes[cls_id]
                            
                        # Chạy qua Postprocessor (có State Machine chống báo giả ngã)
                        label, conf = self.postprocessor.process(
                            tid, raw_label, raw_conf, bbox, 
                            current_time, self.track_recent_preds[tid], time_tracked
                        )

                    current_preds[tid] = (label, conf, bbox)
                    
                    # Ghi log nếu có bất thường
                    if label not in ["normal", "..."]:
                        with open(log_file, mode='a', newline='') as f:
                            csv.writer(f).writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), tid, label, f"{conf:.2f}"])
                
                # Cập nhật kết quả mới nhất cho UI
                latest_results = current_preds
                latest_skeletons = current_skeletons

        # Khởi động các luồng
        t_cam = threading.Thread(target=camera_worker, daemon=True)
        t_ai = threading.Thread(target=ai_worker, daemon=True)
        t_cam.start()
        t_ai.start()

        # Luồng chính (Main Thread): Hiển thị giao diện 30 FPS mượt mà
        while not stop_event.is_set():
            if latest_frame is None:
                continue
            
            display_frame = latest_frame.copy()
            preds = latest_results.copy()
            skeletons = latest_skeletons.copy()
            
            for tid, (label, conf, bbox) in preds.items():
                x1, y1, x2, y2 = map(int, bbox)
                color = (0, 255, 0)
                
                if label == "fighting": color = (0, 0, 255)
                elif label == "falling": color = (0, 165, 255)
                elif label == "loitering": color = (255, 0, 0)
                
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame, f"ID {tid} | {label} ({conf:.2f})", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                if tid in skeletons:
                    for px, py, conf_kp in skeletons[tid]:
                        if conf_kp > 0.3:
                            cv2.circle(display_frame, (int(px), int(py)), 3, color, -1)

            if display:
                cv2.imshow("Realtime Behavior Analytics", display_frame)
                # Đảm bảo vòng lặp main hiển thị mượt 30FPS
                if cv2.waitKey(int(frame_delay * 1000)) & 0xFF == 27: 
                    stop_event.set()
                    break

        # Chờ dọn dẹp
        stop_event.set()
        t_cam.join()
        t_ai.join()
        cap.release()
        cv2.destroyAllWindows()