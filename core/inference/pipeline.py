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
        lstm_weight: str = "",
        classes: List[str] = ["normal", "staggering", "falling"],
        device: str = "cuda",
        seq_len: int = 30,
        loitering_threshold_sec: float = 10.0,
        immobility_time_sec: float = 2.0
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.classes = classes
        
        self.tracker = PoseTracker(pose_weight, self.device)
        self.preprocessor = SequencePreprocessor(seq_len=seq_len, device=self.device)
        self.postprocessor = BehaviorPostprocessor(
            loitering_threshold_sec=loitering_threshold_sec,
            immobility_time_sec=immobility_time_sec
        )
        
        # Tự động ưu tiên load CTR-GCN ONNX hoặc PyTorch nếu có
        self.is_gcn = False
        self.onnx_session = None
        
        gcn_onnx = settings.weights_dir / "classification" / "ctrgcn_best.onnx"
        gcn_pt = settings.weights_dir / "classification" / "ctrgcn_best.pt"
        
        if gcn_onnx.exists():
            try:
                import onnxruntime as ort
                self.onnx_session = ort.InferenceSession(str(gcn_onnx), providers=['CPUExecutionProvider'])
                self.is_gcn = True
                print(f"Loaded CTR-GCN ONNX model: {gcn_onnx}")
            except Exception as e:
                print(f"Khong the mo ONNX: {e}")
                
        if not self.is_gcn and gcn_pt.exists():
            try:
                from core.models.ctr_gcn import CTRGCN
                self.model = CTRGCN(num_classes=3).to(self.device)
                self.model.load_state_dict(torch.load(str(gcn_pt), map_location=self.device, weights_only=False))
                self.model.eval()
                self.is_gcn = True
                print(f"Loaded CTR-GCN PyTorch model: {gcn_pt}")
            except Exception as e:
                print(f"Khong the mo CTR-GCN: {e}")

        if not self.is_gcn:
            # Fallback về LSTM cũ nếu chưa có CTR-GCN
            self.model = SkeletonLSTM(
                input_size=68, 
                hidden_size=128, 
                num_layers=2,
                num_classes=3
            ).to(self.device)
            if lstm_weight and torch.cuda.is_available() or lstm_weight:
                try:
                    self.model.load_state_dict(torch.load(lstm_weight, map_location=self.device, weights_only=False))
                    print(f"Loaded LSTM weights: {lstm_weight}")
                except Exception as e:
                    print(f"Failed to load LSTM weights: {e}")
            self.model.eval()

        self.track_skeletons = defaultdict(lambda: deque(maxlen=seq_len))
        self.track_start_times = {}
        self.track_recent_preds = defaultdict(lambda: deque(maxlen=7))

    def _setup_logger(self):
        log_file = settings.data_dir / "logs" / f"realtime_log_{datetime.now().strftime('%Y%m%d')}.csv"
        if not log_file.exists():
            with open(log_file, mode='w', newline='') as f:
                csv.writer(f).writerow(["Timestamp", "Track_ID", "Behavior", "Confidence"])
        return log_file

    def recognize_from_video(self, source, display=True):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Loi: Khong the mo video {source}")
            return
            
        log_file = self._setup_logger()
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps != fps:
            fps = 30.0
        frame_delay = 1.0 / fps

        # --- KIEN TRUC TACH LUONG (DECOUPLED ARCHITECTURE) ---
        latest_frame = None
        latest_results = {}
        latest_skeletons = {}
        stop_event = threading.Event()
        
        # Bien do hieu nang thoi gian thuc
        ai_latency_ms = 0.0

        # Luong 1: Doc Camera
        def camera_worker():
            nonlocal latest_frame
            while not stop_event.is_set() and cap.isOpened():
                start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    break
                latest_frame = frame
                if isinstance(source, str): 
                    elapsed = time.time() - start_time
                    time.sleep(max(0, frame_delay - elapsed))
            stop_event.set()

        # Luong 2: AI Inference
        def ai_worker():
            nonlocal latest_results, latest_skeletons, ai_latency_ms
            while not stop_event.is_set():
                if latest_frame is None:
                    time.sleep(0.01)
                    continue
                
                t_ai_start = time.time()
                frame_to_process = latest_frame.copy()
                current_time = time.time()
                
                # 1. Pose Tracking
                persons = self.tracker.track_frame(frame_to_process)
                current_ids = [p["track_id"] for p in persons]
                
                # Don dep track cu
                self.postprocessor.cleanup_old_tracks(current_ids)
                for tid in list(self.track_start_times.keys()):
                    if tid not in current_ids:
                        del self.track_start_times[tid]
                        if tid in self.track_recent_preds: del self.track_recent_preds[tid]
                        if tid in self.track_skeletons: del self.track_skeletons[tid]

                current_preds = {}
                current_skeletons = {}
                
                # 2. Action Classification
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
                        if self.is_gcn:
                            # Chuan hoa chuoi khong gian CTR-GCN
                            kps_list = list(self.track_skeletons[tid])
                            arr = np.array(kps_list) # [T, 17, 3]
                            if arr.shape[0] < 30:
                                pad = np.repeat(arr[-1:], 30 - arr.shape[0], axis=0)
                                arr = np.concatenate([arr, pad], axis=0)
                            else:
                                arr = arr[-30:]
                            
                            xy = arr[:, :, :2]
                            center = np.mean(xy[0], axis=0, keepdims=True)
                            xy_c = xy - center
                            scale = np.max(np.linalg.norm(xy[0] - center, axis=1))
                            if scale > 1e-4: xy_c /= scale
                            arr[:, :, :2] = xy_c
                            
                            gcn_in = np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)
                            
                            if self.onnx_session is not None:
                                ort_inputs = {self.onnx_session.get_inputs()[0].name: gcn_in}
                                raw_out = self.onnx_session.run(None, ort_inputs)[0][0]
                                exp_p = np.exp(raw_out - np.max(raw_out))
                                probs = exp_p / exp_p.sum()
                            else:
                                with torch.no_grad():
                                    logits = self.model(torch.from_numpy(gcn_in).to(self.device))
                                    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                            
                            cls_id = int(np.argmax(probs))
                            raw_conf = float(probs[cls_id])
                            raw_label = self.classes[cls_id]
                        else:
                            # LSTM cu
                            x = self.preprocessor(list(self.track_skeletons[tid]))
                            with torch.no_grad():
                                logits = self.model(x)
                                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                                cls_id = int(np.argmax(probs))
                                raw_conf = float(probs[cls_id])
                                raw_label = self.classes[cls_id]
                            
                        # Qua Postprocessor State Machine
                        label, conf = self.postprocessor.process(
                            tid, raw_label, raw_conf, bbox, 
                            current_time, self.track_recent_preds[tid], time_tracked
                        )

                    current_preds[tid] = (label, conf, bbox)
                    
                    if label not in ["normal", "..."]:
                        with open(log_file, mode='a', newline='') as f:
                            csv.writer(f).writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), tid, label, f"{conf:.2f}"])
                
                latest_results = current_preds
                latest_skeletons = current_skeletons
                ai_latency_ms = (time.time() - t_ai_start) * 1000.0

        # Khoi dong luong
        t_cam = threading.Thread(target=camera_worker, daemon=True)
        t_ai = threading.Thread(target=ai_worker, daemon=True)
        t_cam.start()
        t_ai.start()

        # Luong hien thi
        fps_frame_count = 0
        fps_prev_time = time.time()
        display_fps = 0.0

        while not stop_event.is_set():
            if latest_frame is None:
                continue
            
            # Tinh toan Camera Display FPS thuc te
            fps_frame_count += 1
            now = time.time()
            if now - fps_prev_time >= 0.5:
                display_fps = fps_frame_count / (now - fps_prev_time)
                fps_frame_count = 0
                fps_prev_time = now

            display_frame = latest_frame.copy()
            preds = latest_results.copy()
            skeletons = latest_skeletons.copy()
            
            # Ve Bounding box & Keypoints
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

            # Bang Dashboard do hieu nang thoi gian thuc (System Benchmark HUD)
            cv2.rectangle(display_frame, (10, 10), (330, 85), (0, 0, 0), -1)
            cv2.putText(display_frame, f"Camera Display: {display_fps:.1f} FPS", (20, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(display_frame, f"AI Latency:     {ai_latency_ms:.1f} ms", (20, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

            if display:
                cv2.imshow("Realtime Behavior Analytics", display_frame)
                if cv2.waitKey(int(frame_delay * 1000)) & 0xFF == 27: 
                    stop_event.set()
                    break

        stop_event.set()
        t_cam.join()
        t_ai.join()
        cap.release()
        cv2.destroyAllWindows()