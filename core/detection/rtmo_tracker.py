# core/detection/rtmo_tracker.py
import cv2
import numpy as np
import os
try:
    import onnxruntime as ort
except ImportError:
    ort = None

class SimpleIoUTracker:
    """
    Tracker siêu nhẹ dựa trên IoU (Intersection over Union) 
    để dùng kèm với RTMO vì RTMO gốc không có bộ tracker tích hợp như YOLO.
    """
    def __init__(self, iou_threshold=0.3, max_lost=5):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.tracks = []
        self.next_id = 1

    def _iou(self, bbox1, bbox2):
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def update(self, detections):
        # Detections: list of dicts {"bbox": [x1,y1,x2,y2], "keypoints": ..., "conf": ...}
        updated_tracks = []
        
        for det in detections:
            best_iou = 0
            best_track_idx = -1
            
            for i, track in enumerate(self.tracks):
                iou = self._iou(det["bbox"], track["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_track_idx = i

            if best_iou >= self.iou_threshold:
                # Gán ID cũ
                det["track_id"] = self.tracks[best_track_idx]["track_id"]
                det["lost"] = 0
                updated_tracks.append(det)
                self.tracks.pop(best_track_idx)
            else:
                # Tạo ID mới
                det["track_id"] = self.next_id
                self.next_id += 1
                det["lost"] = 0
                updated_tracks.append(det)

        # Xử lý các track bị mất (không khớp với detection nào frame này)
        for track in self.tracks:
            track["lost"] += 1
            if track["lost"] <= self.max_lost:
                updated_tracks.append(track)
                
        self.tracks = updated_tracks
        # Trả về các track đang active (lost == 0)
        return [t for t in self.tracks if t["lost"] == 0]

class RTMOPoseTracker:
    def __init__(self, onnx_model_path: str, device: str = "cpu"):
        if ort is None:
            raise ImportError("Vui lòng cài đặt onnxruntime: pip install onnxruntime")
            
        if not os.path.exists(onnx_model_path):
            print(f"⚠️ Chưa có file RTMO ONNX tại: {onnx_model_path}")
            self.session = None
        else:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
            self.session = ort.InferenceSession(onnx_model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            # input shape RTMO thường là [1, 3, 640, 640]
            
        self.tracker = SimpleIoUTracker()

    def preprocess(self, img):
        # Resize và chuẩn hóa ảnh đưa vào chuẩn [1, 3, 640, 640]
        img_resized = cv2.resize(img, (640, 640))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        img_transpose = np.transpose(img_norm, (2, 0, 1)) # HWC to CHW
        return np.expand_dims(img_transpose, axis=0)

    def track_frame(self, frame, conf_thres: float = 0.4):
        """
        Drop-in replacement hoàn hảo cho hàm track_frame của YOLO cũ.
        Trả về đúng cấu trúc: persons = [{"track_id", "bbox", "bbox_conf", "keypoints"}]
        """
        persons = []
        if self.session is None:
            return persons
            
        h, w = frame.shape[:2]
        input_tensor = self.preprocess(frame)
        
        # Chạy inference ONNX
        # Output của RTMO thường trả về boxes [N, 5] và keypoints [N, 17, 3]
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # NOTE: Cấu trúc output thực tế phụ thuộc vào file export ONNX của RTMO.
        # Ở đây giả định index 0 là keypoints, index 1 là boxes.
        try:
            boxes = outputs[1][0] # [N, 5]
            keypoints = outputs[0][0] # [N, 17, 3]
        except:
            return persons
            
        raw_detections = []
        for i in range(len(boxes)):
            conf = boxes[i][4]
            if conf < conf_thres:
                continue
                
            # Scale lại tọa độ box
            x1 = int(boxes[i][0] * (w / 640))
            y1 = int(boxes[i][1] * (h / 640))
            x2 = int(boxes[i][2] * (w / 640))
            y2 = int(boxes[i][3] * (h / 640))
            
            # Scale lại tọa độ keypoints
            kp = keypoints[i].copy()
            kp[:, 0] *= (w / 640)
            kp[:, 1] *= (h / 640)
            
            raw_detections.append({
                "bbox": [x1, y1, x2, y2],
                "bbox_conf": float(conf),
                "keypoints": kp
            })
            
        # Áp dụng IoU tracking
        active_tracks = self.tracker.update(raw_detections)
        return active_tracks
