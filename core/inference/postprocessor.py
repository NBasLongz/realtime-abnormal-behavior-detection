# core/inference/postprocessor.py
import time
import math
from collections import Counter, deque
from typing import Tuple, Dict, Any

class BehaviorPostprocessor:
    def __init__(self, loitering_threshold_sec: float = 10.0, immobility_time_sec: float = 2.0):
        self.loitering_threshold_sec = loitering_threshold_sec
        
        # --- Configs cho Máy trạng thái (State Machine) Té ngã ---
        self.immobility_time_sec = immobility_time_sec
        # Trạng thái theo track_id: "normal", "falling_detected", "immobile_confirmed"
        self.fall_states: Dict[int, str] = {}
        self.fall_timestamps: Dict[int, float] = {}
        self.last_bboxes: Dict[int, Tuple[float, float]] = {} # Lưu tâm (center_x, center_y)

    def _get_center(self, bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _calculate_distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def cleanup_old_tracks(self, active_tids: list):
        """Xóa bộ nhớ đệm của các ID không còn xuất hiện"""
        for tid in list(self.fall_states.keys()):
            if tid not in active_tids:
                del self.fall_states[tid]
                if tid in self.fall_timestamps: del self.fall_timestamps[tid]
                if tid in self.last_bboxes: del self.last_bboxes[tid]

    def process(self, tid: int, raw_label: str, raw_conf: float, bbox: list, 
                current_time: float, recent_preds: deque, time_tracked: float) -> Tuple[str, float]:
        """
        Xử lý mượt nhãn, áp dụng các ngưỡng an toàn, luật Loitering, 
        và Debounced State Machine cho hành vi Té ngã.
        """
        
        # 1. Bầu chọn nhãn (Majority Voting)
        recent_preds.append((raw_label, raw_conf))
        labels_in_queue = [pred[0] for pred in recent_preds]
        most_common_label = Counter(labels_in_queue).most_common(1)[0][0]
        
        confs_of_winner = [pred[1] for pred in recent_preds if pred[0] == most_common_label]
        final_conf = sum(confs_of_winner) / len(confs_of_winner)
        final_label = most_common_label

        # 2. Ngưỡng tin cậy cơ bản (Confidence Thresholding)
        if final_label == "fighting" and final_conf < 0.6:
            final_label = "normal"
        elif final_label == "falling" and final_conf < 0.5:
            final_label = "normal"

        # 3. Đè nhãn Lảng vảng (Rule-based Loitering)
        if time_tracked > self.loitering_threshold_sec and final_label == "normal":
            final_label = "loitering"
            final_conf = 1.0

        # ==============================================================
        # 4. MÁY TRẠNG THÁI TÉ NGÃ (DEBOUNCED STATE MACHINE)
        # Mục tiêu: Đang đứng -> Mô hình báo ngã -> Chờ X giây xem có bất động không
        # Nếu bất động -> Ngã thật. Nếu cử động mạnh (nhặt đồ xong đứng lên) -> Bỏ qua.
        # ==============================================================
        current_center = self._get_center(bbox)
        
        if tid not in self.fall_states:
            self.fall_states[tid] = "normal"
            self.last_bboxes[tid] = current_center

        state = self.fall_states[tid]
        
        # Tính khoảng cách di chuyển từ frame trước
        dist = self._calculate_distance(current_center, self.last_bboxes[tid])
        self.last_bboxes[tid] = current_center

        if final_label == "falling":
            if state == "normal":
                # Bắt đầu pha ngã nhanh
                self.fall_states[tid] = "falling_detected"
                self.fall_timestamps[tid] = current_time
                final_label = "normal" # Dìm nhãn xuống normal, chưa báo động vội
            
            elif state == "falling_detected":
                time_elapsed = current_time - self.fall_timestamps[tid]
                # Kiểm tra bất động (nếu di chuyển < 10 pixel mỗi frame)
                if dist > 10.0:
                    # Đang cử động quá mạnh, có thể là cúi nhặt đồ hoặc chuyển tư thế
                    self.fall_timestamps[tid] = current_time # Reset timer
                    final_label = "normal"
                else:
                    # Đang bất động
                    if time_elapsed >= self.immobility_time_sec:
                        self.fall_states[tid] = "immobile_confirmed"
                        final_label = "falling" # BẬT CÒI BÁO ĐỘNG NGÃ
                    else:
                        final_label = "normal" # Vẫn đang chờ, chưa đủ thời gian
            
            elif state == "immobile_confirmed":
                # Đã xác nhận ngã thật, duy trì nhãn
                final_label = "falling"
        else:
            # Nếu mô hình không còn báo falling nữa (ví dụ người đó đã đứng dậy)
            if state in ["falling_detected", "immobile_confirmed"]:
                self.fall_states[tid] = "normal" # Reset state

        return final_label, final_conf