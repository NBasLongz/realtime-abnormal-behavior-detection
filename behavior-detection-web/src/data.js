// Bảng màu trạng thái hành vi chuẩn UI/UX Enterprise
export const BEHAVIOR_COLOR = {
  fighting: '#EF4444',  // Rose Red - Nguy hiểm / Đánh nhau
  falling: '#F59E0B',   // Warm Amber - Cảnh báo té ngã
  loitering: '#2563EB', // Tech Blue - Chú ý / Lảng vảng
  normal: '#10B981',    // Emerald Green - Bình thường / An toàn
  walking: '#0D9488',   // Teal - Đi bộ
  running: '#EA580C',   // Deep Orange - Chạy nhanh
  standing: '#8B5CF6',  // Violet - Đứng yên
};

// Bảng màu phân biệt độ phân giải camera (Đa dạng, rõ ràng, không trùng màu)
export const RESOLUTION_COLOR = {
  '1920x1080': '#6366F1', // Indigo - Full HD
  '1280x720': '#10B981',  // Emerald Green - HD 720p
  '854x480': '#F59E0B',   // Warm Amber - SD 480p
  '640x360': '#EC4899',   // Pink - 360p
};

export const RESOLUTION_BADGE = {
  '1920x1080': 'bg-indigo-50 text-indigo-700 border-indigo-200',
  '1280x720': 'bg-emerald-50 text-emerald-700 border-emerald-200',
  '854x480': 'bg-amber-50 text-amber-700 border-amber-200',
  '640x360': 'bg-pink-50 text-pink-700 border-pink-200',
};

const RES = ['1920x1080', '1920x1080', '1280x720', '1280x720', '1920x1080', '854x480'];

// Dữ liệu mẫu đồng bộ với backend
export const seedVideos = () =>
  Array.from({ length: 14 }, (_, i) => {
    const duration = +(18 + ((i * 37) % 55) + (i % 7) / 10).toFixed(1);
    const fps = [30, 30, 25, 24, 29.97][i % 5];
    const res = RES[i % 6];
    return {
      id: i + 1,
      time: Date.now() - i * 3.2 * 36e5,
      file: `camera_${String(i + 1).padStart(2, '0')}.mp4`,
      duration,
      resolution: res,
      fps,
      frames: Math.round(duration * fps),
    };
  });

export const fmtTime = (t) => new Date(t).toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' });
export const fmtDateTime = (t) => new Date(t).toLocaleString('vi-VN');
export const mb = (bytes) => (bytes / 1048576).toFixed(1);
