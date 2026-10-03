# scripts/data_processing/generate_staggering_data.py

import cv2
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings

def extract_stagger_clip(fall_video_path: Path, output_dir: Path) -> bool:
    """
    Trích xuất pha mất thăng bằng / lảo đảo (Pre-fall Staggering phase) từ video té ngã.
    Pha này diễn ra trong 15% - 50% thời lượng video (trước khi người tiếp đất hoàn toàn).
    """
    output_filename = fall_video_path.name.replace("fall_", "stagger_")
    if not output_filename.endswith(".mp4"):
        output_filename = Path(output_filename).stem + ".mp4"
    out_file = output_dir / output_filename

    if out_file.exists() and out_file.stat().st_size > 1000:
        return True

    cap = cv2.VideoCapture(str(fall_video_path))
    if not cap.isOpened():
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total < 30:
        cap.release()
        return False

    # Pha lảo đảo: lấy từ 15% đến 50% thời lượng (thời điểm mất thăng bằng trước khi ngã)
    start_frame = int(total * 0.15)
    end_frame = int(total * 0.52)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(out_file), fourcc, fps, (w, h))

    for _ in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    out.release()
    cap.release()
    return out_file.exists() and out_file.stat().st_size > 1000

def main():
    fall_dir = settings.data_dir / "raw" / "falling"
    stagger_dir = settings.data_dir / "raw" / "staggering"
    stagger_dir.mkdir(parents=True, exist_ok=True)

    fall_videos = sorted(list(fall_dir.glob("*.avi")) + list(fall_dir.glob("*.mp4")))
    if not fall_videos:
        print(f"Không tìm thấy video trong {fall_dir}!")
        return

    print("=" * 60)
    print(f"TRÍCH XUẤT TẬP DỮ LIỆU STAGGERING (LẢO ĐẢO / MẤT THĂNG BẰNG)")
    print(f"Nguồn: {len(fall_videos)} video từ {fall_dir}")
    print(f"Đích:  {stagger_dir}")
    print("=" * 60)

    success = 0
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(extract_stagger_clip, v, stagger_dir): v for v in fall_videos}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting Staggering"):
            if future.result():
                success += 1

    print(f"\nHoàn tất! Đã trích xuất thành công {success}/{len(fall_videos)} video Staggering.")

if __name__ == "__main__":
    main()
