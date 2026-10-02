# scripts/data_processing/download_datasets.py

import os
import sys
import struct
import zlib
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
import cv2
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import settings

HF_VIOLENCE_ZIP_URL = "https://huggingface.co/datasets/khoipd/Violence/resolve/main/Real_Life_Violence_Dataset.zip"
HF_FALL_BASE_URL = "https://huggingface.co/datasets/minhy112/fall-detection-data/resolve/main/raw/MCFD/fall"

def get_zip_central_directory(zip_url: str):
    """
    Reads ZIP Central Directory via HTTP Range request without downloading the full archive.
    """
    req_head = urllib.request.Request(zip_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req_head) as resp:
        content_length = int(resp.headers.get("content-length", 0))

    if content_length <= 0:
        raise ValueError("Could not determine ZIP content length.")

    # Read last 64KB for EOCD
    scan_size = min(content_length, 65536)
    req_tail = urllib.request.Request(
        zip_url,
        headers={"Range": f"bytes={content_length - scan_size}-{content_length - 1}", "User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req_tail) as resp:
        tail_data = resp.read()

    eocd_pos = tail_data.rfind(b"PK\x05\x06")
    if eocd_pos == -1:
        raise ValueError("End of Central Directory (EOCD) signature not found.")

    _, _, _, total_entries, cd_size, cd_offset = struct.unpack(
        "<HHHHII", tail_data[eocd_pos + 4 : eocd_pos + 20]
    )

    # Read the Central Directory
    req_cd = urllib.request.Request(
        zip_url,
        headers={"Range": f"bytes={cd_offset}-{cd_offset + cd_size - 1}", "User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req_cd) as resp:
        cd_data = resp.read()

    entries = []
    p = 0
    while p < len(cd_data):
        if cd_data[p : p + 4] != b"PK\x01\x02":
            break
        comp_method = struct.unpack("<H", cd_data[p + 10 : p + 12])[0]
        comp_size, uncomp_size = struct.unpack("<II", cd_data[p + 20 : p + 28])
        fname_len, extra_len, comment_len = struct.unpack("<HHH", cd_data[p + 28 : p + 34])
        local_offset = struct.unpack("<I", cd_data[p + 42 : p + 46])[0]
        fname = cd_data[p + 46 : p + 46 + fname_len].decode("utf-8", errors="ignore")

        entries.append({
            "filename": fname,
            "comp_method": comp_method,
            "comp_size": comp_size,
            "uncomp_size": uncomp_size,
            "fname_len": fname_len,
            "local_offset": local_offset
        })
        p += 46 + fname_len + extra_len + comment_len

    return entries

def download_zip_entry(zip_url: str, entry: dict, output_file: Path) -> bool:
    """
    Downloads and extracts a single file from a remote zip using a single HTTP Range request.
    """
    if output_file.exists() and output_file.stat().st_size > 0:
        return True

    temp_file = output_file.with_suffix(".tmp")
    try:
        local_offset = entry["local_offset"]
        fname_len = entry["fname_len"]
        comp_size = entry["comp_size"]
        comp_method = entry["comp_method"]

        # Read local header + compressed payload in one request
        end_byte = local_offset + 30 + fname_len + 64 + comp_size
        req = urllib.request.Request(
            zip_url,
            headers={"Range": f"bytes={local_offset}-{end_byte}", "User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            buf = resp.read()

        lfname_len, lextra_len = struct.unpack("<HH", buf[26:30])
        data_start = 30 + lfname_len + lextra_len
        raw = buf[data_start : data_start + comp_size]

        if comp_method == 0:
            payload = raw
        elif comp_method == 8:
            payload = zlib.decompress(raw, -15)
        else:
            return False

        with open(temp_file, "wb") as f:
            f.write(payload)

        # Validate video using OpenCV
        cap = cv2.VideoCapture(str(temp_file))
        is_valid = cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) >= 10
        cap.release()

        if is_valid:
            if output_file.exists():
                output_file.unlink()
            temp_file.rename(output_file)
            return True
        else:
            if temp_file.exists():
                temp_file.unlink()
            return False
    except Exception as e:
        if temp_file.exists():
            temp_file.unlink()
        return False

def download_direct_video(url: str, output_file: Path) -> bool:
    """
    Downloads an individual video file directly via HTTP stream and verifies it.
    """
    if output_file.exists() and output_file.stat().st_size > 0:
        return True

    temp_file = output_file.with_suffix(".tmp")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=40) as resp:
            with open(temp_file, "wb") as f:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)

        # Validate with OpenCV
        cap = cv2.VideoCapture(str(temp_file))
        is_valid = cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) >= 10
        cap.release()

        if is_valid:
            if output_file.exists():
                output_file.unlink()
            temp_file.rename(output_file)
            return True
        else:
            if temp_file.exists():
                temp_file.unlink()
            return False
    except Exception:
        if temp_file.exists():
            temp_file.unlink()
        return False

def download_fighting_and_normal(target_dir: Path, n_fight: int = 147, n_normal: int = 130, max_workers: int = 6):
    fight_dir = target_dir / "fighting"
    normal_dir = target_dir / "normal"
    fight_dir.mkdir(parents=True, exist_ok=True)
    normal_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] Parsing Real Life Violence Dataset entries...")
    entries = get_zip_central_directory(HF_VIOLENCE_ZIP_URL)

    v_entries = [e for e in entries if "/Violence/V_" in e["filename"] and e["filename"].endswith(".mp4")]
    nv_entries = [e for e in entries if "/NonViolence/NV_" in e["filename"] and e["filename"].endswith(".mp4")]

    v_selected = v_entries[:n_fight]
    nv_selected = nv_entries[:n_normal]

    print(f"  -> Found {len(v_entries)} violence videos, selected {len(v_selected)}")
    print(f"  -> Found {len(nv_entries)} non-violence videos, selected {len(nv_selected)}")

    # Download fighting
    print(f"\n[2/3] Downloading FIGHTING dataset ({len(v_selected)} videos)...")
    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, entry in enumerate(v_selected, 1):
            out_file = fight_dir / f"fight_{idx:03d}.mp4"
            tasks.append(executor.submit(download_zip_entry, HF_VIOLENCE_ZIP_URL, entry, out_file))

        success_fight = 0
        for future in tqdm(as_completed(tasks), total=len(tasks), desc="Fighting videos"):
            if future.result():
                success_fight += 1

    print(f"  -> Successfully acquired {success_fight}/{len(v_selected)} fighting videos")

    # Download normal
    print(f"\n[3/3] Downloading NORMAL dataset ({len(nv_selected)} videos)...")
    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, entry in enumerate(nv_selected, 1):
            out_file = normal_dir / f"normal_{idx:03d}.mp4"
            tasks.append(executor.submit(download_zip_entry, HF_VIOLENCE_ZIP_URL, entry, out_file))

        success_normal = 0
        for future in tqdm(as_completed(tasks), total=len(tasks), desc="Normal videos"):
            if future.result():
                success_normal += 1

    print(f"  -> Successfully acquired {success_normal}/{len(nv_selected)} normal videos")

def download_falling(target_dir: Path, n_fall: int = 108, max_workers: int = 5):
    fall_dir = target_dir / "falling"
    fall_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading FALLING dataset from Multiple Cameras Fall Dataset (MCFD)...")

    # MCFD has 22 fall scenarios (chute01 to chute22), each recorded from 8 camera angles
    # We distribute across chutes and cameras to maximize viewpoint diversity
    urls_and_targets = []
    video_idx = 1

    # Camera order prioritizing best angles (cam1, cam2, cam3, cam4, cam5, cam6, cam7, cam8)
    for cam_id in [1, 2, 3, 4, 5, 6, 7, 8]:
        for chute_id in range(1, 23):
            if video_idx > n_fall:
                break
            chute_str = f"chute{chute_id:02d}"
            url = f"{HF_FALL_BASE_URL}/{chute_str}/{chute_str}/cam{cam_id}.avi"
            target_path = fall_dir / f"fall_{video_idx:03d}_chute{chute_id:02d}_cam{cam_id}.avi"
            urls_and_targets.append((url, target_path))
            video_idx += 1
        if video_idx > n_fall:
            break

    print(f"  -> Selected {len(urls_and_targets)} fall video streams across 22 scenarios and multiple viewpoints")

    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for url, target_path in urls_and_targets:
            tasks.append(executor.submit(download_direct_video, url, target_path))

        success_fall = 0
        for future in tqdm(as_completed(tasks), total=len(tasks), desc="Falling videos"):
            if future.result():
                success_fall += 1

    print(f"  -> Successfully acquired {success_fall}/{len(urls_and_targets)} falling videos")

def main():
    parser = argparse.ArgumentParser(description="Download and organize benchmark action datasets for Abnormal Behavior Detection")
    parser.add_argument("--num_fight", type=int, default=147, help="Number of fighting videos (default: 147)")
    parser.add_argument("--num_normal", type=int, default=130, help="Number of normal videos (default: 130)")
    parser.add_argument("--num_fall", type=int, default=108, help="Number of falling videos (default: 108)")
    parser.add_argument("--max_workers", type=int, default=6, help="Concurrent download workers (default: 6)")
    args = parser.parse_args()

    raw_dir = settings.data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BENCHMARK ACTION DATASET DOWNLOADER")
    print("=" * 60)
    print(f"Target Directory: {raw_dir}")
    print(f"Planned quota: Fighting={args.num_fight}, Normal={args.num_normal}, Falling={args.num_fall}")

    download_fighting_and_normal(raw_dir, n_fight=args.num_fight, n_normal=args.num_normal, max_workers=args.max_workers)
    download_falling(raw_dir, n_fall=args.num_fall, max_workers=args.max_workers)

    # Final Summary
    fight_count = len(list((raw_dir / "fighting").glob("*.*")))
    normal_count = len(list((raw_dir / "normal").glob("*.*")))
    fall_count = len(list((raw_dir / "falling").glob("*.*")))

    print("\n" + "=" * 60)
    print("DATASET ACQUISITION SUMMARY:")
    print(f"  - Fighting (Bạo lực / Đánh nhau):  {fight_count} videos in {raw_dir / 'fighting'}")
    print(f"  - Normal   (Sinh hoạt bình thường): {normal_count} videos in {raw_dir / 'normal'}")
    print(f"  - Falling  (Té ngã / Tai nạn):      {fall_count} videos in {raw_dir / 'falling'}")
    print(f"  Total: {fight_count + normal_count + fall_count} videos ready for Pose Sequence Extraction!")
    print("=" * 60)

if __name__ == "__main__":
    main()
