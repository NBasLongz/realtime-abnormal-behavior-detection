# scripts/data_processing/download_weights.py

import sys
from pathlib import Path
import urllib.request
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import settings

MODELS = {
    "yolov8s-pose.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-pose.pt",
    "yolov8s.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt"
}

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file(url: str, output_path: Path):
    if output_path.exists() and output_path.stat().st_size > 1024*1024:
        print(f"{output_path.name} already exists ({round(output_path.stat().st_size / 1024**2, 2)} MB). Skipping.")
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(".tmp")

    print(f"Downloading {output_path.name} from {url}...")
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
            urllib.request.urlretrieve(url, filename=temp_path, reporthook=t.update_to)
        
        if temp_path.exists() and temp_path.stat().st_size > 1024*1024:
            if output_path.exists():
                output_path.unlink()
            temp_path.rename(output_path)
            print(f"Successfully saved {output_path.name}")
            return True
        else:
            if temp_path.exists():
                temp_path.unlink()
            return False
    except Exception as e:
        print(f"Error downloading {output_path.name}: {e}")
        if temp_path.exists():
            temp_path.unlink()
        return False

def main():
    detection_dir = settings.weights_dir / "detection"
    detection_dir.mkdir(parents=True, exist_ok=True)

    print("="* 60)
    print("DOWNLOADING PRE-TRAINED DETECTION WEIGHTS")
    print("="* 60)

    for filename, url in MODELS.items():
        dest = detection_dir / filename
        download_file(url, dest)

    print("\n Weight download process finished.")

if __name__ == "__main__":
    main()
