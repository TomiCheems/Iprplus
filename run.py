"""编辑 IMAGE_FOLDER，然后双击 run.cmd 批量处理。"""
from datetime import datetime
from pathlib import Path
import subprocess
import sys

IMAGE_FOLDER = r"./CCPD2020/"


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    source = Path(IMAGE_FOLDER).expanduser()
    if not source.is_absolute():
        source = root / source
    source = source.resolve()
    if not source.is_dir():
        sys.exit(f"找不到图片文件夹：{source}")
    # Save each run separately so older results can still be inspected.
    result = root / "results" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    subprocess.run(
        [sys.executable, str(root / "pipeline" / "run_batch.py"),
         "--source", str(source), "--output", str(result)],
        check=True,
        cwd=root,
    )
    print(f"输出目录：{result}", flush=True)
