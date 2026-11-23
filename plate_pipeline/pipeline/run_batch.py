"""Detect and segment plates in a photo folder, then export paired datasets."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if not source.is_dir() or not any(
        file.is_file() and file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        for file in source.rglob("*")
    ):
        parser.error("--source must contain at least one supported photo")
    if source == output or source in output.parents:
        parser.error("--output must be outside the input folder")
    output.mkdir(parents=True, exist_ok=False)
    for folder in ("images", "json"):
        (output / folder).mkdir()

    project = Path(__file__).resolve().parent
    detection = output / "detection"
    subprocess.run([sys.executable, str(project / "rectify_only.py"),
                    "--source", str(source), "--output", str(detection)], check=True)
    with TemporaryDirectory() as scratch:
        segments = Path(scratch)
        subprocess.run([sys.executable, str(project / "segment_characters.py"),
                        "--source", str(detection / "rectified"), "--output", str(segments)], check=True)
        plates = read_json(segments / "batch_manifest.json")
        for plate in plates:
            if plate["status"] != "automatic_candidate":
                continue
            folder = Path(plate.pop("output"))
            preview = output / "previews" / folder.name
            preview.mkdir(parents=True)
            for name in ("segmentation_overlay.png", "characters_montage.png",
                         "foreground_mask.png", "filtered_components.png"):
                shutil.copy2(folder / name, preview / name)
            plate["preview"] = preview.relative_to(output).as_posix()
            for row in read_json(folder / "manifest.json"):
                label = row["character"]
                if len(label) != 1 or not label.isalnum():
                    raise ValueError(f"Invalid character label: {label!r}")
                png = folder / row["image"]
                for category, file in (("images", png), ("json", png.with_suffix(".json"))):
                    target = output / category / label / file.name
                    target.parent.mkdir(exist_ok=True)
                    shutil.copy2(file, target)
    result = {"photos": read_json(detection / "results.json"), "plates": plates}
    (output / "batch_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
