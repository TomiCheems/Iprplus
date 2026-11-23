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
    color_folders = {"蓝色": "blue", "绿色": "green"}
    for color in color_folders.values():
        for category in ("images", "json"):
            (output / color / category).mkdir(parents=True)

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
            destination = output / color_folders[plate["plate_color"]]
            rectified = Path(plate["source"])
            for category in ("rectified", "raw_crops"):
                crop = detection / category / rectified.name
                if crop.is_file():
                    crop_folder = destination / category
                    crop_folder.mkdir(exist_ok=True)
                    crop.rename(crop_folder / crop.name)
            plate["source"] = str(destination / "rectified" / rectified.name)
            folder = Path(plate.pop("output"))
            preview = destination / "previews" / folder.name
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
                image_folder = destination / "images" / label
                json_folder = destination / "json" / label
                image_folder.mkdir(exist_ok=True)
                json_folder.mkdir(exist_ok=True)
                shutil.copy2(png, image_folder / png.name)
                row["source_plate"] = plate["source"]
                (json_folder / png.with_suffix(".json").name).write_text(
                    json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
                )
    photos = read_json(detection / "results.json")
    by_crop = {Path(plate["source"]).name: plate for plate in plates}
    for photo in photos:
        for candidate in photo["plates"]:
            plate = by_crop[candidate["file"]]
            rectified = Path(plate["source"])
            candidate["rectified_path"] = rectified.relative_to(output).as_posix()
            raw_crop = rectified.parent.parent / "raw_crops" / rectified.name
            if raw_crop.is_file():
                candidate["raw_crop_path"] = raw_crop.relative_to(output).as_posix()
    (detection / "results.json").write_text(
        json.dumps(photos, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result = {"photos": photos, "plates": plates}
    (output / "batch_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
