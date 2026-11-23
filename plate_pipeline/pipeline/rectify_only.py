"""Detect plates and save perspective-rectified crops, without OCR."""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression_face


def save_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"Cannot encode {path}")
    encoded.tofile(str(path))


def rectify(image, points):
    # Model order: top-left, top-right, bottom-right, bottom-left.
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    if not np.isfinite(points).all() or not cv2.isContourConvex(points):
        raise ValueError("Invalid plate quadrilateral")
    tl, tr, br, bl = points
    width = int(round(max(np.linalg.norm(tr-tl), np.linalg.norm(br-bl))))
    height = int(round(max(np.linalg.norm(bl-tl), np.linalg.norm(br-tr))))
    if min(width, height) < 2 or abs(cv2.contourArea(points)) < 4:
        raise ValueError("Degenerate plate quadrilateral")
    target = np.float32([[0, 0], [width-1, 0],
                         [width-1, height-1], [0, height-1]])
    matrix = cv2.getPerspectiveTransform(points, target)
    return cv2.warpPerspective(image, matrix, (width, height)), matrix


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=root/'imgs')
    parser.add_argument('--output', type=Path, default=root.parent/'outputs'/'plate_rectification')
    parser.add_argument('--weights', type=Path, default=root/'weights'/'plate_detect.pt')
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--conf', type=float, default=0.3)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = attempt_load(str(args.weights), map_location='cpu')
    files = sorted(p for p in args.source.rglob('*')
                   if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}) if args.source.is_dir() else [args.source]
    (args.output / 'rectified').mkdir(parents=True, exist_ok=True)
    records = []
    with torch.inference_mode():
        for index, path in enumerate(files):
            record = {'source': str(path.resolve()), 'plates': []}
            try:
                encoded = np.fromfile(str(path), dtype=np.uint8)
                image = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
            except (OSError, cv2.error):
                image = None
            if image is None:
                record['error'] = 'unreadable image'
                records.append(record)
                print(f'Skipped unreadable image: {path}')
                continue
            start = time.perf_counter()
            padded, ratio, pad = letterbox(image, new_shape=args.img_size)
            tensor = torch.from_numpy(np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1))).float()[None]/255
            detections = non_max_suppression_face(model(tensor)[0], args.conf, 0.5)[0].cpu().numpy()
            annotated = image.copy()
            prefix = f'{index:03d}_{path.stem}'
            for number, detection in enumerate(detections):
                points = detection[5:13].reshape(4, 2).copy()
                points[:, 0] = (points[:, 0]-pad[0])/ratio[0]
                points[:, 1] = (points[:, 1]-pad[1])/ratio[1]
                try:
                    crop, matrix = rectify(image, points)
                except ValueError as error:
                    record.setdefault('rejected', []).append(str(error))
                    continue
                name = f'{prefix}_plate_{number:02d}.png'
                save_image(args.output/'rectified'/name, crop)
                box = detection[:4].copy()
                box[[0, 2]] = (box[[0, 2]]-pad[0])/ratio[0]
                box[[1, 3]] = (box[[1, 3]]-pad[1])/ratio[1]
                x1, y1, x2, y2 = np.rint(box).astype(int)
                raw = image[max(0,y1):min(image.shape[0],y2), max(0,x1):min(image.shape[1],x2)]
                if raw.size:
                    save_image(args.output/'raw_crops'/name, raw)
                cv2.polylines(annotated, [np.rint(points).astype(np.int32)], True, (0,255,0), 2)
                for corner, point in enumerate(points):
                    xy = tuple(np.rint(point).astype(int))
                    cv2.circle(annotated, xy, 3, (0,0,255), -1)
                    cv2.putText(annotated, str(corner), xy, cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,255), 1)
                record['plates'].append({'file': name, 'confidence': float(detection[4]),
                    'corners_tl_tr_br_bl': points.tolist(), 'size': [crop.shape[1],crop.shape[0]],
                    'type': int(detection[13]), 'homography': matrix.tolist()})
            record['seconds'] = time.perf_counter()-start
            save_image(args.output/'annotated'/f'{prefix}.jpg', annotated)
            records.append(record)
            print(f'{path.name}: {len(record["plates"])} plates, {record["seconds"]:.3f}s')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'results.json').write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Saved to {args.output.resolve()}')


if __name__ == '__main__':
    main()
