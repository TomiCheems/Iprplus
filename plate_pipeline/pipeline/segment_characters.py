"""OCR/color classification followed by traditional character segmentation."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from plate_recognition.plate_rec import init_model, get_plate_result
from classical_characters import process


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--ocr-weights', type=Path, default=Path(__file__).parent/'weights/plate_rec_color.pth')
    a = p.parse_args()
    torch.set_num_threads(4)
    model = init_model(torch.device('cpu'), str(a.ocr_weights), True)
    files = [a.source] if a.source.is_file() else sorted(f for f in a.source.rglob('*') if f.suffix.lower() in ('.png','.jpg','.jpeg','.bmp'))
    report = []
    for f in files:
        image = cv2.imdecode(np.fromfile(str(f),np.uint8),cv2.IMREAD_COLOR)
        if image is None:
            report.append({'source': str(f), 'status':'rejected', 'reason':'unreadable'})
            continue
        with torch.inference_mode():
            text, probabilities, color, confidence = get_plate_result(image,torch.device('cpu'),model,True)
        try:
            if confidence < .65 or min(probabilities) < .7:
                raise ValueError('Low OCR/color score; manual review required')
            if not ((color == '蓝色' and len(text)==7) or (color == '绿色' and len(text)==8)):
                raise ValueError('Unsupported plate layout; manual review required')
            target = a.output/f.stem
            records = process(f,target,text,color)
            report.append({'source':str(f.resolve()),'status':'automatic_candidate','plate_text':text,'plate_color':color,'color_score':confidence,'count':len(records),'output':str(target.resolve())})
        except ValueError as error:
            report.append({'source':str(f.resolve()),'status':'rejected','reason':str(error)})
        print(json.dumps(report[-1],ensure_ascii=True))
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'batch_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')

if __name__ == '__main__':
    main()
