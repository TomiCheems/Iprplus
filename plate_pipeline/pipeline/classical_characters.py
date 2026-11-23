"""Color-aware connected-component segmentation of rectified single-row plates."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def segment(image, color, expected):
    if color not in ('蓝色', '绿色') or expected not in (7, 8):
        raise ValueError('Unsupported color or OCR length; manual review required')
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mode = cv2.THRESH_BINARY if color == '蓝色' else cv2.THRESH_BINARY_INV
    mask = cv2.threshold(gray, 0, 255, mode + cv2.THRESH_OTSU)[1]
    # Restrict analysis to the lettering band, avoiding the border and bolts.
    mask[:round(h*(.19 if color == '蓝色' else .12))] = 0
    mask[round(h*.85):] = 0
    mask[:, :round(w*.01)] = 0
    mask[:, round(w*.995):] = 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    components = []
    clean = np.zeros_like(mask)
    for label in range(1, n):
        x, y, cw, ch, area = map(int, stats[label])
        if ch >= h*.38 and area >= h*w*.002 and cw < w*.2:
            components.append([x, y, x+cw, y+ch])
            clean[labels == label] = 255
    components.sort()
    # Join disconnected Chinese radicals (or broken strokes) only across
    # small gaps. Never split the entire plate into equal-width slots.
    while len(components) > expected:
        gaps = [components[i+1][0]-components[i][2] for i in range(len(components)-1)]
        i = int(np.argmin(gaps))
        if gaps[i] > h*.09:
            raise ValueError('Too many glyph-like components; manual review required')
        a, b = components[i:i+2]
        components[i:i+2] = [[min(a[0],b[0]),min(a[1],b[1]),max(a[2],b[2]),max(a[3],b[3])]]
    if len(components) != expected:
        raise ValueError(f'Found {len(components)} glyphs, expected {expected}; no forced labels')
    # A Chinese province glyph can have a short detached left radical (氵).
    # Recover it from the first-character margin without accepting separators.
    first = components[0]
    for x,y,cw,ch,area in stats[1:]:
        if x+cw <= first[0] and first[0]-(x+cw) <= h*.15 and ch >= h*.22 and area >= h*w*.002:
            first[0] = int(min(first[0],x))
    # Province glyphs such as 京 have a detached roof and lower strokes.
    # Merge nearby fragments in both axes, bounded by the next character.
    for x,y,cw,ch,area in stats[1:]:
        if (area >= h*w*.002 and ch >= h*.08 and
                x < first[2]+h*.10 and x+cw > first[0]-h*.10 and
                x+cw < components[1][0]):
            first[:] = [int(min(first[0],x)),int(min(first[1],y)),
                        int(max(first[2],x+cw)),int(max(first[3],y+ch))]
    heights = [b[3]-b[1] for b in components]
    if max(heights) / min(heights) > 1.6:
        raise ValueError('Inconsistent character heights; manual review required')
    return components, mask, clean


def process(source, output, text, color):
    image = cv2.imdecode(np.fromfile(str(source),np.uint8),cv2.IMREAD_COLOR)
    if image is None: raise ValueError(f'Unreadable image: {source}')
    boxes, mask, clean = segment(image, color, len(text))
    output.mkdir(parents=True,exist_ok=True)
    def save(name, data): cv2.imencode('.png', data)[1].tofile(str(output/name))
    save('foreground_mask.png', mask)
    save('filtered_components.png', clean)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',24)
    height, width = image.shape[:2]
    frame_w = max(x2-x1 for x1,_,x2,_ in boxes) + 2*max(2, round(height*.03))
    frame_h = max(y2-y1 for _,y1,_,y2 in boxes) + 2*max(3, round(height*.045))
    frames = []
    for x1,y1,x2,y2 in boxes:
        left = (x1+x2-frame_w)//2
        top = (y1+y2-frame_h)//2
        frames.append((left, top, left+frame_w, top+frame_h))
    pad_left = max(0, -min(x1 for x1,_,_,_ in frames))
    pad_top = max(0, -min(y1 for _,y1,_,_ in frames))
    pad_right = max(0, max(x2 for _,_,x2,_ in frames)-width)
    pad_bottom = max(0, max(y2 for _,_,_,y2 in frames)-height)
    padded = cv2.copyMakeBorder(image,pad_top,pad_bottom,pad_left,pad_right,cv2.BORDER_REPLICATE)
    overlay = Image.fromarray(cv2.cvtColor(padded,cv2.COLOR_BGR2RGB)).resize((padded.shape[1]*4,padded.shape[0]*4))
    draw = ImageDraw.Draw(overlay)
    sheet = Image.new('RGB',(len(text)*145+30,270),'#eeeeee')
    sd = ImageDraw.Draw(sheet)
    records = []
    for i,(char,box,frame) in enumerate(zip(text,boxes,frames)):
        x1,y1,x2,y2 = box
        left,top,right,bottom = frame
        px,py = x1-left,y1-top
        margin_right,margin_bottom = right-x2,bottom-y2
        centered = padded[top+pad_top:bottom+pad_top,left+pad_left:right+pad_left]
        name = f'{source.stem}_char_{i:02d}'
        save(name+'.png',centered)
        offset = [(px-margin_right)/2,(py-margin_bottom)/2]
        item = {'image':name+'.png','character_index':i,'character':char,
                'label_source':'plate_OCR_not_independent_character_recognition',
                'source_plate':str(source.resolve()),'plate_text':text,'plate_color':color,
                'bbox_xyxy_exclusive':box,'output_glyph_bbox_xyxy':[px,py,px+x2-x1,py+y2-y1],
                'crop_bbox_xyxy_exclusive':list(frame),'output_size_wh':[frame_w,frame_h],
                'padding_ltrb':[px,py,margin_right,margin_bottom],'glyph_bbox_center_offset_pixels':offset,
                'method':'color_otsu_connected_components_fragment_merge',
                'review_status':'automatic_candidate'}
        (output/(name+'.json')).write_text(json.dumps(item,ensure_ascii=False,indent=2),encoding='utf8')
        records.append(item)
        lx,ty,rx,by = left+pad_left,top+pad_top,right+pad_left,bottom+pad_top
        draw.rectangle((lx*4,ty*4,rx*4-1,by*4-1),outline='#ff3333',width=3)
        draw.text((lx*4,max(0,ty*4-28)),str(i),font=font,fill='#ff3333')
        picture = Image.fromarray(cv2.cvtColor(centered,cv2.COLOR_BGR2RGB))
        scale = min(120/frame_w,180/frame_h)
        picture = picture.resize((round(frame_w*scale),round(frame_h*scale)),Image.Resampling.LANCZOS)
        # Center the padded glyph inside its display card as well.
        sheet.paste(picture,(20+i*145+(120-picture.width)//2,20+(180-picture.height)//2))
        sd.text((35+i*145,215),f'{i}  {char}',font=font,fill='black')
    overlay.save(output/'segmentation_overlay.png')
    sheet.save(output/'characters_montage.png')
    (output/'manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8')
    return records


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--text',required=True,help='Whole-plate OCR text; labels remain OCR predictions')
    p.add_argument('--color',choices=['蓝色','绿色'],required=True)
    a=p.parse_args()
    r=process(a.source,a.output,a.text,a.color)
    print(f'{len(r)} characters saved; geometric centering checked')
