import numpy as np
import re

def extract_info(json_data: dict,max_n, word):
  annotations = json_data['annotations'] ## list
  annotations = annotations[0]

  bbox = []
  for poly in annotations['polygons']:
    text = poly['text']
    if text in word:
        continue
    word.append(text)
    points = poly['points']
    minX = np.array(points).T[0].min(); minY = np.array(points).T[1].min();
    maxX = np.array(points).T[0].max(); maxY =np.array(points).T[1].max();

    if re.compile('[^ 가-힣a-zA-Z0-9]').sub('', text) == '': ## 특수문자 있으면 제거
      continue 
    text = re.compile('[^ 가-힣a-zA-Z0-9]').sub(' ', text) ## 특수 문자는 제거한 상태로 학습을 시키도록 한다.
    if len(text) < 0 or len(text) > 25: ## max sequence length 보다는 짧아야 하고 최단 길이보다는 길어야 한다.
      continue
    bbox.append({'points': [minX, minY, maxX, maxY], 'text': text})
  return bbox, word




idx = 0
from tqdm import tqdm
import cv2, os, json
FINANCE_FOLDER='/home/guest/speaking_fridgey/ocr_exp_v2/data/finance'

IMAGE=os.path.join(FINANCE_FOLDER, 'images/result/bank/images')
TARGET=os.path.join(FINANCE_FOLDER, 'annotations/result/bank/annotations')

FINANCE_IMAGE_DIR=sorted(os.listdir(IMAGE))
FINANCE_TARGET_DIR=sorted(os.listdir(TARGET))
print(len(FINANCE_IMAGE_DIR), len(FINANCE_TARGET_DIR))
FINANCE_DEST='/home/guest/speaking_fridgey/ocr_exp_v2/data/finance_croped'
os.makedirs(FINANCE_DEST, exist_ok=True)
words = []
with open(os.path.join(FINANCE_DEST, 'new_target_data.txt'), 'w') as ftext:
  loop = tqdm(zip(FINANCE_IMAGE_DIR, FINANCE_TARGET_DIR))
  for image_dir, target_dir in loop:
    if (image_dir.split('.')[0] != target_dir.split('.')[0]):
      print(image_dir, target_dir)
      break
    image = cv2.imread(os.path.join(IMAGE, image_dir))
    with open(os.path.join(TARGET, target_dir), 'r') as f:
      json_data = json.load(f)
    box, words = extract_info(json_data, 0, words)
    for b in box:
      p = b['points']
      p = [int(k) for k in p]
      text = b['text']
      try:
        croped = image[p[1]:p[3], p[0]:p[2]]
        cv2.imwrite(os.path.join(FINANCE_DEST, f"{idx}.png"), croped)
        ftext.write(f"{idx}.png\t{text}\n")
        loop.set_postfix({"IDX": idx, "TEXT": text})
        idx += 1
      except:
        continue
ftext.close()