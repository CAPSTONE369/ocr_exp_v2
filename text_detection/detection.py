import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import *
from loguru import logger
import numpy as np
STD = [0.20037157, 0.18366718, 0.19631825]
MEAN = [0.90890862, 0.91631571, 0.90724233]
BASE = '/content/drive/MyDrive/SpeakingFridgey/model_weights/detection'


def load_weight(weight_name, model):
  pretrained = torch.load(os.path.join(BASE, weight_name))
  model_weight = model.state_dict()
  if 'model_state_dict' in pretrained:
    pretrained = pretrained['model_state_dict']
  available = {key:value for (key, value) in pretrained.items() if key in model_weight and \
                    value.shape == model_weight[key].shape}
  model_weight.update(available)
  model.load_state_dict(model_weight)

  return model

def rm_empty_box(org_image, detected_boxes):
  qualified = []
  for i, bbox in enumerate(detected_boxes):
    xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    croped_image = org_image[ymin:ymax, xmin:xmax,:]
    if croped_image.size>0:
      total_pixels = np.sum(croped_image)
      avg_white_pixels = total_pixels/croped_image.size
      if avg_white_pixels < 250:
        qualified.append(i)
  qualified = np.array(qualified, dtype = np.int32)
  return qualified

def draw_box(image, bboxes, color = (0, 255, 0), thickness = 2):
  for bbox in bboxes:
    xmin, ymin, xmax, ymax = bbox
    image = cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, thickness)
  return image

def make_divisable(value):
  if value % 16 == 0:
    return value
  else:
    a = value // 16
    return (a + 1) * 16

def rescale_for_detect(image):
  C, H, W = image.shape
  
  if H < W:
    new_w = 1024
    new_h = make_divisable(int(H * (1024 / W)))
  else:
    new_h = 1024
    new_w = make_divisable(int(W * (1024 / H)))
  rescale_factor = (W / new_w, H / new_h)
  logger.info(f"NEW_W: {new_w}, NEW_H: {new_h}")
  return (new_w, new_h), rescale_factor

def preprocess(image):
  gray_image = to_gray(image)
  sobel, blurred = sobel_gradient(gray_image)
  morpho = morphology(sobel,1, 1)
  croped, points = crop(morpho, image)
  return croped, points

class DetectBot(object):
  def __init__(self, remove_white=False):
    self.crop = remove_white
    self.cfg = CFG()
    model = CTPN().cuda()
    model = load_weight("CTPN_FINAL_CHECKPOINT.pth", model)
    self.model = model
    self.detector = TextDetector(self.cfg)

  def predict(self, diff_h, diff_w, image: np.ndarray):
    H, W, C = image.shape
    original_image = image
    if H > W:
      new_shape = (2048, 1024) ## (H, W)
    else:
      new_shape = (1024, 2048)
    rescale_factor = (W / new_shape[1], H / new_shape[0])
    image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(new_shape),
        transforms.Normalize(mean = MEAN, std = STD)
      ])(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
      image = image.cuda()
      reg, cls = self.model(image)

    detected_boxes, scores = self.detector((reg, cls), image_size = new_shape)
    print(len(detected_boxes))
    ratio_w, ratio_h = rescale_factor
    size_ = np.array([[ratio_w, ratio_h, ratio_w, ratio_h]])
    detected_boxes *= size_
    detected_boxes += np.array([[diff_w, diff_h, diff_w, diff_h]])
    # detected_boxes = detected_boxes[rm_empty_box(original_image, detected_boxes)]

    return detected_boxes

  def __call__(self, image_path):
    image = cv2.imread(image_path)
    original_image = image
    org_h, org_w, org_c = original_image.shape
    if self.crop:
      croped, points = preprocess(image)
      print(len(points))
    
    else:
      croped = [image];points = [(0, org_w, 0, org_h)];
    all_box = []
    for idx, croped_image in enumerate(croped):
      croped_point = points[idx]
      detected_boxes = self.predict(diff_h=croped_point[2], diff_w=croped_point[0], image=croped_image)
      all_box.extend(detected_boxes)
      original_image = draw_box(original_image, detected_boxes)
    return original_image, all_box



def detect(image_path):
  cfg = CFG()
  image = cv2.imread(image_path)
  original_image = image
  # (new_w, new_h), rescale_factor = rescale_for_detect(image)
  H, W,C = image.shape
  print(H, W)
  if H > W:
    new_shape = (2048, 1024) ## (H, W)
  else:
    new_shape = (1024, 2048)
  rescale_factor = (W / new_shape[1], H / new_shape[0])
  image = transforms.Compose([
      transforms.ToTensor(),
      transforms.Resize(new_shape),
      transforms.Normalize(mean = MEAN, std = STD)
  ])(image)
  image = image.unsqueeze(0)
  model = CTPN().cuda()
  model = load_weight('CTPN_FINAL_CHECKPOINT.pth', model)

  detector = TextDetector(cfg)
  model.eval()
  print(image.shape)
  with torch.no_grad():
    image = image.cuda()
    reg, cls = model(image)
  
  detected_boxes, scores = detector((reg, cls), image_size = new_shape)
  ratio_w, ratio_h = rescale_factor
  size_ = np.array([[ratio_w, ratio_h, ratio_w, ratio_h]])
  detected_boxes *= size_
  detected_boxes = detected_boxes[rm_empty_box(original_image, detected_boxes)]

  drawn_image = draw_box(original_image, detected_boxes)
  return drawn_image