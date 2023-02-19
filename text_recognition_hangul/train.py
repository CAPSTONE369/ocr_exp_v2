from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import CTCLoss, CrossEntropyLoss
import os, sys, re

from loss import FocalLoss, SoftCrossEntropyLoss
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import ConcatDataset
from scheduler import CosineAnnealingWarmUpRestarts

USE_CUDA=torch.cuda.is_available()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from label_converter_hennet import HangulLabelConverter, GeneralLabelConverter
DEVICE=torch.device('cuda:6' if USE_CUDA else 'cpu')

import bisect
import math, warnings
from torch.utils.data import Dataset
from typing import (Generic, Iterable, Iterator, List, Optional, Sequence, Tuple, TypeVar, Union)
T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')
class MultiDataset(Dataset):
    datasets: List[Dataset[T_co]]
    cumulative_sizes: List[int]

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super(MultiDataset, self).__init__()
        self.datasets = list(datasets)
        self.cumulative_sizes = sum([len(x) for x in datasets])
        pointer = {}
        for idx, dataset in enumerate(self.datasets):
            pointer[idx] = 0
        self.pointer = pointer
    
    def __len__(self):
        return self.cumulative_sizes
    
    def __getitem__(self, idx):
        cur_dataset = idx % len(self.datasets)
        ptr = self.pointer[cur_dataset]
        self.pointer[cur_dataset] = (ptr + 1) % len(self.datasets[cur_dataset])
        return self.datasets[cur_dataset][ptr]

def make_loss_weight(train_cfg, label_converter):
  characters = ''.join(label_converter.characters)
  weight = torch.ones(len(characters)) 
  numerics_match = re.finditer(re.compile('[0-9]'), characters)
  for m in numerics_match:
    weight[m.start():m.end()] = train_cfg['NUM_WEIGHT']
  eng_mask = re.finditer(re.compile('[A-Za-z]'), characters)
  for m in eng_mask:
    weight[m.start():m.end()] = train_cfg['ENG_WEIGHT']
  return weight
  
class FocalCE(nn.Module):
  def __init__(self, train_cfg):
    super().__init__()
    self.size_average=True
    self.sequence_normalize=True
    self.sample_normalize=True
    self.train_cfg = train_cfg

  def forward(self, input, target):
    length = torch.where(torch.argmax(target, dim=-1) == 0, 0, 1).sum(dim=1)
    # target = torch.argmax(target, dim=-1)
    batch_size, max_seq_length, class_n = target.size(0), target.size(1), target.size(2)
    mask = torch.zeros(batch_size, max_seq_length, class_n)
    for i in range(batch_size):
      mask[i, :length[i], :].fill_(1) ## 실제 문자가 있는 애들은 1.0 그대로의 가중치를 사용한다.
      mask[i, length[i]:, :].fill_(0.9) ## 공백인 애들은 0.9의 가중치로만 loss를 계산한다. -> 완전히 없애버리면 안됨
      """ masking처리 할 때 공백을 완전히 없애면 안되는 이유
      - 기본적으로 max_seq_length에 비해서 실제 데이터의 문자영역 length는 더 짧은 경우가 많다.
      - 따라서 지워버리면 완전 공백으로 다 채우는 것이 유리하다고 판단하게 될수 있으며, sequence length전체에 backward pass로 loss의 gradient가 전달이 안된다.
      - 결과적으로 모델이 원하는대로 학습이 안 되는 것이다.
      """
    mask = mask.type_as(input)
    # max_length = max(length)
    max_length = max(length) # input.size(1)
    input = input * mask
    output = F.cross_entropy(input, target, reduction ='none')
    output = torch.sum(output)
    """
    target = target[:, :max_length]
    mask = mask[:, :max_length]
    input = input.contiguous().view(-1, input.size(2))
    input = F.log_softmax(input, dim=1)
    target = target.contiguous().view(-1, 1)
    mask = mask.contiguous().view(-1, 1)
    output = - input.gather(1, target.long()) * mask
    output = torch.sum(output)
    """
    if self.sequence_normalize:
      output = output / torch.sum(mask[:,:,0])
    if self.sample_normalize: ## 보통 batch에 대해서만 normalize를 하지 전체 sequence에 대해서는 안함
      output = output / batch_size
    
    return output

    
def make_loss_fn(train_cfg, label_converter):
  if train_cfg['LOSS_FN'] == 'CTC': 
    ## 근데 이 loss function은 RNN과 같은 모듈이 포함되어 있을때 (예측 부분으로) 사용하는 것이 맞다.
    criterion = CTCLoss()
  elif train_cfg['LOSS_FN'] == 'SOFTCE':
    criterion = SoftCrossEntropyLoss()

  elif train_cfg['LOSS_FN'] == 'CE':
    if train_cfg['LOSS_WEIGHT'] and isinstance(label_converter, HangulLabelConverter):
      weight = make_loss_weight(train_cfg, label_converter)
      criterion= CrossEntropyLoss(weight = weight)
    else:
      criterion = CrossEntropyLoss()
  elif train_cfg['LOSS_FN'] == 'FOCAL':
    criterion = FocalCE(train_cfg)
  return criterion

def train_one_epoch(model, train_dataloader, optimizer, \
                 train_cfg, epoch, label_converter, scheduler):
  loop=tqdm(train_dataloader)
  model.train()
  torch.set_grad_enabled(True)
  criterion = make_loss_fn(train_cfg, label_converter)
  outs = []

  for idx, batch in enumerate(loop):
    image, label, text,_ = batch
    # logger.info(f"IMAGE_SHAPE: {image.shape}")
    pred = model(image.to(DEVICE), mode='train', batch_size=image.shape[0]) ## [B, L, C]
    
    if train_cfg['LOSS_FN'] == "CTC":
      pred = F.log_softmax(pred, dim=2).permute(1,0,2)
      label = torch.argmax(label, dim=1)
      B = pred.shape[1];T = pred.shape[0]
      input_lengths = torch.full(size=(B,), fill_value=T, dtype=torch.long)
      target_lengths = torch.randint(low=1, high=pred.shape[2], size=(B,), dtype=torch.long)
      loss = criterion(pred, label.to(DEVICE),input_lengths, target_lengths)
    elif train_cfg['LOSS_FN'] == 'FOCAL':
      loss = criterion(pred, label.to(DEVICE),)
    else:
      loss = criterion(pred, label.to(DEVICE))
      # loss = criterion(pred, label.to(DEVICE), ignore=False)## ignore index 대신에 사용함 -> One hot encoding된 target으로 cross entropy를 하면 ignore_index를 사용할 수 없으니 NULL target인 정답은 예측 제외
      
    # loss = F.cross_entropy(pred, label.to(DEVICE)) ## loss function은 그냥 우선은 cross entropy 사용
  
    outs.append(loss)

    optimizer.zero_grad() # clear gradients
    loss.backward() ## backward pass
    nn.utils.clip_grad_norm_(model.parameters(), 5.0) ## gradient clipping to make the model training converge(방향은 유지하고 gradient의 크기 제한한)
    optimizer.step() ## update parameters
    #label.detach();image.detach();pred.detach();

    loop.set_postfix({"loss": loss.detach().item(), "epoch": epoch, "lr": optimizer.param_groups[0]['lr']})
    pred_text,_,_ = label_converter.decode(pred)
    print(pred_text)
  epoch_metric = torch.mean(torch.stack([x for x in outs]))
  scheduler.step()
  return epoch_metric, model, optimizer, scheduler

def test_one_epoch(model, test_dataloader, converter):
  loop = tqdm(test_dataloader)
  torch.set_grad_enabled(False)
  criterion = CrossEntropyLoss()
  model.eval()

  preds = []
  correct = 0
  targets = []
  for idx, batch in enumerate(loop):
    image, label, text,_ =batch
    pred = model(image.to(DEVICE), mode='eval', batch_size=image.shape[0])
    # loss = F.cross_entropy(pred, label.to(DEVICE))
    loss = criterion(pred, label.to(DEVICE))
    loop.set_postfix({"loss": loss.detach()})
    preds.append(converter.decode(pred))
    targets.append(text)
    #pred.detach();image.detach();label.detach();
  
  for pred, gt in zip(preds, targets):
    if pred == gt:
      correct += 1
  
  accuracy = (correct / len(test_dataloader)) * 100

  return accuracy


  
def debug_model_layer(model, optimizer, train_loader, train_cfg, epoch):

  model.train()
  criterion = make_loss_fn(train_cfg)
  torch.set_grad_enabled(True)

  for i in range(epoch):
    loop = tqdm(train_loader)
    for idx, batch in enumerate(loop):
      image, label, text,_ = batch
      image = image.to(DEVICE)
      label = label.to(DEVICE)

      pred = model(image, mode='eval', batch_size=image.shape[0])
      if train_cfg['LOSS_FN'] == 'FOCAL':
        argm_label = torch.argmax(label, dim=-1)
        length = torch.stack([
        torch.argmin(x, dim=-1) for x in argm_label
        ], dim=0)
        loss = criterion(pred, label.to(DEVICE), length)
      else:
        loss = criterion(pred, label.to(DEVICE))

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      loop.set_postfix({
        "EPOCH" : i+1, "LOSS" : loss
      })
      pred_text,_,_ = train_loader.dataset.label_converter.decode(pred)
      print(pred_text)
  with open("debug_unet_res.txt", 'w') as f:
    for name, param in model.named_parameters():
      if param.requires_grad:
        f.writelines("-"*80 + "\n")
        f.writelines(f"{name}  |  {param} \n")
        f.writelines("-"*80 + '\n')

    


import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import yaml, os
import numpy as np
from torch.utils.data import DataLoader
# import pytorch_lightning as pl
from loguru import logger
import datetime
TODAY=datetime.datetime.now()
TODAY=TODAY.strftime('%Y-%m-%d')
CONFIG_DIR='/home/guest/speaking_fridgey/ocr_exp_v2/text_recognition_hangul/configs'
from dataset import HENDataset, HENDatasetV2, HENDatasetOutdoor, HENDatasetV3
from model.hen_net import HENNet

if __name__ == "__main__":
  torch.autograd.set_detect_anomaly(True)
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type=str)
  parser.add_argument('--exp', type=str)
  args = parser.parse_args()
  config_name=args.config
  config_dir = os.path.join(CONFIG_DIR, config_name)
  exp_name=args.exp

  with open(config_dir, 'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
  data_cfg = cfg['DATA_CFG']
  train_cfg = cfg['TRAIN_CFG']
  model_cfg = cfg['MODEL_CFG']

  os.makedirs(os.path.join(train_cfg['WEIGHT_FOLDER'], exp_name), exist_ok=True)
  os.makedirs(os.path.join(train_cfg['OPTIM_FOLDER'], exp_name), exist_ok=True)

  ratio = data_cfg['RATIO']
  # assert (sum(ratio)== 1 and len(ratio) == len(data_cfg['DATASET']))
  use_datasets = []
  for idx, i in enumerate(data_cfg['DATASET']):
    if 'HENDatasetOutdoor'  == i:
      train_dataset = HENDatasetOutdoor(mode='train', DATA_CFG=data_cfg, ratio = ratio[idx])
      use_datasets.append(train_dataset)

    if 'HENDatasetV2' == i:
      train_dataset = HENDatasetV2(mode='train', DATA_CFG=data_cfg, ratio=ratio[idx])
      use_datasets.append(train_dataset)
    
    if 'HENDatasetV3' == i: ## 한국어 인쇄체 데이터셋이 아닌 경우에는 무조건 한글이 포함되어 있는 데이터만 사용하도록 한다.
      if data_cfg['USE_BANK']:
        train_dataset = HENDatasetV3(mode='train', DATA_CFG=data_cfg, ratio= 1.0,
          base_dir ='/home/guest/speaking_fridgey/ocr_exp_v2/data/finance_croped', target_num=False, target_eng=False, only_hangul=True)
        use_datasets.append(train_dataset)
      if data_cfg['USE_MEDICINE']: ## 의약품 패키지 데이터셋
        train_dataset = HENDatasetV3(mode='train', DATA_CFG=data_cfg, ratio = ratio[idx], 
          base_dir='/home/guest/speaking_fridgey/ocr_exp_v2/data/medicine_croped', target_num= False, target_eng=False, only_hangul=True)
        use_datasets.append(train_dataset)
      if data_cfg['USE_COSMETICS']: ## 화장품 패키지 데이터셋
        train_dataset = HENDatasetV3(mode='train', DATA_CFG=data_cfg, ratio=ratio[idx],
           base_dir='/home/guest/speaking_fridgey/ocr_exp_v2/data/cosmetics_croped', target_num = True, target_eng =False, only_hangul=True)
        use_datasets.append(train_dataset)

  ## MAKE THE CONCATENATED DATASET ##
  train_dataset = MultiDataset(use_datasets) # ConcatDataset(use_datasets) 
  if 'V3' in data_cfg['DATASET']:
    test_dataset = HENDatasetV3(mode='test', DATA_CFG=data_cfg, ratio=1.0) ## TEST WITH DATA WITH V2 or V3
  else:
    test_dataset = HENDatasetV2(mode= 'test', DATA_CFG=data_cfg, ratio=1.0)

  debug_dataset = HENDatasetV2(mode='debug', DATA_CFG=data_cfg, ratio=1.0)

  train_loader = DataLoader(train_dataset, batch_size=train_cfg['BATCH_SIZE'], shuffle=True, drop_last=True)
  test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
  debug_loader = DataLoader(debug_dataset, batch_size = 30, shuffle=True)

  ### LABEL CONVERTER LOAD ###
  if data_cfg['CONVERTER'] == 'general':
    label_converter = GeneralLabelConverter(max_length = data_cfg['MAX_LENGTH'] // 3)
  
  else:
    label_converter = HangulLabelConverter(
      base_character=''.join(data_cfg['BASE_CHARACTERS']), add_num=data_cfg['ADD_NUM'], add_eng=data_cfg['ADD_ENG'],
      add_special=data_cfg['ADD_SPECIAL'], max_length=data_cfg['MAX_LENGTH']
    )
  logger.info(f"CHARACTER NUM: {len(label_converter.char_decoder_dict)} MAX LENGTH: {label_converter.max_length}")
  ### MODEL LOAD ####
  
  model = HENNet(
      img_w=model_cfg['IMG_W'], img_h=model_cfg['IMG_H'], res_in=model_cfg['RES_IN'],
      encoder_layer_num=model_cfg['ENCODER_LAYER_NUM'], 
      make_object_query = model_cfg['MAKE_OBJECT_QUERY'],
      use_resnet=model_cfg['USE_RES'],
      activation=model_cfg['ACTIVATION'],
      seperable_ffn=model_cfg['SEPERABLE_FFN'],
      adaptive_pe=model_cfg['ADAPTIVE_PE'],
      batch_size=model_cfg['BATCH_SIZE'],
      rgb=model_cfg['RGB'],
      tps=model_cfg['TPS'],
      use_conv = model_cfg['USE_CONV'],
      attentional_transformer = model_cfg['ATTENTIONAL_TRANSFORMER'],
      head_num=model_cfg['HEAD_NUM'],
      max_seq_length=label_converter.max_length, # model_cfg['MAX_SEQ_LENGTH'],
      embedding_dim=model_cfg['EMBEDDING_DIM'],
      class_n=len(label_converter.characters)) 
  
  if model_cfg['PRETRAINED'] != '':
    model.load_state_dict(torch.load(model_cfg['PRETRAINED']))
  
  model.to(DEVICE)
  ### OPTIMIZER SETUP ###
  if train_cfg['OPTIMIZER'] == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr = train_cfg['LR'], momentum=train_cfg['MOMENTUM'])
  elif train_cfg['OPTIMIZER'] == 'ADAM':
    optimizer = torch.optim.Adam(model.parameters(), lr = train_cfg['LR'], betas=[0.9, 0.999])
  else:
    raise RuntimeError
  if model_cfg['PRETRAINED_OPTIM'] != '':
    optimizer.load_state_dict(torch.load(model_cfg['PRETRAINED_OPTIM']))
  #optimizer = torch.optim.Adagrad(model.parameters(), lr=1.0) ## Adagrad는 알아서 adaptive learning rate를 찾아가기 때문에 처음 learning rate는 1이어야 한다.
  # for g in optimizer.param_groups:
  #   g['lr'] = train_cfg['LR']
  ### SCHEDULER SETUP ###
  if train_cfg['SCHEDULER'] == 'CYCLIC':
    cycle_momentum=False if isinstance(optimizer, torch.optim.Adam) else True
    scheduler = CyclicLR(optimizer, base_lr = train_cfg['LR'], max_lr=train_cfg['LR'] * 10, \
          step_size_up=2500, step_size_down=None, mode='triangular2', \
          cycle_momentum=cycle_momentum)
  elif train_cfg['SCHEDULER'] == 'STEP':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40
  , gamma=0.5)
  
  elif train_cfg['SCHEDULER'] == 'COSINE':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
  else:
    raise RuntimeError


  ## DEBUGGING TO CHECK IF THE MODEL IS NOT IN THE LOCAL MINIMA ##
  max_acc = 0.0;min_loss=100;
  if train_cfg['DEBUG'] == True:
    logger.info("==== START DEBUGGING ====")
    debug_model_layer(model, optimizer, train_loader, train_cfg, epoch=2)
 

  logger.info("==== START TRAINING ====")
  for epoch in range(train_cfg['EPOCH']):
    epoch_loss, model, optimizer, scheduler = train_one_epoch(model, train_loader, optimizer,  train_cfg, epoch, label_converter, scheduler)
   
    if epoch_loss < min_loss:
      min_loss = epoch_loss
      torch.save(model.state_dict(), os.path.join(train_cfg['WEIGHT_FOLDER'],exp_name, f"{TODAY}_min_loss.pth"))
    if (epoch+1) % train_cfg['EVAL_EPOCH'] == 0:
      logger.info("=== START EVALUATION ===")
      accuracy = test_one_epoch(model, test_loader, label_converter)
      if max_acc < accuracy:
        mac_acc = accuracy
        torch.save(model.state_dict(), os.path.join(train_cfg['WEIGHT_FOLDER'],exp_name, f"{TODAY}_best.pth"))
    torch.save(model.state_dict(), os.path.join(train_cfg['WEIGHT_FOLDER'],exp_name, f"{TODAY}.pth"))
    torch.save(optimizer.state_dict(), os.path.join(train_cfg['OPTIM_FOLDER'],exp_name, f"{TODAY}.pth"))

    if isinstance(train_dataset, torch.utils.data.ConcatDataset):
      for ds in train_dataset.datasets:
        if isinstance(ds, HENDatasetV3):
          ds._shuffle()
        
        
    # scheduler.step()
    logger.info(f"EPOCH: {epoch+1} LR: {scheduler.get_last_lr()} LOSS: {epoch_loss}")




