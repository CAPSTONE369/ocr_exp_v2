## (2) Define the PyTorch Based Dataset
import random, os, json, cv2, re
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
random.seed(42) ## randomness에 일정한 비율(?) 아무튼 random choice등을 할때에 중복된 선택 방지 및 재 구현이 가능하도록 하기 위함이다.
from label_converter_hennet import HangulLabelConverter, GeneralLabelConverter

def load_printed_data(label_data, image_file_dict:dict):
  """ AIHUB의 한글 인쇄체 데이터셋을 사용하기 위한 함수
  """
  annotations = label_data['annotations']
  # idx = random.choice([int(i) for i in range(len(annotations))])
  # annotation = annotations[idx]

  dtype= random.choice([ '글자(음절)', '단어(어절)',])# '문장'])
  if dtype == '문장':
    image_files = image_file_dict['sentence']
  elif dtype == '글자(음절)':
    image_files = image_file_dict['syllable']
  elif dtype == '단어(어절)':
    image_files = image_file_dict['word']

  idx = random.choice([int(i) for i in range(len(image_files))])
  image_fdir = image_files[idx]

  image_fname = image_fdir.split('/')[-1]
  image = cv2.imread(image_fdir)
  #print(image_fdir)
  for anno in annotations:
    if anno['image_id'] == image_fname.split('.')[0]:
      break
  return image, anno['text']
  

def load_outdoor_data(label_folder_dir, image_file_dict: dict):
  """ AIHUB의 야외 한글 간판등 데이터셋을 사용하기 위한 함수
  label_folder_dir: 현재 데이터셋으로 사용하고 싶은 outdoor_1 ~ outdoor_5까지 중에서 해당하는 부분의 json 파일 파일이 저장된 폴더를 의미한다.
  """
  fname= label_folder_dir.split('/')[-1]
  
  image_folder_dir = os.path.join(os.path.dirname(label_folder_dir), fname.replace('target_', ''))
  image_folder_dir = image_folder_dir.replace('/outdoor_target', '')
  image_file_dirs = image_file_dict[image_folder_dir] ## 해당 이미지 폴더에 있는 모든 이미지 경로 배열

  idx = random.choice([int(i) for i in range(len(image_file_dirs))])

  image_file_dir = image_file_dirs[idx]
  json_file_name = image_file_dir.split('/')[-1].split('.')[0] + '.json'

  with open(os.path.join(label_folder_dir, json_file_name), 'r') as f:
    label_data = json.load(f)

  bbox = label_data['annotations'][0]['bbox']
  text = label_data['annotations'][0]['text']

  image = cv2.imread(image_file_dir)
  x, y, w, h = bbox

  if x == None or y == None or w == None or h == None:
    return None, None

  croped_image = image[y:y+h, x:x+w]

  return croped_image, text

### TODO ####
""" Make Batch Balanced Dataset
- Concats multiple datasets
- However, using all the data causes a great burden, so we should train them each with a specific ratio
"""
class BatchBalancedDataset(object):
  def __init__(self, dataset_list, ratio_list: list, loader_args: dict, mode='train'):
    super(BatchBalancedDataset, self).__init__()
    assert sum(ratio_list) == 1 and len(dataset_list) == len(ratio_list)

    self.dataset_len = 0
    self.data_loader_list = []
    self.dataloader_iter_list = []

class BaseDataset(Dataset):
  def __init__(self, mode, DATA_CFG: dict):
    super().__init__()
    self.data_cfg = DATA_CFG
    self.mode = mode
    self.base_folder = DATA_CFG['BASE_FOLDER'] ## 모든 데이터들이 저장이 되어 있는 폴더 
    self.base_characters = DATA_CFG['BASE_CHARACTERS'] ## list
    self.add_num = DATA_CFG['ADD_NUM'] ## 숫자 학습 할지 말지 (bool)
    self.add_eng = DATA_CFG['ADD_ENG'] ## 영어 학습 할지 말지 (bool)
    self.add_special = DATA_CFG['ADD_SPECIAL'] ## 특수 문자 학습 할지 말지 (bool)
    self.max_length = DATA_CFG['MAX_LENGTH'] ## 최대 문자열 길이 (어절로 따지면 25개가 최대)
    self.img_w = DATA_CFG['IMG_W']
    self.img_h = DATA_CFG['IMG_H']
    
    if self.data_cfg['CONVERTER'] == 'general':
      self.label_converter = GeneralLabelConverter(max_length = self.max_length // 3)
    else:
      self.label_converter = HangulLabelConverter(
        base_character=''.join(self.base_characters), add_num=self.add_num, add_eng=self.add_eng,
        add_special=self.add_special, max_length=self.max_length
      )

class HENDatasetOutdoor(BaseDataset):
  def __init__(self, mode, DATA_CFG, ratio):
    super().__init__(mode, DATA_CFG)
    base_dir='/home/guest/ocr_exp_v2/data/croped_outdoor'
    self.base_dir=base_dir
    self.ratio = ratio
    image_files = os.listdir(base_dir)
    self.image_files = list(map(lambda x: os.path.join(base_dir, x), image_files))
    label_dir='/home/guest/speaking_fridgey/ocr_exp_v2/data/croped_outdoor.json'
    with open(label_dir, 'r') as f:
      self.label_data = json.load(f)['annotations']
    self._filter_length()

  def __len__(self):
    if self.mode == 'train':
      return int(len(self.image_files) * self.ratio)
    elif self.mode == 'examine':
      return len(self.image_files)
    else:
      return 100
  
  def _filter_length(self):
    for i in self.label_data:
      text = i['text']
      label = self.label_converter.encode(text, padding=False, one_hot=False)
      if len(label)> self.max_length:
        #print(text)
        # print(i['image'])
        self.label_data.remove(i)
        self.image_files.remove(os.path.join(self.base_dir, i['image']))

  def __getitem__(self, idx):
    if self.mode == 'debug' or self.mode == 'test':
      select = idx
    else:
      select = random.choice(self.image_files)
      # print(select)
    for label in self.label_data: ## 이미지 데이터에 맞는 라벨 데이터를 찾아줄 수 있다.
      if label['image'] == select.split('/')[-1]:
        break
    text = label['text']

    image = cv2.imread(select)
    if self.data_cfg['RGB'] == False:
      image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
      mean, std = self.data_cfg['GRAY_MEAN'], self.data_cfg['GRAY_STD']
    else:
      mean, std = self.data_cfg['MEAN'], self.data_cfg['STD']

    image = cv2.resize(image, (self.img_w, self.img_h))
    
    tensor_image = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=mean, std=std)
    ])(image)
    
    label = self.label_converter.encode(text, padding=True, one_hot=True)

    return tensor_image, label, text, select

class HENDatasetV3(BaseDataset):
  def __init__(self, mode, DATA_CFG, ratio, base_dir, target_num, target_eng, only_hangul=False):
    super().__init__(mode, DATA_CFG)
    self.base_dir=base_dir # '/home/guest/ocr_exp_v2/data/medicine_croped'
    self.data_cfg = DATA_CFG
    self.ratio = ratio
    self.target_num = target_num
    self.target_eng = target_eng
    self.only_hangul = only_hangul
    image_files = os.listdir(self.base_dir)
    image_files = [x for x in image_files if x.split('.')[-1] != 'txt']
    label_file = os.path.join(self.base_dir, 'new_target_data.txt')
    self.image_files = image_files
    with open(label_file, 'r') as f:
      self.label_data = f.readlines()
    self._filter()
    self._shuffle()
  
  def _shuffle(self):
    random.shuffle(self.label_data)

  def __len__(self):
    if self.mode != 'train':
      return 1000
    else:
      return int(len(self.label_data) * self.ratio)
  
  def _filter(self):
    new_label = []
    if len(self.image_files) < len(self.label_data):
      self.label_data = self.label_data[:len(self.image_files)]

    if self.add_eng == False and self.target_eng == True: # self.data_cfg['TARGET_ENG'] == True:
      self.target_eng = False # elf.data_cfg['TARGET_ENG'] = False
    if self.add_num == False and self.target_num == True: # self.data_cfg['TARGET_NUM'] == True:
      self.target_num = False  #self.data_cfg['TARGET_NUM'] = False

    # loop = tqdm(self.label_data)
    for idx, label_info in enumerate(self.label_data):

      image_name, text = label_info.strip('\n').split('\t')
     # image = cv2.imread(os.path.join(self.base_dir, image_name))
      # H, W, C = image.shape
      # if (H > W): ## 세로가 가로에 비해서 길면 미리 
        # loop.set_postfix({"DIFFERNT RATIO" : image_name})
        # print(os.path.join(self.base_dir, image_name))
        # continue
      if self.data_cfg['CONVERTER'] == 'HANGUL':
        if len(text) > self.max_length // 3:
          continue
      else:
        if len(text) > self.max_length:
          continue

      if self.only_hangul:
        han_found = re.findall(re.compile('[가-힣]'), text)
        if len(han_found) == 0:
          continue
      if self.data_cfg['TARGET_BOTH']:
        num_found = re.findall(re.compile('[0-9]'), text)
        eng_found = re.findall(re.compile('[a-zA-Z]'), text)
        if len(num_found)== 0 or len(eng_found) == 0:
          continue
      if self.target_num: # self.data_cfg["TARGET_NUM"]:
        found = re.findall(re.compile('[0-9]'), text)
        if len(found) == 0:
          continue
      if self.target_eng: # self.data_cfg['TARGET_ENG']:
        found = re.findall(re.compile('[a-zA-Z]'), text)
        if len(found) == 0:
          continue

      if self.add_num == False:
        found = re.findall(re.compile('[0-9]'), text)
        if len(found) != 0:
          continue
      if self.add_eng == False:
        found = re.findall(re.compile('[a-zA-Z]'), text)
        if len(found)!=0:
          continue
      new_label.append(label_info)
    self.label_data = new_label

  def __getitem__(self, idx):
    # idx = random.choice([int(i) for i in range(len(self.label_data))])
    label_data = self.label_data[idx]
    image_name, text = label_data.strip('\n').split('\t')

    image = cv2.imread(os.path.join(self.base_dir, image_name))
    if self.data_cfg['RGB'] == False:
      image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      mean = self.data_cfg['GRAY_MEAN']
      std = self.data_cfg['GRAY_STD']
    else:
      mean = self.data_cfg['MEAN'];std = self.data_cfg['STD']
    
    tensor_image = transforms.Compose([
      transforms.ToTensor(),
      transforms.Resize((self.img_h, self.img_w)),
      transforms.Normalize(mean, std)
    ])(image)

    label = self.label_converter.encode(text, padding=True, one_hot=True)

    return tensor_image, label, text, os.path.join(self.base_dir, image_name)

class HENDatasetV2(BaseDataset):
  ## 오직 인쇄체 데이터셋만을 사용하기 위한 데이터셋
  def __init__(self, mode, DATA_CFG, ratio):
    super().__init__(mode, DATA_CFG)
    base_dir=os.path.join(DATA_CFG['BASE_FOLDER'], 'croped_sentence')
    self.base_dir=base_dir
    self.data_cfg = DATA_CFG
    self.ratio = ratio
    data_files = os.listdir(base_dir)
    self.image_files = sorted(x for x in data_files if x.split('.')[-1] == 'png')
    label_data = list(set(data_files) - set(self.image_files))[0]
    self.image_files = [os.path.join(base_dir, x) for x in self.image_files]
    label_data = os.path.join(base_dir, label_data)
    with open(label_data, 'r') as f:
      self.label_data = json.load(f)['annotations']

    self._filter()
  
  def _shuffle(self):
    random.shuffle(self.image_files)

  def _filter(self):
    #out_of_char = f"[^{self.label_converter.char_with_no_tokens}]"
    out_of_char = f"[^ 가-힣a-zA-Z0-9]"
    filtered_label_data = []
    filtered_image_data = []
    for data in self.label_data:
      text = data['text']
      if self.data_cfg['CONVERTER'] == 'HANGUL':
        if len(text) > self.max_length // 3:
          continue
      else:
        if len(text) > self.max_length:
          continue

      if re.search(out_of_char, text):
        continue

      image_name = data['image'].split('/')[-1]
      data['image'] = os.path.join(self.base_dir, image_name)
      filtered_label_data.append(data)
      filtered_image_data.append(os.path.join(self.base_dir, image_name))
    self.label_data = filtered_label_data
    self.image_files = sorted(filtered_image_data)
  

  def __len__(self):
    if self.mode == 'train':
      return int(len(self.image_files) * self.ratio) # len(self.image_files) # int(0.5 * len(self.image_files))
    elif self.mode == 'debug':
      return 3000
    else:
      return 100
    
  def __getitem__(self, idx):
    idx = random.choice([int(i) for i in range(len(self.image_files))])
    #print(self.image_files[idx])
    image = cv2.imread(self.image_files[idx])
    if self.data_cfg['RGB'] == False:
      image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
      mean, std = self.data_cfg['GRAY_MEAN'], self.data_cfg['GRAY_STD']
    else:
      mean, std = self.data_cfg['MEAN'], self.data_cfg['STD']

    # image = cv2.resize(image, (self.img_w, self.img_h))
    
    tensor_image = transforms.Compose([
      transforms.ToTensor(),
       transforms.Resize((self.img_h, self.img_w)),
      transforms.Normalize(mean=mean, std=std)
    ])(image)

    for label_data in self.label_data:
      if label_data['image'] == self.image_files[idx]:
        break


    text = label_data['text']
    label = self.label_converter.encode(text, padding=True, one_hot=True)
    # label = label.type(torch.LongTensor)

    return tensor_image, label, text , self.image_files[idx]




class HENDataset(BaseDataset):
  ## 여러개의 데이터셋을 같이 사용하기 위한 방법이다. (금융, 야외 촬영, 인쇄체 등)
  ## 야외 촬영 데이터셋과 이미지 데이터셋을 모두 같이 사용하기 위한 데이터셋이다.
  def __init__(self, mode, DATA_CFG):
    super().__init__(mode, DATA_CFG)
    
    self.use_outdoor = DATA_CFG['USE_OUTDOOR']
    if DATA_CFG['USE_OUTDOOR'] == True:
      outdoor_dir = os.path.join(self.base_folder, 'outdoor_target')
      self.outdoor_labels = list(map(lambda x: os.path.join(outdoor_dir, x), sorted(os.listdir(outdoor_dir))))
    printed_target = os.path.join(self.base_folder,  'printed_data_info.json')
    with open(printed_target, 'r') as f:
      self.printed_target_data = json.load(f)

    self._make_image_dict()

  def __len__(self):
    if self.mode == 'train':
      items = self.image_file_dict.values()
      n = sum(list(map(lambda x: len(x), items)))
      return n
    else:
      return 100 ## Testing이나 Evaluate할 때는 한번에 하나씩 사용하기 때문에 -> 즉, batch=1으로 inference를 하게 될 것이라는 뜻이다.
  
  def __getitem__(self, idx):
    keys = sorted(list(self.image_file_dict.keys()))
    idx = idx % len(keys)
    
    while True:
      if 'outdoor' in keys[idx]:
        folder_idx = int(keys[idx].split('_')[-1])
        image, text = load_outdoor_data(self.outdoor_labels[folder_idx-1], self.image_file_dict)
      else:
        image, text = load_printed_data(self.printed_target_data, self.image_file_dict)
      #print(text)
      if image is None or text is None:
        continue
      try:
        tensor_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.img_h, self.img_w))
        ])(image)
      except:
        continue
      label = self.label_converter.encode(text, one_hot=True, padding=True)
      if label.shape[0] > self.max_length:
        continue
      else:
        break
    return tensor_image, label, text

  def _make_image_dict(self):
    image_file_dict = {}
    FOLDERS=os.listdir(self.base_folder)
    #print(FOLDERS)
    for folder in FOLDERS:
      if folder not in [
        'word', 'syllable'
      ]:continue
      if 'outdoor' in folder:
        dir = os.path.join(self.base_folder, folder)
        image_file_dict[dir] = list(map(lambda x: os.path.join(dir, x), sorted(os.listdir(dir))))
      elif folder in ['word', 'syllable']:
        dir = os.path.join(self.base_folder, folder)
        if (os.path.isdir(dir) ==False):
          continue
        DTYPES = os.listdir(dir)
        for dtype in DTYPES:
          new_dir = os.path.join(dir, dtype)
          if os.path.isdir(new_dir) == False or 'ipynb' in new_dir:
            continue
          image_file_dict[dtype] = list(map(lambda x: os.path.join(new_dir, x), sorted(os.listdir(new_dir))))
    self.image_file_dict = image_file_dict

