from copy import deepcopy
import logging
import math
import os
import pickle
from pathlib import Path
from sklearn.utils import shuffle
import tqdm
import random

from PIL import Image
import numpy as np
import cv2
import copy
import pandas as pd
import csv
import openpyxl
from pycocotools import mask as maskUtils

# ...............................



# Model..............................

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
from torch import cross
setup_logger()

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg



from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog

def cross_validation(data_dir, part=4):
    train_list = []
    val_list = []
    alldatalist = np.array([i for i in os.listdir(data_dir + '/image')])

    num_data = alldatalist.shape[0]
    
    shuffle_list = np.arange(0, num_data, 1)
    np.random.shuffle(shuffle_list)
    # print(np.random.shuffle(idx_list))
    # shuffle_list = np.random.shuffle(idx_list)
    num_val = num_data // part
    count = 0
    for i in range(0, part-1):
        train = set(copy.deepcopy(shuffle_list))
        val = set(copy.deepcopy(shuffle_list[count:count + num_val]))
        train = train - val
        train_list.append(alldatalist[list(train)])
        val_list.append(alldatalist[list(val)])

        count += num_val
  
    
    train_list.append(alldatalist[shuffle_list[0:count]])
    val_list.append(alldatalist[shuffle_list[count:]])


    return train_list, val_list

def load_fisheye(dataset_dir, datalist):
    """Load a subset of the Balloon dataset.
    dataset_dir: Root directory of the dataset.
    subset: Subset to load: train or val
    """
    # cross_data_path = './CV_weight/BW_iso_alldata_cross_data.pkl'
    # filedata = open(cross_data_path, 'rb')
    # _, split_idx = pickle.load(filedata)
    # filedata.close()
    # val_idx = split_idx[count]
    # tmp_idx = split_idx
    # tmp_idx.pop(count)
    # train_idx = []
    # for i in tmp_idx:
    #     train_idx = train_idx + i
    
    # Train or validation dataset?



    # filenames = ['04561074_20190502_27_down.jpg', '04561074_20190502_27_up.jpg', '07167913_20191022_15_16_17_down.jpg'] # 依照你的需求修改

    D = []
    # Add images
    for idx, filename in enumerate(tqdm.tqdm(datalist)):
        image_path = os.path.join(dataset_dir, 'image', filename)
        ann_path = os.path.join(dataset_dir, 'instance_label', filename[:-4] + '.pkl')
        image = Image.open(image_path)
        height, width = image.size[::-1]
        with open(ann_path, 'rb') as f:
            ann = pickle.load(f)
        

        mask = np.zeros((ann["MapPixel2RegionId"].shape[0], ann["MapPixel2RegionId"].shape[1], len(ann['ListRegion'])), dtype=np.int)
    
        
        annotations = []
        count = 0
        img = cv2.imread(image_path)
        for i in range(len(ann['ListRegion'])):
            t_listRegionId = ann['ListRegion'][i].listRegionId
            count += 1
            for j in range(len(t_listRegionId)):
                bbox = []
                seg = []
                mask[ann['MapPixel2RegionId'] == t_listRegionId[j], i] = 1
                # _mask = np.zeros((ann["MapPixel2RegionId"].shape[0], ann["MapPixel2RegionId"].shape[1]), dtype=np.float)
                # _mask[ann['MapPixel2RegionId'] == j] = 1
                # mask[ann]
                _mask = deepcopy(mask[:, :, i])

            seg_encode = maskUtils.encode(np.asfortranarray(_mask).astype('uint8'))
            # seg_temp = [[int(i[1]), int(i[0])] for i in np.argwhere(_mask == 1)]
            x_min = np.min(np.argwhere(_mask == 1)[:, 1])
            x_max = np.max(np.argwhere(_mask == 1)[:, 1])
            y_min = np.min(np.argwhere(_mask == 1)[:, 0])
            y_max = np.max(np.argwhere(_mask == 1)[:, 0])
            bbox.extend([x_min, y_min, x_max-x_min, y_max-y_min])
            # seg_encode = maskUtils.encode(np.asarray(seg_temp, order="F").astype('uint8'))
            # print(seg_encode)
            
            # img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=(255, 0, 0))
            
            # print(maskUtils.toBbox(seg_encode).flatten().tolist())

        # print(len(seg))
        
        # for a in ann['encoded_masks']:
            # aa = {'bbox': maskUtils.toBbox(bbox).flatten().tolist(), 'bbox_mode': BoxMode.XYWH_ABS, 'category_id': 0, 'segmentation': a}
        # aa = {'bbox': maskUtils.toBbox(bbox).flatten().tolist(), 'bbox_mode': BoxMode.XYWH_ABS, 'category_id': 0, 'segmentation': ann["MapPixel2RegionId"]}
            aa = {'bbox': maskUtils.toBbox(seg_encode).flatten().tolist(), 'bbox_mode': BoxMode.XYWH_ABS, 'category_id': 0, 'segmentation': seg_encode}
            
        # print(bbox)
            annotations.append(aa)
        
        # print(len(annotations))
        # # print("info", filename, ';', count, '\n')
        # cv2.imshow('', img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        dd = {}
        dd.update(
            image_id=idx,  # use file name as a unique image id
            file_name=image_path,
            width=width, height=height,
            annotations=annotations)
        
        # print(dd.keys())
        D.append(dd)
    # print(len(D))
        
    return D

dataset_dir = './data/PA' # 依照你的需求修改

##### part0
# train_list, val_list = cross_validation(dataset_dir, 4)
# allcv = {}
# allcv['train_idx'] = train_list
# allcv['val_idx'] = val_list


# excel_file = pd.ExcelWriter('./CV_weight/PA_cross_data_316.xlsx')
# for i in range(4):
#     d = {}
#     d['Training'] = train_list[i]
#     d['Validation'] = val_list[i]
#     cross_pd = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.items()]))
#     # print([(k, pd.Series(v)) for k, v in d.items()])
#     cross_pd.to_excel(excel_file, sheet_name='part_{}'.format(i+1))
# excel_file.save()


# savecross = open('CV_weight/PA_cross_data_316.pkl', 'wb')
# pickle.dump(allcv, savecross)
# savecross.close()

##### part1~
loadcross = open('CV_weight/PA_cross_data_316.pkl', 'rb')
crossdict = pickle.load(loadcross)
train_list = crossdict['train_idx']
val_list = crossdict['val_idx']
loadcross.close()

part = 0
train_list = np.array(list(train_list[part]) + list(val_list[part]))
iteration = len(train_list) // 2


def load_fisheye_train():
    # return load_fisheye(dataset_dir, train_list[part])
    return load_fisheye(dataset_dir, train_list)

def load_fisheye_val():
    return load_fisheye(dataset_dir, val_list[part])

# 設定資料集
DatasetCatalog.register("PA_train", load_fisheye_train)
# DatasetCatalog.get("BW_train")
# DatasetCatalog.register("PA_val", load_fisheye_val)
MetadataCatalog.get("PA_train").set(thing_classes = ["tooth"])
# MetadataCatalog.get("PA_val").set(thing_classes = ["tooth"])

# test dataset code
import random
from detectron2.utils.visualizer import Visualizer
# dataset_dicts_train = DatasetCatalog.get("PA_train")
# # dataset_dicts_val = DatasetCatalog.get("BW_val")
# # print('Num_training:', len(dataset_dicts_train))
# # print('Num_val:', len(dataset_dicts_val))
# # print(dataset_dicts[0])
# # for d in dataset_dicts_train/:
# for d in random.sample(dataset_dicts_train, 3):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get("PA_train"), scale=0.5)
#     out = visualizer.draw_dataset_dict(d)
#     cv2.imshow("", out.get_image()[:, :, ::-1])
#     cv2.waitKey(0)
# cv2.destroyAllWindows()

# exit()

# Model 設定
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("PA_train",)
# cfg.DATASETS.TEST = ("PA_val",) 
cfg.DATASETS.TEST = () 

cfg.DATALOADER.NUM_WORKERS = 2
cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.MAX_ITER = 40 * iteration # first part
cfg.SOLVER.WEIGHT_DECAY = 0.0001
cfg.SOLVER.GAMMA = 0.1
cfg.SOLVER.STEPS = [] # lr multiplied by gamma at #iter in STEPS
cfg.SOLVER.CHECKPOINT_PERIOD = 10 * iteration # save model every 10 epoches

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)

cfg.MODEL.BACKBONE.FREEZE_AT = 5 # Only head no freeze

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

cfg.TEST.DETECTIONS_PER_IMAGE = iteration

cfg.OUTPUT_DIR = './checkpoints/PAMaskRcnnalldata' # 依照你的需求修改
cfg.MODEL.DEVICE = 'cuda:1'

cfg.INPUT.MASK_FORMAT = "bitmask" # mask mode

print(cfg)

from detectron2.data import transforms as T
from RandomRotation import RandomRotation

augs = [RandomRotation([-15, 15], expand=False), T.RandomFlip(0.5)] # data augmentation
# augs = 0

from detectron2.data import DatasetMapper, build_detection_train_loader, build_detection_test_loader
from detectron2.engine import DefaultTrainer
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
from LossEvalHook import LossEvalHook

class Trainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg, True)
        mapper.tfm_gens = mapper.tfm_gens[0:1] + augs # hack
        # mapper.tfm_gens = mapper.tfm_gens[0:1] # hack
        log_every_n_seconds(
            logging.INFO,
            f'NEW mapper.tfm_gens = {mapper.tfm_gens}',
            n=0,
        )
        return build_detection_train_loader(cfg, mapper=mapper)


    # def build_hooks(self):
    #     mapper = DatasetMapper(self.cfg, True)
    #     mapper.tfm_gens = mapper.tfm_gens[0:1] # hack
    #     log_every_n_seconds(
    #         logging.INFO,
    #         f'NEW mapper.tfm_gens = {mapper.tfm_gens}',
    #         n=0,
    #     )
    #     hooks = super().build_hooks()
    #     hooks.insert(-1, LossEvalHook(
    #         iteration,   # 1 epoch
    #         self.model,
    #         build_detection_test_loader(
    #             self.cfg,
    #             self.cfg.DATASETS.TEST[0],
    #             mapper
    #         )
    #     ))
    #     return hooks

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = Trainer(cfg)
model = trainer.model
for name, value in model.named_parameters():
    print(name)
# trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()


# second part
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
cfg.SOLVER.WARMUP_ITERS = 0
cfg.SOLVER.MAX_ITER = 80 * iteration
cfg.SOLVER.BASE_LR = 0.0001
cfg.MODEL.BACKBONE.FREEZE_AT = 0 # No Freeze

mjp = os.path.join(cfg.OUTPUT_DIR, 'metrics.json')
os.replace(mjp, mjp + '.old')
# trainer = Trainer(cfg)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.start_iter = 40 * iteration
trainer.train()









