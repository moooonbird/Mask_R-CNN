from copy import deepcopy
from inspect import Parameter
import logging
import math
import os
import pickle
from pathlib import Path
import tqdm
import time


from PIL import Image
import numpy as np
import cv2
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



def load_fisheye(dataset_dir, datalist):
    """Load a subset of the Balloon dataset.
    dataset_dir: Root directory of the dataset.
    subset: Subset to load: train or val
    """
    # cross_data_path = './CV_weight/BW_panoptic_cross_data.pkl'
    # filedata = open(cross_data_path, 'rb')
    # _, split_idx = pickle.load(filedata)
    # filedata.close()
    # val_idx = split_idx[count]
    # tmp_idx = split_idx
    # tmp_idx.pop(count)
    # train_idx = []
    # for i in tmp_idx:
    #     train_idx = train_idx + i
    
    # # Train or validation dataset?
    # assert subset in ["train", "val"]
    # alldatalist = os.listdir(dataset_dir + '/image')
    # print("Now:", subset, '\n')
    # if subset == "train":
    #     filenames = [alldatalist[i] for i in train_idx]
    # elif subset == "val":
    #     filenames = [alldatalist[i] for i in val_idx]

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
        
        '''
        ann = {MapPixel2RegionId:~~~~, ListRegion:~~~~}
        [0 1 2 3 4 5


        ]
        '''

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

dataset_dir = './data/CutImage' # 依照你的需求修改
loadcross = open('CV_weight/BW_panoptic_cross_data_214.pkl', 'rb')
crossdict = pickle.load(loadcross)
train_list = crossdict['train_idx']
val_list = crossdict['val_idx']
loadcross.close()


part = 3
# iteration = len(train_list[part]) // 2
iteration = len(train_list[part])

def load_fisheye_train():
    return load_fisheye(dataset_dir, train_list[part])

def load_fisheye_val():
    return load_fisheye(dataset_dir, val_list[part])

# 設定資料集
DatasetCatalog.register("BW_train", load_fisheye_train)
# DatasetCatalog.get("BW_train")
DatasetCatalog.register("BW_val", load_fisheye_val)
MetadataCatalog.get("BW_train").set(thing_classes = ["tooth"])
MetadataCatalog.get("BW_val").set(thing_classes = ["tooth"])

# # test dataset code
# import random
# from detectron2.utils.visualizer import Visualizer
# dataset_dicts_train = DatasetCatalog.get("BW_train")
# # dataset_dicts_val = DatasetCatalog.get("BW_val")
# # print('Num_training:', len(dataset_dicts_train))
# # print('Num_val:', len(dataset_dicts_val))
# # print(dataset_dicts[0])
# # for d in dataset_dicts_train/:
# for d in random.sample(dataset_dicts_train, 3):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get("BW_train"), scale=0.5)
#     out = visualizer.draw_dataset_dict(d)
#     cv2.imshow("", out.get_image()[:, :, ::-1])
#     cv2.waitKey(0)
# cv2.destroyAllWindows()


# Model 設定
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("BW_train",)
cfg.DATASETS.TEST = ("BW_val",) 

cfg.DATALOADER.NUM_WORKERS = 2
cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.MAX_ITER = (40 * iteration) // 2 # first part
cfg.SOLVER.WEIGHT_DECAY = 0.0001
cfg.SOLVER.GAMMA = 0.1
cfg.SOLVER.STEPS = [] # lr multiplied by gamma at #iter in STEPS
cfg.SOLVER.CHECKPOINT_PERIOD = (10 * iteration) // 2 # save model every 10 epoches

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)

cfg.MODEL.BACKBONE.FREEZE_AT = 5 # Only head no freeze

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

cfg.TEST.DETECTIONS_PER_IMAGE = iteration // 2

cfg.OUTPUT_DIR = './checkpoints/BWMaskRcnn' # 依照你的需求修改
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


    def build_hooks(self):
        mapper = DatasetMapper(self.cfg, True)
        mapper.tfm_gens = mapper.tfm_gens[0:1] # hack
        log_every_n_seconds(
            logging.INFO,
            f'NEW mapper.tfm_gens = {mapper.tfm_gens}',
            n=0,
        )
        hooks = super().build_hooks()
        hooks.insert(-1, LossEvalHook(
            iteration // 2,   # 1 epoch
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                mapper
            )
        ))
        return hooks

start_time = time.clock()

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = Trainer(cfg)
model = trainer.model
for name, value in model.named_parameters():
    print(name, value.requires_grad)
# trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()


# second part
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
cfg.SOLVER.WARMUP_ITERS = 0
cfg.SOLVER.MAX_ITER = (80 * iteration) // 2
cfg.SOLVER.BASE_LR = 0.0001
cfg.MODEL.BACKBONE.FREEZE_AT = 0 # No Freeze

mjp = os.path.join(cfg.OUTPUT_DIR, 'metrics.json')
os.replace(mjp, mjp + '.old')
# trainer = Trainer(cfg)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.start_iter = (40 * iteration) // 2
trainer.train()










